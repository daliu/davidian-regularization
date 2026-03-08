#!/usr/bin/env python3
"""
Rigorous Davidian Regularization Experiments

Tests the actual question: Does adjusting k-fold CV scores by the train-validation
gap (Davidian Regularization) lead to better held-out test performance when used
for hyperparameter selection?

Protocol per trial:
  1. Split data into train_pool (80%) and test (20%)
  2. For each hyperparameter config, run k-fold CV on train_pool
  3. Record raw mean validation score AND Davidian-adjusted score
  4. Select best hyperparameters according to each criterion
  5. Retrain on full train_pool with selected hyperparameters
  6. Evaluate on held-out test set
  7. Compare test performance across selection methods

Statistical testing: paired Wilcoxon, paired t-test, Cohen's d, 95% CI.

Usage:
  python rigorous_davidian_experiments.py --mode quick   # 5 seeds, 1 dataset
  python rigorous_davidian_experiments.py --mode full    # all datasets, many seeds
  python rigorous_davidian_experiments.py --mode regime  # fine-grained regime investigation
"""

import argparse
import itertools
import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from data_loaders import (
    load_breast_cancer_dataset,
    load_digits_dataset_binary,
    load_synthetic_imbalanced_dataset,
    load_wine_dataset_binary,
)
from davidian_regularization import (
    conservative_davidian_regularization,
    exponential_decay_regularization,
    inverse_difference_regularization,
    original_davidian_regularization,
    stability_bonus_regularization,
)

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Registry of regularization methods
# ---------------------------------------------------------------------------
REGULARIZATION_METHODS: Dict[str, Callable[[float, float], float]] = {
    'standard_kfold': lambda train_score, val_score: val_score,
    'stability_bonus': stability_bonus_regularization,
    'original_davidian': original_davidian_regularization,
    'conservative_davidian': conservative_davidian_regularization,
    'exponential_decay': exponential_decay_regularization,
    'inverse_difference': inverse_difference_regularization,
}

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TrialResult:
    seed: int
    dataset: str
    model_name: str
    method: str
    selected_params: str
    test_auc: float
    test_accuracy: float


@dataclass
class ExperimentConfig:
    dataset_name: str
    load_fn: Callable
    load_kwargs: Dict[str, Any]
    model_configs: List[Dict[str, Any]]
    n_seeds: int
    methods: List[str]


# ---------------------------------------------------------------------------
# Core evaluation logic
# ---------------------------------------------------------------------------

def kfold_cv_scores(
    X: np.ndarray,
    y: np.ndarray,
    model_class: type,
    model_params: Dict[str, Any],
    k_folds: int = 5,
    random_state: int = 42,
) -> Tuple[float, float]:
    """Run k-fold CV returning (mean_train_score, mean_val_score) as AUC."""
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=random_state)
    train_scores = []
    val_scores = []

    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model = model_class(**model_params)
        model.fit(X_tr, y_tr)

        # Use AUC when possible, fall back to accuracy
        if hasattr(model, 'predict_proba'):
            try:
                tr_proba = model.predict_proba(X_tr)[:, 1]
                val_proba = model.predict_proba(X_val)[:, 1]
                tr_score = roc_auc_score(y_tr, tr_proba)
                val_score = roc_auc_score(y_val, val_proba)
            except Exception:
                tr_score = accuracy_score(y_tr, model.predict(X_tr))
                val_score = accuracy_score(y_val, model.predict(X_val))
        else:
            tr_score = accuracy_score(y_tr, model.predict(X_tr))
            val_score = accuracy_score(y_val, model.predict(X_val))

        train_scores.append(tr_score)
        val_scores.append(val_score)

    return float(np.mean(train_scores)), float(np.mean(val_scores))


def compute_cv_grid(
    X_pool: np.ndarray,
    y_pool: np.ndarray,
    model_class: type,
    base_params: Dict[str, Any],
    param_grid: Dict[str, List[Any]],
    k_folds: int = 5,
    cv_seed: int = 42,
) -> List[Tuple[Dict[str, Any], float, float]]:
    """Compute k-fold CV scores for every hyperparameter config (once).

    Returns list of (config_dict, mean_train_score, mean_val_score).
    """
    param_names = list(param_grid.keys())
    param_combos = list(itertools.product(*param_grid.values()))
    grid_scores = []

    for combo in param_combos:
        config = dict(zip(param_names, combo))
        full_params = {**base_params, **config}
        mean_train, mean_val = kfold_cv_scores(
            X_pool, y_pool, model_class, full_params,
            k_folds=k_folds, random_state=cv_seed,
        )
        grid_scores.append((config, mean_train, mean_val))

    return grid_scores


def select_from_grid(
    grid_scores: List[Tuple[Dict[str, Any], float, float]],
    method_name: str,
) -> Dict[str, Any]:
    """Pick best hyperparameters from pre-computed CV grid using a method."""
    reg_fn = REGULARIZATION_METHODS[method_name]
    best_score = -np.inf
    best_config = None

    for config, mean_train, mean_val in grid_scores:
        adjusted = reg_fn(mean_train, mean_val)
        if adjusted > best_score:
            best_score = adjusted
            best_config = config

    return best_config


def run_single_trial(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    dataset_name: str,
    model_class: type,
    model_name: str,
    base_params: Dict[str, Any],
    param_grid: Dict[str, List[Any]],
    methods: List[str],
    k_folds: int = 5,
    test_size: float = 0.2,
) -> List[TrialResult]:
    """Run one trial: split, CV-score grid once, select per method, evaluate."""
    # Stratified train/test split
    X_pool, X_test, y_pool, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed,
    )

    # Compute CV scores for the full hyperparameter grid ONCE
    grid_scores = compute_cv_grid(
        X_pool, y_pool, model_class, base_params, param_grid,
        k_folds=k_folds, cv_seed=seed,
    )

    # Cache: retrain + evaluate only for distinct selected configs
    eval_cache: Dict[str, Tuple[float, float]] = {}

    results = []
    for method in methods:
        best_config = select_from_grid(grid_scores, method)
        config_key = str(best_config)

        if config_key not in eval_cache:
            full_params = {**base_params, **best_config}
            final_model = model_class(**full_params)
            final_model.fit(X_pool, y_pool)

            test_acc = accuracy_score(y_test, final_model.predict(X_test))
            try:
                test_proba = final_model.predict_proba(X_test)[:, 1]
                test_auc = roc_auc_score(y_test, test_proba)
            except Exception:
                test_auc = float('nan')

            eval_cache[config_key] = (test_auc, test_acc)

        test_auc, test_acc = eval_cache[config_key]

        results.append(TrialResult(
            seed=seed,
            dataset=dataset_name,
            model_name=model_name,
            method=method,
            selected_params=config_key,
            test_auc=test_auc,
            test_accuracy=test_acc,
        ))

    return results


# ---------------------------------------------------------------------------
# Statistical analysis
# ---------------------------------------------------------------------------

def compute_statistics(
    baseline_scores: np.ndarray,
    method_scores: np.ndarray,
    method_name: str,
    metric_name: str = 'test_auc',
) -> Dict[str, Any]:
    """Compute paired statistical tests between baseline and a method."""
    diffs = method_scores - baseline_scores
    n = len(diffs)

    mean_diff = float(np.mean(diffs))
    se_diff = float(np.std(diffs, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    ci_95 = (mean_diff - 1.96 * se_diff, mean_diff + 1.96 * se_diff)

    # Cohen's d (paired)
    sd_diff = float(np.std(diffs, ddof=1)) if n > 1 else 0.0
    cohens_d = mean_diff / sd_diff if sd_diff > 0 else 0.0

    # Paired t-test
    if n > 1 and sd_diff > 0:
        t_stat, t_pval = stats.ttest_rel(method_scores, baseline_scores)
    else:
        t_stat, t_pval = 0.0, 1.0

    # Wilcoxon signed-rank test (needs non-zero differences)
    nonzero_diffs = diffs[diffs != 0]
    if len(nonzero_diffs) >= 10:
        try:
            w_stat, w_pval = stats.wilcoxon(nonzero_diffs)
        except Exception:
            w_stat, w_pval = float('nan'), float('nan')
    else:
        w_stat, w_pval = float('nan'), float('nan')

    # Wins / ties / losses
    wins = int(np.sum(diffs > 0))
    ties = int(np.sum(diffs == 0))
    losses = int(np.sum(diffs < 0))

    return {
        'method': method_name,
        'metric': metric_name,
        'n_trials': n,
        'baseline_mean': float(np.mean(baseline_scores)),
        'method_mean': float(np.mean(method_scores)),
        'mean_diff': mean_diff,
        'se_diff': se_diff,
        'ci_95_low': ci_95[0],
        'ci_95_high': ci_95[1],
        'cohens_d': cohens_d,
        'ttest_stat': float(t_stat),
        'ttest_pval': float(t_pval),
        'wilcoxon_stat': float(w_stat),
        'wilcoxon_pval': float(w_pval),
        'wins': wins,
        'ties': ties,
        'losses': losses,
    }


def verdict(stat_row: Dict[str, Any]) -> str:
    """Return a human-readable verdict string."""
    p = stat_row['ttest_pval']
    d = stat_row['mean_diff']
    if stat_row['ties'] == stat_row['n_trials']:
        return 'IDENTICAL (same hyperparameters selected every time)'
    if p < 0.01:
        direction = 'HELPS' if d > 0 else 'HURTS'
        return f'{direction} (p={p:.4f}, d={stat_row["cohens_d"]:.3f})'
    if p < 0.05:
        direction = 'HELPS (weak)' if d > 0 else 'HURTS (weak)'
        return f'{direction} (p={p:.4f}, d={stat_row["cohens_d"]:.3f})'
    return f'NO SIGNIFICANT EFFECT (p={p:.4f})'


# ---------------------------------------------------------------------------
# Dataset definitions
# ---------------------------------------------------------------------------

def build_synthetic_configs(
    n_seeds: int,
    methods: List[str],
) -> List[ExperimentConfig]:
    """Build experiment configs for synthetic datasets."""
    configs = []
    for n_samples in [500, 2000, 5000]:
        for imbalance in [1.0, 5.0, 10.0, 20.0]:
            name = f'synthetic_n{n_samples}_imb{imbalance:.0f}'
            configs.append(ExperimentConfig(
                dataset_name=name,
                load_fn=load_synthetic_imbalanced_dataset,
                load_kwargs={
                    'total_samples': n_samples,
                    'imbalance_ratio': imbalance,
                    'number_of_features': 20,
                    'informative_features': 15,
                    'redundant_features': 3,
                    # random_state is set per-trial below
                },
                model_configs=[
                    {
                        'model_class': LogisticRegression,
                        'model_name': 'LogisticRegression',
                        'base_params': {'max_iter': 1000, 'solver': 'lbfgs'},
                        'param_grid': {'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]},
                    },
                    {
                        'model_class': GradientBoostingClassifier,
                        'model_name': 'GradientBoosting',
                        'base_params': {'n_estimators': 50},
                        'param_grid': {
                            'learning_rate': [0.01, 0.1, 0.3],
                            'max_depth': [2, 5],
                        },
                    },
                ],
                n_seeds=n_seeds,
                methods=methods,
            ))
    return configs


def build_real_world_configs(
    n_seeds: int,
    methods: List[str],
) -> List[ExperimentConfig]:
    """Build experiment configs for real-world datasets."""
    real_datasets = [
        ('breast_cancer', load_breast_cancer_dataset, {}),
        ('digits_binary', load_digits_dataset_binary, {}),
        ('wine_binary', load_wine_dataset_binary, {}),
    ]
    configs = []
    for ds_name, load_fn, kwargs in real_datasets:
        configs.append(ExperimentConfig(
            dataset_name=ds_name,
            load_fn=load_fn,
            load_kwargs=kwargs,
            model_configs=[
                {
                    'model_class': LogisticRegression,
                    'model_name': 'LogisticRegression',
                    'base_params': {'max_iter': 1000, 'solver': 'lbfgs'},
                    'param_grid': {'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]},
                },
                {
                    'model_class': GradientBoostingClassifier,
                    'model_name': 'GradientBoosting',
                    'base_params': {'n_estimators': 50},
                    'param_grid': {
                        'learning_rate': [0.01, 0.1, 0.3],
                        'max_depth': [2, 5],
                    },
                },
            ],
            n_seeds=n_seeds,
            methods=methods,
        ))
    return configs


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run_experiment_group(
    configs: List[ExperimentConfig],
    k_folds: int = 5,
    test_size: float = 0.2,
) -> pd.DataFrame:
    """Run a group of experiments, return DataFrame of all trial results."""
    all_results: List[TrialResult] = []
    total_configs = sum(len(c.model_configs) for c in configs)
    done = 0

    for cfg in configs:
        # Load dataset once (synthetic datasets get a fixed seed for generation,
        # the trial seed controls the train/test split only).
        if 'random_state' in cfg.load_kwargs:
            X, y, meta = cfg.load_fn(**cfg.load_kwargs)
        else:
            X, y, meta = cfg.load_fn(**cfg.load_kwargs)

        for mc in cfg.model_configs:
            done += 1
            label = f'[{done}/{total_configs}] {cfg.dataset_name} / {mc["model_name"]}'
            print(f'{label}: running {cfg.n_seeds} seeds ...', flush=True)
            t0 = time.time()

            for seed in range(cfg.n_seeds):
                trial_results = run_single_trial(
                    X, y,
                    seed=seed,
                    dataset_name=cfg.dataset_name,
                    model_class=mc['model_class'],
                    model_name=mc['model_name'],
                    base_params=mc['base_params'],
                    param_grid=mc['param_grid'],
                    methods=cfg.methods,
                    k_folds=k_folds,
                    test_size=test_size,
                )
                all_results.extend(trial_results)

            elapsed = time.time() - t0
            print(f'  done in {elapsed:.1f}s', flush=True)

    df = pd.DataFrame([
        {
            'seed': r.seed,
            'dataset': r.dataset,
            'model': r.model_name,
            'method': r.method,
            'selected_params': r.selected_params,
            'test_auc': r.test_auc,
            'test_accuracy': r.test_accuracy,
        }
        for r in all_results
    ])
    return df


def analyze_results(df: pd.DataFrame, baseline: str = 'standard_kfold') -> pd.DataFrame:
    """Produce statistical summary comparing each method to the baseline."""
    summary_rows = []

    for (dataset, model), group in df.groupby(['dataset', 'model']):
        baseline_df = group[group['method'] == baseline]
        if baseline_df.empty:
            continue
        baseline_auc = baseline_df.sort_values('seed')['test_auc'].values
        baseline_acc = baseline_df.sort_values('seed')['test_accuracy'].values

        for method in sorted(group['method'].unique()):
            if method == baseline:
                continue
            method_df = group[group['method'] == method]
            method_auc = method_df.sort_values('seed')['test_auc'].values
            method_acc = method_df.sort_values('seed')['test_accuracy'].values

            if len(method_auc) != len(baseline_auc):
                continue

            # Compute for AUC
            auc_stats = compute_statistics(baseline_auc, method_auc, method, 'test_auc')
            auc_stats['dataset'] = dataset
            auc_stats['model'] = model
            auc_stats['verdict'] = verdict(auc_stats)

            # Compute for accuracy
            acc_stats = compute_statistics(baseline_acc, method_acc, method, 'test_accuracy')
            acc_stats['dataset'] = dataset
            acc_stats['model'] = model
            acc_stats['verdict'] = verdict(acc_stats)

            # How often do methods pick different hyperparameters?
            baseline_params = baseline_df.sort_values('seed')['selected_params'].values
            method_params = method_df.sort_values('seed')['selected_params'].values
            n_different = int(np.sum(baseline_params != method_params))
            auc_stats['n_different_selections'] = n_different
            acc_stats['n_different_selections'] = n_different

            summary_rows.append(auc_stats)
            summary_rows.append(acc_stats)

    return pd.DataFrame(summary_rows)


def print_summary(summary_df: pd.DataFrame) -> None:
    """Print a readable summary table."""
    if summary_df.empty:
        print('No results to summarize.')
        return

    print('\n' + '=' * 100)
    print('RIGOROUS DAVIDIAN REGULARIZATION EXPERIMENT RESULTS')
    print('=' * 100)

    for (dataset, model), group in summary_df.groupby(['dataset', 'model']):
        print(f'\n--- {dataset} / {model} ---')
        auc_rows = group[group['metric'] == 'test_auc']
        for _, row in auc_rows.iterrows():
            diff_pct = row['n_different_selections']
            n = row['n_trials']
            print(
                f"  {row['method']:30s} | "
                f"AUC: {row['method_mean']:.4f} vs baseline {row['baseline_mean']:.4f} | "
                f"diff={row['mean_diff']:+.5f} | "
                f"95%CI=[{row['ci_95_low']:+.5f}, {row['ci_95_high']:+.5f}] | "
                f"p(t)={row['ttest_pval']:.4f} p(W)={row['wilcoxon_pval']:.4f} | "
                f"d={row['cohens_d']:.3f} | "
                f"W/T/L={row['wins']}/{row['ties']}/{row['losses']} | "
                f"diffSel={diff_pct}/{n} | "
                f"{row['verdict']}"
            )

    # Overall summary across all datasets
    print('\n' + '=' * 100)
    print('OVERALL SUMMARY (AUC metric)')
    print('=' * 100)
    auc_summary = summary_df[summary_df['metric'] == 'test_auc']
    for method, method_group in auc_summary.groupby('method'):
        sig_helps = sum(
            (r['ttest_pval'] < 0.05) and (r['mean_diff'] > 0)
            for _, r in method_group.iterrows()
        )
        sig_hurts = sum(
            (r['ttest_pval'] < 0.05) and (r['mean_diff'] < 0)
            for _, r in method_group.iterrows()
        )
        no_effect = len(method_group) - sig_helps - sig_hurts
        avg_diff = method_group['mean_diff'].mean()
        print(
            f"  {method:30s}: "
            f"avg_diff={avg_diff:+.5f} | "
            f"sig_helps={sig_helps} sig_hurts={sig_hurts} no_effect={no_effect} "
            f"(across {len(method_group)} dataset/model combos)"
        )


def sanity_check(df: pd.DataFrame) -> None:
    """When methods select the same hyperparameters, test scores must be identical."""
    baseline = 'standard_kfold'
    issues = 0
    checked = 0

    for (dataset, model, seed), group in df.groupby(['dataset', 'model', 'seed']):
        bl = group[group['method'] == baseline]
        if bl.empty:
            continue
        bl_params = bl.iloc[0]['selected_params']
        bl_auc = bl.iloc[0]['test_auc']

        for _, row in group.iterrows():
            if row['method'] == baseline:
                continue
            if row['selected_params'] == bl_params:
                checked += 1
                if abs(row['test_auc'] - bl_auc) > 1e-10:
                    issues += 1
                    print(
                        f"  SANITY FAIL: {dataset}/{model}/seed={seed}: "
                        f"{row['method']} same params but AUC differs "
                        f"({row['test_auc']:.6f} vs {bl_auc:.6f})"
                    )

    print(f'Sanity check: {checked} same-param pairs checked, {issues} issues found.')


# ---------------------------------------------------------------------------
# Regime investigation
# ---------------------------------------------------------------------------

def build_regime_configs(
    n_seeds: int = 100,
) -> List[ExperimentConfig]:
    """Build fine-grained grid to map where stability_bonus helps."""
    sample_sizes = [500, 1000, 1500, 2000, 3000, 5000]
    imbalance_ratios = [1, 5, 10, 15, 20, 30, 50]
    methods = ['standard_kfold', 'stability_bonus']

    configs = []
    for n_samples in sample_sizes:
        for imbalance in imbalance_ratios:
            name = f'synthetic_n{n_samples}_imb{imbalance}'
            configs.append(ExperimentConfig(
                dataset_name=name,
                load_fn=load_synthetic_imbalanced_dataset,
                load_kwargs={
                    'total_samples': n_samples,
                    'imbalance_ratio': float(imbalance),
                    'number_of_features': 20,
                    'informative_features': 15,
                    'redundant_features': 3,
                },
                model_configs=[{
                    'model_class': LogisticRegression,
                    'model_name': 'LogisticRegression',
                    'base_params': {'max_iter': 1000, 'solver': 'lbfgs'},
                    'param_grid': {'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]},
                }],
                n_seeds=n_seeds,
                methods=methods,
            ))
    return configs


def print_regime_heatmap(summary_df: pd.DataFrame) -> None:
    """Print a text heatmap: rows=sample_size, cols=imbalance_ratio."""
    auc_rows = summary_df[
        (summary_df['metric'] == 'test_auc')
        & (summary_df['method'] == 'stability_bonus')
    ].copy()

    if auc_rows.empty:
        print('No regime results to display.')
        return

    # Parse sample size and imbalance from dataset name
    def parse_name(name):
        # e.g. 'synthetic_n2000_imb20'
        parts = name.split('_')
        n = int(parts[1][1:])   # strip 'n'
        imb = int(parts[2][3:]) # strip 'imb'
        return n, imb

    auc_rows[['n_samples', 'imbalance']] = auc_rows['dataset'].apply(
        lambda x: pd.Series(parse_name(x))
    )

    sample_sizes = sorted(auc_rows['n_samples'].unique())
    imbalances = sorted(auc_rows['imbalance'].unique())

    # Build lookup
    lookup = {}
    for _, row in auc_rows.iterrows():
        lookup[(row['n_samples'], row['imbalance'])] = row

    # Header
    print('\n' + '=' * 100)
    print('REGIME HEATMAP: stability_bonus vs standard_kfold (AUC)')
    print('Rows = sample size, Cols = imbalance ratio')
    print('=' * 100)

    col_width = 18
    header = f'{"n \\\\ imb":>8s}'
    for imb in imbalances:
        header += f'{imb:>{col_width}d}'
    print(header)
    print('-' * (8 + col_width * len(imbalances)))

    for n in sample_sizes:
        row_str = f'{n:>8d}'
        for imb in imbalances:
            entry = lookup.get((n, imb))
            if entry is None:
                row_str += f'{"---":>{col_width}s}'
            else:
                diff = entry['mean_diff']
                p = entry['ttest_pval']
                if entry['ties'] == entry['n_trials']:
                    tag = '='
                elif p < 0.01:
                    tag = 'HELPS' if diff > 0 else 'HURTS'
                elif p < 0.05:
                    tag = 'helps' if diff > 0 else 'hurts'
                else:
                    tag = 'ns'
                cell = f'{diff:+.4f} {tag}'
                row_str += f'{cell:>{col_width}s}'
        print(row_str)

    # Legend
    print()
    print('Legend: HELPS/HURTS = p<0.01, helps/hurts = p<0.05, ns = not significant, = = identical selections')
    print()


def run_regime_investigation(output_dir: str) -> None:
    """Run the fine-grained regime investigation."""
    print('=== REGIME INVESTIGATION ===')
    print('Fine-grained grid: 6 sample sizes x 7 imbalance ratios x 100 seeds')
    print('LogisticRegression only, stability_bonus vs standard_kfold\n')

    configs = build_regime_configs(n_seeds=100)
    print(f'{len(configs)} dataset configs, {len(configs) * 100} total trials\n')

    df = run_experiment_group(configs)

    # Save trials
    trials_path = os.path.join(output_dir, 'trials_regime.csv')
    df.to_csv(trials_path, index=False)
    print(f'\nPer-trial results saved to {trials_path}')

    # Sanity check
    print('\n--- Sanity Check ---')
    sanity_check(df)

    # Statistical analysis
    summary_df = analyze_results(df)
    summary_path = os.path.join(output_dir, 'summary_regime.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f'Summary statistics saved to {summary_path}')

    # Print detailed summary
    print_summary(summary_df)

    # Print heatmap
    print_regime_heatmap(summary_df)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Rigorous Davidian Regularization Experiments',
    )
    parser.add_argument(
        '--mode', choices=['quick', 'full', 'regime'], default='quick',
        help='quick = 5 seeds, 1 dataset; full = all datasets, many seeds; regime = fine-grained regime investigation',
    )
    parser.add_argument(
        '--group', choices=['all', 'synthetic', 'real'], default='all',
        help='Which experiment group to run (default: all)',
    )
    parser.add_argument(
        '--output-dir', default=None,
        help='Directory to save CSV results (default: publication/data/rigorous_results)',
    )
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(
        os.path.dirname(__file__), '..', 'data', 'rigorous_results',
    )
    os.makedirs(output_dir, exist_ok=True)

    if args.mode == 'regime':
        run_regime_investigation(output_dir)
        return

    methods = list(REGULARIZATION_METHODS.keys())

    if args.mode == 'quick':
        print('=== QUICK MODE: 5 seeds, breast_cancer only ===\n')
        configs = [ExperimentConfig(
            dataset_name='breast_cancer',
            load_fn=load_breast_cancer_dataset,
            load_kwargs={},
            model_configs=[{
                'model_class': LogisticRegression,
                'model_name': 'LogisticRegression',
                'base_params': {'max_iter': 1000, 'solver': 'lbfgs'},
                'param_grid': {'C': [0.01, 0.1, 1.0, 10.0, 100.0]},
            }],
            n_seeds=5,
            methods=methods,
        )]
    else:
        print(f'=== FULL MODE (group={args.group}) ===\n')
        synthetic = build_synthetic_configs(n_seeds=100, methods=methods)
        real_world = build_real_world_configs(n_seeds=50, methods=methods)
        if args.group == 'synthetic':
            configs = synthetic
        elif args.group == 'real':
            configs = real_world
        else:
            configs = synthetic + real_world

    # Run experiments
    df = run_experiment_group(configs)

    # Save per-trial results
    suffix = args.mode if args.group == 'all' else f'{args.mode}_{args.group}'
    trials_path = os.path.join(output_dir, f'trials_{suffix}.csv')
    df.to_csv(trials_path, index=False)
    print(f'\nPer-trial results saved to {trials_path}')

    # Sanity check
    print('\n--- Sanity Check ---')
    sanity_check(df)

    # Statistical analysis
    summary_df = analyze_results(df)
    summary_path = os.path.join(output_dir, f'summary_{suffix}.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f'Summary statistics saved to {summary_path}')

    # Print results
    print_summary(summary_df)


if __name__ == '__main__':
    main()
