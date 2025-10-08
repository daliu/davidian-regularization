#!/usr/bin/env python3
"""
Comprehensive AUC-focused experiment for Davidian Regularization.

This experiment:
1. Uses AUC as PRIMARY metric for model selection and evaluation
2. Tests extreme imbalance ratios: 29:1, 39:1, 49:1 (replacing 4:1)
3. Compares multiple Davidian methods: original, conservative, inverse_diff, exponential_decay, stability_bonus
4. Provides clear labeling of all metrics and percentages
5. Runs independent trials per condition with max AUC selection
"""

import sys
import os
import time
import json
import numpy as np
from collections import Counter
from typing import Dict, List, Tuple, Any

# Force single-threaded execution
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

print("COMPREHENSIVE AUC-FOCUSED EXPERIMENT")
print("="*70)
print("PRIMARY METRIC: AUC (Area Under ROC Curve)")
print("SUPPLEMENTAL METRICS: F1-score, Precision, Recall")
print("="*70)

def create_extreme_imbalanced_dataset(n_samples: int, imbalance_ratio: float, 
                                    n_features: int, random_seed: int = 42):
    """Create extremely imbalanced dataset without class reweighting."""
    from sklearn.datasets import make_classification
    
    # Calculate class weights from ratio
    minority_weight = 1.0 / (imbalance_ratio + 1.0)
    majority_weight = 1.0 - minority_weight
    
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=max(2, n_features - 3),
        n_redundant=min(3, n_features // 5),
        n_classes=2,
        weights=[majority_weight, minority_weight],
        class_sep=0.8,  # Clear separation for AUC measurement
        flip_y=0.005,   # Very low noise for clean AUC curves
        random_state=random_seed
    )
    
    actual_counts = dict(Counter(y))
    actual_ratio = actual_counts[0] / actual_counts[1] if actual_counts[1] > 0 else float('inf')
    
    return X, y, {
        'n_samples': n_samples,
        'n_features': n_features,
        'target_imbalance_ratio': imbalance_ratio,
        'actual_imbalance_ratio': actual_ratio,
        'minority_count': actual_counts[1],
        'majority_count': actual_counts[0],
        'minority_percentage': (actual_counts[1] / n_samples) * 100
    }

def create_auc_optimized_models():
    """Create models optimized for AUC performance on imbalanced data."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.svm import SVC
    
    return [
        # Logistic Regression variants
        {'name': 'LogReg_Balanced', 'class': LogisticRegression, 
         'params': {'class_weight': 'balanced', 'random_state': 42, 'max_iter': 2000, 'n_jobs': 1}},
        {'name': 'LogReg_Unbalanced', 'class': LogisticRegression,
         'params': {'class_weight': None, 'random_state': 43, 'max_iter': 2000, 'n_jobs': 1}},
        {'name': 'LogReg_L1_Balanced', 'class': LogisticRegression,
         'params': {'penalty': 'l1', 'solver': 'liblinear', 'class_weight': 'balanced', 'random_state': 44, 'n_jobs': 1}},
        
        # Random Forest variants
        {'name': 'RF_Balanced', 'class': RandomForestClassifier,
         'params': {'class_weight': 'balanced', 'n_estimators': 100, 'random_state': 42, 'n_jobs': 1}},
        {'name': 'RF_Unbalanced', 'class': RandomForestClassifier,
         'params': {'class_weight': None, 'n_estimators': 100, 'random_state': 43, 'n_jobs': 1}},
        {'name': 'RF_Balanced_Deep', 'class': RandomForestClassifier,
         'params': {'class_weight': 'balanced', 'n_estimators': 50, 'max_depth': None, 'random_state': 44, 'n_jobs': 1}},
        
        # Gradient Boosting
        {'name': 'GradBoost_Default', 'class': GradientBoostingClassifier,
         'params': {'n_estimators': 100, 'learning_rate': 0.1, 'random_state': 42}},
        {'name': 'GradBoost_Conservative', 'class': GradientBoostingClassifier,
         'params': {'n_estimators': 50, 'learning_rate': 0.05, 'max_depth': 3, 'random_state': 43}},
        
        # SVM (for smaller datasets)
        {'name': 'SVM_Balanced', 'class': SVC,
         'params': {'class_weight': 'balanced', 'probability': True, 'random_state': 42}},
        
        # Naive Bayes
        {'name': 'NaiveBayes', 'class': GaussianNB, 'params': {}}
    ]

def apply_davidian_selection_methods(train_scores: List[float], val_scores: List[float]) -> Dict[str, List[float]]:
    """
    Apply all Davidian selection methods.
    
    Returns:
        Dictionary mapping method names to selection scores
    """
    selection_scores = {}
    
    # 1. Original Davidian
    original_scores = []
    for train_score, val_score in zip(train_scores, val_scores):
        penalty = abs(train_score - val_score)
        regularized_score = val_score - penalty
        original_scores.append(regularized_score)
    selection_scores['original_davidian'] = original_scores
    
    # 2. Conservative Davidian (0.5 penalty weight)
    conservative_scores = []
    for train_score, val_score in zip(train_scores, val_scores):
        penalty = 0.5 * abs(train_score - val_score)
        regularized_score = val_score - penalty
        conservative_scores.append(regularized_score)
    selection_scores['conservative_davidian'] = conservative_scores
    
    # 3. Inverse Difference (confidence-based)
    inverse_diff_scores = []
    for train_score, val_score in zip(train_scores, val_scores):
        diff = abs(train_score - val_score)
        confidence = 1.0 / (1.0 + diff)
        confidence_score = val_score * confidence
        inverse_diff_scores.append(confidence_score)
    selection_scores['inverse_diff'] = inverse_diff_scores
    
    # 4. Exponential Decay
    exponential_decay_scores = []
    for train_score, val_score in zip(train_scores, val_scores):
        diff = abs(train_score - val_score)
        confidence = np.exp(-diff)
        confidence_score = val_score * confidence
        exponential_decay_scores.append(confidence_score)
    selection_scores['exponential_decay'] = exponential_decay_scores
    
    # 5. Stability Bonus
    stability_bonus_scores = []
    threshold = 0.1
    for train_score, val_score in zip(train_scores, val_scores):
        diff = abs(train_score - val_score)
        if diff < threshold:
            bonus = (threshold - diff) / threshold
            stability_score = val_score * (1.0 + bonus)
        else:
            stability_score = val_score
        stability_bonus_scores.append(stability_score)
    selection_scores['stability_bonus'] = stability_bonus_scores
    
    # 6. Control (regular validation AUC)
    selection_scores['control'] = val_scores.copy()
    
    return selection_scores

def run_auc_focused_trial(X, y, trial_seed: int):
    """
    Run a single trial focused on AUC as primary metric.
    
    Returns:
        Dictionary with results for all selection methods
    """
    from sklearn.model_selection import train_test_split, StratifiedKFold
    from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
    from sklearn.preprocessing import StandardScaler
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=trial_seed, stratify=y
    )
    
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = create_auc_optimized_models()
    
    # K-fold CV (K=5) using AUC as primary metric
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=trial_seed)
    
    model_results = []
    
    for model_config in models:
        try:
            # K-fold CV for train and validation AUC scores
            fold_train_aucs = []
            fold_val_aucs = []
            
            for train_idx, val_idx in cv.split(X_train_scaled, y_train):
                X_fold_train, X_fold_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
                y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
                
                model = model_config['class'](**model_config['params'])
                model.fit(X_fold_train, y_fold_train)
                
                # Get probabilities for AUC calculation
                if hasattr(model, 'predict_proba'):
                    train_proba = model.predict_proba(X_fold_train)[:, 1]
                    val_proba = model.predict_proba(X_fold_val)[:, 1]
                elif hasattr(model, 'decision_function'):
                    train_proba = model.decision_function(X_fold_train)
                    val_proba = model.decision_function(X_fold_val)
                else:
                    # Skip models without probability estimates
                    continue
                
                # Calculate AUC
                train_auc = roc_auc_score(y_fold_train, train_proba)
                val_auc = roc_auc_score(y_fold_val, val_proba)
                
                fold_train_aucs.append(train_auc)
                fold_val_aucs.append(val_auc)
            
            if not fold_train_aucs:  # Skip if no valid folds
                continue
                
            mean_train_auc = np.mean(fold_train_aucs)
            mean_val_auc = np.mean(fold_val_aucs)
            
            # Final model evaluation on test set
            final_model = model_config['class'](**model_config['params'])
            final_model.fit(X_train_scaled, y_train)
            
            test_pred = final_model.predict(X_test_scaled)
            
            # Get test probabilities for AUC
            if hasattr(final_model, 'predict_proba'):
                test_proba = final_model.predict_proba(X_test_scaled)[:, 1]
            elif hasattr(final_model, 'decision_function'):
                test_proba = final_model.decision_function(X_test_scaled)
            else:
                continue
            
            # Calculate comprehensive test metrics
            test_auc = roc_auc_score(y_test, test_proba)
            test_f1 = f1_score(y_test, test_pred, average='binary')
            test_precision = precision_score(y_test, test_pred, average='binary', zero_division=0)
            test_recall = recall_score(y_test, test_pred, average='binary')
            test_accuracy = accuracy_score(y_test, test_pred)
            
            model_results.append({
                'model_name': model_config['name'],
                'cv_train_auc': mean_train_auc,
                'cv_val_auc': mean_val_auc,
                'train_val_auc_diff': abs(mean_train_auc - mean_val_auc),
                'test_auc': test_auc,
                'test_f1': test_f1,
                'test_precision': test_precision,
                'test_recall': test_recall,
                'test_accuracy': test_accuracy
            })
            
        except Exception as e:
            continue
    
    if len(model_results) < 2:
        return None
    
    # Apply all selection methods using AUC scores
    train_aucs = [m['cv_train_auc'] for m in model_results]
    val_aucs = [m['cv_val_auc'] for m in model_results]
    test_aucs = [m['test_auc'] for m in model_results]
    
    selection_scores = apply_davidian_selection_methods(train_aucs, val_aucs)
    
    # Select best model for each method and get their test AUC
    method_results = {}
    
    for method_name, scores in selection_scores.items():
        best_idx = np.argmax(scores)
        best_model = model_results[best_idx]
        
        method_results[method_name] = {
            'selected_model': best_model['model_name'],
            'selection_score': scores[best_idx],
            'test_auc': best_model['test_auc'],
            'test_f1': best_model['test_f1'],
            'test_precision': best_model['test_precision'],
            'test_recall': best_model['test_recall'],
            'test_accuracy': best_model['test_accuracy'],
            'train_val_auc_diff': best_model['train_val_auc_diff']
        }
    
    return {
        'trial_seed': trial_seed,
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'train_class_counts': {int(k): int(v) for k, v in Counter(y_train).items()},
        'test_class_counts': {int(k): int(v) for k, v in Counter(y_test).items()},
        'n_models_evaluated': len(model_results),
        'method_results': method_results
    }

def run_comprehensive_auc_experiment():
    """
    Run the comprehensive 36-item experiment with AUC focus.
    """
    
    # Experimental design
    sample_sizes = [1000, 2000, 5000]  # 3 sample sizes
    imbalance_ratios = [29.0, 39.0, 49.0]  # 3 extreme imbalance ratios (replacing 4:1)
    feature_counts = [10, 15, 20, 25]  # 4 feature counts
    n_trials_per_condition = 10  # Trials per condition
    
    total_conditions = len(sample_sizes) * len(imbalance_ratios) * len(feature_counts)
    
    print(f"\n1. Comprehensive Experimental Design:")
    print(f"   Sample sizes: {sample_sizes}")
    print(f"   Imbalance ratios: {[f'{r:.0f}:1' for r in imbalance_ratios]}")
    print(f"   Feature counts: {feature_counts}")
    print(f"   Trials per condition: {n_trials_per_condition}")
    print(f"   Total conditions: {total_conditions}")
    print(f"   Total trials: {total_conditions * n_trials_per_condition}")
    
    # Create organized output structure
    output_base = 'results/comprehensive_auc'
    os.makedirs(output_base, exist_ok=True)
    
    all_condition_results = []
    condition_counter = 0
    
    for imbalance_ratio in imbalance_ratios:
        for sample_size in sample_sizes:
            for n_features in feature_counts:
                condition_counter += 1
                
                print(f"\n  [{condition_counter}/{total_conditions}] Ratio {imbalance_ratio:.0f}:1, N={sample_size}, Features={n_features}")
                
                # Create dataset for this condition
                X, y, dataset_metadata = create_extreme_imbalanced_dataset(
                    sample_size, imbalance_ratio, n_features, random_seed=42 + condition_counter
                )
                
                print(f"    Actual distribution: {dict(Counter(y))}")
                print(f"    Minority %: {dataset_metadata['minority_percentage']:.2f}%")
                
                # Run trials for this condition
                condition_trials = []
                method_auc_improvements = {
                    'original_davidian': [],
                    'conservative_davidian': [],
                    'inverse_diff': [],
                    'exponential_decay': [],
                    'stability_bonus': []
                }
                
                for trial in range(n_trials_per_condition):
                    trial_seed = np.random.randint(0, 2**31 - 1)
                    trial_result = run_auc_focused_trial(X, y, trial_seed)
                    
                    if trial_result:
                        condition_trials.append(trial_result)
                        
                        # Calculate improvements vs control for each method
                        control_auc = trial_result['method_results']['control']['test_auc']
                        
                        for method_name in method_auc_improvements.keys():
                            method_auc = trial_result['method_results'][method_name]['test_auc']
                            improvement = method_auc - control_auc
                            method_auc_improvements[method_name].append(improvement)
                
                if condition_trials:
                    # Calculate statistics for each method
                    method_statistics = {}
                    
                    for method_name, improvements in method_auc_improvements.items():
                        if improvements:
                            wins = [imp for imp in improvements if imp > 0]
                            losses = [imp for imp in improvements if imp < 0]
                            
                            expected_value = np.mean(improvements)
                            std_dev = np.std(improvements)
                            
                            # Statistical significance
                            from scipy import stats
                            t_stat, p_value = stats.ttest_1samp(improvements, 0)
                            
                            method_statistics[method_name] = {
                                'test_auc_expected_value': float(expected_value),
                                'test_auc_std': float(std_dev),
                                'test_auc_expected_value_pct': float(expected_value * 100),  # CLEAR LABEL: % improvement over control
                                'test_auc_std_pct': float(std_dev * 100),
                                'win_count': int(len(wins)),
                                'loss_count': int(len(losses)),
                                'win_rate_pct': float(len(wins) / len(improvements) * 100),
                                'avg_win_size_pct': float(np.mean(wins) * 100) if wins else 0.0,  # CLEAR LABEL: % AUC improvement when method wins
                                'avg_loss_size_pct': float(np.mean(losses) * 100) if losses else 0.0,  # CLEAR LABEL: % AUC loss when method loses
                                't_statistic': float(t_stat),
                                'p_value': float(p_value) if not np.isnan(p_value) else None,
                                'statistically_significant': bool(p_value < 0.05) if not np.isnan(p_value) else False
                            }
                    
                    condition_result = {
                        'condition_id': condition_counter,
                        'imbalance_ratio': float(imbalance_ratio),
                        'sample_size': int(sample_size),
                        'n_features': int(n_features),
                        'dataset_metadata': {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                                           for k, v in dataset_metadata.items()},
                        'n_successful_trials': int(len(condition_trials)),
                        'method_statistics': method_statistics
                    }
                    
                    all_condition_results.append(condition_result)
                    
                    # Print summary for this condition
                    print(f"    Results ({len(condition_trials)} trials):")
                    
                    # Find best method for this condition
                    best_method = max(method_statistics.items(), 
                                    key=lambda x: x[1]['test_auc_expected_value_pct'])
                    best_method_name, best_stats = best_method
                    
                    print(f"      Best method: {best_method_name}")
                    print(f"        Test AUC Expected Value: {best_stats['test_auc_expected_value_pct']:+.3f}% (improvement over control)")
                    print(f"        Win rate: {best_stats['win_count']}/{len(condition_trials)} ({best_stats['win_rate_pct']:.1f}%)")
                    if best_stats['avg_win_size_pct'] > 0:
                        print(f"        Avg win size: +{best_stats['avg_win_size_pct']:.3f}% AUC improvement")
                    
                    # Show all methods briefly
                    for method_name, stats in method_statistics.items():
                        status = "✅" if stats['test_auc_expected_value_pct'] > 0 else "❌"
                        sig = "*" if stats['statistically_significant'] else ""
                        print(f"        {status} {method_name}: {stats['test_auc_expected_value_pct']:+.3f}% {sig}")
    
    return all_condition_results

def analyze_comprehensive_auc_results(all_results: List[Dict[str, Any]]):
    """Analyze comprehensive AUC results."""
    print(f"\n2. Comprehensive AUC Analysis:")
    print("="*70)
    
    # Method performance summary
    method_names = ['original_davidian', 'conservative_davidian', 'inverse_diff', 
                   'exponential_decay', 'stability_bonus']
    
    print(f"Method Performance Summary (Test AUC Expected Value vs Control):")
    print("-" * 70)
    print(f"{'Method':<20} {'Conditions':<12} {'Avg EV %':<12} {'Win Rate %':<12} {'Best EV %'}")
    print("-" * 70)
    
    method_overall_stats = {}
    
    for method_name in method_names:
        method_evs = []
        method_wins = 0
        total_conditions = 0
        
        for result in all_results:
            if method_name in result['method_statistics']:
                stats = result['method_statistics'][method_name]
                method_evs.append(stats['test_auc_expected_value_pct'])
                method_wins += stats['win_count']
                total_conditions += result['n_successful_trials']
        
        if method_evs:
            avg_ev = np.mean(method_evs)
            best_ev = max(method_evs)
            overall_win_rate = (method_wins / total_conditions) * 100 if total_conditions > 0 else 0
            
            method_overall_stats[method_name] = {
                'avg_expected_value_pct': avg_ev,
                'best_expected_value_pct': best_ev,
                'overall_win_rate_pct': overall_win_rate,
                'conditions_tested': len(method_evs)
            }
            
            print(f"{method_name:<20} {len(method_evs):<12} {avg_ev:+8.3f}% {overall_win_rate:8.1f}% {best_ev:+8.3f}%")
    
    # Find optimal conditions
    print(f"\nOptimal Conditions Analysis:")
    print("-" * 40)
    
    # Find best overall condition
    best_condition = None
    best_overall_ev = -float('inf')
    
    for result in all_results:
        for method_name, stats in result['method_statistics'].items():
            if method_name != 'control' and stats['test_auc_expected_value_pct'] > best_overall_ev:
                best_overall_ev = stats['test_auc_expected_value_pct']
                best_condition = {
                    'method': method_name,
                    'imbalance_ratio': result['imbalance_ratio'],
                    'sample_size': result['sample_size'],
                    'n_features': result['n_features'],
                    'expected_value_pct': stats['test_auc_expected_value_pct'],
                    'win_rate_pct': stats['win_rate_pct'],
                    'p_value': stats['p_value']
                }
    
    if best_condition:
        print(f"Best Overall Configuration:")
        print(f"  Method: {best_condition['method']}")
        print(f"  Imbalance: {best_condition['imbalance_ratio']:.0f}:1")
        print(f"  Sample size: {best_condition['sample_size']}")
        print(f"  Features: {best_condition['n_features']}")
        print(f"  Test AUC Expected Value: {best_condition['expected_value_pct']:+.3f}% (improvement over control)")
        print(f"  Win rate: {best_condition['win_rate_pct']:.1f}%")
        print(f"  Statistical significance: p={best_condition['p_value']:.3f}" if best_condition['p_value'] else "N/A")
    
    # Imbalance ratio analysis
    print(f"\nImbalance Ratio Analysis:")
    print("-" * 30)
    
    by_ratio = {}
    for result in all_results:
        ratio = result['imbalance_ratio']
        if ratio not in by_ratio:
            by_ratio[ratio] = []
        by_ratio[ratio].append(result)
    
    for ratio in sorted(by_ratio.keys()):
        ratio_results = by_ratio[ratio]
        
        # Get best method performance for this ratio
        best_evs = []
        for result in ratio_results:
            best_ev = max(stats['test_auc_expected_value_pct'] 
                         for method_name, stats in result['method_statistics'].items() 
                         if method_name != 'control')
            best_evs.append(best_ev)
        
        avg_best_ev = np.mean(best_evs)
        print(f"  {ratio:.0f}:1 ratio: {avg_best_ev:+.3f}% avg best AUC expected value")

def save_comprehensive_auc_results(all_results: List[Dict[str, Any]]):
    """Save comprehensive AUC results with clear organization."""
    output_base = 'results/comprehensive_auc'
    os.makedirs(output_base, exist_ok=True)
    
    # Save full results
    with open(f'{output_base}/full_results.json', 'w') as f:
        json.dump({
            'experiment_type': 'comprehensive_auc_focused_davidian_regularization',
            'description': 'AUC-focused comparison of multiple Davidian methods on extreme imbalance',
            'primary_metric': 'Test AUC (Area Under ROC Curve)',
            'supplemental_metrics': ['F1-score', 'Precision', 'Recall', 'Accuracy'],
            'methodology': {
                'selection_methods': {
                    'original_davidian': 'val_auc - |train_auc - val_auc|',
                    'conservative_davidian': 'val_auc - 0.5 * |train_auc - val_auc|',
                    'inverse_diff': 'val_auc * (1 / (1 + |train_auc - val_auc|))',
                    'exponential_decay': 'val_auc * exp(-|train_auc - val_auc|)',
                    'stability_bonus': 'val_auc * (1 + bonus) if diff < threshold',
                    'control': 'val_auc (baseline)'
                },
                'evaluation': 'Test AUC performance of selected models',
                'k_folds': 5,
                'percentage_meanings': {
                    'expected_value_pct': 'Average % improvement in test AUC over control method',
                    'win_rate_pct': '% of trials where method outperforms control',
                    'avg_win_size_pct': 'Average % AUC improvement when method wins',
                    'avg_loss_size_pct': 'Average % AUC loss when method loses'
                }
            },
            'results': all_results,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }, f, indent=2)
    
    # Create summary table
    summary_table = []
    for result in all_results:
        for method_name, stats in result['method_statistics'].items():
            if method_name != 'control':
                summary_table.append({
                    'imbalance_ratio': result['imbalance_ratio'],
                    'sample_size': result['sample_size'],
                    'n_features': result['n_features'],
                    'method': method_name,
                    'test_auc_expected_value_pct': stats['test_auc_expected_value_pct'],
                    'test_auc_std_pct': stats['test_auc_std_pct'],
                    'win_rate_pct': stats['win_rate_pct'],
                    'avg_win_size_pct': stats['avg_win_size_pct'],
                    'avg_loss_size_pct': stats['avg_loss_size_pct'],
                    'p_value': stats['p_value'],
                    'statistically_significant': stats['statistically_significant'],
                    'n_trials': result['n_successful_trials']
                })
    
    with open(f'{output_base}/summary_table.json', 'w') as f:
        json.dump(summary_table, f, indent=2)
    
    print(f"\n✅ Comprehensive AUC results saved to:")
    print(f"   - {output_base}/full_results.json")
    print(f"   - {output_base}/summary_table.json")

def main():
    """Run the comprehensive AUC-focused experiment."""
    try:
        start_time = time.time()
        
        print("🔬 COMPREHENSIVE AUC HYPOTHESIS:")
        print("   AUC is the most appropriate metric for imbalanced classification")
        print("   Multiple Davidian methods should be compared systematically")
        print("   Extreme imbalance ratios (29:1, 39:1, 49:1) may show stronger effects")
        
        all_results = run_comprehensive_auc_experiment()
        
        if all_results:
            analyze_comprehensive_auc_results(all_results)
            save_comprehensive_auc_results(all_results)
        else:
            print("❌ No successful experiments completed")
        
        elapsed = time.time() - start_time
        
        print(f"\n{'='*70}")
        print("🎉 COMPREHENSIVE AUC EXPERIMENT COMPLETED!")
        print(f"{'='*70}")
        print(f"Total time: {elapsed:.1f}s")
        print("✅ AUC-focused evaluation implemented")
        print("✅ Multiple Davidian methods compared")
        print("✅ Extreme imbalance ratios tested")
        print("✅ Clear metric labeling provided")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
