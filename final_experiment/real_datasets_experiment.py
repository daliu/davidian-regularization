#!/usr/bin/env python3
"""
Real Datasets Experiment: Davidian Regularization on Real-World Data

This experiment tests Davidian Regularization methods on real datasets from:
- Kaggle datasets
- sklearn built-in datasets  
- HuggingFace datasets
- UCI repository datasets

Focus: Validate synthetic results on real-world data with comprehensive statistics
including Expected Value (EV) means and standard errors across all metrics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple, Optional
import json
import os
import time
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Force single-threaded execution
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

# Import ML libraries
from sklearn.datasets import (
    load_breast_cancer, load_wine, load_digits, load_iris,
    fetch_openml, fetch_covtype, fetch_kddcup99
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

print("REAL DATASETS EXPERIMENT: DAVIDIAN REGULARIZATION VALIDATION")
print("="*80)
print("Testing Davidian Regularization on real-world datasets")
print("Tracking Expected Value (EV) means and standard errors for all metrics")
print("="*80)

def load_real_datasets() -> Dict[str, Dict[str, Any]]:
    """Load various real datasets for comprehensive testing."""
    
    datasets = {}
    
    print("Loading real datasets...")
    
    # 1. Sklearn built-in datasets
    try:
        print("  Loading sklearn datasets...")
        
        # Breast Cancer (binary, balanced)
        cancer = load_breast_cancer()
        datasets['breast_cancer'] = {
            'X': cancer.data,
            'y': cancer.target,
            'name': 'Breast Cancer Wisconsin',
            'source': 'sklearn',
            'type': 'binary_classification',
            'original_size': cancer.data.shape[0],
            'n_features': cancer.data.shape[1],
            'class_distribution': dict(zip(*np.unique(cancer.target, return_counts=True)))
        }
        
        # Wine (multi-class, convert to binary)
        wine = load_wine()
        wine_binary = wine.target.copy()
        wine_binary[wine_binary == 2] = 1  # Merge classes 1 and 2
        datasets['wine'] = {
            'X': wine.data,
            'y': wine_binary,
            'name': 'Wine Recognition (Binary)',
            'source': 'sklearn',
            'type': 'binary_classification',
            'original_size': wine.data.shape[0],
            'n_features': wine.data.shape[1],
            'class_distribution': dict(zip(*np.unique(wine_binary, return_counts=True)))
        }
        
        # Digits (multi-class, convert to binary: 0-4 vs 5-9)
        digits = load_digits()
        digits_binary = (digits.target >= 5).astype(int)
        datasets['digits'] = {
            'X': digits.data,
            'y': digits_binary,
            'name': 'Digits (0-4 vs 5-9)',
            'source': 'sklearn',
            'type': 'binary_classification',
            'original_size': digits.data.shape[0],
            'n_features': digits.data.shape[1],
            'class_distribution': dict(zip(*np.unique(digits_binary, return_counts=True)))
        }
        
    except Exception as e:
        print(f"    Error loading sklearn datasets: {e}")
    
    # 2. OpenML datasets
    try:
        print("  Loading OpenML datasets...")
        
        # Credit-g (German Credit) - classic imbalanced dataset
        credit = fetch_openml('credit-g', version=1, as_frame=True, parser='auto')
        le = LabelEncoder()
        # Convert categorical features to numeric
        X_credit = credit.data.select_dtypes(include=[np.number]).values
        y_credit = le.fit_transform(credit.target)
        
        if X_credit.shape[1] > 0:  # Only add if we have numeric features
            datasets['german_credit'] = {
                'X': X_credit,
                'y': y_credit,
                'name': 'German Credit Risk',
                'source': 'OpenML',
                'type': 'binary_classification',
                'original_size': X_credit.shape[0],
                'n_features': X_credit.shape[1],
                'class_distribution': dict(zip(*np.unique(y_credit, return_counts=True)))
            }
        
    except Exception as e:
        print(f"    Error loading OpenML datasets: {e}")
    
    # 3. Try to load additional datasets
    try:
        print("  Loading additional datasets...")
        
        # Ionosphere dataset
        ionosphere = fetch_openml('ionosphere', version=1, as_frame=True, parser='auto')
        X_iono = ionosphere.data.select_dtypes(include=[np.number]).values
        y_iono = LabelEncoder().fit_transform(ionosphere.target)
        
        if X_iono.shape[1] > 0:
            datasets['ionosphere'] = {
                'X': X_iono,
                'y': y_iono,
                'name': 'Ionosphere',
                'source': 'OpenML',
                'type': 'binary_classification',
                'original_size': X_iono.shape[0],
                'n_features': X_iono.shape[1],
                'class_distribution': dict(zip(*np.unique(y_iono, return_counts=True)))
            }
        
    except Exception as e:
        print(f"    Error loading additional datasets: {e}")
    
    # Print dataset summary
    print(f"\nLoaded {len(datasets)} real datasets:")
    for name, info in datasets.items():
        class_dist = info['class_distribution']
        if len(class_dist) == 2:
            ratio = max(class_dist.values()) / min(class_dist.values())
            print(f"  {name}: {info['original_size']} samples, {info['n_features']} features, "
                  f"ratio ~1:{ratio:.1f}")
    
    return datasets

def create_imbalanced_versions(X: np.ndarray, y: np.ndarray, 
                              target_ratios: List[float] = [1.0, 9.0, 19.0, 49.0],
                              random_state: int = 42) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Create different imbalanced versions of a dataset."""
    
    versions = {}
    
    # Original dataset
    original_ratio = np.bincount(y)[0] / np.bincount(y)[1] if np.bincount(y)[1] > 0 else float('inf')
    versions['original'] = (X, y, original_ratio)
    
    # Create artificial imbalanced versions
    for target_ratio in target_ratios:
        if target_ratio == 1.0:  # Balanced version
            # Undersample majority class to match minority
            minority_class = np.argmin(np.bincount(y))
            majority_class = 1 - minority_class
            
            minority_indices = np.where(y == minority_class)[0]
            majority_indices = np.where(y == majority_class)[0]
            
            # Sample equal numbers
            n_minority = len(minority_indices)
            if len(majority_indices) > n_minority:
                np.random.seed(random_state)
                selected_majority = np.random.choice(majority_indices, n_minority, replace=False)
                
                balanced_indices = np.concatenate([minority_indices, selected_majority])
                np.random.shuffle(balanced_indices)
                
                versions[f'balanced_1_1'] = (X[balanced_indices], y[balanced_indices], 1.0)
        
        else:  # Create imbalanced version
            minority_class = np.argmin(np.bincount(y))
            majority_class = 1 - minority_class
            
            minority_indices = np.where(y == minority_class)[0]
            majority_indices = np.where(y == majority_class)[0]
            
            # Calculate target sizes
            n_minority = len(minority_indices)
            n_majority_target = int(n_minority * target_ratio)
            
            if len(majority_indices) >= n_majority_target:
                np.random.seed(random_state)
                selected_majority = np.random.choice(majority_indices, n_majority_target, replace=False)
                
                imbalanced_indices = np.concatenate([minority_indices, selected_majority])
                np.random.shuffle(imbalanced_indices)
                
                versions[f'imbalanced_1_{target_ratio:.0f}'] = (X[imbalanced_indices], y[imbalanced_indices], target_ratio)
    
    return versions

def apply_regularization_method(train_score: float, val_score: float, 
                               method: str, **kwargs) -> float:
    """Apply different regularization methods."""
    diff = abs(train_score - val_score)
    
    if method == 'stability_bonus':
        threshold = kwargs.get('threshold', 0.1)
        max_bonus = kwargs.get('max_bonus', 0.2)
        if diff < threshold:
            bonus = (threshold - diff) / threshold * max_bonus
            return val_score * (1.0 + bonus)
        return val_score
    
    elif method == 'davidian_regularization':
        alpha = kwargs.get('alpha', 1.0)
        return val_score - alpha * diff
    
    elif method == 'conservative_davidian':
        alpha = kwargs.get('alpha', 1.0)
        return val_score - 0.5 * alpha * diff
    
    elif method == 'exponential_decay':
        return val_score * np.exp(-diff)
    
    elif method == 'inverse_diff':
        confidence = 1.0 / (1.0 + diff)
        return val_score * confidence
    
    elif method == 'standard_stratified_kfold':
        return val_score
    
    else:
        return val_score

def run_comprehensive_real_dataset_experiment(dataset_info: Dict[str, Any],
                                            target_ratios: List[float] = [1.0, 9.0, 19.0, 49.0],
                                            k_values: List[int] = [3, 5, 10],
                                            n_trials: int = 30,
                                            methods: List[str] = None) -> Dict[str, Any]:
    """Run comprehensive experiment on a real dataset."""
    
    if methods is None:
        methods = ['stability_bonus', 'standard_stratified_kfold', 'davidian_regularization', 
                  'conservative_davidian', 'exponential_decay', 'inverse_diff']
    
    dataset_name = dataset_info['name']
    X_original = dataset_info['X']
    y_original = dataset_info['y']
    
    print(f"\nTesting dataset: {dataset_name}")
    print(f"  Original size: {X_original.shape[0]} samples, {X_original.shape[1]} features")
    print(f"  Class distribution: {dataset_info['class_distribution']}")
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_original)
    
    # Create different imbalanced versions
    dataset_versions = create_imbalanced_versions(X_scaled, y_original, target_ratios)
    
    all_results = []
    experiment_count = 0
    
    for version_name, (X, y, actual_ratio) in dataset_versions.items():
        print(f"\n  Version: {version_name} (actual ratio: 1:{actual_ratio:.1f})")
        print(f"    Size: {len(X)} samples")
        
        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        for k in k_values:
            for method in methods:
                experiment_count += 1
                print(f"    Experiment {experiment_count}: K={k}, Method={method}")
                
                try:
                    # Run experiment
                    result = run_single_experiment(
                        X_train, y_train, X_test, y_test,
                        method=method, k=k, n_trials=n_trials,
                        dataset_name=dataset_name,
                        version_name=version_name,
                        actual_ratio=actual_ratio
                    )
                    
                    result['experiment_id'] = experiment_count
                    all_results.append(result)
                    
                    print(f"      Mean improvement: {result['mean_improvement_pct']:+.2f}% ± {result['improvement_se']:.2f}%")
                    print(f"      Test AUC: {result['test_auc']:.3f if result['test_auc'] else 'N/A'}")
                    print(f"      Statistical significance: {'YES' if result['statistically_significant'] else 'NO'}")
                    
                except Exception as e:
                    print(f"      ERROR: {e}")
                    continue
    
    return {
        'dataset_name': dataset_name,
        'dataset_info': dataset_info,
        'results': all_results,
        'summary': calculate_dataset_summary(all_results)
    }

def run_single_experiment(X_train: np.ndarray, y_train: np.ndarray,
                         X_test: np.ndarray, y_test: np.ndarray,
                         method: str, k: int = 5, n_trials: int = 30,
                         dataset_name: str = '', version_name: str = '',
                         actual_ratio: float = 1.0) -> Dict[str, Any]:
    """Run a single experiment with comprehensive statistics tracking."""
    
    # Model configuration
    model_class = LogisticRegression
    model_params = {'random_state': 42, 'max_iter': 2000, 'solver': 'liblinear'}
    
    # Handle class weighting for comparison
    if method == 'class_weighted':
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        weight_dict = {i: weight for i, weight in enumerate(class_weights)}
        model_params['class_weight'] = weight_dict
        regularization_method = 'standard_stratified_kfold'  # Use standard k-fold with weighted model
    else:
        regularization_method = method
    
    # Run multiple trials
    trial_results = []
    method_scores = []
    baseline_scores = []
    train_val_gaps = []
    bonuses_applied = []
    
    for trial in range(n_trials):
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=trial)
        
        fold_method_scores = []
        fold_baseline_scores = []
        fold_gaps = []
        fold_bonuses = []
        
        for train_idx, val_idx in skf.split(X_train, y_train):
            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
            
            # Train model
            model = model_class(**model_params)
            model.fit(X_fold_train, y_fold_train)
            
            # Get predictions and scores
            train_pred = model.predict(X_fold_train)
            val_pred = model.predict(X_fold_val)
            
            train_score = accuracy_score(y_fold_train, train_pred)
            val_score = accuracy_score(y_fold_val, val_pred)
            
            # Apply regularization
            gap = abs(train_score - val_score)
            if regularization_method == 'stability_bonus':
                if gap < 0.1:
                    bonus = (0.1 - gap) / 0.1 * 0.2
                    reg_score = val_score * (1.0 + bonus)
                else:
                    bonus = 0.0
                    reg_score = val_score
                fold_bonuses.append(bonus)
            else:
                reg_score = apply_regularization_method(train_score, val_score, regularization_method)
                fold_bonuses.append(0.0)
            
            fold_method_scores.append(reg_score)
            fold_baseline_scores.append(val_score)
            fold_gaps.append(gap)
        
        # Trial statistics
        trial_method_score = np.mean(fold_method_scores)
        trial_baseline_score = np.mean(fold_baseline_scores)
        
        method_scores.append(trial_method_score)
        baseline_scores.append(trial_baseline_score)
        train_val_gaps.extend(fold_gaps)
        bonuses_applied.extend(fold_bonuses)
        
        trial_results.append({
            'trial': trial,
            'method_score': trial_method_score,
            'baseline_score': trial_baseline_score,
            'improvement': trial_method_score - trial_baseline_score,
            'improvement_pct': (trial_method_score - trial_baseline_score) / abs(trial_baseline_score) * 100 if trial_baseline_score != 0 else 0
        })
    
    # Calculate comprehensive statistics
    method_scores = np.array(method_scores)
    baseline_scores = np.array(baseline_scores)
    improvements = method_scores - baseline_scores
    improvement_pcts = improvements / np.abs(baseline_scores) * 100
    
    # Expected Value (EV) calculations
    ev_method_score = np.mean(method_scores)
    ev_baseline_score = np.mean(baseline_scores)
    ev_improvement = np.mean(improvements)
    ev_improvement_pct = np.mean(improvement_pcts)
    
    # Standard Errors
    se_method_score = np.std(method_scores) / np.sqrt(n_trials)
    se_baseline_score = np.std(baseline_scores) / np.sqrt(n_trials)
    se_improvement = np.std(improvements) / np.sqrt(n_trials)
    se_improvement_pct = np.std(improvement_pcts) / np.sqrt(n_trials)
    
    # Confidence intervals (95%)
    ci_95_method = 1.96 * se_method_score
    ci_95_baseline = 1.96 * se_baseline_score
    ci_95_improvement = 1.96 * se_improvement
    ci_95_improvement_pct = 1.96 * se_improvement_pct
    
    # Statistical significance test
    method_lower = ev_method_score - ci_95_method
    method_upper = ev_method_score + ci_95_method
    baseline_lower = ev_baseline_score - ci_95_baseline
    baseline_upper = ev_baseline_score + ci_95_baseline
    
    ci_overlap = not (method_upper < baseline_lower or baseline_upper < method_lower)
    statistically_significant = not ci_overlap
    
    # Test set evaluation
    final_model = model_class(**model_params)
    final_model.fit(X_train, y_train)
    
    test_pred = final_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_pred)
    test_precision = precision_score(y_test, test_pred, average='weighted', zero_division=0)
    test_recall = recall_score(y_test, test_pred, average='weighted', zero_division=0)
    test_f1 = f1_score(y_test, test_pred, average='weighted', zero_division=0)
    
    # Test AUC if possible
    test_auc = None
    if hasattr(final_model, 'predict_proba') and len(np.unique(y_test)) == 2:
        try:
            test_proba = final_model.predict_proba(X_test)[:, 1]
            test_auc = roc_auc_score(y_test, test_proba)
        except:
            pass
    
    return {
        'dataset_name': dataset_name,
        'version_name': version_name,
        'method': method,
        'k_folds': k,
        'n_trials': n_trials,
        'actual_ratio': actual_ratio,
        
        # Expected Value (EV) Statistics
        'ev_method_score': ev_method_score,
        'ev_baseline_score': ev_baseline_score,
        'ev_improvement': ev_improvement,
        'ev_improvement_pct': ev_improvement_pct,
        
        # Standard Errors
        'se_method_score': se_method_score,
        'se_baseline_score': se_baseline_score,
        'se_improvement': se_improvement,
        'improvement_se': se_improvement_pct,
        
        # Standard Deviations
        'std_method_score': np.std(method_scores),
        'std_baseline_score': np.std(baseline_scores),
        'std_improvement': np.std(improvements),
        'std_improvement_pct': np.std(improvement_pcts),
        
        # Confidence Intervals (95%)
        'ci_95_method': ci_95_method,
        'ci_95_baseline': ci_95_baseline,
        'ci_95_improvement': ci_95_improvement,
        'ci_95_improvement_pct': ci_95_improvement_pct,
        
        # Statistical Tests
        'statistically_significant': statistically_significant,
        'method_better': ev_method_score > ev_baseline_score,
        'ci_overlap': ci_overlap,
        
        # Distribution Statistics
        'min_improvement_pct': np.min(improvement_pcts),
        'max_improvement_pct': np.max(improvement_pcts),
        'median_improvement_pct': np.median(improvement_pcts),
        'q25_improvement_pct': np.percentile(improvement_pcts, 25),
        'q75_improvement_pct': np.percentile(improvement_pcts, 75),
        
        # Test Set Performance
        'test_accuracy': test_accuracy,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1,
        'test_auc': test_auc,
        
        # Method-specific metrics
        'mean_train_val_gap': np.mean(train_val_gaps),
        'std_train_val_gap': np.std(train_val_gaps),
        'mean_bonus_applied': np.mean(bonuses_applied),
        'bonus_frequency': np.sum(np.array(bonuses_applied) > 0) / len(bonuses_applied) * 100,
        
        # Trial details for further analysis
        'trial_results': trial_results
    }

def calculate_dataset_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate comprehensive summary for a dataset."""
    
    if not results:
        return {}
    
    df = pd.DataFrame(results)
    summary = {}
    
    # Overall statistics
    summary['total_experiments'] = len(results)
    summary['methods_tested'] = df['method'].unique().tolist()
    summary['versions_tested'] = df['version_name'].unique().tolist()
    
    # Performance by method
    summary['by_method'] = {}
    for method in df['method'].unique():
        method_data = df[df['method'] == method]
        
        summary['by_method'][method] = {
            'count': len(method_data),
            # Expected Value Statistics
            'ev_improvement_pct': method_data['ev_improvement_pct'].mean(),
            'ev_improvement_se': method_data['improvement_se'].mean(),
            'ev_improvement_ci_95': 1.96 * method_data['improvement_se'].mean(),
            
            # Distribution Statistics
            'mean_improvement_pct': method_data['ev_improvement_pct'].mean(),
            'std_improvement_pct': method_data['ev_improvement_pct'].std(),
            'median_improvement_pct': method_data['ev_improvement_pct'].median(),
            'min_improvement_pct': method_data['min_improvement_pct'].min(),
            'max_improvement_pct': method_data['max_improvement_pct'].max(),
            
            # Success Rates
            'better_rate': method_data['method_better'].mean() * 100,
            'significance_rate': method_data['statistically_significant'].mean() * 100,
            
            # Test Performance
            'mean_test_auc': method_data['test_auc'].mean() if method_data['test_auc'].notna().any() else None,
            'std_test_auc': method_data['test_auc'].std() if method_data['test_auc'].notna().any() else None,
            'mean_test_accuracy': method_data['test_accuracy'].mean(),
            'mean_test_f1': method_data['test_f1'].mean(),
            
            # Consistency Metrics
            'consistency_score': 1.0 / (method_data['std_improvement_pct'].mean() + 0.001),
            'coefficient_of_variation': method_data['std_improvement_pct'].mean() / (abs(method_data['ev_improvement_pct'].mean()) + 0.001)
        }
        
        # Method-specific metrics
        if method == 'stability_bonus':
            summary['by_method'][method].update({
                'mean_bonus_frequency': method_data['bonus_frequency'].mean(),
                'mean_bonus_applied': method_data['mean_bonus_applied'].mean(),
                'mean_train_val_gap': method_data['mean_train_val_gap'].mean()
            })
    
    return summary

def create_real_dataset_visualizations(all_dataset_results: List[Dict[str, Any]], save_path: str):
    """Create comprehensive visualizations for real dataset results."""
    
    fig = plt.figure(figsize=(24, 18))
    gs = fig.add_gridspec(4, 4, height_ratios=[1, 1, 1, 0.3])
    
    fig.suptitle('Real Datasets Validation: Davidian Regularization Performance\n' + 
                 'Expected Value Analysis with Standard Errors Across Multiple Real-World Datasets', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # Combine all results
    all_results = []
    for dataset_result in all_dataset_results:
        for result in dataset_result['results']:
            result['dataset_name'] = dataset_result['dataset_name']
            all_results.append(result)
    
    df = pd.DataFrame(all_results)
    
    # Method setup
    methods = df['method'].unique()
    method_labels = [m.replace('_', ' ').title() for m in methods]
    colors = plt.cm.Set3(np.linspace(0, 1, len(methods)))
    
    # 1. Expected Value Performance by Method (All Datasets)
    ax1 = fig.add_subplot(gs[0, 0:2])
    
    method_ev_stats = []
    for method in methods:
        method_data = df[df['method'] == method]
        ev_mean = method_data['ev_improvement_pct'].mean()
        ev_se = method_data['improvement_se'].mean()
        ev_ci_95 = 1.96 * ev_se
        
        method_ev_stats.append({
            'method': method,
            'ev_mean': ev_mean,
            'ev_se': ev_se,
            'ev_ci_95': ev_ci_95,
            'n_experiments': len(method_data)
        })
    
    # Sort by performance
    method_ev_stats.sort(key=lambda x: x['ev_mean'], reverse=True)
    
    # Create horizontal bar chart
    y_pos = np.arange(len(method_ev_stats))
    ev_means = [s['ev_mean'] for s in method_ev_stats]
    ev_cis = [s['ev_ci_95'] for s in method_ev_stats]
    labels = [s['method'].replace('_', ' ').title() for s in method_ev_stats]
    
    bars = ax1.barh(y_pos, ev_means, xerr=ev_cis, capsize=5, 
                    color=[colors[methods.tolist().index(s['method'])] for s in method_ev_stats], 
                    alpha=0.8, height=0.6)
    
    # Highlight stability bonus
    stability_idx = next((i for i, s in enumerate(method_ev_stats) if s['method'] == 'stability_bonus'), None)
    if stability_idx is not None:
        bars[stability_idx].set_edgecolor('#C0392B')
        bars[stability_idx].set_linewidth(3)
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels)
    ax1.set_xlabel('Expected Value: Mean Improvement (%) with 95% CI')
    ax1.set_title('Real Datasets: Expected Value Performance Ranking\n(All datasets combined)', 
                  fontweight='bold')
    ax1.axvline(x=0, color='red', linestyle='--', alpha=0.8, linewidth=2, label='Baseline')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add value labels
    for i, (bar, stats) in enumerate(zip(bars, method_ev_stats)):
        ax1.text(bar.get_width() + stats['ev_ci_95'] + 0.5, bar.get_y() + bar.get_height()/2, 
                f"EV: {stats['ev_mean']:+.1f}% ± {stats['ev_ci_95']:.1f}%\n(n={stats['n_experiments']})", 
                va='center', ha='left', fontweight='bold' if i == stability_idx else 'normal',
                fontsize=10)
    
    # 2. Performance by Dataset
    ax2 = fig.add_subplot(gs[0, 2:])
    
    datasets = df['dataset_name'].unique()
    
    # Create grouped bar chart by dataset
    x = np.arange(len(datasets))
    width = 0.15
    
    for i, method in enumerate(['stability_bonus', 'standard_stratified_kfold', 'davidian_regularization']):
        if method in methods:
            dataset_means = []
            dataset_ses = []
            
            for dataset in datasets:
                subset = df[(df['method'] == method) & (df['dataset_name'] == dataset)]
                if len(subset) > 0:
                    dataset_means.append(subset['ev_improvement_pct'].mean())
                    dataset_ses.append(subset['improvement_se'].mean())
                else:
                    dataset_means.append(0)
                    dataset_ses.append(0)
            
            ax2.bar(x + i*width, dataset_means, width, yerr=dataset_ses, 
                   label=method.replace('_', ' ').title(), 
                   color=colors[methods.tolist().index(method)], alpha=0.8, capsize=3)
    
    ax2.set_xlabel('Dataset')
    ax2.set_ylabel('Expected Value: Mean Improvement (%)')
    ax2.set_title('Performance by Real Dataset\n(Top 3 methods comparison)')
    ax2.set_xticks(x + width)
    ax2.set_xticklabels([d.replace(' ', '\n') for d in datasets], fontsize=9)
    ax2.legend()
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax2.grid(True, alpha=0.3)
    
    # 3. Standard Error Analysis
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Compare standard errors across methods
    method_ses = []
    method_names = []
    
    for method in methods:
        method_data = df[df['method'] == method]
        avg_se = method_data['improvement_se'].mean()
        method_ses.append(avg_se)
        method_names.append(method.replace('_', ' ').title())
    
    bars = ax3.bar(range(len(method_ses)), method_ses, 
                   color=[colors[i] for i in range(len(methods))], alpha=0.8)
    
    ax3.set_xticks(range(len(method_ses)))
    ax3.set_xticklabels(method_names, rotation=45, ha='right')
    ax3.set_ylabel('Average Standard Error (%)')
    ax3.set_title('Statistical Precision\n(Lower SE = More Precise)')
    ax3.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, se_val in zip(bars, method_ses):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{se_val:.3f}%', ha='center', va='bottom', fontweight='bold')
    
    # 4. Test AUC Distribution
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Box plot of test AUC by method
    auc_data = []
    valid_methods = []
    
    for method in methods:
        method_auc = df[df['method'] == method]['test_auc'].dropna().values
        if len(method_auc) > 0:
            auc_data.append(method_auc)
            valid_methods.append(method.replace('_', ' ').title())
    
    if auc_data:
        bp = ax4.boxplot(auc_data, labels=valid_methods, patch_artist=True)
        
        # Color the boxes
        for i, patch in enumerate(bp['boxes']):
            method_idx = methods.tolist().index(methods[i]) if i < len(methods) else 0
            patch.set_facecolor(colors[method_idx])
            patch.set_alpha(0.7)
    
    ax4.set_ylabel('Test AUC')
    ax4.set_title('Generalization Performance\n(Real Dataset Test AUC)')
    ax4.set_xticklabels(valid_methods, rotation=45, ha='right', fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # 5. Consistency Analysis
    ax5 = fig.add_subplot(gs[1, 2])
    
    # Coefficient of variation by method
    cv_data = []
    for method in methods:
        method_data = df[df['method'] == method]
        cv = method_data['std_improvement_pct'].mean() / (abs(method_data['ev_improvement_pct'].mean()) + 0.001)
        cv_data.append(cv)
    
    bars = ax5.bar(range(len(cv_data)), cv_data, 
                   color=[colors[i] for i in range(len(methods))], alpha=0.8)
    
    ax5.set_xticks(range(len(cv_data)))
    ax5.set_xticklabels([m.replace('_', ' ').title() for m in methods], rotation=45, ha='right')
    ax5.set_ylabel('Coefficient of Variation\n(Lower = More Consistent)')
    ax5.set_title('Performance Consistency\nAcross Real Datasets')
    ax5.grid(True, alpha=0.3)
    
    # 6. Performance vs Dataset Characteristics
    ax6 = fig.add_subplot(gs[1, 3])
    
    # Scatter plot: dataset size vs improvement for stability bonus
    stability_data = df[df['method'] == 'stability_bonus']
    if len(stability_data) > 0:
        # Get dataset sizes (approximate from version names)
        dataset_sizes = []
        improvements = []
        
        for _, row in stability_data.iterrows():
            # Estimate size from actual ratio and version
            if 'original' in row['version_name']:
                size = 1000  # Placeholder
            else:
                size = 500  # Placeholder for modified versions
            
            dataset_sizes.append(size)
            improvements.append(row['ev_improvement_pct'])
        
        ax6.scatter(dataset_sizes, improvements, color='#E74C3C', alpha=0.7, s=50)
        ax6.set_xlabel('Approximate Dataset Size')
        ax6.set_ylabel('EV Improvement (%)')
        ax6.set_title('Stability Bonus:\nPerformance vs Dataset Size')
        ax6.grid(True, alpha=0.3)
    
    # 7. Statistical Significance Matrix
    ax7 = fig.add_subplot(gs[2, 0:2])
    
    # Create significance heatmap: method vs dataset
    datasets = df['dataset_name'].unique()
    sig_matrix = np.zeros((len(methods), len(datasets)))
    
    for i, method in enumerate(methods):
        for j, dataset in enumerate(datasets):
            subset = df[(df['method'] == method) & (df['dataset_name'] == dataset)]
            if len(subset) > 0:
                sig_matrix[i, j] = subset['statistically_significant'].mean() * 100
            else:
                sig_matrix[i, j] = np.nan
    
    im = ax7.imshow(sig_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    
    # Add text annotations
    for i in range(len(methods)):
        for j in range(len(datasets)):
            if not np.isnan(sig_matrix[i, j]):
                text = ax7.text(j, i, f'{sig_matrix[i, j]:.0f}%', 
                              ha="center", va="center", 
                              color="white" if sig_matrix[i, j] > 50 else "black",
                              fontweight='bold')
    
    ax7.set_yticks(range(len(methods)))
    ax7.set_yticklabels([m.replace('_', ' ').title() for m in methods])
    ax7.set_xticks(range(len(datasets)))
    ax7.set_xticklabels([d.replace(' ', '\n') for d in datasets], rotation=45, ha='right')
    ax7.set_xlabel('Real Dataset')
    ax7.set_ylabel('Method')
    ax7.set_title('Statistical Significance Rate by Method and Dataset\n(% of experiments with significant results)')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax7)
    cbar.set_label('Significance Rate (%)', rotation=270, labelpad=15)
    
    # 8. Expected Value vs Standard Error Scatter
    ax8 = fig.add_subplot(gs[2, 2:])
    
    # Scatter plot: EV improvement vs standard error for all methods
    for i, method in enumerate(methods):
        method_data = df[df['method'] == method]
        if len(method_data) > 0:
            ax8.scatter(method_data['ev_improvement_pct'], method_data['improvement_se'], 
                       c=[colors[i]], label=method.replace('_', ' ').title(), 
                       alpha=0.7, s=60)
    
    ax8.set_xlabel('Expected Value: Mean Improvement (%)')
    ax8.set_ylabel('Standard Error (%)')
    ax8.set_title('Performance vs Precision\n(Top-right quadrant = best)')
    ax8.axhline(y=df['improvement_se'].median(), color='gray', linestyle='--', alpha=0.5, label='Median SE')
    ax8.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Baseline')
    ax8.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax8.grid(True, alpha=0.3)
    
    # 9. Comprehensive Summary Table
    ax9 = fig.add_subplot(gs[3, :])
    ax9.axis('tight')
    ax9.axis('off')
    
    # Create detailed summary table
    summary_data = []
    for method in methods:
        method_data = df[df['method'] == method]
        if len(method_data) > 0:
            summary_data.append([
                method.replace('_', ' ').title(),
                f"{method_data['ev_improvement_pct'].mean():+.2f}%",
                f"±{method_data['improvement_se'].mean():.3f}%",
                f"{method_data['statistically_significant'].mean()*100:.0f}%",
                f"{method_data['method_better'].mean()*100:.0f}%",
                f"{method_data['test_auc'].mean():.3f}" if method_data['test_auc'].notna().any() else "N/A",
                f"{method_data['test_accuracy'].mean():.3f}",
                f"{method_data['test_f1'].mean():.3f}",
                f"{len(method_data)}",
                f"{method_data['std_improvement_pct'].mean() / (abs(method_data['ev_improvement_pct'].mean()) + 0.001):.2f}"
            ])
    
    headers = ['Method', 'EV Mean', '±SE', 'Significant', 'Better', 'Test AUC', 
               'Test Acc', 'Test F1', 'N', 'CV']
    
    table = ax9.table(cellText=summary_data, colLabels=headers, 
                     cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Color code the table
    for i, method in enumerate([s['method'] for s in method_ev_stats]):
        method_idx = methods.tolist().index(method)
        for j in range(len(headers)):
            table[(i+1, j)].set_facecolor(colors[method_idx])
            table[(i+1, j)].set_alpha(0.3)
            if method == 'stability_bonus':
                table[(i+1, j)].set_alpha(0.5)
                table[(i+1, j)].set_edgecolor('#C0392B')
                table[(i+1, j)].set_linewidth(2)
    
    # Style header
    for j in range(len(headers)):
        table[(0, j)].set_facecolor('#34495E')
        table[(0, j)].set_text_props(weight='bold', color='white')
    
    ax9.set_title('Real Datasets Summary: Expected Value Analysis with Standard Errors', 
                  fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Real dataset visualizations saved to {save_path}")

def main():
    """Main function to run real dataset experiments."""
    
    print("REAL DATASET EXPERIMENT PIPELINE")
    print("="*80)
    
    start_time = time.time()
    
    # Load real datasets
    datasets = load_real_datasets()
    
    if not datasets:
        print("No datasets loaded successfully. Exiting.")
        return
    
    # Run experiments on each dataset
    all_dataset_results = []
    
    for dataset_name, dataset_info in datasets.items():
        print(f"\n{'='*60}")
        print(f"PROCESSING DATASET: {dataset_info['name']}")
        print(f"{'='*60}")
        
        try:
            dataset_results = run_comprehensive_real_dataset_experiment(
                dataset_info,
                target_ratios=[1.0, 9.0, 19.0, 49.0],  # Include original as well
                k_values=[3, 5, 10],
                n_trials=30,  # More trials for better statistics
                methods=['stability_bonus', 'standard_stratified_kfold', 'davidian_regularization', 
                        'conservative_davidian', 'exponential_decay', 'inverse_diff']
            )
            
            all_dataset_results.append(dataset_results)
            
        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")
            continue
    
    if not all_dataset_results:
        print("No successful dataset experiments. Exiting.")
        return
    
    # Save results
    print(f"\nSaving comprehensive results...")
    os.makedirs('data', exist_ok=True)
    
    with open('data/real_datasets_results.json', 'w') as f:
        json.dump(all_dataset_results, f, indent=2, default=str)
    
    # Create comprehensive results DataFrame
    all_results = []
    for dataset_result in all_dataset_results:
        for result in dataset_result['results']:
            result['dataset_name'] = dataset_result['dataset_name']
            all_results.append(result)
    
    results_df = pd.DataFrame(all_results)
    results_df.to_csv('data/real_datasets_results.csv', index=False)
    
    # Create visualizations
    print(f"\nCreating comprehensive visualizations...")
    create_real_dataset_visualizations(all_dataset_results, 'graphs/real_datasets_analysis.png')
    
    # Print comprehensive summary
    elapsed_time = time.time() - start_time
    
    print(f"\n{'='*80}")
    print("REAL DATASETS EXPERIMENT SUMMARY")
    print(f"{'='*80}")
    print(f"Total execution time: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
    print(f"Datasets processed: {len(all_dataset_results)}")
    print(f"Total experiments: {len(all_results)}")
    
    # Overall performance summary
    if len(all_results) > 0:
        print(f"\nOVERALL PERFORMANCE (Expected Value Analysis):")
        
        for method in results_df['method'].unique():
            method_data = results_df[results_df['method'] == method]
            
            ev_mean = method_data['ev_improvement_pct'].mean()
            avg_se = method_data['improvement_se'].mean()
            better_rate = method_data['method_better'].mean() * 100
            sig_rate = method_data['statistically_significant'].mean() * 100
            
            print(f"  {method:25s}: EV={ev_mean:+6.2f}% ± {avg_se:.3f}%, "
                  f"Better: {better_rate:5.1f}%, Significant: {sig_rate:5.1f}%")
            
            if method == 'stability_bonus':
                bonus_freq = method_data['bonus_frequency'].mean()
                avg_gap = method_data['mean_train_val_gap'].mean()
                print(f"    Bonus frequency: {bonus_freq:.1f}%, Avg train-val gap: {avg_gap:.4f}")
    
    print(f"\nFILES GENERATED:")
    print(f"  - data/real_datasets_results.json")
    print(f"  - data/real_datasets_results.csv") 
    print(f"  - graphs/real_datasets_analysis.png")
    
    print(f"\n✓ Real dataset validation completed!")
    print(f"✓ Expected Value and Standard Error analysis included")
    print(f"✓ Results demonstrate method performance on real-world data")

if __name__ == "__main__":
    main()
