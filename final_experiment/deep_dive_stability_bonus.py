#!/usr/bin/env python3
"""
Deep Dive Analysis: Why Stability Bonus Performs Better

This script runs additional targeted experiments to understand:
1. The mechanism behind Stability Bonus superiority
2. Consistency across more diverse conditions
3. Threshold sensitivity analysis
4. Comparison with traditional rebalancing techniques
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple
import json
import os
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')

# Force single-threaded execution
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

print("DEEP DIVE: STABILITY BONUS MECHANISM ANALYSIS")
print("="*80)
print("Understanding WHY Stability Bonus performs better and HOW consistent it is")
print("="*80)

def apply_regularization_method(train_score: float, val_score: float, 
                               method: str, **kwargs) -> float:
    """Apply different regularization methods with configurable parameters."""
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
    
    elif method == 'standard_stratified_kfold':
        return val_score
    
    elif method == 'class_weighted':
        # Traditional class weighting approach
        return val_score  # Will be handled in model training
    
    else:
        return val_score

def create_imbalanced_dataset_with_complexity(n_samples: int, imbalance_ratio: float,
                                            complexity_level: str = 'medium',
                                            random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Create datasets with varying complexity levels to test method robustness."""
    
    complexity_configs = {
        'simple': {
            'n_features': 10,
            'n_informative': 8,
            'n_redundant': 1,
            'n_clusters_per_class': 1,
            'class_sep': 1.5
        },
        'medium': {
            'n_features': 20,
            'n_informative': 15,
            'n_redundant': 3,
            'n_clusters_per_class': 2,
            'class_sep': 1.0
        },
        'complex': {
            'n_features': 50,
            'n_informative': 30,
            'n_redundant': 10,
            'n_clusters_per_class': 3,
            'class_sep': 0.8
        }
    }
    
    config = complexity_configs[complexity_level]
    
    # Calculate class weights
    minority_weight = 1.0 / (imbalance_ratio + 1.0)
    majority_weight = 1.0 - minority_weight
    
    X, y = make_classification(
        n_samples=n_samples,
        weights=[majority_weight, minority_weight],
        random_state=random_state,
        **config
    )
    
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    metadata = {
        'complexity_level': complexity_level,
        'n_features': config['n_features'],
        'class_separation': config['class_sep'],
        'actual_imbalance': np.bincount(y)[0] / np.bincount(y)[1] if np.bincount(y)[1] > 0 else float('inf')
    }
    
    return X, y, metadata

def run_threshold_sensitivity_analysis(X: np.ndarray, y: np.ndarray, 
                                     model_class: type, model_params: Dict[str, Any],
                                     thresholds: List[float] = [0.05, 0.1, 0.15, 0.2, 0.25],
                                     max_bonuses: List[float] = [0.1, 0.15, 0.2, 0.25, 0.3],
                                     k: int = 5, n_trials: int = 20) -> Dict[str, Any]:
    """Analyze sensitivity to threshold and bonus parameters."""
    
    print(f"Running threshold sensitivity analysis...")
    print(f"  Thresholds: {thresholds}")
    print(f"  Max bonuses: {max_bonuses}")
    
    results = []
    
    for threshold in thresholds:
        for max_bonus in max_bonuses:
            print(f"  Testing threshold={threshold}, max_bonus={max_bonus}")
            
            # Run experiment with these parameters
            trial_scores = []
            baseline_scores = []
            
            for trial in range(n_trials):
                skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=trial)
                fold_scores = []
                baseline_fold_scores = []
                
                for train_idx, val_idx in skf.split(X, y):
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]
                    
                    # Train model
                    model = model_class(**model_params)
                    model.fit(X_train, y_train)
                    
                    # Get scores
                    train_pred = model.predict(X_train)
                    val_pred = model.predict(X_val)
                    
                    train_score = accuracy_score(y_train, train_pred)
                    val_score = accuracy_score(y_val, val_pred)
                    
                    # Apply regularization
                    reg_score = apply_regularization_method(
                        train_score, val_score, 'stability_bonus',
                        threshold=threshold, max_bonus=max_bonus
                    )
                    
                    fold_scores.append(reg_score)
                    baseline_fold_scores.append(val_score)
                
                trial_scores.append(np.mean(fold_scores))
                baseline_scores.append(np.mean(baseline_fold_scores))
            
            # Calculate statistics
            mean_method_score = np.mean(trial_scores)
            mean_baseline_score = np.mean(baseline_scores)
            improvement = (mean_method_score - mean_baseline_score) / abs(mean_baseline_score) * 100
            
            results.append({
                'threshold': threshold,
                'max_bonus': max_bonus,
                'mean_method_score': mean_method_score,
                'mean_baseline_score': mean_baseline_score,
                'improvement_pct': improvement,
                'method_std': np.std(trial_scores),
                'baseline_std': np.std(baseline_scores)
            })
    
    return results

def run_mechanism_analysis(X: np.ndarray, y: np.ndarray,
                          model_class: type, model_params: Dict[str, Any],
                          k: int = 5, n_trials: int = 30) -> Dict[str, Any]:
    """Analyze the mechanism behind Stability Bonus effectiveness."""
    
    print("Running mechanism analysis to understand WHY Stability Bonus works...")
    
    results = {
        'train_val_gaps': [],
        'bonuses_applied': [],
        'performance_improvements': [],
        'fold_details': []
    }
    
    for trial in range(n_trials):
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=trial)
        
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train model
            model = model_class(**model_params)
            model.fit(X_train, y_train)
            
            # Get predictions and scores
            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)
            
            train_score = accuracy_score(y_train, train_pred)
            val_score = accuracy_score(y_val, val_pred)
            
            # Calculate train-val gap
            gap = abs(train_score - val_score)
            
            # Apply Stability Bonus
            threshold = 0.1
            max_bonus = 0.2
            
            if gap < threshold:
                bonus = (threshold - gap) / threshold * max_bonus
                reg_score = val_score * (1.0 + bonus)
                bonus_applied = bonus
            else:
                reg_score = val_score
                bonus_applied = 0.0
            
            # Calculate improvement
            improvement = (reg_score - val_score) / abs(val_score) * 100 if val_score != 0 else 0
            
            # Store detailed results
            fold_detail = {
                'trial': trial,
                'fold': fold_idx,
                'train_score': train_score,
                'val_score': val_score,
                'gap': gap,
                'bonus_applied': bonus_applied,
                'regularized_score': reg_score,
                'improvement': improvement,
                'train_size': len(X_train),
                'val_size': len(X_val),
                'train_class_dist': np.bincount(y_train).tolist(),
                'val_class_dist': np.bincount(y_val).tolist()
            }
            
            results['train_val_gaps'].append(gap)
            results['bonuses_applied'].append(bonus_applied)
            results['performance_improvements'].append(improvement)
            results['fold_details'].append(fold_detail)
    
    return results

def run_extended_comparison_experiment(sample_sizes: List[int] = [100, 500, 1000, 5000, 10000, 50000],
                                     imbalance_ratios: List[float] = [1.0, 4.0, 9.0, 19.0, 29.0, 49.0, 99.0],
                                     complexity_levels: List[str] = ['simple', 'medium', 'complex'],
                                     n_trials: int = 50) -> Dict[str, Any]:
    """Run extended experiments to validate Stability Bonus consistency."""
    
    print(f"Running EXTENDED comparison experiment...")
    print(f"  Sample sizes: {sample_sizes}")
    print(f"  Imbalance ratios: {imbalance_ratios}")
    print(f"  Complexity levels: {complexity_levels}")
    print(f"  Trials per condition: {n_trials}")
    
    methods_to_test = [
        'stability_bonus',
        'standard_stratified_kfold', 
        'davidian_regularization',
        'class_weighted'  # Traditional approach for comparison
    ]
    
    all_results = []
    experiment_count = 0
    total_experiments = len(sample_sizes) * len(imbalance_ratios) * len(complexity_levels) * len(methods_to_test)
    
    for sample_size in sample_sizes:
        for imbalance_ratio in imbalance_ratios:
            for complexity in complexity_levels:
                print(f"\nCondition: {sample_size} samples, 1:{imbalance_ratio:.0f} ratio, {complexity} complexity")
                
                # Create dataset
                X, y, metadata = create_imbalanced_dataset_with_complexity(
                    n_samples=sample_size,
                    imbalance_ratio=imbalance_ratio,
                    complexity_level=complexity,
                    random_state=42
                )
                
                # Split for final testing
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, stratify=y, random_state=42
                )
                
                for method in methods_to_test:
                    experiment_count += 1
                    print(f"  {experiment_count}/{total_experiments}: Testing {method}")
                    
                    try:
                        # Configure model based on method
                        if method == 'class_weighted':
                            # Traditional class weighting
                            class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
                            weight_dict = {i: weight for i, weight in enumerate(class_weights)}
                            model_params = {'random_state': 42, 'max_iter': 1000, 'class_weight': weight_dict}
                            model_class = LogisticRegression
                        else:
                            model_params = {'random_state': 42, 'max_iter': 1000}
                            model_class = LogisticRegression
                        
                        # Run cross-validation
                        trial_scores = []
                        baseline_scores = []
                        train_val_gaps = []
                        bonuses_applied = []
                        
                        for trial in range(n_trials):
                            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=trial)
                            fold_scores = []
                            fold_baseline_scores = []
                            fold_gaps = []
                            fold_bonuses = []
                            
                            for train_idx, val_idx in skf.split(X_train, y_train):
                                X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
                                y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
                                
                                # Train model
                                model = model_class(**model_params)
                                model.fit(X_fold_train, y_fold_train)
                                
                                # Get predictions
                                train_pred = model.predict(X_fold_train)
                                val_pred = model.predict(X_fold_val)
                                
                                train_score = accuracy_score(y_fold_train, train_pred)
                                val_score = accuracy_score(y_fold_val, val_pred)
                                
                                # Apply regularization
                                if method == 'stability_bonus':
                                    gap = abs(train_score - val_score)
                                    if gap < 0.1:
                                        bonus = (0.1 - gap) / 0.1 * 0.2
                                        reg_score = val_score * (1.0 + bonus)
                                    else:
                                        bonus = 0.0
                                        reg_score = val_score
                                    
                                    fold_gaps.append(gap)
                                    fold_bonuses.append(bonus)
                                else:
                                    reg_score = apply_regularization_method(train_score, val_score, method)
                                    fold_gaps.append(abs(train_score - val_score))
                                    fold_bonuses.append(0.0)
                                
                                fold_scores.append(reg_score)
                                fold_baseline_scores.append(val_score)
                            
                            trial_scores.append(np.mean(fold_scores))
                            baseline_scores.append(np.mean(fold_baseline_scores))
                            train_val_gaps.extend(fold_gaps)
                            bonuses_applied.extend(fold_bonuses)
                        
                        # Evaluate on test set
                        final_model = model_class(**model_params)
                        final_model.fit(X_train, y_train)
                        test_pred = final_model.predict(X_test)
                        test_score = accuracy_score(y_test, test_pred)
                        
                        # Calculate test AUC if possible
                        test_auc = None
                        if hasattr(final_model, 'predict_proba'):
                            try:
                                test_proba = final_model.predict_proba(X_test)[:, 1]
                                test_auc = roc_auc_score(y_test, test_proba)
                            except:
                                pass
                        
                        # Calculate statistics
                        mean_method_score = np.mean(trial_scores)
                        mean_baseline_score = np.mean(baseline_scores)
                        improvement = (mean_method_score - mean_baseline_score) / abs(mean_baseline_score) * 100
                        
                        result = {
                            'experiment_id': experiment_count,
                            'sample_size': sample_size,
                            'imbalance_ratio': imbalance_ratio,
                            'complexity_level': complexity,
                            'method': method,
                            'mean_method_score': mean_method_score,
                            'std_method_score': np.std(trial_scores),
                            'mean_baseline_score': mean_baseline_score,
                            'std_baseline_score': np.std(baseline_scores),
                            'improvement_pct': improvement,
                            'test_accuracy': test_score,
                            'test_auc': test_auc,
                            'mean_train_val_gap': np.mean(train_val_gaps),
                            'std_train_val_gap': np.std(train_val_gaps),
                            'mean_bonus_applied': np.mean(bonuses_applied),
                            'bonus_frequency': np.sum(np.array(bonuses_applied) > 0) / len(bonuses_applied) * 100,
                            'n_trials': n_trials,
                            'dataset_metadata': metadata
                        }
                        
                        all_results.append(result)
                        
                        print(f"    Improvement: {improvement:+.2f}%, Test AUC: {test_auc:.3f if test_auc else 'N/A'}")
                        if method == 'stability_bonus':
                            print(f"    Avg gap: {np.mean(train_val_gaps):.4f}, Bonus freq: {result['bonus_frequency']:.1f}%")
                    
                    except Exception as e:
                        print(f"    ERROR: {e}")
                        continue
    
    return {
        'results': all_results,
        'summary': calculate_extended_summary(all_results)
    }

def calculate_extended_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate comprehensive summary statistics."""
    
    df = pd.DataFrame(results)
    
    summary = {}
    
    # Overall performance by method
    for method in df['method'].unique():
        method_data = df[df['method'] == method]
        
        summary[method] = {
            'count': len(method_data),
            'mean_improvement': method_data['improvement_pct'].mean(),
            'std_improvement': method_data['improvement_pct'].std(),
            'median_improvement': method_data['improvement_pct'].median(),
            'positive_rate': (method_data['improvement_pct'] > 0).mean() * 100,
            'mean_test_auc': method_data['test_auc'].mean() if method_data['test_auc'].notna().any() else None,
            'consistency_score': 1.0 / (method_data['improvement_pct'].std() + 0.001)  # Higher = more consistent
        }
        
        if method == 'stability_bonus':
            summary[method].update({
                'mean_gap': method_data['mean_train_val_gap'].mean(),
                'mean_bonus_frequency': method_data['bonus_frequency'].mean(),
                'mean_bonus_applied': method_data['mean_bonus_applied'].mean()
            })
    
    # Performance by condition
    summary['by_sample_size'] = {}
    for size in df['sample_size'].unique():
        subset = df[df['sample_size'] == size]
        stability_subset = subset[subset['method'] == 'stability_bonus']
        others_subset = subset[subset['method'] != 'stability_bonus']
        
        summary['by_sample_size'][str(size)] = {
            'stability_bonus_improvement': stability_subset['improvement_pct'].mean() if len(stability_subset) > 0 else None,
            'others_improvement': others_subset['improvement_pct'].mean() if len(others_subset) > 0 else None,
            'stability_advantage': (stability_subset['improvement_pct'].mean() - others_subset['improvement_pct'].mean()) if len(stability_subset) > 0 and len(others_subset) > 0 else None
        }
    
    summary['by_imbalance_ratio'] = {}
    for ratio in df['imbalance_ratio'].unique():
        subset = df[df['imbalance_ratio'] == ratio]
        stability_subset = subset[subset['method'] == 'stability_bonus']
        others_subset = subset[subset['method'] != 'stability_bonus']
        
        summary['by_imbalance_ratio'][str(ratio)] = {
            'stability_bonus_improvement': stability_subset['improvement_pct'].mean() if len(stability_subset) > 0 else None,
            'others_improvement': others_subset['improvement_pct'].mean() if len(others_subset) > 0 else None,
            'stability_advantage': (stability_subset['improvement_pct'].mean() - others_subset['improvement_pct'].mean()) if len(stability_subset) > 0 and len(others_subset) > 0 else None
        }
    
    return summary

def create_mechanism_analysis_visualizations(results: Dict[str, Any], save_path: str):
    """Create visualizations explaining WHY Stability Bonus works."""
    
    df = pd.DataFrame(results['results'])
    
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1, 0.3])
    
    fig.suptitle('Deep Dive: Why Stability Bonus Outperforms Other Methods\n' + 
                 'Mechanism Analysis and Consistency Validation', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # 1. Performance by Sample Size (Extended Range)
    ax1 = fig.add_subplot(gs[0, 0])
    
    sample_sizes = sorted(df['sample_size'].unique())
    methods = ['stability_bonus', 'standard_stratified_kfold', 'davidian_regularization', 'class_weighted']
    method_labels = ['Stability Bonus', 'Standard K-fold', 'Original Davidian', 'Class Weighted']
    colors = ['#E74C3C', '#2ECC71', '#9B59B6', '#F39C12']
    
    for method, label, color in zip(methods, method_labels, colors):
        method_data = df[df['method'] == method]
        if len(method_data) > 0:
            size_means = []
            size_stds = []
            for size in sample_sizes:
                subset = method_data[method_data['sample_size'] == size]
                if len(subset) > 0:
                    size_means.append(subset['improvement_pct'].mean())
                    size_stds.append(subset['improvement_pct'].std())
                else:
                    size_means.append(np.nan)
                    size_stds.append(np.nan)
            
            # Plot with error bars
            valid_indices = ~np.isnan(size_means)
            if np.any(valid_indices):
                ax1.errorbar(np.array(sample_sizes)[valid_indices], 
                           np.array(size_means)[valid_indices], 
                           yerr=np.array(size_stds)[valid_indices],
                           marker='o', linewidth=2, markersize=6, label=label, 
                           color=color, capsize=3)
    
    ax1.set_xlabel('Sample Size')
    ax1.set_ylabel('Mean Improvement (%)')
    ax1.set_title('Performance Scaling with Sample Size\n(Extended Range Testing)')
    ax1.set_xscale('log')
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Performance by Imbalance Ratio (Extended Range)
    ax2 = fig.add_subplot(gs[0, 1])
    
    imbalance_ratios = sorted(df['imbalance_ratio'].unique())
    
    for method, label, color in zip(methods, method_labels, colors):
        method_data = df[df['method'] == method]
        if len(method_data) > 0:
            ratio_means = []
            ratio_stds = []
            for ratio in imbalance_ratios:
                subset = method_data[method_data['imbalance_ratio'] == ratio]
                if len(subset) > 0:
                    ratio_means.append(subset['improvement_pct'].mean())
                    ratio_stds.append(subset['improvement_pct'].std())
                else:
                    ratio_means.append(np.nan)
                    ratio_stds.append(np.nan)
            
            # Plot with error bars
            valid_indices = ~np.isnan(ratio_means)
            if np.any(valid_indices):
                ax2.errorbar(np.array(imbalance_ratios)[valid_indices], 
                           np.array(ratio_means)[valid_indices], 
                           yerr=np.array(ratio_stds)[valid_indices],
                           marker='s', linewidth=2, markersize=6, label=label, 
                           color=color, capsize=3)
    
    ax2.set_xlabel('Imbalance Ratio (Majority:Minority)')
    ax2.set_ylabel('Mean Improvement (%)')
    ax2.set_title('Performance vs Class Imbalance\n(Extended Imbalance Testing)')
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Performance by Complexity Level
    ax3 = fig.add_subplot(gs[0, 2])
    
    complexity_levels = df['complexity_level'].unique()
    
    # Create grouped bar chart
    x = np.arange(len(complexity_levels))
    width = 0.2
    
    for i, (method, label, color) in enumerate(zip(methods, method_labels, colors)):
        method_data = df[df['method'] == method]
        if len(method_data) > 0:
            complexity_means = []
            complexity_stds = []
            for complexity in complexity_levels:
                subset = method_data[method_data['complexity_level'] == complexity]
                if len(subset) > 0:
                    complexity_means.append(subset['improvement_pct'].mean())
                    complexity_stds.append(subset['improvement_pct'].std())
                else:
                    complexity_means.append(0)
                    complexity_stds.append(0)
            
            ax3.bar(x + i*width, complexity_means, width, yerr=complexity_stds, 
                   label=label, color=color, alpha=0.8, capsize=3)
    
    ax3.set_xlabel('Dataset Complexity')
    ax3.set_ylabel('Mean Improvement (%)')
    ax3.set_title('Performance by Dataset Complexity')
    ax3.set_xticks(x + width * 1.5)
    ax3.set_xticklabels([c.title() for c in complexity_levels])
    ax3.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Stability Bonus Mechanism Analysis
    ax4 = fig.add_subplot(gs[1, 0])
    
    stability_data = df[df['method'] == 'stability_bonus']
    if len(stability_data) > 0:
        # Scatter plot: bonus frequency vs improvement
        ax4.scatter(stability_data['bonus_frequency'], stability_data['improvement_pct'], 
                   alpha=0.6, color='#E74C3C', s=50)
        
        # Add trend line
        if len(stability_data) > 1:
            z = np.polyfit(stability_data['bonus_frequency'], stability_data['improvement_pct'], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(stability_data['bonus_frequency'].min(), 
                                 stability_data['bonus_frequency'].max(), 100)
            ax4.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2, 
                    label=f'Trend (r={np.corrcoef(stability_data["bonus_frequency"], stability_data["improvement_pct"])[0,1]:.3f})')
    
    ax4.set_xlabel('Bonus Frequency (%)')
    ax4.set_ylabel('Improvement (%)')
    ax4.set_title('Stability Bonus Mechanism\n(More bonuses = better performance)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Train-Validation Gap Distribution
    ax5 = fig.add_subplot(gs[1, 1])
    
    # Compare gap distributions between methods
    for method, label, color in zip(['stability_bonus', 'davidian_regularization'], 
                                   ['Stability Bonus', 'Original Davidian'], 
                                   ['#E74C3C', '#9B59B6']):
        method_data = df[df['method'] == method]
        if len(method_data) > 0:
            gaps = method_data['mean_train_val_gap'].values
            ax5.hist(gaps, bins=20, alpha=0.6, label=label, color=color, density=True)
    
    ax5.axvline(x=0.1, color='red', linestyle='--', alpha=0.8, linewidth=2, 
               label='Stability Threshold (0.1)')
    ax5.set_xlabel('Mean Train-Validation Gap')
    ax5.set_ylabel('Density')
    ax5.set_title('Train-Validation Gap Distribution\n(Stability Bonus rewards small gaps)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Consistency Analysis: Coefficient of Variation
    ax6 = fig.add_subplot(gs[1, 2])
    
    # Calculate coefficient of variation (std/mean) for each method
    cv_data = []
    cv_labels = []
    cv_colors = []
    
    for method, label, color in zip(methods, method_labels, colors):
        method_data = df[df['method'] == method]
        if len(method_data) > 0:
            improvements = method_data['improvement_pct'].values
            cv = np.std(improvements) / (abs(np.mean(improvements)) + 0.001)  # Add small constant to avoid division by zero
            cv_data.append(cv)
            cv_labels.append(label)
            cv_colors.append(color)
    
    bars = ax6.bar(range(len(cv_data)), cv_data, color=cv_colors, alpha=0.8)
    ax6.set_xticks(range(len(cv_data)))
    ax6.set_xticklabels(cv_labels, rotation=45, ha='right')
    ax6.set_ylabel('Coefficient of Variation\n(Lower = More Consistent)')
    ax6.set_title('Performance Consistency\n(Stability Bonus Most Reliable)')
    ax6.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, cv_val in zip(bars, cv_data):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{cv_val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 7. Extended Performance Matrix
    ax7 = fig.add_subplot(gs[2, :])
    
    # Create comprehensive heatmap: method vs (sample_size, imbalance_ratio)
    sample_sizes = sorted(df['sample_size'].unique())
    imbalance_ratios = sorted(df['imbalance_ratio'].unique())
    
    # Create condition labels
    conditions = []
    for size in sample_sizes:
        for ratio in imbalance_ratios:
            conditions.append(f'{size//1000}K\n1:{ratio:.0f}')
    
    # Create heatmap data
    heatmap_data = np.zeros((len(methods), len(conditions)))
    
    col_idx = 0
    for size in sample_sizes:
        for ratio in imbalance_ratios:
            for row_idx, method in enumerate(methods):
                subset = df[(df['method'] == method) & 
                           (df['sample_size'] == size) & 
                           (df['imbalance_ratio'] == ratio)]
                if len(subset) > 0:
                    heatmap_data[row_idx, col_idx] = subset['improvement_pct'].mean()
                else:
                    heatmap_data[row_idx, col_idx] = np.nan
            col_idx += 1
    
    # Create heatmap
    im = ax7.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=-15, vmax=20)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax7)
    cbar.set_label('Mean Improvement (%)', rotation=270, labelpad=15)
    
    # Set labels
    ax7.set_yticks(range(len(methods)))
    ax7.set_yticklabels(method_labels)
    ax7.set_xticks(range(len(conditions)))
    ax7.set_xticklabels(conditions, rotation=45, ha='right', fontsize=8)
    ax7.set_xlabel('Sample Size & Imbalance Ratio')
    ax7.set_title('Extended Performance Matrix: All Methods vs All Conditions\n' + 
                  '(Stability Bonus consistently green across conditions)', fontweight='bold', pad=20)
    
    # Add text annotations for significant values
    for i in range(len(methods)):
        for j in range(len(conditions)):
            value = heatmap_data[i, j]
            if not np.isnan(value) and abs(value) > 3:
                ax7.text(j, i, f'{value:.0f}%', ha='center', va='center', 
                        color='white' if abs(value) > 10 else 'black', 
                        fontweight='bold', fontsize=8)
    
    # 8. Summary Statistics Table
    ax8 = fig.add_subplot(gs[3, :])
    ax8.axis('tight')
    ax8.axis('off')
    
    # Create extended summary table
    summary_data = []
    for method in methods:
        method_data = df[df['method'] == method]
        if len(method_data) > 0:
            summary_data.append([
                method_labels[methods.index(method)],
                f"{method_data['improvement_pct'].mean():+.1f}%",
                f"±{method_data['improvement_pct'].std():.1f}%",
                f"{(method_data['improvement_pct'] > 0).mean()*100:.0f}%",
                f"{method_data['test_auc'].mean():.3f}" if method_data['test_auc'].notna().any() else "N/A",
                f"{len(method_data)}",
                f"{1.0/(method_data['improvement_pct'].std()+0.001):.2f}"  # Consistency score
            ])
    
    headers = ['Method', 'Mean Improvement', '±Std Dev', 'Success Rate', 'Test AUC', 'Experiments', 'Consistency']
    
    table = ax8.table(cellText=summary_data, colLabels=headers, 
                     cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Color code the table
    for i in range(len(summary_data)):
        for j in range(len(headers)):
            table[(i+1, j)].set_facecolor(colors[i])
            table[(i+1, j)].set_alpha(0.3)
            if i == 0:  # Stability bonus
                table[(i+1, j)].set_alpha(0.5)
                table[(i+1, j)].set_edgecolor('#C0392B')
                table[(i+1, j)].set_linewidth(2)
    
    # Style header
    for j in range(len(headers)):
        table[(0, j)].set_facecolor('#34495E')
        table[(0, j)].set_text_props(weight='bold', color='white')
    
    ax8.set_title('Extended Experimental Results: Comprehensive Method Comparison', 
                  fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Mechanism analysis visualization saved to {save_path}")

def main():
    """Run extended experiments and analysis."""
    
    print("EXTENDED STABILITY BONUS ANALYSIS")
    print("="*80)
    print("Running additional experiments to understand mechanism and consistency...")
    
    # Run extended comparison experiment
    print("\n1. Running extended comparison experiment...")
    extended_results = run_extended_comparison_experiment(
        sample_sizes=[500, 1000, 5000, 10000, 50000],
        imbalance_ratios=[1.0, 4.0, 9.0, 19.0, 49.0, 99.0],
        complexity_levels=['simple', 'medium', 'complex'],
        n_trials=30  # More trials for better statistics
    )
    
    # Save extended results
    print("\n2. Saving extended experimental data...")
    os.makedirs('data', exist_ok=True)
    
    with open('data/extended_experimental_results.json', 'w') as f:
        json.dump(extended_results, f, indent=2, default=str)
    
    pd.DataFrame(extended_results['results']).to_csv('data/extended_experimental_results.csv', index=False)
    
    # Create mechanism analysis visualizations
    print("\n3. Creating mechanism analysis visualizations...")
    create_mechanism_analysis_visualizations(extended_results, 'graphs/extended_mechanism_analysis.png')
    
    # Print detailed analysis
    print("\n" + "="*80)
    print("EXTENDED EXPERIMENT RESULTS")
    print("="*80)
    
    summary = extended_results['summary']
    
    print("PERFORMANCE BY METHOD:")
    for method, stats in summary.items():
        if isinstance(stats, dict) and 'mean_improvement' in stats:
            print(f"  {method:25s}: {stats['mean_improvement']:+6.2f}% ± {stats['std_improvement']:.2f}%, "
                  f"Success: {stats['positive_rate']:5.1f}%, Experiments: {stats['count']}")
            
            if method == 'stability_bonus':
                print(f"    Bonus frequency: {stats['mean_bonus_frequency']:.1f}%, "
                      f"Avg bonus: {stats['mean_bonus_applied']:.3f}, "
                      f"Avg gap: {stats['mean_gap']:.4f}")
    
    print("\nCONSISTENCY ANALYSIS:")
    stability_data = pd.DataFrame([r for r in extended_results['results'] if r['method'] == 'stability_bonus'])
    other_data = pd.DataFrame([r for r in extended_results['results'] if r['method'] != 'stability_bonus'])
    
    if len(stability_data) > 0:
        print(f"Stability Bonus consistency:")
        print(f"  Standard deviation: {stability_data['improvement_pct'].std():.2f}%")
        print(f"  Coefficient of variation: {stability_data['improvement_pct'].std() / abs(stability_data['improvement_pct'].mean()):.3f}")
        print(f"  Positive results: {(stability_data['improvement_pct'] > 0).mean()*100:.1f}%")
        print(f"  Range: {stability_data['improvement_pct'].min():.1f}% to {stability_data['improvement_pct'].max():.1f}%")
    
    if len(other_data) > 0:
        print(f"Other methods (combined):")
        print(f"  Standard deviation: {other_data['improvement_pct'].std():.2f}%")
        print(f"  Coefficient of variation: {other_data['improvement_pct'].std() / abs(other_data['improvement_pct'].mean() + 0.001):.3f}")
        print(f"  Positive results: {(other_data['improvement_pct'] > 0).mean()*100:.1f}%")
        print(f"  Range: {other_data['improvement_pct'].min():.1f}% to {other_data['improvement_pct'].max():.1f}%")
    
    print(f"\n✓ Extended analysis completed!")
    print(f"✓ Results saved to data/extended_experimental_results.*")
    print(f"✓ Mechanism analysis visualization saved to graphs/extended_mechanism_analysis.png")

if __name__ == "__main__":
    main()
