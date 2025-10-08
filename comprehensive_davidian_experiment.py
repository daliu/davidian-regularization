#!/usr/bin/env python3
"""
Comprehensive Davidian Regularization Experiment

This experiment systematically tests Davidian Regularization as an alternative to 
minority class rebalancing across multiple dimensions:

1. Data sources: sklearn make_classification with varying complexity
2. Sample sizes: 50, 500, 5000, 50000
3. Class imbalance ratios: 1:1 (control), 1:9, 1:19, 1:29, 1:49
4. K-fold values: 3, 4, 5, 10
5. Trial counts: 5, 10, 15, 25
6. Models: Linear Regression, Naive Bayes, Gradient Boosted Trees
7. Regularization methods: Original, Conservative, Inverse_diff, Exponential_decay, Stability_bonus, No regularization

The goal is to demonstrate that Davidian Regularization can be used in lieu of 
minority class rebalancing to improve model generalization on imbalanced datasets.
"""

import os
import time
import json
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Force single-threaded execution for reproducibility
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'

# Import required libraries
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler

print("COMPREHENSIVE DAVIDIAN REGULARIZATION EXPERIMENT")
print("="*80)
print("Testing Davidian Regularization in lieu of minority class rebalancing")
print("="*80)


def create_imbalanced_dataset(n_samples: int, imbalance_ratio: float, 
                             n_features: int = 20, n_informative: int = 15,
                             n_redundant: int = 3, n_clusters_per_class: int = 1,
                             random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Create an imbalanced binary classification dataset.
    
    Args:
        n_samples: Total number of samples
        imbalance_ratio: Ratio of majority to minority class (e.g., 19.0 for 95/5 split)
        n_features: Total number of features
        n_informative: Number of informative features
        n_redundant: Number of redundant features
        n_clusters_per_class: Number of clusters per class
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X, y, metadata)
    """
    # Calculate class weights from ratio
    minority_weight = 1.0 / (imbalance_ratio + 1.0)
    majority_weight = 1.0 - minority_weight
    
    # Generate the dataset
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_clusters_per_class=n_clusters_per_class,
        weights=[majority_weight, minority_weight],
        flip_y=0.01,  # Small amount of label noise
        random_state=random_state
    )
    
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Calculate actual class distribution
    class_counts = Counter(y)
    actual_ratio = class_counts[0] / class_counts[1] if class_counts[1] > 0 else float('inf')
    
    metadata = {
        'n_samples': n_samples,
        'n_features': n_features,
        'target_imbalance_ratio': imbalance_ratio,
        'actual_imbalance_ratio': actual_ratio,
        'class_counts': dict(class_counts),
        'minority_percentage': (class_counts[1] / n_samples) * 100,
        'majority_percentage': (class_counts[0] / n_samples) * 100
    }
    
    return X, y, metadata


def apply_davidian_regularization_variant(train_score: float, val_score: float, 
                                        method: str = 'davidian_regularization', 
                                        alpha: float = 1.0) -> float:
    """
    Apply different variants of Davidian Regularization.
    
    Args:
        train_score: Training score
        val_score: Validation score
        method: Regularization method
        alpha: Penalty weight
        
    Returns:
        Regularized validation score
    """
    diff = abs(train_score - val_score)
    
    if method == 'davidian_regularization':
        # Davidian Regularization: val_score - α * |train_score - val_score|
        return val_score - alpha * diff
    
    elif method == 'conservative_davidian':
        # Conservative Davidian: val_score - 0.5 * α * |train_score - val_score|
        return val_score - 0.5 * alpha * diff
    
    elif method == 'inverse_diff':
        # Inverse Difference: val_score * (1 / (1 + |train_score - val_score|))
        confidence = 1.0 / (1.0 + diff)
        return val_score * confidence
    
    elif method == 'exponential_decay':
        # Exponential Decay: val_score * exp(-|train_score - val_score|)
        confidence = np.exp(-diff)
        return val_score * confidence
    
    elif method == 'stability_bonus':
        # Stability Bonus: val_score * (1 + bonus) if diff < threshold
        stability_threshold = 0.1
        if diff < stability_threshold:
            bonus = (stability_threshold - diff) / stability_threshold * 0.2  # Max 20% bonus
            return val_score * (1.0 + bonus)
        else:
            return val_score
    
    elif method == 'standard_stratified_kfold':
        # Standard Stratified K-fold (Control): No regularization - just return validation score
        return val_score
    
    else:
        raise ValueError(f"Unknown regularization method: {method}")


def run_stratified_kfold_experiment(X: np.ndarray, y: np.ndarray, 
                                   model_class: type, model_params: Dict[str, Any],
                                   k: int = 5, n_trials: int = 10,
                                   regularization_method: str = 'davidian_regularization',
                                   alpha: float = 1.0,
                                   random_state: int = 42) -> Dict[str, Any]:
    """
    Run stratified k-fold cross-validation experiment with Davidian Regularization.
    
    This function runs multiple trials of k-fold cross-validation and selects the best
    model based on regularized validation scores. The PRIMARY METRIC is the maximum
    regularized score across all trials (winner-takes-all). Secondary metrics include
    mean scores with standard deviations for confidence interval estimation.
    
    Args:
        X: Feature matrix
        y: Target vector
        model_class: Model class to use
        model_params: Model parameters
        k: Number of folds
        n_trials: Number of trials
        regularization_method: Davidian regularization variant
        alpha: Regularization penalty weight
        random_state: Random seed
        
    Returns:
        Dictionary containing experiment results with max and mean metrics
    """
    trial_results = []
    all_regularized_scores = []
    all_val_scores = []
    all_train_scores = []
    
    for trial in range(n_trials):
        # Create stratified k-fold split
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state + trial)
        
        fold_results = []
        trial_regularized_scores = []
        trial_val_scores = []
        trial_train_scores = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train model
            model = model_class(**model_params)
            model.fit(X_train, y_train)
            
            # Get predictions
            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)
            
            # Calculate scores
            train_score = accuracy_score(y_train, train_pred)
            val_score = accuracy_score(y_val, val_pred)
            
            # Apply Davidian regularization
            regularized_score = apply_davidian_regularization_variant(
                train_score, val_score, regularization_method, alpha
            )
            
            fold_results.append({
                'fold': fold_idx,
                'train_score': train_score,
                'val_score': val_score,
                'regularized_score': regularized_score,
                'penalty': abs(train_score - val_score) * alpha
            })
            
            trial_regularized_scores.append(regularized_score)
            trial_val_scores.append(val_score)
            trial_train_scores.append(train_score)
        
        # Calculate trial statistics
        trial_result = {
            'trial': trial,
            'fold_results': fold_results,
            'mean_regularized_score': np.mean(trial_regularized_scores),
            'mean_val_score': np.mean(trial_val_scores),
            'mean_train_score': np.mean(trial_train_scores),
            'std_regularized_score': np.std(trial_regularized_scores),
            'std_val_score': np.std(trial_val_scores),
            'std_train_score': np.std(trial_train_scores)
        }
        
        trial_results.append(trial_result)
        all_regularized_scores.append(trial_result['mean_regularized_score'])
        all_val_scores.append(trial_result['mean_val_score'])
        all_train_scores.append(trial_result['mean_train_score'])
    
    # Calculate mean performance and confidence intervals (focus on generalizability)
    mean_regularized_score = np.mean(all_regularized_scores)
    std_regularized_score = np.std(all_regularized_scores)
    se_regularized_score = std_regularized_score / np.sqrt(n_trials)  # Standard error
    ci_95_regularized = 1.96 * se_regularized_score  # 95% confidence interval
    
    mean_val_score = np.mean(all_val_scores)
    std_val_score = np.std(all_val_scores)
    se_val_score = std_val_score / np.sqrt(n_trials)
    ci_95_val = 1.96 * se_val_score
    
    return {
        'trial_results': trial_results,
        # PRIMARY METRICS (Mean performance with confidence intervals)
        'mean_regularized_score': mean_regularized_score,
        'std_regularized_score': std_regularized_score,
        'se_regularized_score': se_regularized_score,
        'ci_95_regularized_score': ci_95_regularized,
        'mean_val_score': mean_val_score,
        'std_val_score': std_val_score,
        'se_val_score': se_val_score,
        'ci_95_val_score': ci_95_val,
        'mean_train_score': np.mean(all_train_scores),
        'std_train_score': np.std(all_train_scores),
        # Distribution metrics for analysis
        'min_regularized_score': np.min(all_regularized_scores),
        'max_regularized_score': np.max(all_regularized_scores),
        'median_regularized_score': np.median(all_regularized_scores),
        # Experiment configuration
        'regularization_method': regularization_method,
        'k': k,
        'n_trials': n_trials,
        'alpha': alpha
    }


def run_random_sampling_baseline(X: np.ndarray, y: np.ndarray,
                                model_class: type, model_params: Dict[str, Any],
                                n_trials: int = 10, test_size: float = 0.2,
                                random_state: int = 42) -> Dict[str, Any]:
    """
    Run random holdout validation baseline for comparison.
    
    This is the CONTROL/BASELINE method that uses traditional random train-validation
    splits (stratified). All Davidian regularization methods are compared against this.
    The PRIMARY METRIC is the maximum validation score (winner-takes-all).
    
    Args:
        X: Feature matrix
        y: Target vector
        model_class: Model class to use
        model_params: Model parameters
        n_trials: Number of random splits
        test_size: Fraction for validation
        random_state: Random seed
        
    Returns:
        Dictionary containing baseline results with max and mean metrics
    """
    trial_results = []
    all_val_scores = []
    all_train_scores = []
    
    for trial in range(n_trials):
        # Random stratified split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_state + trial,
            stratify=y
        )
        
        # Train model
        model = model_class(**model_params)
        model.fit(X_train, y_train)
        
        # Get predictions and scores
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        
        train_score = accuracy_score(y_train, train_pred)
        val_score = accuracy_score(y_val, val_pred)
        
        trial_results.append({
            'trial': trial,
            'train_score': train_score,
            'val_score': val_score
        })
        
        all_val_scores.append(val_score)
        all_train_scores.append(train_score)
    
    # Calculate mean performance and confidence intervals for baseline
    mean_val_score = np.mean(all_val_scores)
    std_val_score = np.std(all_val_scores)
    se_val_score = std_val_score / np.sqrt(n_trials)  # Standard error
    ci_95_val = 1.96 * se_val_score  # 95% confidence interval
    
    mean_train_score = np.mean(all_train_scores)
    std_train_score = np.std(all_train_scores)
    
    return {
        'trial_results': trial_results,
        # PRIMARY METRICS (Mean performance with confidence intervals)
        'mean_val_score': mean_val_score,
        'std_val_score': std_val_score,
        'se_val_score': se_val_score,
        'ci_95_val_score': ci_95_val,
        'mean_train_score': mean_train_score,
        'std_train_score': std_train_score,
        # Distribution metrics for analysis
        'min_val_score': np.min(all_val_scores),
        'max_val_score': np.max(all_val_scores),
        'median_val_score': np.median(all_val_scores),
        # Experiment configuration
        'n_trials': n_trials,
        'method': 'random_holdout_validation'
    }


def evaluate_model_on_test_set(X_train: np.ndarray, y_train: np.ndarray,
                              X_test: np.ndarray, y_test: np.ndarray,
                              model_class: type, model_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate model on held-out test set and calculate comprehensive metrics.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        model_class: Model class
        model_params: Model parameters
        
    Returns:
        Dictionary containing test set evaluation metrics
    """
    # Train model on full training set
    model = model_class(**model_params)
    model.fit(X_train, y_train)
    
    # Get predictions
    y_pred = model.predict(X_test)
    
    # Calculate comprehensive metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }
    
    # Add AUC if model supports probability prediction
    if hasattr(model, 'predict_proba'):
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
            metrics['auc'] = roc_auc_score(y_test, y_proba)
        except:
            metrics['auc'] = None
    else:
        metrics['auc'] = None
    
    # Add classification report
    metrics['classification_report'] = classification_report(y_test, y_pred, output_dict=True)
    
    return metrics


def run_comprehensive_experiment(sample_sizes: List[int] = [50, 500, 5000, 50000],
                               imbalance_ratios: List[float] = [1.0, 9.0, 19.0, 29.0, 49.0],
                               k_values: List[int] = [3, 4, 5, 10],
                               trial_counts: List[int] = [5, 10, 15, 25],
                               regularization_methods: List[str] = ['davidian_regularization', 'conservative_davidian', 'inverse_diff', 'exponential_decay', 'stability_bonus', 'standard_stratified_kfold'],
                               model_types: List[str] = ['logistic', 'naive_bayes', 'gradient_boosting'],
                               max_experiments: int = 100) -> Dict[str, Any]:
    """
    Run comprehensive Davidian Regularization experiment across all parameter combinations.
    
    Args:
        sample_sizes: List of sample sizes to test
        imbalance_ratios: List of class imbalance ratios
        k_values: List of k-fold values
        trial_counts: List of trial counts
        regularization_methods: List of regularization methods
        model_types: List of model types
        max_experiments: Maximum number of experiments to run
        
    Returns:
        Dictionary containing all experimental results
    """
    # Model configurations
    model_configs = {
        'logistic': {
            'class': LogisticRegression,
            'params': {'random_state': 42, 'max_iter': 1000, 'n_jobs': 1, 'solver': 'liblinear'}
        },
        'naive_bayes': {
            'class': GaussianNB,
            'params': {}
        },
        'gradient_boosting': {
            'class': GradientBoostingClassifier,
            'params': {'random_state': 42, 'n_estimators': 50}  # Reduced for speed
        }
    }
    
    all_results = []
    experiment_count = 0
    
    print(f"Starting comprehensive experiment with max {max_experiments} experiments...")
    print(f"Parameter space: {len(sample_sizes)} sample sizes × {len(imbalance_ratios)} ratios × {len(k_values)} k-values × {len(trial_counts)} trial counts × {len(regularization_methods)} methods × {len(model_types)} models")
    
    start_time = time.time()
    
    # Iterate through all parameter combinations
    for sample_size in sample_sizes:
        for imbalance_ratio in imbalance_ratios:
            for model_type in model_types:
                for k in k_values:
                    for n_trials in trial_counts:
                        for reg_method in regularization_methods:
                            
                            if experiment_count >= max_experiments:
                                print(f"Reached maximum experiment limit ({max_experiments})")
                                break
                            
                            experiment_count += 1
                            print(f"\nExperiment {experiment_count}/{max_experiments}")
                            print(f"  Sample size: {sample_size}, Ratio: 1:{imbalance_ratio:.0f}, Model: {model_type}")
                            print(f"  K-folds: {k}, Trials: {n_trials}, Method: {reg_method}")
                            
                            try:
                                # Create dataset
                                X, y, dataset_metadata = create_imbalanced_dataset(
                                    n_samples=sample_size,
                                    imbalance_ratio=imbalance_ratio,
                                    random_state=42 + experiment_count
                                )
                                
                                # Split into train/test
                                X_train, X_test, y_train, y_test = train_test_split(
                                    X, y, test_size=0.2, random_state=42 + experiment_count,
                                    stratify=y
                                )
                                
                                # Get model configuration
                                model_config = model_configs[model_type]
                                
                                # Run Davidian regularization experiment
                                davidian_results = run_stratified_kfold_experiment(
                                    X_train, y_train,
                                    model_config['class'], model_config['params'],
                                    k=k, n_trials=n_trials,
                                    regularization_method=reg_method,
                                    random_state=42 + experiment_count
                                )
                                
                                # Run random holdout validation baseline
                                random_results = run_random_sampling_baseline(
                                    X_train, y_train,
                                    model_config['class'], model_config['params'],
                                    n_trials=n_trials,
                                    random_state=42 + experiment_count
                                )
                                
                                # Evaluate on test set
                                test_metrics = evaluate_model_on_test_set(
                                    X_train, y_train, X_test, y_test,
                                    model_config['class'], model_config['params']
                                )
                                
                                # Calculate comparison metrics (focus on mean performance and generalizability)
                                mean_method_score = davidian_results['mean_regularized_score']
                                mean_baseline_score = random_results['mean_val_score']
                                
                                # Method confidence intervals
                                method_ci_95 = davidian_results['ci_95_regularized_score']
                                baseline_ci_95 = random_results['ci_95_val_score']
                                
                                # Performance comparison with statistical significance
                                mean_improvement = mean_method_score - mean_baseline_score
                                mean_improvement_pct = (mean_improvement / abs(mean_baseline_score)) * 100 if mean_baseline_score != 0 else 0
                                
                                # Statistical significance test (non-overlapping confidence intervals)
                                method_lower = mean_method_score - method_ci_95
                                method_upper = mean_method_score + method_ci_95
                                baseline_lower = mean_baseline_score - baseline_ci_95
                                baseline_upper = mean_baseline_score + baseline_ci_95
                                
                                # Check if confidence intervals overlap
                                ci_overlap = not (method_upper < baseline_lower or baseline_upper < method_lower)
                                statistically_significant = not ci_overlap
                                method_better = mean_method_score > mean_baseline_score
                                
                                # Store results
                                experiment_result = {
                                    'experiment_id': experiment_count,
                                    'parameters': {
                                        'sample_size': sample_size,
                                        'imbalance_ratio': imbalance_ratio,
                                        'model_type': model_type,
                                        'k': k,
                                        'n_trials': n_trials,
                                        'regularization_method': reg_method
                                    },
                                    'dataset_metadata': dataset_metadata,
                                    'method_results': davidian_results,
                                    'baseline_results': random_results,
                                    'test_metrics': test_metrics,
                                    'comparison': {
                                        # MEAN PERFORMANCE METRICS (Primary focus on generalizability)
                                        'mean_method_score': mean_method_score,
                                        'mean_baseline_score': mean_baseline_score,
                                        'mean_improvement': mean_improvement,
                                        'mean_improvement_pct': mean_improvement_pct,
                                        'method_better': method_better,
                                        # CONFIDENCE INTERVALS AND STATISTICAL SIGNIFICANCE
                                        'method_ci_95': method_ci_95,
                                        'baseline_ci_95': baseline_ci_95,
                                        'method_ci_lower': method_lower,
                                        'method_ci_upper': method_upper,
                                        'baseline_ci_lower': baseline_lower,
                                        'baseline_ci_upper': baseline_upper,
                                        'ci_overlap': ci_overlap,
                                        'statistically_significant': statistically_significant,
                                        # VARIABILITY METRICS
                                        'method_std': davidian_results['std_regularized_score'],
                                        'baseline_std': random_results['std_val_score'],
                                        'method_se': davidian_results['se_regularized_score'],
                                        'baseline_se': random_results['se_val_score'],
                                        # TEST SET METRICS
                                        'test_auc': test_metrics.get('auc', None),
                                        'test_accuracy': test_metrics['accuracy'],
                                        'test_f1': test_metrics['f1_score']
                                    }
                                }
                                
                                all_results.append(experiment_result)
                                
                                print(f"  Method (mean): {mean_method_score:.4f} ± {method_ci_95:.4f}")
                                print(f"  Baseline (mean): {mean_baseline_score:.4f} ± {baseline_ci_95:.4f}")
                                print(f"  Test AUC: {test_metrics.get('auc', 'N/A')}")
                                print(f"  Mean improvement: {mean_improvement_pct:+.2f}%")
                                print(f"  Statistically significant: {'YES' if statistically_significant else 'NO'}")
                                print(f"  Method better: {'YES' if method_better else 'NO'}")
                                
                            except Exception as e:
                                print(f"  ERROR: {str(e)}")
                                continue
                        
                        if experiment_count >= max_experiments:
                            break
                    if experiment_count >= max_experiments:
                        break
                if experiment_count >= max_experiments:
                    break
            if experiment_count >= max_experiments:
                break
        if experiment_count >= max_experiments:
            break
    
    elapsed_time = time.time() - start_time
    
    # Compile summary statistics
    summary = {
        'total_experiments': len(all_results),
        'elapsed_time_seconds': elapsed_time,
        'parameters_tested': {
            'sample_sizes': sample_sizes,
            'imbalance_ratios': imbalance_ratios,
            'k_values': k_values,
            'trial_counts': trial_counts,
            'regularization_methods': regularization_methods,
            'model_types': model_types
        },
        'overall_statistics': calculate_overall_statistics(all_results)
    }
    
    return {
        'summary': summary,
        'results': all_results
    }


def calculate_overall_statistics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate overall statistics across all experiments."""
    if not results:
        return {}
    
    # Extract MEAN PERFORMANCE metrics (focus on generalizability)
    mean_improvements = [r['comparison']['mean_improvement_pct'] for r in results]
    method_better_count = [r['comparison']['method_better'] for r in results]
    statistically_significant_count = [r['comparison']['statistically_significant'] for r in results]
    mean_method_scores = [r['comparison']['mean_method_score'] for r in results]
    mean_baseline_scores = [r['comparison']['mean_baseline_score'] for r in results]
    
    # Extract CONFIDENCE INTERVAL metrics
    method_cis = [r['comparison']['method_ci_95'] for r in results]
    baseline_cis = [r['comparison']['baseline_ci_95'] for r in results]
    
    # Extract TEST metrics
    test_aucs = [r['comparison']['test_auc'] for r in results if r['comparison']['test_auc'] is not None]
    test_accuracies = [r['comparison']['test_accuracy'] for r in results]
    test_f1s = [r['comparison']['test_f1'] for r in results]
    
    # Count improvements and statistical significance
    total_better = sum(method_better_count)
    total_significant = sum(statistically_significant_count)
    
    # Statistics by regularization method (focus on mean performance and significance)
    method_stats = defaultdict(lambda: {
        'improvements': [], 'better_count': [], 'significant_count': [], 
        'method_scores': [], 'baseline_scores': [], 'method_cis': [], 'baseline_cis': []
    })
    
    for result in results:
        method = result['parameters']['regularization_method']
        comp = result['comparison']
        method_stats[method]['improvements'].append(comp['mean_improvement_pct'])
        method_stats[method]['better_count'].append(comp['method_better'])
        method_stats[method]['significant_count'].append(comp['statistically_significant'])
        method_stats[method]['method_scores'].append(comp['mean_method_score'])
        method_stats[method]['baseline_scores'].append(comp['mean_baseline_score'])
        method_stats[method]['method_cis'].append(comp['method_ci_95'])
        method_stats[method]['baseline_cis'].append(comp['baseline_ci_95'])
    
    method_summary = {}
    for method, stats in method_stats.items():
        n_experiments = len(stats['improvements'])
        method_summary[method] = {
            'count': n_experiments,
            # MEAN PERFORMANCE METRICS
            'mean_improvement': np.mean(stats['improvements']),
            'std_improvement': np.std(stats['improvements']),
            'median_improvement': np.median(stats['improvements']),
            'improvement_ci_95': 1.96 * np.std(stats['improvements']) / np.sqrt(n_experiments),
            # SUCCESS RATES
            'better_rate': sum(stats['better_count']) / n_experiments * 100,
            'significance_rate': sum(stats['significant_count']) / n_experiments * 100,
            # SCORE COMPARISONS
            'mean_method_score': np.mean(stats['method_scores']),
            'mean_baseline_score': np.mean(stats['baseline_scores']),
            'mean_method_ci': np.mean(stats['method_cis']),
            'mean_baseline_ci': np.mean(stats['baseline_cis']),
            # GENERALIZABILITY INDICATORS
            'score_stability': np.std(stats['method_scores']),  # Lower is more stable
            'relative_stability': np.std(stats['method_scores']) / np.std(stats['baseline_scores']) if np.std(stats['baseline_scores']) > 0 else float('inf')
        }
    
    return {
        # OVERALL PERFORMANCE METRICS (Mean-based, not winner-takes-all)
        'overall_better_rate': (total_better / len(results)) * 100,
        'overall_significance_rate': (total_significant / len(results)) * 100,
        'mean_improvement': np.mean(mean_improvements),
        'median_improvement': np.median(mean_improvements),
        'std_improvement': np.std(mean_improvements),
        'improvement_ci_95': 1.96 * np.std(mean_improvements) / np.sqrt(len(mean_improvements)),
        # SCORE COMPARISONS
        'overall_mean_method_score': np.mean(mean_method_scores),
        'overall_mean_baseline_score': np.mean(mean_baseline_scores),
        'overall_mean_method_ci': np.mean(method_cis),
        'overall_mean_baseline_ci': np.mean(baseline_cis),
        # TEST SET METRICS
        'mean_test_auc': np.mean(test_aucs) if test_aucs else None,
        'mean_test_accuracy': np.mean(test_accuracies),
        'mean_test_f1': np.mean(test_f1s),
        # BY METHOD ANALYSIS (focus on generalizability)
        'by_regularization_method': method_summary
    }


def create_comprehensive_results_table(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Create a comprehensive results table for analysis."""
    rows = []
    
    for result in results:
        params = result['parameters']
        comparison = result['comparison']
        test_metrics = result['test_metrics']
        
        row = {
            'experiment_id': result['experiment_id'],
            'sample_size': params['sample_size'],
            'imbalance_ratio': f"1:{params['imbalance_ratio']:.0f}",
            'model_type': params['model_type'],
            'k_folds': params['k'],
            'n_trials': params['n_trials'],
            'regularization_method': params['regularization_method'],
            # MEAN PERFORMANCE METRICS (Primary focus)
            'mean_method_score': comparison['mean_method_score'],
            'mean_baseline_score': comparison['mean_baseline_score'],
            'mean_improvement_pct': comparison['mean_improvement_pct'],
            'method_better': comparison['method_better'],
            'statistically_significant': comparison['statistically_significant'],
            # CONFIDENCE INTERVALS
            'method_ci_95': comparison['method_ci_95'],
            'baseline_ci_95': comparison['baseline_ci_95'],
            'ci_overlap': comparison['ci_overlap'],
            # VARIABILITY METRICS
            'method_std': comparison['method_std'],
            'baseline_std': comparison['baseline_std'],
            # TEST SET METRICS
            'test_accuracy': comparison['test_accuracy'],
            'test_f1': comparison['test_f1'],
            'test_auc': comparison['test_auc'],
            # DATASET INFO
            'minority_percentage': result['dataset_metadata']['minority_percentage']
        }
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def save_results(results: Dict[str, Any], filename: str = 'comprehensive_davidian_results.json'):
    """Save results to JSON file."""
    # Convert numpy types to native Python types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj
    
    results_serializable = convert_numpy(results)
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"Results saved to {filename}")


def main():
    """Main experiment execution."""
    print("Starting comprehensive Davidian Regularization experiment...")
    
    # Define experimental parameters
    # Reduced parameter space for initial testing - can be expanded
    sample_sizes = [500, 5000]  # Start with smaller set
    imbalance_ratios = [1.0, 9.0, 19.0]  # 1:1, 1:9, 1:19
    k_values = [3, 5]  # Reduced set
    trial_counts = [5, 10]  # Reduced set
    regularization_methods = ['davidian_regularization', 'conservative_davidian', 'inverse_diff', 'exponential_decay', 'stability_bonus', 'standard_stratified_kfold']
    model_types = ['logistic', 'naive_bayes', 'gradient_boosting']
    
    # Run comprehensive experiment
    results = run_comprehensive_experiment(
        sample_sizes=sample_sizes,
        imbalance_ratios=imbalance_ratios,
        k_values=k_values,
        trial_counts=trial_counts,
        regularization_methods=regularization_methods,
        model_types=model_types,
        max_experiments=50  # Limit for initial run
    )
    
    # Save results
    save_results(results, 'results/comprehensive_davidian_results.json')
    
    # Create and save results table
    results_df = create_comprehensive_results_table(results['results'])
    results_df.to_csv('results/comprehensive_davidian_results.csv', index=False)
    
    # Print summary
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    
    summary = results['summary']
    overall_stats = summary['overall_statistics']
    
    print(f"Total experiments completed: {summary['total_experiments']}")
    print(f"Total elapsed time: {summary['elapsed_time_seconds']:.1f} seconds")
    print(f"Overall method better rate vs baseline: {overall_stats['overall_better_rate']:.1f}%")
    print(f"Overall statistical significance rate: {overall_stats['overall_significance_rate']:.1f}%")
    print(f"Mean improvement vs baseline: {overall_stats['mean_improvement']:+.2f}% ± {overall_stats['improvement_ci_95']:.2f}%")
    print(f"Mean test AUC: {overall_stats.get('mean_test_auc', 'N/A')}")
    print(f"Mean test accuracy: {overall_stats['mean_test_accuracy']:.4f}")
    
    print("\nPerformance by regularization method (vs Random Holdout Baseline):")
    for method, stats in overall_stats['by_regularization_method'].items():
        print(f"  {method:25s}: {stats['better_rate']:5.1f}% better rate, {stats['significance_rate']:5.1f}% significant, {stats['mean_improvement']:+6.2f}% ± {stats['improvement_ci_95']:.2f}% improvement")
    
    print(f"\nDetailed results saved to:")
    print(f"  - results/comprehensive_davidian_results.json")
    print(f"  - results/comprehensive_davidian_results.csv")


if __name__ == "__main__":
    main()
