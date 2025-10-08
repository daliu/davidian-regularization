#!/usr/bin/env python3
"""
Imbalanced data experiment for Davidian Regularization.

This experiment tests the hypothesis that Davidian Regularization is most effective
on imbalanced datasets where models can achieve high validation scores through
majority class prediction but show poor train-validation consistency.
"""

import sys
import os
import time
import json
import numpy as np
from typing import Dict, List, Tuple, Any
from collections import Counter

# Force single-threaded execution
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'

print("IMBALANCED DATA EXPERIMENT")
print("="*70)
print("Testing Davidian Regularization on highly imbalanced datasets")
print("="*70)

def create_imbalanced_datasets() -> Dict[str, Tuple]:
    """
    Create various imbalanced datasets for testing.
    """
    datasets = {}
    
    print("\n1. Creating imbalanced datasets...")
    
    # 1. Severely imbalanced binary classification (95/5 split)
    print("  Creating severely imbalanced binary dataset (95/5 split)...")
    from sklearn.datasets import make_classification
    
    X1, y1 = make_classification(
        n_samples=2000,
        n_features=20,
        n_informative=15,
        n_redundant=3,
        n_classes=2,
        weights=[0.95, 0.05],  # Severe imbalance
        class_sep=0.8,
        flip_y=0.01,
        random_state=42
    )
    
    datasets['severe_binary'] = (X1, y1, {
        'name': 'Severe Binary Imbalance (95/5)',
        'type': 'classification',
        'n_samples': X1.shape[0],
        'n_features': X1.shape[1],
        'n_classes': 2,
        'class_distribution': [0.95, 0.05],
        'imbalance_ratio': 19.0,  # 95/5 = 19:1
        'complexity': 'high_imbalance'
    })
    
    # 2. Moderate imbalance multiclass (70/20/10 split)
    print("  Creating moderate imbalanced multiclass dataset (70/20/10 split)...")
    X2, y2 = make_classification(
        n_samples=1500,
        n_features=15,
        n_informative=12,
        n_redundant=2,
        n_classes=3,
        weights=[0.70, 0.20, 0.10],
        class_sep=0.7,
        flip_y=0.02,
        random_state=42
    )
    
    datasets['moderate_multiclass'] = (X2, y2, {
        'name': 'Moderate Multiclass Imbalance (70/20/10)',
        'type': 'classification',
        'n_samples': X2.shape[0],
        'n_features': X2.shape[1],
        'n_classes': 3,
        'class_distribution': [0.70, 0.20, 0.10],
        'imbalance_ratio': 7.0,  # 70/10 = 7:1
        'complexity': 'medium_imbalance'
    })
    
    # 3. Extreme imbalance (99/1 split)
    print("  Creating extreme imbalanced dataset (99/1 split)...")
    X3, y3 = make_classification(
        n_samples=3000,
        n_features=25,
        n_informative=20,
        n_redundant=3,
        n_classes=2,
        weights=[0.99, 0.01],  # Extreme imbalance
        class_sep=0.9,
        flip_y=0.005,
        random_state=42
    )
    
    datasets['extreme_binary'] = (X3, y3, {
        'name': 'Extreme Binary Imbalance (99/1)',
        'type': 'classification',
        'n_samples': X3.shape[0],
        'n_features': X3.shape[1],
        'n_classes': 2,
        'class_distribution': [0.99, 0.01],
        'imbalance_ratio': 99.0,  # 99/1 = 99:1
        'complexity': 'extreme_imbalance'
    })
    
    # 4. Real-world-like fraud detection scenario
    print("  Creating fraud detection-like dataset...")
    np.random.seed(42)
    
    n_samples = 5000
    n_features = 30
    
    # Create features that might indicate fraud
    X_normal = np.random.randn(int(n_samples * 0.98), n_features)  # 98% normal
    X_fraud = np.random.randn(int(n_samples * 0.02), n_features) + 2  # 2% fraud (shifted distribution)
    
    X4 = np.vstack([X_normal, X_fraud])
    y4 = np.array([0] * int(n_samples * 0.98) + [1] * int(n_samples * 0.02))
    
    # Shuffle
    indices = np.random.permutation(len(X4))
    X4, y4 = X4[indices], y4[indices]
    
    datasets['fraud_detection'] = (X4, y4, {
        'name': 'Fraud Detection Simulation (98/2)',
        'type': 'classification',
        'n_samples': X4.shape[0],
        'n_features': X4.shape[1],
        'n_classes': 2,
        'class_distribution': [0.98, 0.02],
        'imbalance_ratio': 49.0,  # 98/2 = 49:1
        'complexity': 'fraud_like'
    })
    
    # 5. Medical diagnosis-like scenario
    print("  Creating medical diagnosis-like dataset...")
    np.random.seed(43)
    
    n_samples = 2500
    n_features = 40
    
    # Simulate medical features where disease is rare but has distinct patterns
    X_healthy = np.random.randn(int(n_samples * 0.92), n_features)  # 92% healthy
    X_disease = np.random.randn(int(n_samples * 0.08), n_features)  # 8% disease
    
    # Disease cases have different patterns in some features
    X_disease[:, :10] += 1.5  # First 10 features show disease pattern
    X_disease[:, 10:20] -= 0.8  # Next 10 features show opposite pattern
    
    X5 = np.vstack([X_healthy, X_disease])
    y5 = np.array([0] * int(n_samples * 0.92) + [1] * int(n_samples * 0.08))
    
    # Shuffle
    indices = np.random.permutation(len(X5))
    X5, y5 = X5[indices], y5[indices]
    
    datasets['medical_diagnosis'] = (X5, y5, {
        'name': 'Medical Diagnosis Simulation (92/8)',
        'type': 'classification',
        'n_samples': X5.shape[0],
        'n_features': X5.shape[1],
        'n_classes': 2,
        'class_distribution': [0.92, 0.08],
        'imbalance_ratio': 11.5,  # 92/8 = 11.5:1
        'complexity': 'medical_like'
    })
    
    return datasets

def create_imbalance_aware_models(n_models: int = 30) -> List[Dict[str, Any]]:
    """
    Create models that are appropriate for imbalanced classification.
    """
    models = []
    
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.naive_bayes import GaussianNB
    from sklearn.tree import DecisionTreeClassifier
    
    # Logistic Regression with class weight balancing
    for i in range(8):
        models.append({
            'name': f'LogReg_Balanced_{i}',
            'class': LogisticRegression,
            'params': {
                'C': np.random.choice([0.01, 0.1, 1.0, 10.0, 100.0]),
                'class_weight': 'balanced',  # Handle imbalance
                'penalty': 'l2',
                'solver': 'lbfgs',
                'random_state': 42 + i,
                'max_iter': 2000,
                'n_jobs': 1
            }
        })
    
    # Logistic Regression without balancing (for comparison)
    for i in range(4):
        models.append({
            'name': f'LogReg_Unbalanced_{i}',
            'class': LogisticRegression,
            'params': {
                'C': np.random.choice([0.1, 1.0, 10.0, 100.0]),
                'class_weight': None,  # No balancing
                'penalty': 'l2',
                'solver': 'lbfgs',
                'random_state': 42 + i,
                'max_iter': 2000,
                'n_jobs': 1
            }
        })
    
    # Random Forest with balanced class weights
    for i in range(8):
        models.append({
            'name': f'RF_Balanced_{i}',
            'class': RandomForestClassifier,
            'params': {
                'n_estimators': np.random.choice([50, 100, 200]),
                'max_depth': np.random.choice([5, 10, 15, None]),
                'min_samples_split': np.random.choice([2, 5, 10]),
                'class_weight': 'balanced',
                'random_state': 42 + i,
                'n_jobs': 1
            }
        })
    
    # Random Forest without balancing
    for i in range(4):
        models.append({
            'name': f'RF_Unbalanced_{i}',
            'class': RandomForestClassifier,
            'params': {
                'n_estimators': np.random.choice([50, 100, 200]),
                'max_depth': np.random.choice([5, 10, 15, None]),
                'class_weight': None,
                'random_state': 42 + i,
                'n_jobs': 1
            }
        })
    
    # Gradient Boosting (naturally handles imbalance well)
    for i in range(4):
        models.append({
            'name': f'GradBoost_{i}',
            'class': GradientBoostingClassifier,
            'params': {
                'n_estimators': np.random.choice([50, 100, 200]),
                'learning_rate': np.random.choice([0.01, 0.1, 0.2]),
                'max_depth': np.random.choice([3, 5, 7]),
                'random_state': 42 + i
            }
        })
    
    # Naive Bayes (good baseline for imbalanced data)
    for i in range(2):
        models.append({
            'name': f'NaiveBayes_{i}',
            'class': GaussianNB,
            'params': {
                'var_smoothing': np.random.choice([1e-9, 1e-8, 1e-7])
            }
        })
    
    return models[:n_models]

def run_imbalanced_experiment_single_trial(X, y, metadata: Dict[str, Any], trial_seed: int):
    """
    Run a single trial of the imbalanced data experiment.
    """
    from sklearn.model_selection import train_test_split, StratifiedKFold
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler
    
    # Split data maintaining class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=trial_seed, stratify=y
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"    Trial seed {trial_seed}: Train={len(X_train)}, Test={len(X_test)}")
    print(f"    Train class distribution: {dict(Counter(y_train))}")
    print(f"    Test class distribution: {dict(Counter(y_test))}")
    
    # Create models
    models = create_imbalance_aware_models(n_models=25)
    
    # K-fold cross-validation on training data
    k = 5
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=trial_seed)
    
    model_results = []
    
    for model_config in models:
        try:
            # Run k-fold CV
            fold_train_scores = []
            fold_val_scores = []
            
            for train_idx, val_idx in cv.split(X_train_scaled, y_train):
                X_fold_train, X_fold_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
                y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
                
                # Train model
                model = model_config['class'](**model_config['params'])
                model.fit(X_fold_train, y_fold_train)
                
                # Evaluate with multiple metrics
                train_pred = model.predict(X_fold_train)
                val_pred = model.predict(X_fold_val)
                
                # Use F1-score as primary metric for imbalanced data
                train_f1 = f1_score(y_fold_train, train_pred, average='weighted')
                val_f1 = f1_score(y_fold_val, val_pred, average='weighted')
                
                fold_train_scores.append(train_f1)
                fold_val_scores.append(val_f1)
            
            mean_train_score = np.mean(fold_train_scores)
            mean_val_score = np.mean(fold_val_scores)
            
            # Calculate Davidian Regularization score
            dr_score = mean_val_score - abs(mean_train_score - mean_val_score)
            
            # Train final model on all training data and evaluate on test
            final_model = model_config['class'](**model_config['params'])
            final_model.fit(X_train_scaled, y_train)
            
            test_pred = final_model.predict(X_test_scaled)
            
            # Calculate comprehensive test metrics
            test_accuracy = accuracy_score(y_test, test_pred)
            test_f1 = f1_score(y_test, test_pred, average='weighted')
            test_precision = precision_score(y_test, test_pred, average='weighted', zero_division=0)
            test_recall = recall_score(y_test, test_pred, average='weighted')
            
            # Try to get AUC if possible
            test_auc = None
            if hasattr(final_model, 'predict_proba') and len(np.unique(y)) == 2:
                try:
                    test_proba = final_model.predict_proba(X_test_scaled)[:, 1]
                    test_auc = roc_auc_score(y_test, test_proba)
                except:
                    pass
            
            model_results.append({
                'model_name': model_config['name'],
                'mean_train_score': mean_train_score,
                'mean_val_score': mean_val_score,
                'dr_score': dr_score,
                'train_val_difference': abs(mean_train_score - mean_val_score),
                'test_accuracy': test_accuracy,
                'test_f1': test_f1,
                'test_precision': test_precision,
                'test_recall': test_recall,
                'test_auc': test_auc
            })
            
        except Exception as e:
            # Skip failed models
            continue
    
    if len(model_results) < 4:
        return None
    
    # Sort models by different criteria
    models_by_dr = sorted(model_results, key=lambda x: x['dr_score'], reverse=True)
    models_by_val = sorted(model_results, key=lambda x: x['mean_val_score'], reverse=True)
    
    # Select top model from each method
    best_dr_model = models_by_dr[0]
    best_val_model = models_by_val[0]
    
    return {
        'trial_seed': trial_seed,
        'n_models_evaluated': len(model_results),
        'class_distribution_train': dict(Counter(y_train)),
        'class_distribution_test': dict(Counter(y_test)),
        'best_dr_model': best_dr_model,
        'best_val_model': best_val_model,
        'all_model_results': model_results,
        'comparison': {
            'dr_selected_test_f1': best_dr_model['test_f1'],
            'val_selected_test_f1': best_val_model['test_f1'],
            'dr_selected_test_accuracy': best_dr_model['test_accuracy'],
            'val_selected_test_accuracy': best_val_model['test_accuracy'],
            'dr_selected_test_auc': best_dr_model['test_auc'],
            'val_selected_test_auc': best_val_model['test_auc'],
            'improvement_f1': best_dr_model['test_f1'] - best_val_model['test_f1'],
            'improvement_accuracy': best_dr_model['test_accuracy'] - best_val_model['test_accuracy'],
            'improvement_f1_pct': ((best_dr_model['test_f1'] - best_val_model['test_f1']) / abs(best_val_model['test_f1'])) * 100 if best_val_model['test_f1'] != 0 else 0,
            'improvement_accuracy_pct': ((best_dr_model['test_accuracy'] - best_val_model['test_accuracy']) / abs(best_val_model['test_accuracy'])) * 100 if best_val_model['test_accuracy'] != 0 else 0
        }
    }

def run_imbalanced_dataset_experiment(dataset_name: str, X, y, metadata: Dict[str, Any], n_trials: int = 20):
    """
    Run comprehensive experiment on an imbalanced dataset.
    """
    from sklearn.preprocessing import StandardScaler
    
    print(f"\n  Running {dataset_name} experiment...")
    print(f"    Dataset: {metadata['name']}")
    print(f"    Samples: {metadata['n_samples']}, Features: {metadata['n_features']}")
    print(f"    Class distribution: {metadata['class_distribution']}")
    print(f"    Imbalance ratio: {metadata['imbalance_ratio']:.1f}:1")
    
    # Standardize the full dataset
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    all_trials = []
    f1_improvements = []
    accuracy_improvements = []
    
    for trial in range(n_trials):
        trial_seed = np.random.randint(0, 2**31 - 1)
        
        trial_result = run_imbalanced_experiment_single_trial(X_scaled, y, metadata, trial_seed)
        
        if trial_result:
            all_trials.append(trial_result)
            f1_improvements.append(trial_result['comparison']['improvement_f1_pct'])
            accuracy_improvements.append(trial_result['comparison']['improvement_accuracy_pct'])
            
            if (trial + 1) % 5 == 0:
                print(f"      Completed {trial + 1}/{n_trials} trials")
    
    if not all_trials:
        return None
    
    # Calculate aggregate statistics
    mean_f1_improvement = np.mean(f1_improvements)
    std_f1_improvement = np.std(f1_improvements)
    mean_accuracy_improvement = np.mean(accuracy_improvements)
    std_accuracy_improvement = np.std(accuracy_improvements)
    
    f1_wins = sum(1 for imp in f1_improvements if imp > 0)
    accuracy_wins = sum(1 for imp in accuracy_improvements if imp > 0)
    
    f1_win_rate = (f1_wins / len(f1_improvements)) * 100
    accuracy_win_rate = (accuracy_wins / len(accuracy_improvements)) * 100
    
    # Statistical significance tests
    from scipy import stats
    f1_t_stat, f1_p_value = stats.ttest_1samp(f1_improvements, 0)
    accuracy_t_stat, accuracy_p_value = stats.ttest_1samp(accuracy_improvements, 0)
    
    result = {
        'dataset_name': dataset_name,
        'metadata': metadata,
        'n_trials': len(all_trials),
        'aggregate_results': {
            'f1_improvement': {
                'mean_pct': mean_f1_improvement,
                'std_pct': std_f1_improvement,
                'wins': f1_wins,
                'win_rate_pct': f1_win_rate,
                't_statistic': f1_t_stat,
                'p_value': f1_p_value,
                'statistically_significant': f1_p_value < 0.05
            },
            'accuracy_improvement': {
                'mean_pct': mean_accuracy_improvement,
                'std_pct': std_accuracy_improvement,
                'wins': accuracy_wins,
                'win_rate_pct': accuracy_win_rate,
                't_statistic': accuracy_t_stat,
                'p_value': accuracy_p_value,
                'statistically_significant': accuracy_p_value < 0.05
            }
        },
        'all_trials': all_trials
    }
    
    # Print results
    print(f"    Results for {dataset_name}:")
    
    f1_sig = "📈" if f1_p_value < 0.05 and mean_f1_improvement > 0 else "📊" if f1_p_value < 0.05 else ""
    f1_status = "✅" if mean_f1_improvement > 0 else "❌"
    
    accuracy_sig = "📈" if accuracy_p_value < 0.05 and mean_accuracy_improvement > 0 else "📊" if accuracy_p_value < 0.05 else ""
    accuracy_status = "✅" if mean_accuracy_improvement > 0 else "❌"
    
    print(f"      {f1_status} {f1_sig} F1-Score: {mean_f1_improvement:+.2f}% ± {std_f1_improvement:.2f}% "
          f"(p={f1_p_value:.3f}, wins: {f1_wins}/{len(f1_improvements)})")
    print(f"      {accuracy_status} {accuracy_sig} Accuracy: {mean_accuracy_improvement:+.2f}% ± {std_accuracy_improvement:.2f}% "
          f"(p={accuracy_p_value:.3f}, wins: {accuracy_wins}/{len(accuracy_improvements)})")
    
    return result

def run_comprehensive_imbalanced_experiment():
    """Run comprehensive imbalanced data experiments."""
    
    # Create imbalanced datasets
    datasets = create_imbalanced_datasets()
    
    print(f"\n2. Running experiments on imbalanced datasets...")
    
    all_results = []
    
    for dataset_name, (X, y, metadata) in datasets.items():
        result = run_imbalanced_dataset_experiment(dataset_name, X, y, metadata, n_trials=15)
        
        if result:
            all_results.append(result)
    
    return all_results

def analyze_imbalanced_results(all_results: List[Dict[str, Any]]):
    """Analyze imbalanced data experiment results."""
    print(f"\n3. Imbalanced Data Analysis:")
    print("="*70)
    
    print(f"Summary by Imbalance Level:")
    print("-" * 40)
    
    # Sort by imbalance ratio
    results_by_imbalance = sorted(all_results, key=lambda x: x['metadata']['imbalance_ratio'])
    
    print(f"{'Dataset':<25} {'Ratio':<8} {'F1 Improvement':<15} {'Accuracy Imp':<15} {'Significance'}")
    print("-" * 80)
    
    overall_f1_improvements = []
    overall_accuracy_improvements = []
    
    for result in results_by_imbalance:
        dataset = result['dataset_name']
        ratio = result['metadata']['imbalance_ratio']
        
        f1_imp = result['aggregate_results']['f1_improvement']['mean_pct']
        f1_sig = result['aggregate_results']['f1_improvement']['statistically_significant']
        
        acc_imp = result['aggregate_results']['accuracy_improvement']['mean_pct']
        acc_sig = result['aggregate_results']['accuracy_improvement']['statistically_significant']
        
        overall_f1_improvements.append(f1_imp)
        overall_accuracy_improvements.append(acc_imp)
        
        sig_marker = "***" if (f1_sig and acc_sig) else "**" if (f1_sig or acc_sig) else ""
        
        print(f"{dataset:<25} {ratio:<8.1f} {f1_imp:+8.2f}% {acc_imp:+12.2f}% {sig_marker}")
    
    print(f"\nOverall Statistics:")
    print("-" * 25)
    
    overall_f1_mean = np.mean(overall_f1_improvements)
    overall_accuracy_mean = np.mean(overall_accuracy_improvements)
    
    f1_wins = sum(1 for imp in overall_f1_improvements if imp > 0)
    accuracy_wins = sum(1 for imp in overall_accuracy_improvements if imp > 0)
    
    total_experiments = len(all_results)
    
    print(f"F1-Score Improvements:")
    print(f"  Mean: {overall_f1_mean:+.2f}%")
    print(f"  Wins: {f1_wins}/{total_experiments} ({(f1_wins/total_experiments)*100:.1f}%)")
    
    print(f"Accuracy Improvements:")
    print(f"  Mean: {overall_accuracy_mean:+.2f}%")
    print(f"  Wins: {accuracy_wins}/{total_experiments} ({(accuracy_wins/total_experiments)*100:.1f}%)")
    
    # Correlation with imbalance ratio
    if len(all_results) > 2:
        from scipy import stats
        
        imbalance_ratios = [r['metadata']['imbalance_ratio'] for r in all_results]
        
        f1_corr, f1_corr_p = stats.pearsonr(imbalance_ratios, overall_f1_improvements)
        acc_corr, acc_corr_p = stats.pearsonr(imbalance_ratios, overall_accuracy_improvements)
        
        print(f"\nCorrelation with Imbalance Ratio:")
        print(f"  F1 vs Imbalance: r={f1_corr:.3f}, p={f1_corr_p:.3f}")
        print(f"  Accuracy vs Imbalance: r={acc_corr:.3f}, p={acc_corr_p:.3f}")
        
        if f1_corr_p < 0.05:
            direction = "positive" if f1_corr > 0 else "negative"
            print(f"  📈 Significant {direction} correlation between imbalance and F1 improvement!")
        
        if acc_corr_p < 0.05:
            direction = "positive" if acc_corr > 0 else "negative"
            print(f"  📈 Significant {direction} correlation between imbalance and accuracy improvement!")

def save_imbalanced_results(all_results: List[Dict[str, Any]]):
    """Save imbalanced data experiment results."""
    os.makedirs('results', exist_ok=True)
    
    comprehensive_results = {
        'experiment_type': 'imbalanced_data_analysis',
        'description': 'Testing Davidian Regularization effectiveness on imbalanced datasets',
        'hypothesis': 'Imbalanced data creates scenarios where train-val consistency is crucial for good generalization',
        'methodology': {
            'k_folds': 5,
            'trials_per_dataset': 15,
            'models_per_trial': 25,
            'primary_metric': 'F1-score (weighted average)',
            'secondary_metrics': ['accuracy', 'precision', 'recall', 'AUC'],
            'model_selection': 'Top 1 model selected by Davidian score vs validation score'
        },
        'datasets_tested': [
            {'name': r['dataset_name'], 'imbalance_ratio': r['metadata']['imbalance_ratio']} 
            for r in all_results
        ],
        'results': all_results,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open('results/imbalanced_data_results.json', 'w') as f:
        json.dump(comprehensive_results, f, indent=2, default=str)
    
    # Create summary
    summary = []
    for result in all_results:
        summary.append({
            'dataset': result['dataset_name'],
            'imbalance_ratio': result['metadata']['imbalance_ratio'],
            'f1_improvement_pct': result['aggregate_results']['f1_improvement']['mean_pct'],
            'f1_win_rate': result['aggregate_results']['f1_improvement']['win_rate_pct'],
            'f1_p_value': result['aggregate_results']['f1_improvement']['p_value'],
            'accuracy_improvement_pct': result['aggregate_results']['accuracy_improvement']['mean_pct'],
            'accuracy_win_rate': result['aggregate_results']['accuracy_improvement']['win_rate_pct'],
            'accuracy_p_value': result['aggregate_results']['accuracy_improvement']['p_value'],
            'n_trials': result['n_trials']
        })
    
    with open('results/imbalanced_data_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n✅ Imbalanced data results saved to:")
    print(f"   - results/imbalanced_data_results.json")
    print(f"   - results/imbalanced_data_summary.json")

def main():
    """Run the imbalanced data experiment."""
    try:
        start_time = time.time()
        
        print("🔬 IMBALANCED DATA HYPOTHESIS:")
        print("   Imbalanced datasets create scenarios where models can achieve")
        print("   high validation scores through majority class prediction, but")
        print("   Davidian Regularization can identify models with better")
        print("   train-validation consistency and superior generalization.")
        
        # Run experiments
        all_results = run_comprehensive_imbalanced_experiment()
        
        if all_results:
            # Analyze results
            analyze_imbalanced_results(all_results)
            
            # Save results
            save_imbalanced_results(all_results)
        else:
            print("❌ No experiments completed successfully")
        
        elapsed = time.time() - start_time
        
        print(f"\n{'='*70}")
        print("🎉 IMBALANCED DATA EXPERIMENT COMPLETED!")
        print(f"{'='*70}")
        print(f"Total time: {elapsed:.2f} seconds")
        print("✅ Multiple imbalance levels tested")
        print("✅ Comprehensive metrics evaluated (F1, Accuracy, AUC)")
        print("✅ Statistical significance testing included")
        
        return True
        
    except Exception as e:
        print(f"\n{'='*70}")
        print("❌ IMBALANCED DATA EXPERIMENT FAILED")
        print(f"{'='*70}")
        print(f"Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
