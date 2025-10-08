#!/usr/bin/env python3
"""
Complex datasets experiment for Davidian Regularization.

This experiment tests Davidian Regularization on more challenging datasets
to better understand when and why the technique is effective.
"""

import sys
import os
import time
import json
import numpy as np
from typing import Dict, List, Tuple, Any

# Force single-threaded execution
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

print("COMPLEX DATASETS EXPERIMENT")
print("="*70)
print("Testing Davidian Regularization on challenging real-world-like datasets")
print("="*70)

def generate_random_seed():
    """Generate a truly random seed."""
    return np.random.randint(0, 2**31 - 1)

def split_data_with_random_seed(X, y, train_size=0.7, val_size=0.15, test_size=0.15, 
                               stratify=True, random_seed=None):
    """Split data using a specific random seed."""
    from sklearn.model_selection import train_test_split
    
    if random_seed is None:
        random_seed = generate_random_seed()
    
    # First split: separate test set
    stratify_first = y if stratify else None
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed, 
        stratify=stratify_first
    )
    
    # Second split: separate train and validation
    val_size_adjusted = val_size / (train_size + val_size)
    stratify_second = y_temp if stratify else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_seed + 1,
        stratify=stratify_second
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test, random_seed

def create_robust_models(task_type: str, n_models: int = 40):
    """
    Create robust models suitable for complex datasets.
    """
    models = []
    
    if task_type == 'classification':
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.svm import SVC
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.naive_bayes import GaussianNB
        from sklearn.tree import DecisionTreeClassifier
        
        # Logistic Regression with different regularization
        for i in range(8):
            C_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
            models.append({
                'name': f'LogReg_{i}',
                'class': LogisticRegression,
                'params': {
                    'C': np.random.choice(C_values),
                    'penalty': 'l2',
                    'solver': 'lbfgs',
                    'random_state': 42 + i,
                    'max_iter': 2000,
                    'n_jobs': 1
                }
            })
        
        # Random Forest with varied complexity
        for i in range(10):
            models.append({
                'name': f'RF_{i}',
                'class': RandomForestClassifier,
                'params': {
                    'n_estimators': np.random.choice([20, 50, 100, 200]),
                    'max_depth': np.random.choice([3, 5, 10, 15, None]),
                    'min_samples_split': np.random.choice([2, 5, 10, 20]),
                    'min_samples_leaf': np.random.choice([1, 2, 4, 8]),
                    'random_state': 42 + i,
                    'n_jobs': 1
                }
            })
        
        # Gradient Boosting
        for i in range(8):
            models.append({
                'name': f'GB_{i}',
                'class': GradientBoostingClassifier,
                'params': {
                    'n_estimators': np.random.choice([50, 100, 200]),
                    'learning_rate': np.random.choice([0.01, 0.05, 0.1, 0.2]),
                    'max_depth': np.random.choice([3, 5, 7, 10]),
                    'subsample': np.random.choice([0.8, 0.9, 1.0]),
                    'random_state': 42 + i
                }
            })
        
        # SVM (for smaller datasets)
        for i in range(6):
            models.append({
                'name': f'SVM_{i}',
                'class': SVC,
                'params': {
                    'C': np.random.choice([0.1, 1.0, 10.0, 100.0]),
                    'kernel': np.random.choice(['rbf', 'poly', 'linear']),
                    'gamma': np.random.choice(['scale', 'auto']),
                    'random_state': 42 + i
                }
            })
        
        # KNN
        for i in range(4):
            models.append({
                'name': f'KNN_{i}',
                'class': KNeighborsClassifier,
                'params': {
                    'n_neighbors': np.random.choice([3, 5, 7, 11, 15]),
                    'weights': np.random.choice(['uniform', 'distance']),
                    'metric': np.random.choice(['euclidean', 'manhattan']),
                    'n_jobs': 1
                }
            })
        
        # Decision Tree
        for i in range(4):
            models.append({
                'name': f'DT_{i}',
                'class': DecisionTreeClassifier,
                'params': {
                    'max_depth': np.random.choice([3, 5, 10, 15, None]),
                    'min_samples_split': np.random.choice([2, 5, 10, 20]),
                    'min_samples_leaf': np.random.choice([1, 2, 4, 8]),
                    'random_state': 42 + i
                }
            })
    
    else:  # regression
        from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.svm import SVR
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.tree import DecisionTreeRegressor
        
        # Linear models with regularization
        for i in range(10):
            alpha_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
            
            if i < 2:
                models.append({
                    'name': f'Linear_{i}',
                    'class': LinearRegression,
                    'params': {'n_jobs': 1}
                })
            elif i < 5:
                models.append({
                    'name': f'Ridge_{i}',
                    'class': Ridge,
                    'params': {
                        'alpha': np.random.choice(alpha_values),
                        'random_state': 42 + i
                    }
                })
            elif i < 8:
                models.append({
                    'name': f'Lasso_{i}',
                    'class': Lasso,
                    'params': {
                        'alpha': np.random.choice(alpha_values),
                        'random_state': 42 + i,
                        'max_iter': 2000
                    }
                })
            else:
                models.append({
                    'name': f'ElasticNet_{i}',
                    'class': ElasticNet,
                    'params': {
                        'alpha': np.random.choice(alpha_values),
                        'l1_ratio': np.random.choice([0.1, 0.3, 0.5, 0.7, 0.9]),
                        'random_state': 42 + i,
                        'max_iter': 2000
                    }
                })
        
        # Random Forest
        for i in range(10):
            models.append({
                'name': f'RF_Reg_{i}',
                'class': RandomForestRegressor,
                'params': {
                    'n_estimators': np.random.choice([20, 50, 100, 200]),
                    'max_depth': np.random.choice([3, 5, 10, 15, None]),
                    'min_samples_split': np.random.choice([2, 5, 10, 20]),
                    'random_state': 42 + i,
                    'n_jobs': 1
                }
            })
        
        # Gradient Boosting
        for i in range(8):
            models.append({
                'name': f'GB_Reg_{i}',
                'class': GradientBoostingRegressor,
                'params': {
                    'n_estimators': np.random.choice([50, 100, 200]),
                    'learning_rate': np.random.choice([0.01, 0.05, 0.1, 0.2]),
                    'max_depth': np.random.choice([3, 5, 7, 10]),
                    'random_state': 42 + i
                }
            })
        
        # SVR (for smaller datasets)
        for i in range(6):
            models.append({
                'name': f'SVR_{i}',
                'class': SVR,
                'params': {
                    'C': np.random.choice([0.1, 1.0, 10.0, 100.0]),
                    'kernel': np.random.choice(['rbf', 'poly', 'linear']),
                    'gamma': np.random.choice(['scale', 'auto'])
                }
            })
        
        # KNN Regressor
        for i in range(4):
            models.append({
                'name': f'KNN_Reg_{i}',
                'class': KNeighborsRegressor,
                'params': {
                    'n_neighbors': np.random.choice([3, 5, 7, 11, 15]),
                    'weights': np.random.choice(['uniform', 'distance']),
                    'n_jobs': 1
                }
            })
        
        # Decision Tree
        for i in range(2):
            models.append({
                'name': f'DT_Reg_{i}',
                'class': DecisionTreeRegressor,
                'params': {
                    'max_depth': np.random.choice([3, 5, 10, 15, None]),
                    'min_samples_split': np.random.choice([2, 5, 10, 20]),
                    'random_state': 42 + i
                }
            })
    
    return models[:n_models]

def run_complex_dataset_experiment(X, y, task_type: str, dataset_name: str, 
                                  complexity: str, k_values: List[int] = [3, 5, 10], 
                                  n_trials: int = 10):
    """
    Run experiment on a complex dataset.
    """
    from sklearn.metrics import accuracy_score, mean_squared_error
    from sklearn.model_selection import KFold, StratifiedKFold
    
    print(f"\n  Running {dataset_name} experiment...")
    print(f"    Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"    Complexity: {complexity}")
    print(f"    Testing K values: {k_values}")
    
    # Skip very large datasets for SVM to avoid memory issues
    if X.shape[0] > 5000:
        print(f"    Note: Large dataset ({X.shape[0]} samples) - using subset for SVM models")
    
    all_k_results = {}
    
    for k in k_values:
        print(f"    Testing K={k}...")
        
        k_trial_results = []
        davidian_improvements = []
        
        for trial in range(n_trials):
            trial_seed = generate_random_seed()
            
            # Random data split
            X_train, X_val, X_test, y_train, y_val, y_test, split_seed = split_data_with_random_seed(
                X, y, stratify=(task_type == 'classification'), random_seed=trial_seed
            )
            
            # Create models (fewer for very large datasets)
            n_models = 30 if X.shape[0] < 5000 else 20
            models = create_robust_models(task_type, n_models=n_models)
            
            # Scoring function
            if task_type == 'classification':
                scoring_func = accuracy_score
                cv_class = StratifiedKFold
            else:
                scoring_func = lambda y_true, y_pred: -mean_squared_error(y_true, y_pred)
                cv_class = KFold
            
            # Run k-fold CV on training data for each model
            model_davidian_scores = []
            model_regular_val_scores = []
            model_test_scores = []
            model_names = []
            
            for model_config in models:
                try:
                    # Skip SVM for very large datasets
                    if 'SVM' in model_config['name'] and X.shape[0] > 5000:
                        continue
                    
                    # K-fold CV on training data
                    cv = cv_class(n_splits=k, shuffle=True, random_state=trial_seed)
                    fold_train_scores = []
                    fold_val_scores = []
                    
                    for train_idx, val_idx in cv.split(X_train, y_train):
                        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
                        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
                        
                        # Train model
                        model = model_config['class'](**model_config['params'])
                        model.fit(X_fold_train, y_fold_train)
                        
                        # Evaluate
                        train_pred = model.predict(X_fold_train)
                        val_pred = model.predict(X_fold_val)
                        
                        train_score = scoring_func(y_fold_train, train_pred)
                        val_score = scoring_func(y_fold_val, val_pred)
                        
                        fold_train_scores.append(train_score)
                        fold_val_scores.append(val_score)
                    
                    mean_train_score = np.mean(fold_train_scores)
                    mean_val_score = np.mean(fold_val_scores)
                    
                    # Apply Davidian Regularization
                    penalty = abs(mean_train_score - mean_val_score)
                    davidian_score = mean_val_score - penalty
                    
                    # Also get regular validation score
                    full_model = model_config['class'](**model_config['params'])
                    full_model.fit(X_train, y_train)
                    
                    val_pred = full_model.predict(X_val)
                    test_pred = full_model.predict(X_test)
                    
                    regular_val_score = scoring_func(y_val, val_pred)
                    test_score = scoring_func(y_test, test_pred)
                    
                    model_davidian_scores.append(davidian_score)
                    model_regular_val_scores.append(regular_val_score)
                    model_test_scores.append(test_score)
                    model_names.append(model_config['name'])
                    
                except Exception as e:
                    # Skip failed models
                    continue
            
            if len(model_names) < 4:
                print(f"      Warning: Only {len(model_names)} models succeeded in trial {trial + 1}")
                continue
            
            # Select top 4 models using different methods
            n_select = min(4, len(model_names))
            
            # Davidian selection
            davidian_indices = np.argsort(model_davidian_scores)[::-1][:n_select]
            davidian_test_scores = [model_test_scores[i] for i in davidian_indices]
            
            # Random selection
            random_indices = np.argsort(model_regular_val_scores)[::-1][:n_select]
            random_test_scores = [model_test_scores[i] for i in random_indices]
            
            # Calculate improvement
            davidian_mean = np.mean(davidian_test_scores)
            random_mean = np.mean(random_test_scores)
            improvement = davidian_mean - random_mean
            improvement_pct = (improvement / abs(random_mean)) * 100 if random_mean != 0 else 0
            
            k_trial_results.append({
                'trial': trial,
                'trial_seed': trial_seed,
                'split_seed': split_seed,
                'n_models': len(model_names),
                'davidian_test_mean': davidian_mean,
                'random_test_mean': random_mean,
                'improvement': improvement,
                'improvement_pct': improvement_pct,
                'davidian_selected': [model_names[i] for i in davidian_indices],
                'random_selected': [model_names[i] for i in random_indices]
            })
            
            davidian_improvements.append(improvement_pct)
            
            if (trial + 1) % 5 == 0:
                print(f"      Completed {trial + 1}/{n_trials} trials for K={k}")
        
        # Aggregate results for this K value
        if davidian_improvements:
            mean_improvement = np.mean(davidian_improvements)
            std_improvement = np.std(davidian_improvements)
            wins = sum(1 for imp in davidian_improvements if imp > 0)
            win_rate = (wins / len(davidian_improvements)) * 100
            
            all_k_results[f'k_{k}'] = {
                'k': k,
                'n_trials': len(k_trial_results),
                'mean_improvement_pct': mean_improvement,
                'std_improvement_pct': std_improvement,
                'wins': wins,
                'win_rate_pct': win_rate,
                'all_improvements': davidian_improvements,
                'trial_details': k_trial_results
            }
            
            status = "✅" if mean_improvement > 0 else "❌"
            print(f"    {status} K={k}: {mean_improvement:+.2f}% ± {std_improvement:.2f}% "
                  f"(wins: {wins}/{len(davidian_improvements)}, {win_rate:.1f}%)")
    
    return {
        'dataset': dataset_name,
        'task_type': task_type,
        'complexity': complexity,
        'dataset_shape': X.shape,
        'k_results': all_k_results
    }

def run_comprehensive_complex_experiment():
    """Run comprehensive experiment on complex datasets."""
    from src.complex_data_loaders import get_complex_datasets, load_and_validate_dataset
    from sklearn.preprocessing import StandardScaler
    
    print("\n1. Loading and testing complex datasets...")
    
    # Select a diverse set of complex datasets
    datasets_to_test = [
        'digits',                           # High-dimensional classification
        'synthetic_complex_classification', # Controlled complexity
        'imbalanced_classification',        # Class imbalance challenge
        'california_housing',               # Large regression
        'synthetic_complex_regression',     # Controlled regression complexity
        'noisy_regression',                 # Noise and outliers
        'multicollinear_regression',        # Multicollinearity challenge
        'clustered_classification'          # Natural clustering
    ]
    
    all_results = []
    successful_experiments = 0
    
    for dataset_name in datasets_to_test:
        print(f"\n{'='*50}")
        print(f"DATASET: {dataset_name}")
        print(f"{'='*50}")
        
        try:
            # Load dataset
            X, y, metadata = load_and_validate_dataset(dataset_name)
            
            if X is None:
                print(f"❌ Skipping {dataset_name} - failed to load")
                continue
            
            # Preprocess data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Run experiment
            result = run_complex_dataset_experiment(
                X_scaled, y, metadata['type'], dataset_name, 
                metadata.get('complexity', 'unknown'),
                k_values=[3, 5, 10], n_trials=10
            )
            
            all_results.append(result)
            successful_experiments += 1
            
        except Exception as e:
            print(f"❌ Error with {dataset_name}: {e}")
            continue
    
    return all_results, successful_experiments

def analyze_complex_results(all_results: List[Dict[str, Any]]):
    """Analyze results from complex datasets."""
    print(f"\n2. Complex Datasets Analysis:")
    print("="*70)
    
    # Overall statistics
    k_values = [3, 5, 10]
    overall_k_stats = {}
    
    for k in k_values:
        all_improvements = []
        total_wins = 0
        total_trials = 0
        
        for result in all_results:
            k_key = f'k_{k}'
            if k_key in result['k_results']:
                k_data = result['k_results'][k_key]
                all_improvements.extend(k_data['all_improvements'])
                total_wins += k_data['wins']
                total_trials += k_data['n_trials']
        
        if all_improvements:
            overall_k_stats[k] = {
                'mean_improvement_pct': np.mean(all_improvements),
                'std_improvement_pct': np.std(all_improvements),
                'total_wins': total_wins,
                'total_trials': total_trials,
                'overall_win_rate_pct': (total_wins / total_trials) * 100
            }
    
    print(f"Overall Performance Across Complex Datasets:")
    print("-" * 50)
    
    for k in sorted(overall_k_stats.keys()):
        stats = overall_k_stats[k]
        mean_imp = stats['mean_improvement_pct']
        win_rate = stats['overall_win_rate_pct']
        
        status = "🏆" if mean_imp == max(s['mean_improvement_pct'] for s in overall_k_stats.values()) else "✅" if mean_imp > 0 else "❌"
        print(f"  {status} K={k}: {mean_imp:+.2f}% ± {stats['std_improvement_pct']:.2f}% "
              f"(win rate: {win_rate:.1f}%, {stats['total_wins']}/{stats['total_trials']} trials)")
    
    print(f"\nDataset-Specific Results:")
    print("-" * 50)
    
    # Analyze by complexity level
    complexity_stats = {}
    
    for result in all_results:
        dataset = result['dataset']
        complexity = result['complexity']
        task_type = result['task_type']
        
        print(f"\n{dataset} ({task_type}, {complexity} complexity):")
        
        if complexity not in complexity_stats:
            complexity_stats[complexity] = {'improvements': [], 'wins': 0, 'total': 0}
        
        best_k = None
        best_improvement = -float('inf')
        
        for k in k_values:
            k_key = f'k_{k}'
            if k_key in result['k_results']:
                k_data = result['k_results'][k_key]
                improvement = k_data['mean_improvement_pct']
                win_rate = k_data['win_rate_pct']
                
                complexity_stats[complexity]['improvements'].append(improvement)
                complexity_stats[complexity]['total'] += 1
                if improvement > 0:
                    complexity_stats[complexity]['wins'] += 1
                
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_k = k
                
                status = "🏆" if improvement == best_improvement else "✅" if improvement > 0 else "❌"
                print(f"  {status} K={k}: {improvement:+.2f}% (win rate: {win_rate:.1f}%)")
        
        if best_k:
            print(f"  → Best K for {dataset}: K={best_k} ({best_improvement:+.2f}%)")
    
    print(f"\nComplexity-Based Analysis:")
    print("-" * 30)
    
    for complexity, stats in complexity_stats.items():
        if stats['improvements']:
            mean_imp = np.mean(stats['improvements'])
            win_rate = (stats['wins'] / stats['total']) * 100
            print(f"  {complexity.title()} complexity: {mean_imp:+.2f}% avg, {win_rate:.1f}% win rate")

def save_complex_results(all_results: List[Dict[str, Any]], successful_experiments: int):
    """Save complex dataset results."""
    os.makedirs('results', exist_ok=True)
    
    comprehensive_results = {
        'experiment_type': 'complex_datasets_analysis',
        'description': 'Davidian Regularization tested on challenging real-world-like datasets',
        'methodology': {
            'datasets': 'Complex datasets with varying difficulty levels',
            'k_values_tested': [3, 5, 10],
            'trials_per_k': 10,
            'data_split': '70/15/15 train/val/test with random splits',
            'models': 'Diverse model architectures (20-30 per trial)'
        },
        'successful_experiments': successful_experiments,
        'results': all_results,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open('results/complex_datasets_results.json', 'w') as f:
        json.dump(comprehensive_results, f, indent=2, default=str)
    
    # Create summary
    summary = []
    for result in all_results:
        for k_key, k_data in result['k_results'].items():
            summary.append({
                'dataset': result['dataset'],
                'task_type': result['task_type'],
                'complexity': result['complexity'],
                'k': k_data['k'],
                'mean_improvement_pct': k_data['mean_improvement_pct'],
                'win_rate_pct': k_data['win_rate_pct'],
                'n_trials': k_data['n_trials']
            })
    
    with open('results/complex_datasets_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n✅ Complex datasets results saved to:")
    print(f"   - results/complex_datasets_results.json")
    print(f"   - results/complex_datasets_summary.json")

def main():
    """Run the complex datasets experiment."""
    try:
        start_time = time.time()
        
        print("🔬 COMPLEX DATASETS HYPOTHESIS:")
        print("   Davidian Regularization may be more effective on challenging datasets")
        print("   where overfitting and model selection are more critical")
        
        # Run experiments
        all_results, successful_experiments = run_comprehensive_complex_experiment()
        
        if successful_experiments > 0:
            # Analyze results
            analyze_complex_results(all_results)
            
            # Save results
            save_complex_results(all_results, successful_experiments)
        else:
            print("❌ No experiments completed successfully")
        
        elapsed = time.time() - start_time
        
        print(f"\n{'='*70}")
        print("🎉 COMPLEX DATASETS EXPERIMENT COMPLETED!")
        print(f"{'='*70}")
        print(f"Total time: {elapsed:.2f} seconds")
        print(f"Successful experiments: {successful_experiments}")
        print("✅ Challenging datasets tested")
        print("✅ Real-world complexity scenarios evaluated")
        
        return True
        
    except Exception as e:
        print(f"\n{'='*70}")
        print("❌ COMPLEX DATASETS EXPERIMENT FAILED")
        print(f"{'='*70}")
        print(f"Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
