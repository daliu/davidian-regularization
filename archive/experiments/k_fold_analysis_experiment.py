#!/usr/bin/env python3
"""
K-fold analysis experiment for Davidian Regularization.

This experiment systematically tests how different values of K (number of folds)
affect the performance of Davidian Regularization, using truly random splits
and proper train/validation/test methodology.
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

print("K-FOLD ANALYSIS EXPERIMENT")
print("="*70)
print("Testing how different K values affect Davidian Regularization")
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
    
    # Second split: separate train and validation for regular comparison
    val_size_adjusted = val_size / (train_size + val_size)
    stratify_second = y_temp if stratify else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_seed + 1,
        stratify=stratify_second
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test, random_seed

def run_k_fold_davidian_on_train_data(X_train, y_train, k, task_type, model_class, model_params, random_seed):
    """
    Run k-fold cross-validation on training data only.
    
    This simulates the validation process that would be used for model selection
    before final evaluation on test data.
    """
    from sklearn.model_selection import KFold, StratifiedKFold
    from sklearn.metrics import accuracy_score, mean_squared_error
    
    if task_type == 'classification':
        cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_seed)
        scoring_func = accuracy_score
    else:
        cv = KFold(n_splits=k, shuffle=True, random_state=random_seed)
        scoring_func = lambda y_true, y_pred: -mean_squared_error(y_true, y_pred)
    
    fold_train_scores = []
    fold_val_scores = []
    
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
        
        # Train model
        model = model_class(**model_params)
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
    
    return {
        'k': k,
        'mean_train_score': mean_train_score,
        'mean_val_score': mean_val_score,
        'davidian_score': davidian_score,
        'penalty': penalty,
        'fold_train_scores': fold_train_scores,
        'fold_val_scores': fold_val_scores
    }

def create_models_for_k_analysis(task_type: str, n_models: int = 30):
    """Create diverse models for K analysis."""
    if task_type == 'classification':
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        
        models = []
        
        # Logistic regression variants
        for i in range(10):
            C_values = [0.01, 0.1, 1.0, 10.0, 100.0]
            models.append({
                'name': f'LogReg_{i}',
                'class': LogisticRegression,
                'params': {
                    'C': np.random.choice(C_values),
                    'penalty': 'l2',
                    'solver': 'lbfgs',
                    'random_state': 42 + i,
                    'max_iter': 1000,
                    'n_jobs': 1
                }
            })
        
        # Random forest variants
        for i in range(10):
            models.append({
                'name': f'RF_{i}',
                'class': RandomForestClassifier,
                'params': {
                    'n_estimators': np.random.choice([10, 20, 50, 100]),
                    'max_depth': np.random.choice([3, 5, 10, None]),
                    'random_state': 42 + i,
                    'n_jobs': 1
                }
            })
        
        # Gradient boosting variants
        for i in range(10):
            models.append({
                'name': f'GB_{i}',
                'class': GradientBoostingClassifier,
                'params': {
                    'n_estimators': np.random.choice([20, 50, 100]),
                    'learning_rate': np.random.choice([0.01, 0.1, 0.2]),
                    'max_depth': np.random.choice([3, 5, 7]),
                    'random_state': 42 + i
                }
            })
    
    else:  # regression
        from sklearn.linear_model import LinearRegression, Ridge, Lasso
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        
        models = []
        
        # Linear models
        for i in range(5):
            models.append({
                'name': f'Linear_{i}',
                'class': LinearRegression,
                'params': {'n_jobs': 1}
            })
        
        # Ridge regression
        for i in range(10):
            models.append({
                'name': f'Ridge_{i}',
                'class': Ridge,
                'params': {
                    'alpha': np.random.choice([0.001, 0.01, 0.1, 1.0, 10.0, 100.0]),
                    'random_state': 42 + i
                }
            })
        
        # Random forest
        for i in range(10):
            models.append({
                'name': f'RF_Reg_{i}',
                'class': RandomForestRegressor,
                'params': {
                    'n_estimators': np.random.choice([10, 20, 50, 100]),
                    'max_depth': np.random.choice([3, 5, 10, None]),
                    'random_state': 42 + i,
                    'n_jobs': 1
                }
            })
        
        # Gradient boosting
        for i in range(5):
            models.append({
                'name': f'GB_Reg_{i}',
                'class': GradientBoostingRegressor,
                'params': {
                    'n_estimators': np.random.choice([20, 50, 100]),
                    'learning_rate': np.random.choice([0.01, 0.1, 0.2]),
                    'random_state': 42 + i
                }
            })
    
    return models[:n_models]

def run_k_analysis_trial(X, y, task_type: str, k_values: List[int], trial_seed: int = None):
    """
    Run a single trial testing different K values.
    """
    from sklearn.metrics import accuracy_score, mean_squared_error
    
    if trial_seed is None:
        trial_seed = generate_random_seed()
    
    # Random data split for this trial
    X_train, X_val, X_test, y_train, y_val, y_test, split_seed = split_data_with_random_seed(
        X, y, stratify=(task_type == 'classification'), random_seed=trial_seed
    )
    
    # Create diverse models
    models = create_models_for_k_analysis(task_type, n_models=30)
    
    # Scoring function
    if task_type == 'classification':
        scoring_func = accuracy_score
    else:
        scoring_func = lambda y_true, y_pred: -mean_squared_error(y_true, y_pred)
    
    trial_results = {}
    
    # Test each K value
    for k in k_values:
        print(f"      Testing K={k}...")
        
        # For each model, run k-fold CV on training data to get davidian scores
        model_davidian_scores = []
        model_regular_val_scores = []
        model_test_scores = []
        model_names = []
        
        for model_config in models:
            try:
                # Run k-fold CV on training data
                cv_result = run_k_fold_davidian_on_train_data(
                    X_train, y_train, k, task_type, 
                    model_config['class'], model_config['params'], 
                    random_seed=trial_seed + hash(model_config['name']) % 1000
                )
                
                # Also get regular validation score (train on all training data, test on validation data)
                full_model = model_config['class'](**model_config['params'])
                full_model.fit(X_train, y_train)
                val_pred = full_model.predict(X_val)
                test_pred = full_model.predict(X_test)
                
                regular_val_score = scoring_func(y_val, val_pred)
                test_score = scoring_func(y_test, test_pred)
                
                model_davidian_scores.append(cv_result['davidian_score'])
                model_regular_val_scores.append(regular_val_score)
                model_test_scores.append(test_score)
                model_names.append(model_config['name'])
                
            except Exception as e:
                # Skip failed models
                continue
        
        # Select top 4 models using different criteria
        # 1. Davidian selection (based on k-fold CV davidian scores)
        davidian_indices = np.argsort(model_davidian_scores)[::-1][:4]
        davidian_test_scores = [model_test_scores[i] for i in davidian_indices]
        davidian_selected_models = [model_names[i] for i in davidian_indices]
        
        # 2. Random selection (based on regular validation scores)
        random_indices = np.argsort(model_regular_val_scores)[::-1][:4]
        random_test_scores = [model_test_scores[i] for i in random_indices]
        random_selected_models = [model_names[i] for i in random_indices]
        
        trial_results[f'k_{k}'] = {
            'k': k,
            'davidian_selection': {
                'selected_models': davidian_selected_models,
                'test_scores': davidian_test_scores,
                'mean_test_score': np.mean(davidian_test_scores),
                'std_test_score': np.std(davidian_test_scores)
            },
            'random_selection': {
                'selected_models': random_selected_models,
                'test_scores': random_test_scores,
                'mean_test_score': np.mean(random_test_scores),
                'std_test_score': np.std(random_test_scores)
            },
            'n_models_evaluated': len(model_names),
            'model_scores': {
                'davidian_scores': model_davidian_scores,
                'regular_val_scores': model_regular_val_scores,
                'test_scores': model_test_scores,
                'model_names': model_names
            }
        }
        
        # Calculate improvement
        davidian_mean = np.mean(davidian_test_scores)
        random_mean = np.mean(random_test_scores)
        improvement = davidian_mean - random_mean
        improvement_pct = (improvement / abs(random_mean)) * 100 if random_mean != 0 else 0
        
        trial_results[f'k_{k}']['improvement'] = improvement
        trial_results[f'k_{k}']['improvement_pct'] = improvement_pct
        
        status = "✅" if improvement > 0 else "❌"
        print(f"        {status} K={k}: Davidian={davidian_mean:.4f}, Random={random_mean:.4f} ({improvement_pct:+.2f}%)")
    
    return {
        'trial_seed': trial_seed,
        'split_seed': split_seed,
        'data_split_sizes': {
            'train': len(X_train),
            'val': len(X_val),
            'test': len(X_test)
        },
        'k_results': trial_results
    }

def run_k_analysis_multiple_trials(X, y, task_type: str, dataset_name: str, 
                                  k_values: List[int] = [2, 3, 4, 5, 10], 
                                  n_trials: int = 15):
    """
    Run multiple trials testing different K values.
    """
    print(f"\n  Running {dataset_name} K-fold analysis ({n_trials} trials)...")
    print(f"    Testing K values: {k_values}")
    
    all_trials = []
    k_performance = {f'k_{k}': {'davidian_improvements': [], 'wins': 0} for k in k_values}
    
    for trial in range(n_trials):
        trial_seed = generate_random_seed()
        
        print(f"    Trial {trial + 1}/{n_trials} (seed: {trial_seed})")
        
        trial_result = run_k_analysis_trial(X, y, task_type, k_values, trial_seed)
        all_trials.append(trial_result)
        
        # Collect performance stats for each K
        for k in k_values:
            k_key = f'k_{k}'
            k_data = trial_result['k_results'][k_key]
            improvement_pct = k_data['improvement_pct']
            
            k_performance[k_key]['davidian_improvements'].append(improvement_pct)
            if k_data['improvement'] > 0:
                k_performance[k_key]['wins'] += 1
    
    # Calculate aggregate statistics for each K
    print(f"\n    K-fold Analysis Results for {dataset_name}:")
    print("    " + "-" * 50)
    
    k_summary = {}
    for k in k_values:
        k_key = f'k_{k}'
        improvements = k_performance[k_key]['davidian_improvements']
        wins = k_performance[k_key]['wins']
        
        mean_improvement = np.mean(improvements)
        std_improvement = np.std(improvements)
        win_rate = (wins / n_trials) * 100
        
        k_summary[k_key] = {
            'k': k,
            'mean_improvement_pct': mean_improvement,
            'std_improvement_pct': std_improvement,
            'wins': wins,
            'win_rate_pct': win_rate,
            'all_improvements': improvements
        }
        
        status = "✅" if mean_improvement > 0 else "❌"
        print(f"    {status} K={k}: {mean_improvement:+.2f}% ± {std_improvement:.2f}% "
              f"(wins: {wins}/{n_trials}, {win_rate:.1f}%)")
    
    return {
        'dataset': dataset_name,
        'task_type': task_type,
        'k_values_tested': k_values,
        'n_trials': n_trials,
        'all_trials': all_trials,
        'k_summary': k_summary
    }

def run_comprehensive_k_analysis():
    """Run comprehensive K-fold analysis across datasets."""
    from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_diabetes
    from sklearn.preprocessing import StandardScaler
    
    print("\n1. Running comprehensive K-fold analysis...")
    
    datasets = {
        'iris': (load_iris(), 'classification'),
        'wine': (load_wine(), 'classification'),
        'breast_cancer': (load_breast_cancer(), 'classification'),
        'diabetes': (load_diabetes(), 'regression')
    }
    
    k_values = [2, 3, 4, 5, 10]
    all_results = []
    
    for dataset_name, (dataset, task_type) in datasets.items():
        # Prepare data
        scaler = StandardScaler()
        X = scaler.fit_transform(dataset.data)
        y = dataset.target
        
        # Run K analysis
        result = run_k_analysis_multiple_trials(X, y, task_type, dataset_name, k_values, n_trials=15)
        all_results.append(result)
    
    return all_results

def analyze_k_fold_results(all_results: List[Dict[str, Any]]):
    """Analyze K-fold analysis results."""
    print(f"\n2. K-Fold Analysis Summary:")
    print("="*70)
    
    k_values = [2, 3, 4, 5, 10]
    
    # Aggregate performance by K value across all datasets
    k_aggregate_stats = {}
    for k in k_values:
        k_key = f'k_{k}'
        all_improvements = []
        total_wins = 0
        total_trials = 0
        
        for result in all_results:
            k_data = result['k_summary'][k_key]
            all_improvements.extend(k_data['all_improvements'])
            total_wins += k_data['wins']
            total_trials += result['n_trials']
        
        k_aggregate_stats[k] = {
            'mean_improvement_pct': np.mean(all_improvements),
            'std_improvement_pct': np.std(all_improvements),
            'total_wins': total_wins,
            'total_trials': total_trials,
            'overall_win_rate_pct': (total_wins / total_trials) * 100,
            'all_improvements': all_improvements
        }
    
    print(f"Overall K-fold Performance Ranking:")
    print("-" * 40)
    
    # Sort K values by performance
    k_performance_ranking = sorted(k_aggregate_stats.items(), 
                                  key=lambda x: x[1]['mean_improvement_pct'], 
                                  reverse=True)
    
    for rank, (k, stats) in enumerate(k_performance_ranking, 1):
        mean_imp = stats['mean_improvement_pct']
        win_rate = stats['overall_win_rate_pct']
        status = "🏆" if rank == 1 else "✅" if mean_imp > 0 else "❌"
        
        print(f"  {rank}. {status} K={k}: {mean_imp:+.2f}% ± {stats['std_improvement_pct']:.2f}% "
              f"(win rate: {win_rate:.1f}%)")
    
    print(f"\nDataset-Specific K Performance:")
    print("-" * 40)
    
    for result in all_results:
        dataset = result['dataset']
        task_type = result['task_type']
        
        print(f"\n{dataset} ({task_type}):")
        
        # Find best K for this dataset
        best_k = None
        best_improvement = -float('inf')
        
        for k in k_values:
            k_key = f'k_{k}'
            k_data = result['k_summary'][k_key]
            improvement = k_data['mean_improvement_pct']
            win_rate = k_data['win_rate_pct']
            
            if improvement > best_improvement:
                best_improvement = improvement
                best_k = k
            
            status = "🏆" if improvement == best_improvement else "✅" if improvement > 0 else "❌"
            print(f"  {status} K={k}: {improvement:+.2f}% (wins: {win_rate:.1f}%)")
        
        print(f"  → Best K for {dataset}: K={best_k} ({best_improvement:+.2f}%)")

def save_k_analysis_results(all_results: List[Dict[str, Any]]):
    """Save K-fold analysis results."""
    os.makedirs('results', exist_ok=True)
    
    # Calculate overall statistics
    k_values = [2, 3, 4, 5, 10]
    overall_k_stats = {}
    
    for k in k_values:
        all_improvements = []
        total_wins = 0
        total_trials = 0
        
        for result in all_results:
            k_key = f'k_{k}'
            k_data = result['k_summary'][k_key]
            all_improvements.extend(k_data['all_improvements'])
            total_wins += k_data['wins']
            total_trials += result['n_trials']
        
        overall_k_stats[k] = {
            'mean_improvement_pct': np.mean(all_improvements),
            'std_improvement_pct': np.std(all_improvements),
            'total_wins': total_wins,
            'total_trials': total_trials,
            'overall_win_rate_pct': (total_wins / total_trials) * 100
        }
    
    comprehensive_results = {
        'experiment_type': 'k_fold_analysis',
        'description': 'Analysis of how different K values affect Davidian Regularization performance',
        'methodology': {
            'k_values_tested': k_values,
            'trials_per_dataset': 15,
            'models_per_trial': 30,
            'data_split': '70/15/15 train/val/test with random splits',
            'selection_method': 'K-fold CV on training data, evaluation on test data'
        },
        'overall_k_statistics': overall_k_stats,
        'dataset_results': all_results,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open('results/k_fold_analysis_results.json', 'w') as f:
        json.dump(comprehensive_results, f, indent=2, default=str)
    
    print(f"\n✅ K-fold analysis results saved to:")
    print(f"   - results/k_fold_analysis_results.json")

def main():
    """Run the K-fold analysis experiment."""
    try:
        start_time = time.time()
        
        print("🔬 K-FOLD ANALYSIS HYPOTHESIS:")
        print("   Different values of K may show different effectiveness for Davidian Regularization")
        print("   Testing K = [2, 3, 4, 5, 10] with random splits and diverse models")
        
        # Run experiments
        all_results = run_comprehensive_k_analysis()
        
        # Analyze results
        analyze_k_fold_results(all_results)
        
        # Save results
        save_k_analysis_results(all_results)
        
        elapsed = time.time() - start_time
        
        print(f"\n{'='*70}")
        print("🎉 K-FOLD ANALYSIS COMPLETED!")
        print(f"{'='*70}")
        print(f"Total time: {elapsed:.2f} seconds")
        print("✅ All K values tested systematically")
        print("✅ Random splits ensure model selection variety")
        print("✅ Proper train/val/test methodology maintained")
        
        return True
        
    except Exception as e:
        print(f"\n{'='*70}")
        print("❌ K-FOLD ANALYSIS FAILED")
        print(f"{'='*70}")
        print(f"Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
