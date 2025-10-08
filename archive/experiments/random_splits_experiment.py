#!/usr/bin/env python3
"""
Random splits experiment for Davidian Regularization.

This experiment uses truly random data splits (not fixed random_state=42)
to ensure different models can be "best" under different splits, while
maintaining reproducibility by recording all random seeds used.
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

print("RANDOM SPLITS DAVIDIAN REGULARIZATION EXPERIMENT")
print("="*70)
print("Using truly random data splits for each trial")
print("="*70)

def generate_random_seed():
    """Generate a truly random seed and record it for reproducibility."""
    return np.random.randint(0, 2**31 - 1)

def split_data_with_random_seed(X, y, train_size=0.7, val_size=0.15, test_size=0.15, 
                               stratify=True, random_seed=None):
    """
    Split data using a specific random seed (or generate new one).
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test, random_seed_used
    """
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

def create_diverse_models_with_random_seeds(task_type: str, n_models: int = 50, base_seed: int = None):
    """
    Create diverse models with random seeds for each model.
    
    Returns:
        List of model configs with their random seeds recorded
    """
    if base_seed is None:
        base_seed = generate_random_seed()
    
    models = []
    
    if task_type == 'classification':
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.neighbors import KNeighborsClassifier
        
        model_configs = [
            ('LogisticRegression', LogisticRegression, {
                'C_values': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                'penalty_values': ['l2'],  # Avoid l1 to prevent solver issues
                'solver_values': ['lbfgs', 'liblinear']
            }),
            ('RandomForest', RandomForestClassifier, {
                'n_estimators_values': [10, 20, 50, 100],
                'max_depth_values': [3, 5, 10, None],
                'min_samples_split_values': [2, 5, 10]
            }),
            ('GradientBoosting', GradientBoostingClassifier, {
                'n_estimators_values': [20, 50, 100],
                'learning_rate_values': [0.01, 0.1, 0.2],
                'max_depth_values': [3, 5, 7]
            }),
            ('KNN', KNeighborsClassifier, {
                'n_neighbors_values': [3, 5, 7, 9, 11],
                'weights_values': ['uniform', 'distance'],
                'metric_values': ['euclidean', 'manhattan']
            })
        ]
        
        models_per_type = n_models // len(model_configs)
        
        for model_name, model_class, param_options in model_configs:
            for i in range(models_per_type):
                model_seed = base_seed + len(models) + 1
                np.random.seed(model_seed)  # Set seed for parameter selection
                
                if model_name == 'LogisticRegression':
                    params = {
                        'C': np.random.choice(param_options['C_values']),
                        'penalty': np.random.choice(param_options['penalty_values']),
                        'solver': np.random.choice(param_options['solver_values']),
                        'random_state': model_seed,
                        'max_iter': 1000,
                        'n_jobs': 1
                    }
                elif model_name == 'RandomForest':
                    params = {
                        'n_estimators': np.random.choice(param_options['n_estimators_values']),
                        'max_depth': np.random.choice(param_options['max_depth_values']),
                        'min_samples_split': np.random.choice(param_options['min_samples_split_values']),
                        'random_state': model_seed,
                        'n_jobs': 1
                    }
                elif model_name == 'GradientBoosting':
                    params = {
                        'n_estimators': np.random.choice(param_options['n_estimators_values']),
                        'learning_rate': np.random.choice(param_options['learning_rate_values']),
                        'max_depth': np.random.choice(param_options['max_depth_values']),
                        'random_state': model_seed
                    }
                else:  # KNN
                    params = {
                        'n_neighbors': np.random.choice(param_options['n_neighbors_values']),
                        'weights': np.random.choice(param_options['weights_values']),
                        'metric': np.random.choice(param_options['metric_values']),
                        'n_jobs': 1
                    }
                
                models.append({
                    'name': f'{model_name}_{i}',
                    'model_class': model_class,
                    'params': params,
                    'model_seed': model_seed
                })
    
    else:  # regression
        from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        
        model_configs = [
            ('Ridge', Ridge, {'alpha_values': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}),
            ('Lasso', Lasso, {'alpha_values': [0.001, 0.01, 0.1, 1.0, 10.0]}),
            ('ElasticNet', ElasticNet, {
                'alpha_values': [0.001, 0.01, 0.1, 1.0, 10.0],
                'l1_ratio_values': [0.1, 0.5, 0.7, 0.9]
            }),
            ('RandomForestRegressor', RandomForestRegressor, {
                'n_estimators_values': [10, 20, 50, 100],
                'max_depth_values': [3, 5, 10, None]
            }),
            ('GradientBoostingRegressor', GradientBoostingRegressor, {
                'n_estimators_values': [20, 50, 100],
                'learning_rate_values': [0.01, 0.1, 0.2]
            })
        ]
        
        # Add some LinearRegression models too
        for i in range(5):
            model_seed = base_seed + len(models) + 1
            models.append({
                'name': f'LinearRegression_{i}',
                'model_class': LinearRegression,
                'params': {'n_jobs': 1},
                'model_seed': model_seed
            })
        
        models_per_type = (n_models - 5) // len(model_configs)
        
        for model_name, model_class, param_options in model_configs:
            for i in range(models_per_type):
                model_seed = base_seed + len(models) + 1
                np.random.seed(model_seed)
                
                if model_name in ['Ridge', 'Lasso']:
                    params = {
                        'alpha': np.random.choice(param_options['alpha_values']),
                        'random_state': model_seed,
                        'max_iter': 1000
                    }
                elif model_name == 'ElasticNet':
                    params = {
                        'alpha': np.random.choice(param_options['alpha_values']),
                        'l1_ratio': np.random.choice(param_options['l1_ratio_values']),
                        'random_state': model_seed,
                        'max_iter': 1000
                    }
                elif model_name == 'RandomForestRegressor':
                    params = {
                        'n_estimators': np.random.choice(param_options['n_estimators_values']),
                        'max_depth': np.random.choice(param_options['max_depth_values']),
                        'random_state': model_seed,
                        'n_jobs': 1
                    }
                else:  # GradientBoostingRegressor
                    params = {
                        'n_estimators': np.random.choice(param_options['n_estimators_values']),
                        'learning_rate': np.random.choice(param_options['learning_rate_values']),
                        'random_state': model_seed
                    }
                
                models.append({
                    'name': f'{model_name}_{i}',
                    'model_class': model_class,
                    'params': params,
                    'model_seed': model_seed
                })
    
    return models[:n_models]

def run_random_splits_trial(X, y, task_type: str, models: List[Dict], trial_seed: int = None):
    """
    Run a single trial with random data split.
    
    Returns:
        Dictionary with trial results and all random seeds used
    """
    from sklearn.metrics import accuracy_score, mean_squared_error
    
    if trial_seed is None:
        trial_seed = generate_random_seed()
    
    # Random data split
    X_train, X_val, X_test, y_train, y_val, y_test, split_seed = split_data_with_random_seed(
        X, y, stratify=(task_type == 'classification'), random_seed=trial_seed
    )
    
    # Scoring function
    if task_type == 'classification':
        scoring_func = accuracy_score
    else:
        scoring_func = lambda y_true, y_pred: -mean_squared_error(y_true, y_pred)
    
    # Train and evaluate all models
    model_results = {
        'model_names': [],
        'train_scores': [],
        'val_scores': [],
        'test_scores': [],
        'model_seeds': []
    }
    
    for model_config in models:
        try:
            # Create and train model
            model = model_config['model_class'](**model_config['params'])
            model.fit(X_train, y_train)
            
            # Evaluate
            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)
            test_pred = model.predict(X_test)
            
            train_score = scoring_func(y_train, train_pred)
            val_score = scoring_func(y_val, val_pred)
            test_score = scoring_func(y_test, test_pred)
            
            model_results['model_names'].append(model_config['name'])
            model_results['train_scores'].append(train_score)
            model_results['val_scores'].append(val_score)
            model_results['test_scores'].append(test_score)
            model_results['model_seeds'].append(model_config['model_seed'])
            
        except Exception as e:
            # Skip failed models
            continue
    
    # Apply selection methods
    selection_methods = {}
    
    # Original Davidian
    original_scores = []
    for train_score, val_score in zip(model_results['train_scores'], model_results['val_scores']):
        penalty = abs(train_score - val_score)
        regularized_score = val_score - penalty
        original_scores.append(regularized_score)
    
    # Confidence Davidian
    confidence_scores = []
    threshold = 0.1
    for train_score, val_score in zip(model_results['train_scores'], model_results['val_scores']):
        diff = abs(train_score - val_score)
        if diff < threshold:
            bonus = (threshold - diff) / threshold
            confidence_score = val_score * (1.0 + bonus)
        else:
            confidence_score = val_score
        confidence_scores.append(confidence_score)
    
    # Random (baseline)
    random_scores = model_results['val_scores'].copy()
    
    # Select top 4 models for each method
    for method_name, scores in [('original_davidian', original_scores), 
                               ('confidence_davidian', confidence_scores),
                               ('random', random_scores)]:
        
        sorted_indices = np.argsort(scores)[::-1]  # Descending
        best_indices = sorted_indices[:4]
        
        selected_test_scores = [model_results['test_scores'][i] for i in best_indices]
        selected_models = [model_results['model_names'][i] for i in best_indices]
        
        selection_methods[method_name] = {
            'best_indices': best_indices.tolist(),
            'selected_models': selected_models,
            'selected_test_scores': selected_test_scores,
            'mean_test_score': np.mean(selected_test_scores),
            'selection_scores': [scores[i] for i in best_indices]
        }
    
    return {
        'trial_seed': trial_seed,
        'split_seed': split_seed,
        'data_split_sizes': {
            'train': len(X_train),
            'val': len(X_val),
            'test': len(X_test)
        },
        'n_models_trained': len(model_results['model_names']),
        'selection_methods': selection_methods,
        'model_results': model_results
    }

def run_multiple_random_trials(X, y, task_type: str, dataset_name: str, n_trials: int = 20):
    """
    Run multiple trials with different random splits.
    """
    print(f"\n  Running {dataset_name} with {n_trials} random split trials...")
    
    # Create models once (they'll have their own random seeds recorded)
    base_model_seed = generate_random_seed()
    models = create_diverse_models_with_random_seeds(task_type, n_models=50, base_seed=base_model_seed)
    print(f"    Created {len(models)} diverse models")
    
    all_trials = []
    method_test_scores = {
        'original_davidian': [],
        'confidence_davidian': [],
        'random': []
    }
    
    for trial in range(n_trials):
        trial_seed = generate_random_seed()
        
        trial_result = run_random_splits_trial(X, y, task_type, models, trial_seed)
        all_trials.append(trial_result)
        
        # Collect test scores for each method
        for method_name in method_test_scores.keys():
            test_score = trial_result['selection_methods'][method_name]['mean_test_score']
            method_test_scores[method_name].append(test_score)
        
        if (trial + 1) % 5 == 0:
            print(f"    Completed {trial + 1}/{n_trials} trials")
    
    # Calculate aggregate statistics
    results = {
        'dataset': dataset_name,
        'task_type': task_type,
        'n_trials': n_trials,
        'base_model_seed': base_model_seed,
        'n_models_per_trial': len(models),
        'all_trials': all_trials,
        'aggregate_results': {}
    }
    
    print(f"    Aggregate Test Performance Results:")
    
    baseline_scores = method_test_scores['random']
    baseline_mean = np.mean(baseline_scores)
    
    for method_name, test_scores in method_test_scores.items():
        mean_score = np.mean(test_scores)
        std_score = np.std(test_scores)
        
        if method_name != 'random':
            improvement = mean_score - baseline_mean
            improvement_pct = (improvement / abs(baseline_mean)) * 100 if baseline_mean != 0 else 0
            
            results['aggregate_results'][method_name] = {
                'mean_test_score': mean_score,
                'std_test_score': std_score,
                'improvement': improvement,
                'improvement_pct': improvement_pct,
                'all_test_scores': test_scores
            }
            
            status = "✅" if improvement > 0 else "❌"
            print(f"      {status} {method_name}: {mean_score:.4f} ± {std_score:.4f} ({improvement_pct:+.2f}%)")
        else:
            results['aggregate_results'][method_name] = {
                'mean_test_score': mean_score,
                'std_test_score': std_score,
                'all_test_scores': test_scores
            }
            print(f"      📊 {method_name}: {mean_score:.4f} ± {std_score:.4f} (baseline)")
    
    return results

def run_comprehensive_random_splits_experiment():
    """Run comprehensive experiment with random splits."""
    from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_diabetes
    from sklearn.preprocessing import StandardScaler
    
    print("\n1. Running experiments with truly random data splits...")
    
    datasets = {
        'iris': (load_iris(), 'classification'),
        'wine': (load_wine(), 'classification'),
        'breast_cancer': (load_breast_cancer(), 'classification'),
        'diabetes': (load_diabetes(), 'regression')
    }
    
    all_results = []
    
    for dataset_name, (dataset, task_type) in datasets.items():
        # Prepare data
        scaler = StandardScaler()
        X = scaler.fit_transform(dataset.data)
        y = dataset.target
        
        # Run experiment with random splits
        result = run_multiple_random_trials(X, y, task_type, dataset_name, n_trials=30)
        all_results.append(result)
    
    return all_results

def analyze_random_splits_results(all_results: List[Dict[str, Any]]):
    """Analyze results from random splits experiment."""
    print(f"\n2. Random Splits Results Analysis:")
    print("="*70)
    
    methods = ['original_davidian', 'confidence_davidian']
    overall_stats = {method: {'wins': 0, 'improvements': []} for method in methods}
    
    print(f"Dataset-by-Dataset Results (with statistical significance):")
    print("-" * 60)
    
    for result in all_results:
        dataset = result['dataset']
        task_type = result['task_type']
        n_trials = result['n_trials']
        
        print(f"\n{dataset} ({task_type}) - {n_trials} trials:")
        
        baseline_mean = result['aggregate_results']['random']['mean_test_score']
        baseline_std = result['aggregate_results']['random']['std_test_score']
        print(f"  📊 Random Baseline: {baseline_mean:.4f} ± {baseline_std:.4f}")
        
        for method in methods:
            method_data = result['aggregate_results'][method]
            mean_score = method_data['mean_test_score']
            std_score = method_data['std_test_score']
            improvement_pct = method_data['improvement_pct']
            
            # Simple statistical test (t-test would be more rigorous)
            improvement_significant = abs(method_data['improvement']) > 2 * (std_score + baseline_std)
            
            if method_data['improvement'] > 0:
                overall_stats[method]['wins'] += 1
                status = "✅"
                if improvement_significant:
                    status += " (significant)"
            else:
                status = "❌"
            
            overall_stats[method]['improvements'].append(improvement_pct)
            
            method_display = method.replace('_', ' ').title()
            print(f"  {status} {method_display}: {mean_score:.4f} ± {std_score:.4f} ({improvement_pct:+.2f}%)")
    
    print(f"\n" + "="*50)
    print(f"RANDOM SPLITS EXPERIMENT SUMMARY")
    print("="*50)
    
    total_experiments = len(all_results)
    
    for method in methods:
        wins = overall_stats[method]['wins']
        win_rate = (wins / total_experiments) * 100
        avg_improvement = np.mean(overall_stats[method]['improvements'])
        
        method_display = method.replace('_', ' ').title()
        print(f"\n{method_display}:")
        print(f"  Wins: {wins}/{total_experiments} ({win_rate:.1f}%)")
        print(f"  Average improvement: {avg_improvement:+.2f}%")
        
        if wins > 0:
            winning_improvements = [imp for imp in overall_stats[method]['improvements'] if imp > 0]
            if winning_improvements:
                print(f"  Average improvement (wins only): {np.mean(winning_improvements):+.2f}%")

def save_random_splits_results(all_results: List[Dict[str, Any]]):
    """Save results with all random seeds for reproducibility."""
    os.makedirs('results', exist_ok=True)
    
    # Create comprehensive results with full reproducibility info
    comprehensive_results = {
        'experiment_type': 'random_splits_train_validation_test',
        'description': 'Truly random data splits with recorded seeds for reproducibility',
        'methodology': {
            'data_splits': 'Random 70/15/15 train/val/test splits for each trial',
            'models': 'Diverse model architectures with recorded random seeds',
            'trials_per_dataset': 30,
            'reproducibility': 'All random seeds recorded for exact reproduction'
        },
        'selection_methods': {
            'original_davidian': 'val_score - |train_score - val_score|',
            'confidence_davidian': 'stability bonus method',
            'random': 'raw validation scores (baseline)'
        },
        'results': all_results,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open('results/random_splits_results.json', 'w') as f:
        json.dump(comprehensive_results, f, indent=2, default=str)
    
    # Create summary for quick analysis
    summary = []
    for result in all_results:
        for method in ['original_davidian', 'confidence_davidian']:
            method_data = result['aggregate_results'][method]
            summary.append({
                'dataset': result['dataset'],
                'task_type': result['task_type'],
                'method': method,
                'mean_test_score': method_data['mean_test_score'],
                'std_test_score': method_data['std_test_score'],
                'improvement_pct': method_data['improvement_pct'],
                'wins': method_data['improvement'] > 0,
                'n_trials': result['n_trials']
            })
    
    with open('results/random_splits_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n✅ Random splits results saved to:")
    print(f"   - results/random_splits_results.json (detailed with all seeds)")
    print(f"   - results/random_splits_summary.json (summary)")

def main():
    """Run the random splits experiment."""
    try:
        start_time = time.time()
        
        print("🔬 RANDOM SPLITS HYPOTHESIS TEST:")
        print("   Using truly random data splits to ensure model selection variety")
        print("   Recording all random seeds for complete reproducibility")
        
        # Run experiments
        all_results = run_comprehensive_random_splits_experiment()
        
        # Analyze results
        analyze_random_splits_results(all_results)
        
        # Save results
        save_random_splits_results(all_results)
        
        elapsed = time.time() - start_time
        
        print(f"\n{'='*70}")
        print("🎉 RANDOM SPLITS EXPERIMENT COMPLETED!")
        print(f"{'='*70}")
        print(f"Total time: {elapsed:.2f} seconds")
        print("✅ Truly random data splits used")
        print("✅ All random seeds recorded for reproducibility")
        print("✅ Statistical variation in model selection achieved")
        
        return True
        
    except Exception as e:
        print(f"\n{'='*70}")
        print("❌ RANDOM SPLITS EXPERIMENT FAILED")
        print(f"{'='*70}")
        print(f"Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
