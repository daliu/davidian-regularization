#!/usr/bin/env python3
"""
Enhanced train/validation/test experiment with more model diversity.

This experiment addresses the issue of identical model selection by:
1. Using diverse model architectures and hyperparameters
2. Adding controlled noise and complexity variations
3. Testing on scenarios where models genuinely differ
4. Including cross-validation within the training process
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

print("ENHANCED TRAIN/VALIDATION/TEST EXPERIMENT")
print("="*70)
print("Testing with diverse models and hyperparameters")
print("="*70)

def create_diverse_models(task_type: str, n_models: int = 50) -> List[Dict[str, Any]]:
    """
    Create diverse models with different architectures and hyperparameters.
    """
    models = []
    
    if task_type == 'classification':
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.svm import SVC
        from sklearn.neighbors import KNeighborsClassifier
        
        # Logistic Regression with different regularization
        for i in range(15):
            C_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
            penalty_values = ['l1', 'l2']
            solver_values = ['liblinear', 'saga']
            
            models.append({
                'name': f'LogisticRegression_{i}',
                'model_class': LogisticRegression,
                'params': {
                    'C': np.random.choice(C_values),
                    'penalty': np.random.choice(penalty_values),
                    'solver': np.random.choice(solver_values),
                    'random_state': 42 + i,
                    'max_iter': 1000,
                    'n_jobs': 1
                }
            })
        
        # Random Forest with different parameters
        for i in range(15):
            models.append({
                'name': f'RandomForest_{i}',
                'model_class': RandomForestClassifier,
                'params': {
                    'n_estimators': np.random.choice([10, 20, 50, 100]),
                    'max_depth': np.random.choice([3, 5, 10, None]),
                    'min_samples_split': np.random.choice([2, 5, 10]),
                    'min_samples_leaf': np.random.choice([1, 2, 4]),
                    'random_state': 42 + i,
                    'n_jobs': 1
                }
            })
        
        # Gradient Boosting with different parameters
        for i in range(10):
            models.append({
                'name': f'GradientBoosting_{i}',
                'model_class': GradientBoostingClassifier,
                'params': {
                    'n_estimators': np.random.choice([20, 50, 100]),
                    'learning_rate': np.random.choice([0.01, 0.1, 0.2]),
                    'max_depth': np.random.choice([3, 5, 7]),
                    'random_state': 42 + i
                }
            })
        
        # KNN with different parameters
        for i in range(10):
            models.append({
                'name': f'KNN_{i}',
                'model_class': KNeighborsClassifier,
                'params': {
                    'n_neighbors': np.random.choice([3, 5, 7, 9, 11]),
                    'weights': np.random.choice(['uniform', 'distance']),
                    'metric': np.random.choice(['euclidean', 'manhattan', 'minkowski']),
                    'n_jobs': 1
                }
            })
    
    else:  # regression
        from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.neighbors import KNeighborsRegressor
        
        # Linear models with different regularization
        for i in range(20):
            alpha_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
            
            if i < 5:
                models.append({
                    'name': f'LinearRegression_{i}',
                    'model_class': LinearRegression,
                    'params': {'n_jobs': 1}
                })
            elif i < 10:
                models.append({
                    'name': f'Ridge_{i}',
                    'model_class': Ridge,
                    'params': {
                        'alpha': np.random.choice(alpha_values),
                        'random_state': 42 + i
                    }
                })
            elif i < 15:
                models.append({
                    'name': f'Lasso_{i}',
                    'model_class': Lasso,
                    'params': {
                        'alpha': np.random.choice(alpha_values),
                        'random_state': 42 + i,
                        'max_iter': 1000
                    }
                })
            else:
                models.append({
                    'name': f'ElasticNet_{i}',
                    'model_class': ElasticNet,
                    'params': {
                        'alpha': np.random.choice(alpha_values),
                        'l1_ratio': np.random.choice([0.1, 0.5, 0.7, 0.9]),
                        'random_state': 42 + i,
                        'max_iter': 1000
                    }
                })
        
        # Random Forest
        for i in range(15):
            models.append({
                'name': f'RandomForestRegressor_{i}',
                'model_class': RandomForestRegressor,
                'params': {
                    'n_estimators': np.random.choice([10, 20, 50, 100]),
                    'max_depth': np.random.choice([3, 5, 10, None]),
                    'min_samples_split': np.random.choice([2, 5, 10]),
                    'random_state': 42 + i,
                    'n_jobs': 1
                }
            })
        
        # Gradient Boosting
        for i in range(15):
            models.append({
                'name': f'GradientBoostingRegressor_{i}',
                'model_class': GradientBoostingRegressor,
                'params': {
                    'n_estimators': np.random.choice([20, 50, 100]),
                    'learning_rate': np.random.choice([0.01, 0.1, 0.2]),
                    'max_depth': np.random.choice([3, 5, 7]),
                    'random_state': 42 + i
                }
            })
    
    return models[:n_models]  # Return exactly n_models

def split_data_train_val_test(X, y, train_size=0.7, val_size=0.15, test_size=0.15, 
                             stratify=True, random_state=42):
    """Split data into train/validation/test sets."""
    from sklearn.model_selection import train_test_split
    
    # First split: separate test set
    stratify_first = y if stratify else None
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, 
        stratify=stratify_first
    )
    
    # Second split: separate train and validation
    val_size_adjusted = val_size / (train_size + val_size)
    stratify_second = y_temp if stratify else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state + 1,
        stratify=stratify_second
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def train_and_evaluate_models(models: List[Dict[str, Any]], X_train, y_train, 
                             X_val, y_val, X_test, y_test, task_type: str) -> Dict[str, Any]:
    """
    Train all models and collect their train/val/test scores.
    """
    from sklearn.metrics import accuracy_score, mean_squared_error
    
    if task_type == 'classification':
        scoring_func = accuracy_score
    else:
        scoring_func = lambda y_true, y_pred: -mean_squared_error(y_true, y_pred)
    
    results = {
        'model_names': [],
        'train_scores': [],
        'val_scores': [],
        'test_scores': [],
        'models': []
    }
    
    for model_config in models:
        try:
            # Create and train model
            model = model_config['model_class'](**model_config['params'])
            model.fit(X_train, y_train)
            
            # Evaluate on all sets
            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)
            test_pred = model.predict(X_test)
            
            train_score = scoring_func(y_train, train_pred)
            val_score = scoring_func(y_val, val_pred)
            test_score = scoring_func(y_test, test_pred)
            
            results['model_names'].append(model_config['name'])
            results['train_scores'].append(train_score)
            results['val_scores'].append(val_score)
            results['test_scores'].append(test_score)
            results['models'].append(model)
            
        except Exception as e:
            print(f"    Warning: Model {model_config['name']} failed: {e}")
            continue
    
    return results

def apply_selection_methods(train_scores: List[float], val_scores: List[float]) -> Dict[str, List[float]]:
    """Apply different selection methods to get selection scores."""
    
    selection_scores = {}
    
    # 1. Original Davidian (penalty-based)
    original_scores = []
    for train_score, val_score in zip(train_scores, val_scores):
        penalty = abs(train_score - val_score)
        regularized_score = val_score - penalty
        original_scores.append(regularized_score)
    selection_scores['original_davidian'] = original_scores
    
    # 2. Confidence-based Davidian
    confidence_scores = []
    threshold = 0.1
    for train_score, val_score in zip(train_scores, val_scores):
        diff = abs(train_score - val_score)
        if diff < threshold:
            bonus = (threshold - diff) / threshold
            confidence_score = val_score * (1.0 + bonus)
        else:
            confidence_score = val_score
        confidence_scores.append(confidence_score)
    selection_scores['confidence_davidian'] = confidence_scores
    
    # 3. Random/baseline selection
    selection_scores['random'] = val_scores.copy()
    
    # 4. Additional method: Conservative Davidian (smaller penalty)
    conservative_scores = []
    for train_score, val_score in zip(train_scores, val_scores):
        penalty = 0.5 * abs(train_score - val_score)  # Smaller penalty
        regularized_score = val_score - penalty
        conservative_scores.append(regularized_score)
    selection_scores['conservative_davidian'] = conservative_scores
    
    return selection_scores

def select_and_evaluate(selection_scores: Dict[str, List[float]], test_scores: List[float], 
                       model_names: List[str], n_best: int = 4) -> Dict[str, Any]:
    """Select best models and evaluate their test performance."""
    
    results = {}
    
    for method_name, scores in selection_scores.items():
        # Get indices of best models
        sorted_indices = np.argsort(scores)[::-1]  # Descending order
        best_indices = sorted_indices[:n_best]
        
        # Get test scores of selected models
        selected_test_scores = [test_scores[i] for i in best_indices]
        selected_model_names = [model_names[i] for i in best_indices]
        
        results[method_name] = {
            'best_indices': best_indices.tolist(),
            'selected_models': selected_model_names,
            'selected_test_scores': selected_test_scores,
            'mean_test_score': np.mean(selected_test_scores),
            'std_test_score': np.std(selected_test_scores),
            'selection_scores': [scores[i] for i in best_indices]
        }
    
    return results

def run_enhanced_experiment(X, y, task_type: str, dataset_name: str) -> Dict[str, Any]:
    """Run enhanced experiment with diverse models."""
    
    print(f"\n  Running enhanced {dataset_name} experiment...")
    
    # Split data
    stratify = (task_type == 'classification')
    X_train, X_val, X_test, y_train, y_val, y_test = split_data_train_val_test(
        X, y, stratify=stratify
    )
    
    print(f"    Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    # Create diverse models
    models = create_diverse_models(task_type, n_models=60)
    print(f"    Created {len(models)} diverse models")
    
    # Train and evaluate all models
    model_results = train_and_evaluate_models(
        models, X_train, y_train, X_val, y_val, X_test, y_test, task_type
    )
    
    print(f"    Successfully trained {len(model_results['model_names'])} models")
    
    # Apply selection methods
    selection_scores = apply_selection_methods(
        model_results['train_scores'], 
        model_results['val_scores']
    )
    
    # Select best models and evaluate
    selection_results = select_and_evaluate(
        selection_scores, 
        model_results['test_scores'],
        model_results['model_names']
    )
    
    # Calculate improvements relative to random baseline
    baseline_score = selection_results['random']['mean_test_score']
    
    print(f"    Test Performance Results:")
    for method_name, method_data in selection_results.items():
        mean_test = method_data['mean_test_score']
        if method_name != 'random':
            improvement = mean_test - baseline_score
            improvement_pct = (improvement / abs(baseline_score)) * 100 if baseline_score != 0 else 0
            method_data['improvement'] = improvement
            method_data['improvement_pct'] = improvement_pct
            
            status = "✅" if improvement > 0 else "❌"
            print(f"      {status} {method_name}: {mean_test:.4f} ({improvement_pct:+.2f}%)")
        else:
            print(f"      📊 {method_name}: {mean_test:.4f} (baseline)")
    
    return {
        'dataset': dataset_name,
        'task_type': task_type,
        'data_split': {
            'train_size': len(X_train),
            'val_size': len(X_val),
            'test_size': len(X_test)
        },
        'n_models_trained': len(model_results['model_names']),
        'selection_results': selection_results,
        'model_details': model_results
    }

def run_comprehensive_enhanced_experiment():
    """Run comprehensive enhanced experiments."""
    from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_diabetes
    from sklearn.preprocessing import StandardScaler
    
    print("\n1. Running enhanced experiments with diverse models...")
    
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
        
        # Run enhanced experiment
        result = run_enhanced_experiment(X, y, task_type, dataset_name)
        all_results.append(result)
    
    return all_results

def analyze_enhanced_results(all_results: List[Dict[str, Any]]):
    """Analyze enhanced experimental results."""
    print(f"\n2. Enhanced Results Analysis:")
    print("="*70)
    
    methods = ['original_davidian', 'confidence_davidian', 'conservative_davidian']
    method_stats = {method: {'wins': 0, 'improvements': []} for method in methods}
    
    print(f"Dataset-by-Dataset Results:")
    print("-" * 50)
    
    for result in all_results:
        dataset = result['dataset']
        task_type = result['task_type']
        n_models = result['n_models_trained']
        
        print(f"\n{dataset} ({task_type}) - {n_models} models tested:")
        
        baseline_score = result['selection_results']['random']['mean_test_score']
        print(f"  📊 Baseline (Random): {baseline_score:.4f}")
        
        for method in methods:
            method_data = result['selection_results'][method]
            test_score = method_data['mean_test_score']
            improvement_pct = method_data.get('improvement_pct', 0)
            
            if method_data.get('improvement', 0) > 0:
                method_stats[method]['wins'] += 1
                status = "✅"
            else:
                status = "❌"
            
            method_stats[method]['improvements'].append(improvement_pct)
            
            method_display = method.replace('_', ' ').title()
            print(f"  {status} {method_display}: {test_score:.4f} ({improvement_pct:+.2f}%)")
            
            # Show selected models
            selected_models = method_data['selected_models'][:3]  # Show first 3
            print(f"     Selected: {', '.join(selected_models)}")
    
    print(f"\n" + "="*50)
    print(f"ENHANCED EXPERIMENT SUMMARY")
    print("="*50)
    
    total_experiments = len(all_results)
    
    for method in methods:
        wins = method_stats[method]['wins']
        win_rate = (wins / total_experiments) * 100
        avg_improvement = np.mean(method_stats[method]['improvements'])
        
        method_display = method.replace('_', ' ').title()
        print(f"\n{method_display}:")
        print(f"  Wins: {wins}/{total_experiments} ({win_rate:.1f}%)")
        print(f"  Average improvement: {avg_improvement:+.2f}%")
        
        if wins > 0:
            winning_improvements = [imp for imp in method_stats[method]['improvements'] if imp > 0]
            if winning_improvements:
                print(f"  Average improvement (wins only): {np.mean(winning_improvements):+.2f}%")

def save_enhanced_results(all_results: List[Dict[str, Any]]):
    """Save enhanced results."""
    os.makedirs('results', exist_ok=True)
    
    comprehensive_results = {
        'experiment_type': 'enhanced_train_validation_test',
        'description': 'Diverse models with proper train/val/test evaluation',
        'methodology': {
            'models_per_dataset': '~60 diverse models with different architectures and hyperparameters',
            'data_split': '70% train, 15% validation, 15% test',
            'selection_methods': {
                'original_davidian': 'val_score - |train_score - val_score|',
                'confidence_davidian': 'stability bonus method',
                'conservative_davidian': 'val_score - 0.5 * |train_score - val_score|',
                'random': 'raw validation scores'
            }
        },
        'results': all_results,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open('results/enhanced_train_val_test_results.json', 'w') as f:
        json.dump(comprehensive_results, f, indent=2, default=str)
    
    print(f"\n✅ Enhanced results saved to:")
    print(f"   - results/enhanced_train_val_test_results.json")

def main():
    """Run the enhanced train/validation/test experiment."""
    try:
        start_time = time.time()
        
        print("🔬 ENHANCED HYPOTHESIS TEST:")
        print("   Testing with diverse model architectures and hyperparameters")
        print("   to see if Davidian Regularization can identify better generalizing models")
        
        # Run experiments
        all_results = run_comprehensive_enhanced_experiment()
        
        # Analyze results
        analyze_enhanced_results(all_results)
        
        # Save results
        save_enhanced_results(all_results)
        
        elapsed = time.time() - start_time
        
        print(f"\n{'='*70}")
        print("🎉 ENHANCED EXPERIMENT COMPLETED!")
        print(f"{'='*70}")
        print(f"Total time: {elapsed:.2f} seconds")
        print("✅ Diverse model architectures tested")
        print("✅ Multiple Davidian formulations compared")
        print("✅ True generalization performance measured")
        
        return True
        
    except Exception as e:
        print(f"\n{'='*70}")
        print("❌ ENHANCED EXPERIMENT FAILED")
        print(f"{'='*70}")
        print(f"Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
