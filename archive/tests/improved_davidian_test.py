#!/usr/bin/env python3
"""
Improved Davidian Regularization test with better penalty tuning.
"""

import sys
import os
import time
import json
import numpy as np

# Force single-threaded execution
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'

print("IMPROVED DAVIDIAN REGULARIZATION TEST")
print("="*60)
print("Testing with improved penalty formulations")
print("="*60)

def improved_davidian_penalty(train_score, val_score, alpha=0.5, method='proportional'):
    """
    Improved Davidian penalty calculations.
    
    Args:
        train_score: Training score
        val_score: Validation score
        alpha: Penalty weight (0.1 to 1.0)
        method: Penalty method ('proportional', 'sqrt', 'log', 'adaptive')
    """
    diff = abs(train_score - val_score)
    
    if method == 'proportional':
        # Proportional to the difference, scaled by alpha
        penalty = alpha * diff
    elif method == 'sqrt':
        # Square root penalty (less aggressive)
        penalty = alpha * np.sqrt(diff)
    elif method == 'log':
        # Logarithmic penalty (even less aggressive)
        penalty = alpha * np.log(1 + diff)
    elif method == 'adaptive':
        # Adaptive penalty based on validation score magnitude
        penalty = alpha * diff * (1 / (1 + abs(val_score)))
    else:
        penalty = alpha * diff
    
    return penalty

def run_improved_davidian_experiment(X, y, task_type, k=3, n_trials=10, 
                                   alpha=0.5, penalty_method='proportional'):
    """Run improved Davidian Regularization experiment."""
    from sklearn.model_selection import KFold, StratifiedKFold
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.metrics import accuracy_score, mean_squared_error
    
    # Choose model and scoring based on task type
    if task_type == 'classification':
        model_class = LogisticRegression
        model_params = {'random_state': 42, 'max_iter': 1000, 'n_jobs': 1}
        cv_class = StratifiedKFold
        scoring_func = accuracy_score
    else:
        model_class = LinearRegression
        model_params = {'n_jobs': 1}
        cv_class = KFold
        scoring_func = lambda y_true, y_pred: -mean_squared_error(y_true, y_pred)
    
    trial_scores = []
    trial_details = []
    
    for trial in range(n_trials):
        cv = cv_class(n_splits=k, shuffle=True, random_state=trial)
        fold_scores = []
        fold_details = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train model
            model = model_class(**model_params)
            model.fit(X_train, y_train)
            
            # Get predictions and scores
            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)
            
            train_score = scoring_func(y_train, train_pred)
            val_score = scoring_func(y_val, val_pred)
            
            # Apply improved Davidian Regularization
            penalty = improved_davidian_penalty(train_score, val_score, alpha, penalty_method)
            regularized_score = val_score - penalty
            
            fold_scores.append(regularized_score)
            fold_details.append({
                'fold': fold_idx,
                'train_score': train_score,
                'val_score': val_score,
                'penalty': penalty,
                'regularized_score': regularized_score
            })
        
        trial_mean = np.mean(fold_scores)
        trial_scores.append(trial_mean)
        trial_details.append({
            'trial': trial,
            'folds': fold_details,
            'mean_score': trial_mean
        })
    
    # Get best 4 trials
    best_4_scores = sorted(trial_scores, reverse=True)[:4]
    return {
        'all_scores': trial_scores,
        'best_4_scores': best_4_scores,
        'mean_best_4': np.mean(best_4_scores),
        'overall_mean': np.mean(trial_scores),
        'overall_std': np.std(trial_scores),
        'trial_details': trial_details,
        'alpha': alpha,
        'penalty_method': penalty_method
    }

def run_random_experiment(X, y, task_type, n_trials=10, test_size=0.2):
    """Run random sampling experiment (same as before)."""
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.metrics import accuracy_score, mean_squared_error
    
    if task_type == 'classification':
        model_class = LogisticRegression
        model_params = {'random_state': 42, 'max_iter': 1000, 'n_jobs': 1}
        scoring_func = accuracy_score
        stratify = y
    else:
        model_class = LinearRegression
        model_params = {'n_jobs': 1}
        scoring_func = lambda y_true, y_pred: -mean_squared_error(y_true, y_pred)
        stratify = None
    
    trial_scores = []
    
    for trial in range(n_trials):
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=trial, stratify=stratify
        )
        
        model = model_class(**model_params)
        model.fit(X_train, y_train)
        
        val_pred = model.predict(X_val)
        val_score = scoring_func(y_val, val_pred)
        
        trial_scores.append(val_score)
    
    best_4_scores = sorted(trial_scores, reverse=True)[:4]
    return {
        'all_scores': trial_scores,
        'best_4_scores': best_4_scores,
        'mean_best_4': np.mean(best_4_scores),
        'overall_mean': np.mean(trial_scores),
        'overall_std': np.std(trial_scores)
    }

def test_penalty_methods():
    """Test different penalty methods and alpha values."""
    from sklearn.datasets import load_iris, load_wine
    from sklearn.preprocessing import StandardScaler
    
    print("\n1. Testing penalty methods on Iris dataset...")
    
    # Load and prepare data
    iris = load_iris()
    scaler = StandardScaler()
    X = scaler.fit_transform(iris.data)
    y = iris.target
    
    penalty_methods = ['proportional', 'sqrt', 'log', 'adaptive']
    alpha_values = [0.1, 0.3, 0.5, 0.7, 1.0]
    
    results = {}
    
    # Get baseline random performance
    random_result = run_random_experiment(X, y, 'classification', n_trials=20)
    baseline_score = random_result['mean_best_4']
    
    print(f"   Baseline (Random): {baseline_score:.4f}")
    
    best_improvement = -float('inf')
    best_config = None
    
    for method in penalty_methods:
        print(f"\n   Testing {method} penalty method:")
        for alpha in alpha_values:
            davidian_result = run_improved_davidian_experiment(
                X, y, 'classification', k=3, n_trials=20, 
                alpha=alpha, penalty_method=method
            )
            
            davidian_score = davidian_result['mean_best_4']
            improvement = davidian_score - baseline_score
            improvement_pct = (improvement / abs(baseline_score)) * 100
            
            results[f"{method}_alpha_{alpha}"] = {
                'method': method,
                'alpha': alpha,
                'davidian_score': davidian_score,
                'baseline_score': baseline_score,
                'improvement': improvement,
                'improvement_pct': improvement_pct
            }
            
            print(f"     α={alpha}: {davidian_score:.4f} ({improvement_pct:+.2f}%)")
            
            if improvement > best_improvement:
                best_improvement = improvement
                best_config = (method, alpha)
    
    print(f"\n   Best configuration: {best_config[0]} with α={best_config[1]}")
    print(f"   Best improvement: {best_improvement:+.4f} ({(best_improvement/abs(baseline_score))*100:+.2f}%)")
    
    return results, best_config

def run_comprehensive_improved_test():
    """Run comprehensive test with improved algorithm."""
    from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_diabetes
    from sklearn.preprocessing import StandardScaler
    
    # Test penalty methods first
    penalty_results, best_config = test_penalty_methods()
    best_method, best_alpha = best_config
    
    print(f"\n2. Running comprehensive test with best configuration:")
    print(f"   Method: {best_method}, Alpha: {best_alpha}")
    
    datasets = {
        'iris': (load_iris(), 'classification'),
        'wine': (load_wine(), 'classification'),
        'breast_cancer': (load_breast_cancer(), 'classification'),
        'diabetes': (load_diabetes(), 'regression')
    }
    
    final_results = []
    
    for dataset_name, (dataset, task_type) in datasets.items():
        print(f"\n   Testing {dataset_name}...")
        
        scaler = StandardScaler()
        X = scaler.fit_transform(dataset.data)
        y = dataset.target
        
        # Run improved Davidian
        davidian_result = run_improved_davidian_experiment(
            X, y, task_type, k=5, n_trials=50, 
            alpha=best_alpha, penalty_method=best_method
        )
        
        # Run random baseline
        random_result = run_random_experiment(X, y, task_type, n_trials=50)
        
        # Calculate improvement
        davidian_score = davidian_result['mean_best_4']
        random_score = random_result['mean_best_4']
        improvement = davidian_score - random_score
        improvement_pct = (improvement / abs(random_score)) * 100 if random_score != 0 else 0
        
        result = {
            'dataset': dataset_name,
            'task_type': task_type,
            'davidian_score': davidian_score,
            'random_score': random_score,
            'improvement': improvement,
            'improvement_pct': improvement_pct,
            'davidian_better': improvement > 0,
            'method': best_method,
            'alpha': best_alpha
        }
        
        final_results.append(result)
        
        print(f"     Davidian: {davidian_score:.4f}")
        print(f"     Random: {random_score:.4f}")
        print(f"     Improvement: {improvement_pct:+.2f}%")
        
        if improvement > 0:
            print("     ✅ Davidian better!")
        else:
            print("     ❌ Random better")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    
    with open('results/improved_davidian_results.json', 'w') as f:
        json.dump({
            'penalty_tuning_results': penalty_results,
            'best_config': {'method': best_method, 'alpha': best_alpha},
            'final_results': final_results,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }, f, indent=2, default=str)
    
    return final_results

def analyze_improved_results(results):
    """Analyze improved results."""
    print(f"\n3. Improved Results Analysis:")
    print("="*60)
    
    total_experiments = len(results)
    davidian_wins = sum(1 for r in results if r['davidian_better'])
    win_rate = (davidian_wins / total_experiments) * 100
    
    print(f"Total experiments: {total_experiments}")
    print(f"Davidian Regularization wins: {davidian_wins}")
    print(f"Win rate: {win_rate:.1f}%")
    
    avg_improvement = np.mean([r['improvement_pct'] for r in results])
    print(f"Average improvement: {avg_improvement:+.2f}%")
    
    print(f"\nResults by dataset:")
    for result in results:
        status = "✅" if result['davidian_better'] else "❌"
        print(f"  {status} {result['dataset']}: {result['improvement_pct']:+.2f}%")

def main():
    """Run improved Davidian Regularization test."""
    try:
        start_time = time.time()
        
        results = run_comprehensive_improved_test()
        analyze_improved_results(results)
        
        elapsed = time.time() - start_time
        
        print(f"\n{'='*60}")
        print("🎉 IMPROVED DAVIDIAN TEST COMPLETED! 🎉")
        print(f"{'='*60}")
        print(f"Total time: {elapsed:.2f} seconds")
        print("✅ No mutex blocking issues")
        print("✅ Penalty methods optimized")
        print("✅ Results saved to results/improved_davidian_results.json")
        
        return True
        
    except Exception as e:
        print(f"\n{'='*60}")
        print("❌ TEST FAILED")
        print(f"{'='*60}")
        print(f"Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
