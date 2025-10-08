#!/usr/bin/env python3
"""
Definitive train/validation/test experiment for Davidian Regularization.

This experiment implements proper data splitting to avoid validation data leakage:
1. Train models on training data
2. Select best models using validation data with different selection criteria
3. Evaluate final selected models on unseen test data

This tests the hypothesis that Davidian Regularization improves generalization
to truly unseen data, even if validation performance appears worse.
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

print("TRAIN/VALIDATION/TEST DAVIDIAN REGULARIZATION EXPERIMENT")
print("="*70)
print("Testing generalization to unseen test data")
print("="*70)

def split_data_train_val_test(X, y, train_size=0.7, val_size=0.15, test_size=0.15, 
                             stratify=True, random_state=42):
    """
    Split data into train/validation/test sets.
    
    Args:
        X: Features
        y: Labels
        train_size: Fraction for training (default 0.7)
        val_size: Fraction for validation (default 0.15)  
        test_size: Fraction for testing (default 0.15)
        stratify: Whether to stratify split for classification
        random_state: Random seed
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    from sklearn.model_selection import train_test_split
    
    # First split: separate test set
    test_size_adjusted = test_size
    stratify_first = y if stratify else None
    
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size_adjusted, random_state=random_state, 
        stratify=stratify_first
    )
    
    # Second split: separate train and validation from remaining data
    val_size_adjusted = val_size / (train_size + val_size)
    stratify_second = y_temp if stratify else None
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state + 1,
        stratify=stratify_second
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def original_davidian_selection(train_scores: List[float], val_scores: List[float], 
                               alpha: float = 1.0) -> List[float]:
    """
    Original Davidian Regularization: penalize validation scores.
    
    regularized_score = val_score - α * |train_score - val_score|
    """
    regularized_scores = []
    for train_score, val_score in zip(train_scores, val_scores):
        penalty = alpha * abs(train_score - val_score)
        regularized_score = val_score - penalty
        regularized_scores.append(regularized_score)
    
    return regularized_scores

def confidence_based_selection(train_scores: List[float], val_scores: List[float], 
                              threshold: float = 0.1) -> List[float]:
    """
    Confidence-based Davidian: reward stable models.
    """
    confidence_scores = []
    for train_score, val_score in zip(train_scores, val_scores):
        diff = abs(train_score - val_score)
        if diff < threshold:
            bonus = (threshold - diff) / threshold
            confidence_score = val_score * (1.0 + bonus)
        else:
            confidence_score = val_score
        confidence_scores.append(confidence_score)
    
    return confidence_scores

def random_selection(val_scores: List[float]) -> List[float]:
    """
    Random/baseline selection: use raw validation scores.
    """
    return val_scores.copy()

def run_model_trials(X_train, y_train, X_val, y_val, X_test, y_test, 
                    task_type: str, n_trials: int = 50) -> Dict[str, Any]:
    """
    Run multiple model training trials and collect train/val/test scores.
    
    Returns:
        Dictionary with lists of train_scores, val_scores, test_scores, and models
    """
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.metrics import accuracy_score, mean_squared_error
    
    # Choose model and scoring based on task type
    if task_type == 'classification':
        model_class = LogisticRegression
        model_params = {'random_state': 42, 'max_iter': 1000, 'n_jobs': 1}
        scoring_func = accuracy_score
    else:
        model_class = LinearRegression
        model_params = {'n_jobs': 1}
        scoring_func = lambda y_true, y_pred: -mean_squared_error(y_true, y_pred)
    
    train_scores = []
    val_scores = []
    test_scores = []
    models = []
    
    for trial in range(n_trials):
        # Create model with slight variations (different random states for initialization)
        model_params_trial = model_params.copy()
        if 'random_state' in model_params_trial:
            model_params_trial['random_state'] = 42 + trial
        
        model = model_class(**model_params_trial)
        
        # Train on training data only
        model.fit(X_train, y_train)
        
        # Evaluate on all three sets
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)
        
        train_score = scoring_func(y_train, train_pred)
        val_score = scoring_func(y_val, val_pred)
        test_score = scoring_func(y_test, test_pred)
        
        train_scores.append(train_score)
        val_scores.append(val_score)
        test_scores.append(test_score)
        models.append(model)
    
    return {
        'train_scores': train_scores,
        'val_scores': val_scores,
        'test_scores': test_scores,
        'models': models,
        'n_trials': n_trials
    }

def select_best_models(selection_scores: List[float], test_scores: List[float], 
                      n_best: int = 4) -> Dict[str, Any]:
    """
    Select best models based on selection scores and return their test performance.
    """
    # Get indices of best models according to selection criteria
    sorted_indices = np.argsort(selection_scores)[::-1]  # Descending order
    best_indices = sorted_indices[:n_best]
    
    # Get test scores of selected models
    selected_test_scores = [test_scores[i] for i in best_indices]
    
    return {
        'best_indices': best_indices.tolist(),
        'selected_test_scores': selected_test_scores,
        'mean_test_score': np.mean(selected_test_scores),
        'std_test_score': np.std(selected_test_scores),
        'selection_scores': [selection_scores[i] for i in best_indices]
    }

def run_train_val_test_experiment(X, y, task_type: str, dataset_name: str, 
                                 n_trials: int = 100) -> Dict[str, Any]:
    """
    Run complete train/validation/test experiment comparing selection methods.
    """
    print(f"\n  Running {dataset_name} experiment...")
    
    # Split data
    stratify = (task_type == 'classification')
    X_train, X_val, X_test, y_train, y_val, y_test = split_data_train_val_test(
        X, y, stratify=stratify
    )
    
    print(f"    Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    # Run model trials
    trial_results = run_model_trials(
        X_train, y_train, X_val, y_val, X_test, y_test, 
        task_type, n_trials
    )
    
    print(f"    Completed {n_trials} model trials")
    
    # Apply different selection methods
    selection_methods = {}
    
    # 1. Original Davidian (penalty-based)
    original_scores = original_davidian_selection(
        trial_results['train_scores'], 
        trial_results['val_scores'], 
        alpha=1.0
    )
    selection_methods['original_davidian'] = select_best_models(
        original_scores, trial_results['test_scores']
    )
    
    # 2. Confidence-based Davidian
    confidence_scores = confidence_based_selection(
        trial_results['train_scores'], 
        trial_results['val_scores'], 
        threshold=0.1
    )
    selection_methods['confidence_davidian'] = select_best_models(
        confidence_scores, trial_results['test_scores']
    )
    
    # 3. Random/baseline selection
    random_scores = random_selection(trial_results['val_scores'])
    selection_methods['random'] = select_best_models(
        random_scores, trial_results['test_scores']
    )
    
    # Compare methods
    results = {
        'dataset': dataset_name,
        'task_type': task_type,
        'data_split': {
            'train_size': len(X_train),
            'val_size': len(X_val), 
            'test_size': len(X_test)
        },
        'n_trials': n_trials,
        'selection_methods': selection_methods,
        'raw_scores': trial_results
    }
    
    # Print results
    print(f"    Test Performance Results:")
    for method_name, method_results in selection_methods.items():
        mean_test = method_results['mean_test_score']
        print(f"      {method_name}: {mean_test:.4f}")
    
    # Calculate improvements
    baseline_score = selection_methods['random']['mean_test_score']
    
    for method_name in ['original_davidian', 'confidence_davidian']:
        method_score = selection_methods[method_name]['mean_test_score']
        improvement = method_score - baseline_score
        improvement_pct = (improvement / abs(baseline_score)) * 100 if baseline_score != 0 else 0
        
        results['selection_methods'][method_name]['improvement'] = improvement
        results['selection_methods'][method_name]['improvement_pct'] = improvement_pct
        
        status = "✅" if improvement > 0 else "❌"
        print(f"      {status} {method_name} vs random: {improvement_pct:+.2f}%")
    
    return results

def run_comprehensive_train_val_test():
    """
    Run comprehensive train/validation/test experiments.
    """
    from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_diabetes
    from sklearn.preprocessing import StandardScaler
    
    print("\n1. Loading datasets and running experiments...")
    
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
        
        # Run experiment
        result = run_train_val_test_experiment(X, y, task_type, dataset_name, n_trials=100)
        all_results.append(result)
    
    return all_results

def analyze_train_val_test_results(all_results: List[Dict[str, Any]]):
    """
    Analyze and summarize train/validation/test results.
    """
    print(f"\n2. Train/Validation/Test Results Analysis:")
    print("="*70)
    
    # Overall statistics
    total_experiments = len(all_results)
    
    method_wins = {
        'original_davidian': 0,
        'confidence_davidian': 0
    }
    
    method_improvements = {
        'original_davidian': [],
        'confidence_davidian': []
    }
    
    print(f"Dataset-by-Dataset Results:")
    print("-" * 50)
    
    for result in all_results:
        dataset = result['dataset']
        task_type = result['task_type']
        
        print(f"\n{dataset} ({task_type}):")
        
        baseline_score = result['selection_methods']['random']['mean_test_score']
        print(f"  Baseline (Random): {baseline_score:.4f}")
        
        for method in ['original_davidian', 'confidence_davidian']:
            method_data = result['selection_methods'][method]
            test_score = method_data['mean_test_score']
            improvement_pct = method_data['improvement_pct']
            
            if method_data['improvement'] > 0:
                method_wins[method] += 1
                status = "✅"
            else:
                status = "❌"
            
            method_improvements[method].append(improvement_pct)
            
            method_display = method.replace('_', ' ').title()
            print(f"  {status} {method_display}: {test_score:.4f} ({improvement_pct:+.2f}%)")
    
    print(f"\n" + "="*50)
    print(f"OVERALL SUMMARY")
    print("="*50)
    
    for method in ['original_davidian', 'confidence_davidian']:
        wins = method_wins[method]
        win_rate = (wins / total_experiments) * 100
        avg_improvement = np.mean(method_improvements[method])
        
        method_display = method.replace('_', ' ').title()
        print(f"\n{method_display}:")
        print(f"  Wins: {wins}/{total_experiments} ({win_rate:.1f}%)")
        print(f"  Average improvement: {avg_improvement:+.2f}%")
        
        if wins > 0:
            winning_improvements = [imp for imp in method_improvements[method] if imp > 0]
            if winning_improvements:
                print(f"  Average improvement (wins only): {np.mean(winning_improvements):+.2f}%")
    
    # Task-specific analysis
    print(f"\nTask-Specific Analysis:")
    print("-" * 30)
    
    classification_results = [r for r in all_results if r['task_type'] == 'classification']
    regression_results = [r for r in all_results if r['task_type'] == 'regression']
    
    for task_type, task_results in [('Classification', classification_results), 
                                   ('Regression', regression_results)]:
        if task_results:
            print(f"\n{task_type} Tasks:")
            
            for method in ['original_davidian', 'confidence_davidian']:
                task_improvements = [r['selection_methods'][method]['improvement_pct'] 
                                   for r in task_results]
                task_wins = sum(1 for r in task_results 
                               if r['selection_methods'][method]['improvement'] > 0)
                
                avg_imp = np.mean(task_improvements)
                win_rate = (task_wins / len(task_results)) * 100
                
                method_display = method.replace('_', ' ').title()
                print(f"  {method_display}: {win_rate:.1f}% win rate, {avg_imp:+.2f}% avg improvement")

def save_train_val_test_results(all_results: List[Dict[str, Any]]):
    """Save results with proper documentation."""
    os.makedirs('results', exist_ok=True)
    
    # Create comprehensive results dictionary
    comprehensive_results = {
        'experiment_type': 'train_validation_test',
        'description': 'Proper evaluation on unseen test data after model selection',
        'methodology': {
            'data_split': '70% train, 15% validation, 15% test',
            'model_selection': 'Based on validation data using different criteria',
            'evaluation': 'Final test performance on unseen data',
            'n_trials_per_dataset': 100
        },
        'selection_methods': {
            'original_davidian': 'val_score - |train_score - val_score|',
            'confidence_davidian': 'stability bonus for small train-val differences',
            'random': 'raw validation scores (baseline)'
        },
        'results': all_results,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Save detailed results
    with open('results/train_val_test_results.json', 'w') as f:
        json.dump(comprehensive_results, f, indent=2, default=str)
    
    # Create summary for easy analysis
    summary = []
    for result in all_results:
        for method in ['original_davidian', 'confidence_davidian']:
            summary.append({
                'dataset': result['dataset'],
                'task_type': result['task_type'],
                'method': method,
                'test_score': result['selection_methods'][method]['mean_test_score'],
                'baseline_score': result['selection_methods']['random']['mean_test_score'],
                'improvement_pct': result['selection_methods'][method]['improvement_pct'],
                'wins': result['selection_methods'][method]['improvement'] > 0
            })
    
    with open('results/train_val_test_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n✅ Results saved to:")
    print(f"   - results/train_val_test_results.json (detailed)")
    print(f"   - results/train_val_test_summary.json (summary)")

def main():
    """Run the definitive train/validation/test experiment."""
    try:
        start_time = time.time()
        
        print("🔬 HYPOTHESIS TEST:")
        print("   H₀: Davidian Regularization shows no improvement on unseen test data")
        print("   H₁: Davidian Regularization improves generalization to test data")
        
        # Run experiments
        all_results = run_comprehensive_train_val_test()
        
        # Analyze results
        analyze_train_val_test_results(all_results)
        
        # Save results
        save_train_val_test_results(all_results)
        
        elapsed = time.time() - start_time
        
        print(f"\n{'='*70}")
        print("🎉 TRAIN/VALIDATION/TEST EXPERIMENT COMPLETED!")
        print(f"{'='*70}")
        print(f"Total time: {elapsed:.2f} seconds")
        print("✅ Proper data splitting implemented")
        print("✅ Both original and confidence-based methods tested")
        print("✅ True generalization performance measured")
        print("✅ Results saved for paper writeup")
        
        return True
        
    except Exception as e:
        print(f"\n{'='*70}")
        print("❌ EXPERIMENT FAILED")
        print(f"{'='*70}")
        print(f"Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
