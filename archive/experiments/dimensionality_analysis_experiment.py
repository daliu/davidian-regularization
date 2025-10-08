#!/usr/bin/env python3
"""
Dimensionality analysis experiment for Davidian Regularization.

This experiment systematically tests how feature dimensionality affects
the effectiveness of Davidian Regularization, with the hypothesis that
higher dimensionality leads to more overfitting variance and thus
better performance for our selection method.
"""

import sys
import os
import time
import json
import numpy as np
from typing import Dict, List, Tuple, Any
from scipy import stats

# Force single-threaded execution
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

print("DIMENSIONALITY ANALYSIS EXPERIMENT")
print("="*70)
print("Testing how feature dimensionality affects Davidian Regularization")
print("="*70)

def generate_random_seed():
    """Generate a truly random seed."""
    return np.random.randint(0, 2**31 - 1)

def create_controlled_dimensionality_datasets(n_samples: int = 1000, 
                                            dimensionalities: List[int] = [5, 10, 20, 50, 100, 200, 500],
                                            task_type: str = 'classification') -> Dict[str, Tuple]:
    """
    Create datasets with controlled dimensionality for systematic testing.
    
    Args:
        n_samples: Number of samples (fixed across all datasets)
        dimensionalities: List of feature dimensions to test
        task_type: 'classification' or 'regression'
    
    Returns:
        Dictionary mapping dimensionality to (X, y, metadata)
    """
    datasets = {}
    
    for n_features in dimensionalities:
        print(f"  Creating {task_type} dataset: {n_samples} samples, {n_features} features...")
        
        if task_type == 'classification':
            from sklearn.datasets import make_classification
            
            # Adjust parameters based on dimensionality
            n_informative = min(n_features, max(2, n_features // 2))
            n_redundant = min(n_features - n_informative, max(0, n_features // 4))
            n_classes = min(5, max(2, n_features // 20 + 2))  # More classes for higher dimensions
            
            X, y = make_classification(
                n_samples=n_samples,
                n_features=n_features,
                n_informative=n_informative,
                n_redundant=n_redundant,
                n_classes=n_classes,
                n_clusters_per_class=1,
                class_sep=0.8,  # Moderate difficulty
                flip_y=0.01,    # 1% label noise
                random_state=42
            )
            
            metadata = {
                'name': f'Synthetic Classification (D={n_features})',
                'type': 'classification',
                'n_samples': n_samples,
                'n_features': n_features,
                'n_classes': n_classes,
                'n_informative': n_informative,
                'n_redundant': n_redundant,
                'dimensionality_ratio': n_features / n_samples,
                'complexity': 'controlled'
            }
            
        else:  # regression
            from sklearn.datasets import make_regression
            
            n_informative = min(n_features, max(1, n_features // 2))
            
            X, y = make_regression(
                n_samples=n_samples,
                n_features=n_features,
                n_informative=n_informative,
                noise=0.1,  # Low noise for controlled comparison
                bias=0.0,
                random_state=42
            )
            
            metadata = {
                'name': f'Synthetic Regression (D={n_features})',
                'type': 'regression',
                'n_samples': n_samples,
                'n_features': n_features,
                'n_informative': n_informative,
                'dimensionality_ratio': n_features / n_samples,
                'complexity': 'controlled'
            }
        
        datasets[n_features] = (X, y, metadata)
    
    return datasets

def create_dimensionality_appropriate_models(task_type: str, n_features: int, n_models: int = 25):
    """
    Create models appropriate for different dimensionalities.
    
    For high-dimensional data, we focus on models that handle dimensionality well.
    """
    models = []
    
    if task_type == 'classification':
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.naive_bayes import GaussianNB
        from sklearn.neighbors import KNeighborsClassifier
        
        # Logistic Regression (good for high dimensions)
        for i in range(8):
            C_values = [0.001, 0.01, 0.1, 1.0, 10.0] if n_features > 100 else [0.1, 1.0, 10.0, 100.0]
            models.append({
                'name': f'LogReg_{i}',
                'class': LogisticRegression,
                'params': {
                    'C': np.random.choice(C_values),
                    'penalty': 'l2',
                    'solver': 'lbfgs' if n_features < 100 else 'saga',
                    'random_state': 42 + i,
                    'max_iter': 2000,
                    'n_jobs': 1
                }
            })
        
        # Random Forest (handles high dimensions well)
        for i in range(10):
            # Adjust parameters for dimensionality
            max_features_options = ['sqrt', 'log2'] if n_features > 50 else ['sqrt', 'log2', None]
            n_estimators_options = [50, 100] if n_features > 200 else [20, 50, 100, 200]
            
            models.append({
                'name': f'RF_{i}',
                'class': RandomForestClassifier,
                'params': {
                    'n_estimators': np.random.choice(n_estimators_options),
                    'max_features': np.random.choice(max_features_options),
                    'max_depth': np.random.choice([5, 10, 15, None]),
                    'min_samples_split': np.random.choice([2, 5, 10]),
                    'random_state': 42 + i,
                    'n_jobs': 1
                }
            })
        
        # Naive Bayes (good baseline for high dimensions)
        for i in range(3):
            models.append({
                'name': f'NB_{i}',
                'class': GaussianNB,
                'params': {
                    'var_smoothing': np.random.choice([1e-9, 1e-8, 1e-7, 1e-6])
                }
            })
        
        # KNN (only for lower dimensions due to curse of dimensionality)
        if n_features < 100:
            for i in range(4):
                models.append({
                    'name': f'KNN_{i}',
                    'class': KNeighborsClassifier,
                    'params': {
                        'n_neighbors': np.random.choice([3, 5, 7, 11]),
                        'weights': np.random.choice(['uniform', 'distance']),
                        'n_jobs': 1
                    }
                })
    
    else:  # regression
        from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.neighbors import KNeighborsRegressor
        
        # Linear models with regularization (essential for high dimensions)
        for i in range(5):
            alpha_values = [0.01, 0.1, 1.0, 10.0, 100.0] if n_features > 100 else [0.001, 0.01, 0.1, 1.0, 10.0]
            
            if i < 2 and n_features < 50:  # Only use unregularized linear for low dimensions
                models.append({
                    'name': f'Linear_{i}',
                    'class': LinearRegression,
                    'params': {'n_jobs': 1}
                })
            else:
                models.append({
                    'name': f'Ridge_{i}',
                    'class': Ridge,
                    'params': {
                        'alpha': np.random.choice(alpha_values),
                        'random_state': 42 + i
                    }
                })
        
        # Lasso (excellent for high dimensions with sparsity)
        for i in range(8):
            alpha_values = [0.01, 0.1, 1.0, 10.0] if n_features > 100 else [0.001, 0.01, 0.1, 1.0]
            models.append({
                'name': f'Lasso_{i}',
                'class': Lasso,
                'params': {
                    'alpha': np.random.choice(alpha_values),
                    'random_state': 42 + i,
                    'max_iter': 3000
                }
            })
        
        # ElasticNet (good for high dimensions)
        for i in range(6):
            models.append({
                'name': f'ElasticNet_{i}',
                'class': ElasticNet,
                'params': {
                    'alpha': np.random.choice([0.01, 0.1, 1.0, 10.0]),
                    'l1_ratio': np.random.choice([0.1, 0.3, 0.5, 0.7, 0.9]),
                    'random_state': 42 + i,
                    'max_iter': 3000
                }
            })
        
        # Random Forest (handles high dimensions)
        for i in range(6):
            max_features_options = ['sqrt', 'log2'] if n_features > 50 else ['sqrt', 'log2', None]
            models.append({
                'name': f'RF_Reg_{i}',
                'class': RandomForestRegressor,
                'params': {
                    'n_estimators': np.random.choice([50, 100] if n_features > 200 else [20, 50, 100]),
                    'max_features': np.random.choice(max_features_options),
                    'max_depth': np.random.choice([5, 10, None]),
                    'random_state': 42 + i,
                    'n_jobs': 1
                }
            })
    
    return models[:n_models]

def run_dimensionality_experiment(X, y, metadata: Dict[str, Any], n_trials: int = 15):
    """
    Run Davidian Regularization experiment for a specific dimensionality.
    """
    from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
    from sklearn.metrics import accuracy_score, mean_squared_error
    from sklearn.preprocessing import StandardScaler
    
    dataset_name = metadata['name']
    task_type = metadata['type']
    n_features = metadata['n_features']
    dimensionality_ratio = metadata['dimensionality_ratio']
    
    print(f"    Dataset: {dataset_name}")
    print(f"    Dimensionality ratio (features/samples): {dimensionality_ratio:.3f}")
    
    # Standardize features (crucial for high dimensions)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Use K=5 based on previous findings
    k = 5
    
    trial_results = []
    davidian_improvements = []
    
    for trial in range(n_trials):
        trial_seed = generate_random_seed()
        
        # Random split
        stratify = y if task_type == 'classification' else None
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.3, random_state=trial_seed, stratify=stratify
        )
        
        # Create appropriate models for this dimensionality
        models = create_dimensionality_appropriate_models(task_type, n_features, n_models=25)
        
        # Scoring function
        if task_type == 'classification':
            scoring_func = accuracy_score
            cv_class = StratifiedKFold
        else:
            scoring_func = lambda y_true, y_pred: -mean_squared_error(y_true, y_pred)
            cv_class = KFold
        
        # Evaluate models with k-fold CV
        model_davidian_scores = []
        model_regular_scores = []
        model_test_scores = []
        model_names = []
        
        for model_config in models:
            try:
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
                
                # Also get test performance
                final_model = model_config['class'](**model_config['params'])
                final_model.fit(X_train, y_train)
                test_pred = final_model.predict(X_test)
                test_score = scoring_func(y_test, test_pred)
                
                model_davidian_scores.append(davidian_score)
                model_regular_scores.append(mean_val_score)
                model_test_scores.append(test_score)
                model_names.append(model_config['name'])
                
            except Exception as e:
                # Skip failed models (common in high dimensions)
                continue
        
        if len(model_names) < 4:
            continue
        
        # Select top 4 models
        n_select = min(4, len(model_names))
        
        # Davidian selection
        davidian_indices = np.argsort(model_davidian_scores)[::-1][:n_select]
        davidian_test_scores = [model_test_scores[i] for i in davidian_indices]
        
        # Regular selection
        regular_indices = np.argsort(model_regular_scores)[::-1][:n_select]
        regular_test_scores = [model_test_scores[i] for i in regular_indices]
        
        # Calculate improvement
        davidian_mean = np.mean(davidian_test_scores)
        regular_mean = np.mean(regular_test_scores)
        improvement = davidian_mean - regular_mean
        improvement_pct = (improvement / abs(regular_mean)) * 100 if regular_mean != 0 else 0
        
        trial_results.append({
            'trial': trial,
            'trial_seed': trial_seed,
            'n_models_evaluated': len(model_names),
            'davidian_test_mean': davidian_mean,
            'regular_test_mean': regular_mean,
            'improvement': improvement,
            'improvement_pct': improvement_pct,
            'avg_train_val_difference': np.mean([abs(ds + abs(rs - ds) - rs) for ds, rs in zip(model_davidian_scores, model_regular_scores)])
        })
        
        davidian_improvements.append(improvement_pct)
    
    # Calculate statistics
    if davidian_improvements:
        mean_improvement = np.mean(davidian_improvements)
        std_improvement = np.std(davidian_improvements)
        wins = sum(1 for imp in davidian_improvements if imp > 0)
        win_rate = (wins / len(davidian_improvements)) * 100
        
        # Statistical significance test
        t_stat, p_value = stats.ttest_1samp(davidian_improvements, 0)
        significant = p_value < 0.05
        
        result = {
            'dimensionality': n_features,
            'dimensionality_ratio': dimensionality_ratio,
            'n_trials': len(trial_results),
            'mean_improvement_pct': mean_improvement,
            'std_improvement_pct': std_improvement,
            'wins': wins,
            'win_rate_pct': win_rate,
            'all_improvements': davidian_improvements,
            't_statistic': t_stat,
            'p_value': p_value,
            'statistically_significant': significant,
            'trial_details': trial_results,
            'metadata': metadata
        }
        
        significance_marker = "📈" if significant and mean_improvement > 0 else "📊" if significant else ""
        status = "✅" if mean_improvement > 0 else "❌"
        
        print(f"    {status} {significance_marker} D={n_features}: {mean_improvement:+.2f}% ± {std_improvement:.2f}% "
              f"(p={p_value:.3f}, wins: {wins}/{len(davidian_improvements)})")
        
        return result
    
    return None

def run_comprehensive_dimensionality_analysis():
    """Run comprehensive dimensionality analysis."""
    print("\n1. Creating controlled dimensionality datasets...")
    
    # Test both classification and regression
    dimensionalities = [5, 10, 20, 50, 100, 200, 500]
    n_samples = 1000  # Fixed sample size
    
    all_results = []
    
    for task_type in ['classification', 'regression']:
        print(f"\n{'='*50}")
        print(f"TASK TYPE: {task_type.upper()}")
        print(f"{'='*50}")
        
        # Create datasets
        datasets = create_controlled_dimensionality_datasets(
            n_samples=n_samples, 
            dimensionalities=dimensionalities, 
            task_type=task_type
        )
        
        print(f"\n  Running experiments for {task_type}...")
        
        for n_features in dimensionalities:
            X, y, metadata = datasets[n_features]
            
            print(f"\n  Testing dimensionality {n_features}:")
            
            result = run_dimensionality_experiment(X, y, metadata, n_trials=15)
            
            if result:
                result['task_type'] = task_type
                all_results.append(result)
    
    return all_results

def analyze_dimensionality_results(all_results: List[Dict[str, Any]]):
    """Analyze dimensionality experiment results."""
    print(f"\n2. Dimensionality Analysis Results:")
    print("="*70)
    
    # Separate by task type
    classification_results = [r for r in all_results if r['task_type'] == 'classification']
    regression_results = [r for r in all_results if r['task_type'] == 'regression']
    
    for task_type, results in [('Classification', classification_results), ('Regression', regression_results)]:
        if not results:
            continue
            
        print(f"\n{task_type} Tasks:")
        print("-" * 40)
        
        # Sort by dimensionality
        results_sorted = sorted(results, key=lambda x: x['dimensionality'])
        
        print(f"{'Dimensions':<12} {'Ratio':<8} {'Improvement':<12} {'Win Rate':<10} {'Significance':<12}")
        print("-" * 60)
        
        for result in results_sorted:
            dim = result['dimensionality']
            ratio = result['dimensionality_ratio']
            improvement = result['mean_improvement_pct']
            win_rate = result['win_rate_pct']
            p_value = result['p_value']
            significant = result['statistically_significant']
            
            sig_marker = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
            status = "✅" if improvement > 0 else "❌"
            
            print(f"{dim:<12} {ratio:<8.3f} {improvement:+8.2f}% {win_rate:8.1f}% {sig_marker:<12}")
    
    # Correlation analysis
    print(f"\nCorrelation Analysis:")
    print("-" * 30)
    
    for task_type, results in [('Classification', classification_results), ('Regression', regression_results)]:
        if len(results) < 3:
            continue
            
        dimensionalities = [r['dimensionality'] for r in results]
        improvements = [r['mean_improvement_pct'] for r in results]
        ratios = [r['dimensionality_ratio'] for r in results]
        
        # Correlation between dimensionality and improvement
        corr_dim, p_dim = stats.pearsonr(dimensionalities, improvements)
        corr_ratio, p_ratio = stats.pearsonr(ratios, improvements)
        
        print(f"\n{task_type}:")
        print(f"  Dimensionality vs Improvement: r={corr_dim:.3f}, p={p_dim:.3f}")
        print(f"  Dim/Sample Ratio vs Improvement: r={corr_ratio:.3f}, p={p_ratio:.3f}")
        
        if p_dim < 0.05:
            direction = "positive" if corr_dim > 0 else "negative"
            print(f"  📈 Significant {direction} correlation with dimensionality!")
        
        if p_ratio < 0.05:
            direction = "positive" if corr_ratio > 0 else "negative"
            print(f"  📈 Significant {direction} correlation with dimensionality ratio!")
    
    # Find optimal dimensionality ranges
    print(f"\nOptimal Dimensionality Ranges:")
    print("-" * 35)
    
    for task_type, results in [('Classification', classification_results), ('Regression', regression_results)]:
        if not results:
            continue
            
        # Find best performing dimensionality ranges
        positive_results = [r for r in results if r['mean_improvement_pct'] > 0]
        significant_results = [r for r in results if r['statistically_significant'] and r['mean_improvement_pct'] > 0]
        
        print(f"\n{task_type}:")
        if positive_results:
            best_dims = [r['dimensionality'] for r in positive_results]
            print(f"  Positive improvement dimensions: {sorted(best_dims)}")
            
            if significant_results:
                sig_dims = [r['dimensionality'] for r in significant_results]
                print(f"  Statistically significant dimensions: {sorted(sig_dims)}")
            else:
                print(f"  No statistically significant improvements")
        else:
            print(f"  No positive improvements found")

def save_dimensionality_results(all_results: List[Dict[str, Any]]):
    """Save dimensionality analysis results."""
    os.makedirs('results', exist_ok=True)
    
    # Calculate summary statistics
    classification_results = [r for r in all_results if r['task_type'] == 'classification']
    regression_results = [r for r in all_results if r['task_type'] == 'regression']
    
    summary_stats = {}
    
    for task_type, results in [('classification', classification_results), ('regression', regression_results)]:
        if results:
            dimensionalities = [r['dimensionality'] for r in results]
            improvements = [r['mean_improvement_pct'] for r in results]
            ratios = [r['dimensionality_ratio'] for r in results]
            
            # Correlation analysis
            corr_dim, p_dim = stats.pearsonr(dimensionalities, improvements) if len(results) > 2 else (0, 1)
            corr_ratio, p_ratio = stats.pearsonr(ratios, improvements) if len(results) > 2 else (0, 1)
            
            summary_stats[task_type] = {
                'dimensionalities_tested': dimensionalities,
                'mean_improvements': improvements,
                'dimensionality_ratios': ratios,
                'correlation_dimensionality': {'r': corr_dim, 'p': p_dim},
                'correlation_ratio': {'r': corr_ratio, 'p': p_ratio},
                'best_dimensionality': dimensionalities[np.argmax(improvements)] if improvements else None,
                'best_improvement': max(improvements) if improvements else None,
                'significant_results': [r for r in results if r['statistically_significant'] and r['mean_improvement_pct'] > 0]
            }
    
    comprehensive_results = {
        'experiment_type': 'dimensionality_analysis',
        'description': 'Systematic analysis of how feature dimensionality affects Davidian Regularization',
        'methodology': {
            'sample_size': 1000,
            'dimensionalities_tested': [5, 10, 20, 50, 100, 200, 500],
            'k_folds': 5,
            'trials_per_dimensionality': 15,
            'models_per_trial': 25,
            'statistical_testing': 'One-sample t-test against null hypothesis of 0% improvement'
        },
        'summary_statistics': summary_stats,
        'detailed_results': all_results,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open('results/dimensionality_analysis_results.json', 'w') as f:
        json.dump(comprehensive_results, f, indent=2, default=str)
    
    # Create summary table
    summary_table = []
    for result in all_results:
        summary_table.append({
            'task_type': result['task_type'],
            'dimensionality': result['dimensionality'],
            'dimensionality_ratio': result['dimensionality_ratio'],
            'mean_improvement_pct': result['mean_improvement_pct'],
            'std_improvement_pct': result['std_improvement_pct'],
            'win_rate_pct': result['win_rate_pct'],
            'p_value': result['p_value'],
            'statistically_significant': result['statistically_significant'],
            'n_trials': result['n_trials']
        })
    
    with open('results/dimensionality_summary_table.json', 'w') as f:
        json.dump(summary_table, f, indent=2, default=str)
    
    print(f"\n✅ Dimensionality analysis results saved to:")
    print(f"   - results/dimensionality_analysis_results.json")
    print(f"   - results/dimensionality_summary_table.json")

def main():
    """Run the dimensionality analysis experiment."""
    try:
        start_time = time.time()
        
        print("🔬 DIMENSIONALITY HYPOTHESIS:")
        print("   Higher feature dimensionality may increase overfitting variance,")
        print("   making Davidian Regularization more effective at identifying")
        print("   models with better generalization properties.")
        
        # Run experiments
        all_results = run_comprehensive_dimensionality_analysis()
        
        if all_results:
            # Analyze results
            analyze_dimensionality_results(all_results)
            
            # Save results
            save_dimensionality_results(all_results)
        else:
            print("❌ No experiments completed successfully")
        
        elapsed = time.time() - start_time
        
        print(f"\n{'='*70}")
        print("🎉 DIMENSIONALITY ANALYSIS COMPLETED!")
        print(f"{'='*70}")
        print(f"Total time: {elapsed:.2f} seconds")
        print("✅ Systematic dimensionality testing completed")
        print("✅ Statistical significance testing included")
        print("✅ Correlation analysis performed")
        
        return True
        
    except Exception as e:
        print(f"\n{'='*70}")
        print("❌ DIMENSIONALITY ANALYSIS FAILED")
        print(f"{'='*70}")
        print(f"Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
