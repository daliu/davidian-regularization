#!/usr/bin/env python3
"""
Direct comparison: Davidian Regularization on Imbalanced vs Balanced Data.

This experiment directly tests whether Davidian Regularization provides
additional value beyond simply balancing the dataset through subsampling.
"""

import sys
import os
import time
import json
import numpy as np
from collections import Counter

# Force single-threaded execution
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

print("BALANCED vs IMBALANCED COMPARISON")
print("="*50)
print("Direct comparison of Davidian Regularization effectiveness")
print("="*50)

def create_matched_datasets(n_total: int = 2000, imbalance_ratio: float = 19.0, 
                           n_features: int = 10, random_seed: int = 42):
    """
    Create perfectly matched imbalanced and balanced datasets.
    """
    from sklearn.datasets import make_classification
    
    minority_weight = 1.0 / (imbalance_ratio + 1.0)
    majority_weight = 1.0 - minority_weight
    
    # Create imbalanced dataset
    X_imbalanced, y_imbalanced = make_classification(
        n_samples=n_total,
        n_features=n_features,
        n_informative=max(2, n_features - 2),
        n_redundant=min(2, n_features // 5),
        n_classes=2,
        weights=[majority_weight, minority_weight],
        class_sep=0.7,
        flip_y=0.01,
        random_state=random_seed
    )
    
    # Create balanced version by subsampling majority class
    minority_indices = np.where(y_imbalanced == 1)[0]
    majority_indices = np.where(y_imbalanced == 0)[0]
    
    n_minority = len(minority_indices)
    
    # Randomly select majority samples to match minority count
    np.random.seed(random_seed + 1)
    selected_majority_indices = np.random.choice(majority_indices, n_minority, replace=False)
    
    # Create balanced dataset
    balanced_indices = np.concatenate([minority_indices, selected_majority_indices])
    np.random.shuffle(balanced_indices)
    
    X_balanced = X_imbalanced[balanced_indices]
    y_balanced = y_imbalanced[balanced_indices]
    
    return X_imbalanced, y_imbalanced, X_balanced, y_balanced, {
        'n_features': n_features,
        'imbalance_ratio': imbalance_ratio,
        'original_total': n_total,
        'balanced_total': len(X_balanced),
        'minority_count': n_minority,
        'majority_count_original': len(majority_indices),
        'majority_count_balanced': n_minority
    }

def run_comparison_trial(X_imbalanced, y_imbalanced, X_balanced, y_balanced, trial_seed: int):
    """
    Run a single trial comparing both datasets.
    """
    from sklearn.model_selection import train_test_split, StratifiedKFold
    from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.naive_bayes import GaussianNB
    
    models = [
        {'name': 'LogReg_Balanced', 'class': LogisticRegression, 
         'params': {'class_weight': 'balanced', 'random_state': 42, 'max_iter': 1000, 'n_jobs': 1}},
        {'name': 'LogReg_None', 'class': LogisticRegression,
         'params': {'class_weight': None, 'random_state': 43, 'max_iter': 1000, 'n_jobs': 1}},
        {'name': 'RF_Balanced', 'class': RandomForestClassifier,
         'params': {'class_weight': 'balanced', 'n_estimators': 50, 'random_state': 42, 'n_jobs': 1}},
        {'name': 'RF_None', 'class': RandomForestClassifier,
         'params': {'class_weight': None, 'n_estimators': 50, 'random_state': 43, 'n_jobs': 1}},
        {'name': 'GradBoost', 'class': GradientBoostingClassifier,
         'params': {'n_estimators': 50, 'random_state': 42}},
        {'name': 'NaiveBayes', 'class': GaussianNB, 'params': {}}
    ]
    
    results = {}
    
    for dataset_name, X, y in [('imbalanced', X_imbalanced, y_imbalanced), 
                              ('balanced', X_balanced, y_balanced)]:
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=trial_seed, stratify=y
        )
        
        # Standardize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # K-fold CV (K=5)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=trial_seed)
        
        model_performances = []
        
        for model_config in models:
            try:
                # K-fold CV for Davidian score
                fold_train_f1s = []
                fold_val_f1s = []
                
                for train_idx, val_idx in cv.split(X_train_scaled, y_train):
                    X_fold_train, X_fold_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
                    y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
                    
                    model = model_config['class'](**model_config['params'])
                    model.fit(X_fold_train, y_fold_train)
                    
                    train_pred = model.predict(X_fold_train)
                    val_pred = model.predict(X_fold_val)
                    
                    train_f1 = f1_score(y_fold_train, train_pred, average='binary')
                    val_f1 = f1_score(y_fold_val, val_pred, average='binary')
                    
                    fold_train_f1s.append(train_f1)
                    fold_val_f1s.append(val_f1)
                
                mean_train_f1 = np.mean(fold_train_f1s)
                mean_val_f1 = np.mean(fold_val_f1s)
                
                # Davidian score
                dr_score = mean_val_f1 - abs(mean_train_f1 - mean_val_f1)
                
                # Test performance
                final_model = model_config['class'](**model_config['params'])
                final_model.fit(X_train_scaled, y_train)
                test_pred = final_model.predict(X_test_scaled)
                
                test_f1 = f1_score(y_test, test_pred, average='binary')
                test_accuracy = accuracy_score(y_test, test_pred)
                test_precision = precision_score(y_test, test_pred, average='binary', zero_division=0)
                test_recall = recall_score(y_test, test_pred, average='binary')
                
                model_performances.append({
                    'model_name': model_config['name'],
                    'davidian_score': dr_score,
                    'cv_val_f1': mean_val_f1,
                    'train_val_diff': abs(mean_train_f1 - mean_val_f1),
                    'test_f1': test_f1,
                    'test_accuracy': test_accuracy,
                    'test_precision': test_precision,
                    'test_recall': test_recall
                })
                
            except Exception as e:
                continue
        
        if model_performances:
            # Select best models
            best_davidian = max(model_performances, key=lambda x: x['davidian_score'])
            best_f1_val = max(model_performances, key=lambda x: x['cv_val_f1'])
            
            results[dataset_name] = {
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'train_class_dist': dict(Counter(y_train)),
                'test_class_dist': dict(Counter(y_test)),
                'davidian_selected': best_davidian,
                'f1_validation_selected': best_f1_val,
                'f1_improvement': best_davidian['test_f1'] - best_f1_val['test_f1'],
                'accuracy_improvement': best_davidian['test_accuracy'] - best_f1_val['test_accuracy']
            }
    
    return {
        'trial_seed': trial_seed,
        'results': results
    }

def run_focused_comparison():
    """Run focused comparison experiment."""
    
    print(f"\n1. Running focused comparison (Imbalanced vs Balanced):")
    
    # Focus on the best-performing configuration from previous results
    optimal_config = {
        'n_total': 2000,
        'imbalance_ratio': 19.0,
        'n_features': 10,
        'n_trials': 20  # More trials for better statistics
    }
    
    print(f"   Configuration: {optimal_config}")
    
    # Create datasets
    X_imbalanced, y_imbalanced, X_balanced, y_balanced, metadata = create_matched_datasets(
        optimal_config['n_total'], optimal_config['imbalance_ratio'], 
        optimal_config['n_features'], random_seed=42
    )
    
    print(f"   Imbalanced distribution: {dict(Counter(y_imbalanced))}")
    print(f"   Balanced distribution: {dict(Counter(y_balanced))}")
    
    # Run trials
    all_trials = []
    for trial in range(optimal_config['n_trials']):
        trial_seed = np.random.randint(0, 2**31 - 1)
        trial_result = run_comparison_trial(X_imbalanced, y_imbalanced, X_balanced, y_balanced, trial_seed)
        if trial_result and 'imbalanced' in trial_result['results'] and 'balanced' in trial_result['results']:
            all_trials.append(trial_result)
        
        if (trial + 1) % 5 == 0:
            print(f"     Completed {trial + 1}/{optimal_config['n_trials']} trials")
    
    if not all_trials:
        return None
    
    # Calculate comparative statistics
    imbalanced_f1_improvements = [t['results']['imbalanced']['f1_improvement'] for t in all_trials]
    balanced_f1_improvements = [t['results']['balanced']['f1_improvement'] for t in all_trials]
    
    # Direct comparison: imbalanced vs balanced performance
    imbalanced_vs_balanced = []
    for trial in all_trials:
        imb_best_f1 = trial['results']['imbalanced']['davidian_selected']['test_f1']
        bal_best_f1 = trial['results']['balanced']['davidian_selected']['test_f1']
        imbalanced_vs_balanced.append(imb_best_f1 - bal_best_f1)
    
    return {
        'configuration': optimal_config,
        'metadata': metadata,
        'n_trials': len(all_trials),
        'imbalanced_analysis': {
            'f1_expected_value_pct': float(np.mean(imbalanced_f1_improvements) * 100),
            'f1_std_pct': float(np.std(imbalanced_f1_improvements) * 100),
            'f1_wins': int(sum(1 for imp in imbalanced_f1_improvements if imp > 0)),
            'f1_win_rate_pct': float(sum(1 for imp in imbalanced_f1_improvements if imp > 0) / len(all_trials) * 100)
        },
        'balanced_analysis': {
            'f1_expected_value_pct': float(np.mean(balanced_f1_improvements) * 100),
            'f1_std_pct': float(np.std(balanced_f1_improvements) * 100),
            'f1_wins': int(sum(1 for imp in balanced_f1_improvements if imp > 0)),
            'f1_win_rate_pct': float(sum(1 for imp in balanced_f1_improvements if imp > 0) / len(all_trials) * 100)
        },
        'direct_comparison': {
            'imbalanced_vs_balanced_f1_diff_pct': float(np.mean(imbalanced_vs_balanced) * 100),
            'imbalanced_vs_balanced_std_pct': float(np.std(imbalanced_vs_balanced) * 100),
            'imbalanced_better_count': int(sum(1 for diff in imbalanced_vs_balanced if diff > 0)),
            'imbalanced_better_rate_pct': float(sum(1 for diff in imbalanced_vs_balanced if diff > 0) / len(all_trials) * 100)
        },
        'all_trials': all_trials
    }

def analyze_comparison_results(results):
    """Analyze the comparison results."""
    print(f"\n2. Comparison Analysis:")
    print("="*50)
    
    config = results['configuration']
    imb_analysis = results['imbalanced_analysis']
    bal_analysis = results['balanced_analysis']
    direct_comp = results['direct_comparison']
    
    print(f"Configuration: {config['imbalance_ratio']:.0f}:1 ratio, N={config['n_total']}, {config['n_features']} features")
    print(f"Trials: {results['n_trials']}")
    
    print(f"\nDavidian Regularization Performance:")
    print("-" * 40)
    print(f"{'Dataset Type':<15} {'F1 EV':<10} {'Win Rate':<10} {'Wins'}")
    print("-" * 45)
    print(f"{'Imbalanced':<15} {imb_analysis['f1_expected_value_pct']:+8.3f}% {imb_analysis['f1_win_rate_pct']:8.1f}% {imb_analysis['f1_wins']}/{results['n_trials']}")
    print(f"{'Balanced':<15} {bal_analysis['f1_expected_value_pct']:+8.3f}% {bal_analysis['f1_win_rate_pct']:8.1f}% {bal_analysis['f1_wins']}/{results['n_trials']}")
    
    # Direct comparison
    print(f"\nDirect Comparison (Imbalanced vs Balanced):")
    print("-" * 45)
    
    advantage = direct_comp['imbalanced_vs_balanced_f1_diff_pct']
    advantage_rate = direct_comp['imbalanced_better_rate_pct']
    
    print(f"Imbalanced advantage: {advantage:+.3f}% ± {direct_comp['imbalanced_vs_balanced_std_pct']:.3f}%")
    print(f"Imbalanced better: {direct_comp['imbalanced_better_count']}/{results['n_trials']} trials ({advantage_rate:.1f}%)")
    
    # Statistical significance
    from scipy import stats
    imbalanced_improvements = [t['results']['imbalanced']['f1_improvement'] for t in results['all_trials']]
    balanced_improvements = [t['results']['balanced']['f1_improvement'] for t in results['all_trials']]
    
    # Test if imbalanced is significantly better than balanced
    t_stat, p_value = stats.ttest_rel(imbalanced_improvements, balanced_improvements)
    
    print(f"\nStatistical Test (Paired t-test):")
    print(f"  H₀: No difference between imbalanced and balanced")
    print(f"  H₁: Imbalanced performs better than balanced")
    print(f"  t-statistic: {t_stat:.3f}")
    print(f"  p-value: {p_value:.3f}")
    
    if p_value < 0.05:
        if t_stat > 0:
            print("  📈 SIGNIFICANT: Imbalanced data significantly better!")
        else:
            print("  📉 SIGNIFICANT: Balanced data significantly better!")
    else:
        print("  📊 No significant difference between approaches")
    
    # Practical recommendation
    print(f"\nPractical Recommendation:")
    print("-" * 30)
    
    if imb_analysis['f1_expected_value_pct'] > 0.5 and imb_analysis['f1_expected_value_pct'] > bal_analysis['f1_expected_value_pct']:
        print("✅ RECOMMENDATION: Use Davidian Regularization on IMBALANCED data")
        print(f"   Expected F1 benefit: {imb_analysis['f1_expected_value_pct']:+.3f}%")
        print(f"   Advantage over balanced approach: {advantage:+.3f}%")
    elif bal_analysis['f1_expected_value_pct'] > 0.5:
        print("⚖️  RECOMMENDATION: Use Davidian Regularization on BALANCED data")
        print(f"   Expected F1 benefit: {bal_analysis['f1_expected_value_pct']:+.3f}%")
    else:
        print("❌ NOT RECOMMENDED: Use traditional methods instead")
        print("   Both approaches show negative or minimal expected value")

def main():
    """Run the comparison experiment."""
    try:
        start_time = time.time()
        
        print("🔬 COMPARISON HYPOTHESIS:")
        print("   Davidian Regularization may work better on imbalanced data")
        print("   than on artificially balanced data (subsampled majority class)")
        
        results = run_focused_comparison()
        
        if results:
            analyze_comparison_results(results)
            
            # Save results
            os.makedirs('results/comparison', exist_ok=True)
            with open('results/comparison/imbalanced_vs_balanced.json', 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"\n✅ Comparison results saved to results/comparison/imbalanced_vs_balanced.json")
        
        elapsed = time.time() - start_time
        
        print(f"\n{'='*50}")
        print("🎉 COMPARISON COMPLETED!")
        print(f"{'='*50}")
        print(f"Time: {elapsed:.1f}s")
        print("✅ Direct comparison performed")
        print("✅ Statistical significance tested")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
