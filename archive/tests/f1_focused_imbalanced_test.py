#!/usr/bin/env python3
"""
F1-score focused imbalanced data test for Davidian Regularization.

Focus specifically on F1-score as the primary metric for imbalanced datasets,
with expected value analysis and sample size/trial effects.
"""

import sys
import os
import time
import json
import numpy as np
from collections import Counter
from typing import Dict, List, Tuple, Any

# Force single-threaded execution
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

print("F1-FOCUSED IMBALANCED DATA TEST")
print("="*50)
print("Expected Value Analysis using F1-score for imbalanced datasets")
print("="*50)

def create_imbalanced_dataset(n_samples: int, imbalance_ratio: float, random_seed: int = 42):
    """Create imbalanced dataset with specified parameters."""
    from sklearn.datasets import make_classification
    
    minority_weight = 1.0 / (imbalance_ratio + 1.0)
    majority_weight = 1.0 - minority_weight
    
    X, y = make_classification(
        n_samples=n_samples,
        n_features=20,
        n_informative=15,
        n_redundant=3,
        n_classes=2,
        weights=[majority_weight, minority_weight],
        class_sep=0.7,
        flip_y=0.01,
        random_state=random_seed
    )
    
    return X, y, {
        'n_samples': n_samples,
        'imbalance_ratio': imbalance_ratio,
        'majority_pct': majority_weight * 100,
        'minority_pct': minority_weight * 100
    }

def create_models_for_imbalanced():
    """Create models specifically good for imbalanced data."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.naive_bayes import GaussianNB
    
    return [
        # Balanced models (should handle imbalance better)
        {'name': 'LogReg_Balanced', 'class': LogisticRegression, 
         'params': {'class_weight': 'balanced', 'random_state': 42, 'max_iter': 1000, 'n_jobs': 1}},
        {'name': 'RF_Balanced', 'class': RandomForestClassifier,
         'params': {'class_weight': 'balanced', 'n_estimators': 100, 'random_state': 42, 'n_jobs': 1}},
        {'name': 'GB_Default', 'class': GradientBoostingClassifier,
         'params': {'n_estimators': 100, 'learning_rate': 0.1, 'random_state': 42}},
        
        # Unbalanced models (for comparison - should show larger train-val differences)
        {'name': 'LogReg_Unbalanced', 'class': LogisticRegression,
         'params': {'class_weight': None, 'random_state': 43, 'max_iter': 1000, 'n_jobs': 1}},
        {'name': 'RF_Unbalanced', 'class': RandomForestClassifier,
         'params': {'class_weight': None, 'n_estimators': 100, 'random_state': 43, 'n_jobs': 1}},
        
        # Different regularization strengths
        {'name': 'LogReg_Strong_Reg', 'class': LogisticRegression,
         'params': {'C': 0.1, 'class_weight': 'balanced', 'random_state': 44, 'max_iter': 1000, 'n_jobs': 1}},
        {'name': 'LogReg_Weak_Reg', 'class': LogisticRegression,
         'params': {'C': 10.0, 'class_weight': 'balanced', 'random_state': 45, 'max_iter': 1000, 'n_jobs': 1}},
        
        # Naive Bayes (good baseline)
        {'name': 'NaiveBayes', 'class': GaussianNB, 'params': {}}
    ]

def run_f1_trial(X, y, trial_seed: int):
    """Run a single trial focused on F1-score."""
    from sklearn.model_selection import train_test_split, StratifiedKFold
    from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
    from sklearn.preprocessing import StandardScaler
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=trial_seed, stratify=y
    )
    
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = create_models_for_imbalanced()
    
    # K-fold CV (K=5) focused on F1-score
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=trial_seed)
    
    model_results = []
    
    for model_config in models:
        try:
            # K-fold CV for Davidian score using F1
            fold_train_f1s = []
            fold_val_f1s = []
            
            for train_idx, val_idx in cv.split(X_train_scaled, y_train):
                X_fold_train, X_fold_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
                y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
                
                model = model_config['class'](**model_config['params'])
                model.fit(X_fold_train, y_fold_train)
                
                train_pred = model.predict(X_fold_train)
                val_pred = model.predict(X_fold_val)
                
                # Use F1-score as primary metric
                train_f1 = f1_score(y_fold_train, train_pred, average='weighted')
                val_f1 = f1_score(y_fold_val, val_pred, average='weighted')
                
                fold_train_f1s.append(train_f1)
                fold_val_f1s.append(val_f1)
            
            mean_train_f1 = np.mean(fold_train_f1s)
            mean_val_f1 = np.mean(fold_val_f1s)
            
            # Davidian score based on F1
            dr_score = mean_val_f1 - abs(mean_train_f1 - mean_val_f1)
            
            # Final model evaluation on test (comprehensive metrics)
            final_model = model_config['class'](**model_config['params'])
            final_model.fit(X_train_scaled, y_train)
            test_pred = final_model.predict(X_test_scaled)
            
            test_f1 = f1_score(y_test, test_pred, average='weighted')
            test_f1_macro = f1_score(y_test, test_pred, average='macro')  # Also track macro F1
            test_precision = precision_score(y_test, test_pred, average='weighted', zero_division=0)
            test_recall = recall_score(y_test, test_pred, average='weighted')
            test_accuracy = accuracy_score(y_test, test_pred)
            
            model_results.append({
                'model_name': model_config['name'],
                'cv_train_f1': mean_train_f1,
                'cv_val_f1': mean_val_f1,
                'train_val_diff': abs(mean_train_f1 - mean_val_f1),
                'davidian_score': dr_score,
                'test_f1_weighted': test_f1,
                'test_f1_macro': test_f1_macro,
                'test_precision': test_precision,
                'test_recall': test_recall,
                'test_accuracy': test_accuracy
            })
            
        except Exception as e:
            continue
    
    if len(model_results) < 2:
        return None
    
    # Select best models based on F1-focused criteria
    best_davidian = max(model_results, key=lambda x: x['davidian_score'])
    best_f1_validation = max(model_results, key=lambda x: x['cv_val_f1'])
    
    return {
        'trial_seed': trial_seed,
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'train_class_counts': {int(k): int(v) for k, v in Counter(y_train).items()},
        'test_class_counts': {int(k): int(v) for k, v in Counter(y_test).items()},
        'davidian_selected': {
            'model': best_davidian['model_name'],
            'davidian_score': best_davidian['davidian_score'],
            'train_val_diff': best_davidian['train_val_diff'],
            'test_f1_weighted': best_davidian['test_f1_weighted'],
            'test_f1_macro': best_davidian['test_f1_macro'],
            'test_precision': best_davidian['test_precision'],
            'test_recall': best_davidian['test_recall'],
            'test_accuracy': best_davidian['test_accuracy']
        },
        'f1_validation_selected': {
            'model': best_f1_validation['model_name'],
            'cv_val_f1': best_f1_validation['cv_val_f1'],
            'train_val_diff': best_f1_validation['train_val_diff'],
            'test_f1_weighted': best_f1_validation['test_f1_weighted'],
            'test_f1_macro': best_f1_validation['test_f1_macro'],
            'test_precision': best_f1_validation['test_precision'],
            'test_recall': best_f1_validation['test_recall'],
            'test_accuracy': best_f1_validation['test_accuracy']
        },
        'f1_improvements': {
            'weighted_f1_diff': best_davidian['test_f1_weighted'] - best_f1_validation['test_f1_weighted'],
            'macro_f1_diff': best_davidian['test_f1_macro'] - best_f1_validation['test_f1_macro'],
            'precision_diff': best_davidian['test_precision'] - best_f1_validation['test_precision'],
            'recall_diff': best_davidian['test_recall'] - best_f1_validation['test_recall'],
            'accuracy_diff': best_davidian['test_accuracy'] - best_f1_validation['test_accuracy']
        }
    }

def run_focused_f1_experiment():
    """Run focused F1-score experiment with systematic parameter variation."""
    
    # Focused experimental design based on partial results
    configurations = [
        # Test the promising 9:1 and 19:1 ratios with varying sample sizes and trials
        {'ratio': 9.0, 'samples': 1000, 'trials': 10},   # Showed +0.57% F1 EV
        {'ratio': 9.0, 'samples': 2000, 'trials': 15},   # Test larger sample
        {'ratio': 9.0, 'samples': 2000, 'trials': 25},   # Test more trials
        
        {'ratio': 19.0, 'samples': 1000, 'trials': 10},  # Showed +0.57% F1 EV  
        {'ratio': 19.0, 'samples': 1000, 'trials': 20},  # Test more trials
        {'ratio': 19.0, 'samples': 2000, 'trials': 15},  # Test larger sample
        {'ratio': 19.0, 'samples': 3000, 'trials': 15},  # Test even larger sample
        
        # Test extreme imbalance
        {'ratio': 49.0, 'samples': 2000, 'trials': 15},  # 98/2 split
        {'ratio': 49.0, 'samples': 5000, 'trials': 20},  # Larger sample
    ]
    
    print(f"\n1. Running focused F1-score experiments:")
    print(f"   Total configurations: {len(configurations)}")
    
    all_results = []
    
    for i, config in enumerate(configurations):
        ratio = config['ratio']
        samples = config['samples']
        trials = config['trials']
        
        print(f"\n  [{i+1}/{len(configurations)}] Ratio {ratio:.0f}:1, N={samples}, {trials} trials")
        
        # Create dataset
        X, y, dataset_info = create_imbalanced_dataset(samples, ratio, random_seed=42 + i)
        
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        print(f"    Actual distribution: {dict(Counter(y))}")
        
        # Run trials
        trial_results = []
        for trial in range(trials):
            trial_seed = np.random.randint(0, 2**31 - 1)
            result = run_f1_trial(X_scaled, y, trial_seed)
            if result:
                trial_results.append(result)
        
        if trial_results:
            # F1-score analysis (weighted F1 as primary)
            f1_improvements = [t['f1_improvements']['weighted_f1_diff'] for t in trial_results]
            f1_macro_improvements = [t['f1_improvements']['macro_f1_diff'] for t in trial_results]
            
            f1_wins = [imp for imp in f1_improvements if imp > 0]
            f1_losses = [imp for imp in f1_improvements if imp < 0]
            
            # Expected value
            f1_ev = np.mean(f1_improvements) * 100
            f1_std = np.std(f1_improvements) * 100
            
            # Statistical significance
            from scipy import stats
            f1_t_stat, f1_p_value = stats.ttest_1samp(f1_improvements, 0)
            
            result_summary = {
                'configuration': config,
                'dataset_info': dataset_info,
                'n_successful_trials': len(trial_results),
                'f1_analysis': {
                    'expected_value_pct': f1_ev,
                    'std_pct': f1_std,
                    'win_rate_pct': len(f1_wins) / len(trial_results) * 100,
                    'avg_win_size_pct': np.mean(f1_wins) * 100 if f1_wins else 0,
                    'avg_loss_size_pct': np.mean(f1_losses) * 100 if f1_losses else 0,
                    'wins': len(f1_wins),
                    'losses': len(f1_losses),
                    't_statistic': f1_t_stat,
                    'p_value': f1_p_value,
                    'significant': f1_p_value < 0.05
                },
                'trial_details': trial_results
            }
            
            all_results.append(result_summary)
            
            # Print results
            f1_status = "✅" if f1_ev > 0 else "❌"
            f1_sig = "***" if f1_p_value < 0.001 else "**" if f1_p_value < 0.01 else "*" if f1_p_value < 0.05 else ""
            
            print(f"    {f1_status} F1 Expected Value: {f1_ev:+.3f}% ± {f1_std:.3f}% {f1_sig}")
            print(f"       Win rate: {len(f1_wins)}/{len(trial_results)} ({len(f1_wins)/len(trial_results)*100:.1f}%)")
            if f1_wins:
                print(f"       Avg win size: +{np.mean(f1_wins)*100:.3f}%")
            if f1_losses:
                print(f"       Avg loss size: {np.mean(f1_losses)*100:.3f}%")
    
    return all_results

def analyze_f1_results(all_results: List[Dict[str, Any]]):
    """Analyze F1-focused results."""
    print(f"\n2. F1-Score Expected Value Analysis:")
    print("="*60)
    
    # Create detailed table
    print(f"{'Ratio':<8} {'Samples':<8} {'Trials':<8} {'F1 EV':<10} {'±Std':<8} {'Win%':<8} {'Avg Win':<10} {'p-val':<8} {'Sig'}")
    print("-" * 85)
    
    for result in all_results:
        config = result['configuration']
        f1_analysis = result['f1_analysis']
        
        ratio = config['ratio']
        samples = config['samples']
        trials = result['n_successful_trials']
        ev = f1_analysis['expected_value_pct']
        std = f1_analysis['std_pct']
        win_rate = f1_analysis['win_rate_pct']
        avg_win = f1_analysis['avg_win_size_pct']
        p_val = f1_analysis['p_value']
        
        sig_marker = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        
        print(f"{ratio:<8.0f} {samples:<8} {trials:<8} {ev:+8.3f}% {std:<8.3f} {win_rate:<8.1f} {avg_win:<10.3f} {p_val:<8.3f} {sig_marker}")
    
    # Find patterns
    print(f"\nPattern Analysis:")
    print("-" * 25)
    
    # Group by imbalance ratio
    by_ratio = {}
    for result in all_results:
        ratio = result['configuration']['ratio']
        if ratio not in by_ratio:
            by_ratio[ratio] = []
        by_ratio[ratio].append(result)
    
    for ratio in sorted(by_ratio.keys()):
        ratio_results = by_ratio[ratio]
        ratio_evs = [r['f1_analysis']['expected_value_pct'] for r in ratio_results]
        
        print(f"\nRatio {ratio:.0f}:1 ({len(ratio_results)} configurations):")
        print(f"  Expected value range: {min(ratio_evs):+.3f}% to {max(ratio_evs):+.3f}%")
        print(f"  Average expected value: {np.mean(ratio_evs):+.3f}%")
        
        positive_configs = [r for r in ratio_results if r['f1_analysis']['expected_value_pct'] > 0]
        if positive_configs:
            print(f"  Positive EV configurations: {len(positive_configs)}/{len(ratio_results)}")
            
            # Find best configuration for this ratio
            best_config = max(positive_configs, key=lambda x: x['f1_analysis']['expected_value_pct'])
            best_samples = best_config['configuration']['samples']
            best_trials = best_config['configuration']['trials']
            best_ev = best_config['f1_analysis']['expected_value_pct']
            
            print(f"  Best configuration: N={best_samples}, trials={best_trials} → {best_ev:+.3f}% F1 EV")
    
    # Overall recommendations
    print(f"\nRecommendations:")
    print("-" * 20)
    
    best_overall = max(all_results, key=lambda x: x['f1_analysis']['expected_value_pct'])
    best_config = best_overall['configuration']
    best_f1_analysis = best_overall['f1_analysis']
    
    if best_f1_analysis['expected_value_pct'] > 0.1:  # Meaningful threshold
        print("✅ STRONG RECOMMENDATION: Use Davidian Regularization")
        print(f"   Optimal: {best_config['ratio']:.0f}:1 ratio, N={best_config['samples']}, {best_config['trials']} trials")
        print(f"   Expected F1 improvement: +{best_f1_analysis['expected_value_pct']:.3f}%")
        print(f"   Win rate: {best_f1_analysis['win_rate_pct']:.1f}%")
        print(f"   Statistical significance: p={best_f1_analysis['p_value']:.3f}")
    elif best_f1_analysis['expected_value_pct'] > 0:
        print("⚖️  MARGINAL RECOMMENDATION: Small positive expected value")
        print(f"   Best case: +{best_f1_analysis['expected_value_pct']:.3f}% F1 improvement")
    else:
        print("❌ NOT RECOMMENDED: Negative expected value")

def main():
    """Run F1-focused imbalanced data experiment."""
    try:
        start_time = time.time()
        
        print("🔬 F1-FOCUSED HYPOTHESIS:")
        print("   F1-score is more appropriate for imbalanced data evaluation")
        print("   Larger samples and more trials should reveal stronger benefits")
        
        all_results = run_focused_f1_experiment()
        
        if all_results:
            analyze_f1_results(all_results)
            
            # Save results (simplified to avoid JSON issues)
            os.makedirs('results', exist_ok=True)
            
            # Create summary table
            summary_table = []
            for result in all_results:
                config = result['configuration']
                f1_analysis = result['f1_analysis']
                
                summary_table.append({
                    'imbalance_ratio': float(config['ratio']),
                    'sample_size': int(config['samples']),
                    'n_trials': int(result['n_successful_trials']),
                    'f1_expected_value_pct': float(f1_analysis['expected_value_pct']),
                    'f1_std_pct': float(f1_analysis['std_pct']),
                    'f1_win_rate_pct': float(f1_analysis['win_rate_pct']),
                    'f1_avg_win_size_pct': float(f1_analysis['avg_win_size_pct']),
                    'f1_p_value': float(f1_analysis['p_value']) if not np.isnan(f1_analysis['p_value']) else None,
                    'f1_significant': bool(f1_analysis['significant'])
                })
            
            with open('results/f1_focused_imbalanced_summary.json', 'w') as f:
                json.dump(summary_table, f, indent=2)
            
            print(f"\n✅ F1-focused results saved to results/f1_focused_imbalanced_summary.json")
        
        elapsed = time.time() - start_time
        
        print(f"\n{'='*50}")
        print("🎉 F1-FOCUSED EXPERIMENT COMPLETED!")
        print(f"{'='*50}")
        print(f"Time: {elapsed:.1f}s")
        print("✅ F1-score expected value analysis")
        print("✅ Sample size and trial effects measured")
        
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
