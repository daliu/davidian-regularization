#!/usr/bin/env python3
"""
Sample size and trials analysis for Davidian Regularization on imbalanced data.

This experiment systematically varies:
1. Sample sizes: [500, 1000, 2000, 5000, 10000]
2. Number of trials: [5, 10, 20, 50]
3. Imbalance ratios: [4:1, 9:1, 19:1, 49:1]

Testing the hypothesis that larger sample sizes and more trials reveal
stronger benefits of Davidian Regularization on imbalanced datasets.
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

print("SAMPLE SIZE & TRIALS ANALYSIS")
print("="*60)
print("Testing how sample size and trial count affect Davidian Regularization")
print("="*60)

def create_imbalanced_dataset(n_samples: int, imbalance_ratio: float, random_seed: int = 42):
    """
    Create an imbalanced dataset with specified sample size and ratio.
    
    Args:
        n_samples: Total number of samples
        imbalance_ratio: Ratio of majority to minority class (e.g., 19.0 for 95/5 split)
        random_seed: Random seed for reproducibility
    """
    from sklearn.datasets import make_classification
    
    # Calculate class weights from ratio
    # If ratio is 19:1, then majority = 19/(19+1) = 0.95, minority = 1/(19+1) = 0.05
    minority_weight = 1.0 / (imbalance_ratio + 1.0)
    majority_weight = 1.0 - minority_weight
    
    X, y = make_classification(
        n_samples=n_samples,
        n_features=20,  # Fixed feature count for fair comparison
        n_informative=15,
        n_redundant=3,
        n_classes=2,
        weights=[majority_weight, minority_weight],
        class_sep=0.7,  # Moderate difficulty
        flip_y=0.01,    # 1% label noise
        random_state=random_seed
    )
    
    actual_ratio = np.sum(y == 0) / np.sum(y == 1)
    
    return X, y, {
        'n_samples': n_samples,
        'n_features': X.shape[1],
        'target_ratio': imbalance_ratio,
        'actual_ratio': actual_ratio,
        'majority_class_pct': majority_weight * 100,
        'minority_class_pct': minority_weight * 100
    }

def create_efficient_models():
    """Create a small set of efficient models for quick testing."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.naive_bayes import GaussianNB
    
    return [
        {'name': 'LogReg_Balanced', 'class': LogisticRegression, 
         'params': {'class_weight': 'balanced', 'random_state': 42, 'max_iter': 1000, 'n_jobs': 1}},
        {'name': 'LogReg_Unbalanced', 'class': LogisticRegression,
         'params': {'class_weight': None, 'random_state': 43, 'max_iter': 1000, 'n_jobs': 1}},
        {'name': 'RF_Balanced', 'class': RandomForestClassifier,
         'params': {'class_weight': 'balanced', 'n_estimators': 50, 'random_state': 42, 'n_jobs': 1}},
        {'name': 'RF_Unbalanced', 'class': RandomForestClassifier,
         'params': {'class_weight': None, 'n_estimators': 50, 'random_state': 43, 'n_jobs': 1}},
        {'name': 'GradBoost', 'class': GradientBoostingClassifier,
         'params': {'n_estimators': 50, 'learning_rate': 0.1, 'random_state': 42}},
        {'name': 'NaiveBayes', 'class': GaussianNB, 'params': {}}
    ]

def run_single_trial(X, y, trial_seed: int):
    """Run a single trial and return detailed metrics."""
    from sklearn.model_selection import train_test_split, StratifiedKFold
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    from sklearn.preprocessing import StandardScaler
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=trial_seed, stratify=y
    )
    
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = create_efficient_models()
    
    # K-fold CV (K=5)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=trial_seed)
    
    model_results = []
    
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
                
                train_f1 = f1_score(y_fold_train, train_pred, average='weighted')
                val_f1 = f1_score(y_fold_val, val_pred, average='weighted')
                
                fold_train_f1s.append(train_f1)
                fold_val_f1s.append(val_f1)
            
            mean_train_f1 = np.mean(fold_train_f1s)
            mean_val_f1 = np.mean(fold_val_f1s)
            
            # Davidian score
            dr_score = mean_val_f1 - abs(mean_train_f1 - mean_val_f1)
            
            # Final model evaluation on test
            final_model = model_config['class'](**model_config['params'])
            final_model.fit(X_train_scaled, y_train)
            test_pred = final_model.predict(X_test_scaled)
            
            test_accuracy = accuracy_score(y_test, test_pred)
            test_f1 = f1_score(y_test, test_pred, average='weighted')
            test_precision = precision_score(y_test, test_pred, average='weighted', zero_division=0)
            test_recall = recall_score(y_test, test_pred, average='weighted')
            
            model_results.append({
                'model_name': model_config['name'],
                'cv_train_f1': mean_train_f1,
                'cv_val_f1': mean_val_f1,
                'train_val_diff': abs(mean_train_f1 - mean_val_f1),
                'davidian_score': dr_score,
                'test_accuracy': test_accuracy,
                'test_f1': test_f1,
                'test_precision': test_precision,
                'test_recall': test_recall
            })
            
        except Exception as e:
            continue
    
    if len(model_results) < 2:
        return None
    
    # Select best models
    best_davidian = max(model_results, key=lambda x: x['davidian_score'])
    best_validation = max(model_results, key=lambda x: x['cv_val_f1'])
    
    return {
        'trial_seed': trial_seed,
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'train_class_counts': {int(k): int(v) for k, v in Counter(y_train).items()},
        'test_class_counts': {int(k): int(v) for k, v in Counter(y_test).items()},
        'davidian_selected': {
            'model': best_davidian['model_name'],
            'test_accuracy': best_davidian['test_accuracy'],
            'test_f1': best_davidian['test_f1'],
            'train_val_diff': best_davidian['train_val_diff']
        },
        'validation_selected': {
            'model': best_validation['model_name'],
            'test_accuracy': best_validation['test_accuracy'],
            'test_f1': best_validation['test_f1'],
            'train_val_diff': best_validation['train_val_diff']
        },
        'improvements': {
            'accuracy': best_davidian['test_accuracy'] - best_validation['test_accuracy'],
            'f1': best_davidian['test_f1'] - best_validation['test_f1']
        }
    }

def run_sample_size_trials_experiment():
    """Run systematic experiment varying sample sizes and trial counts."""
    
    # Experimental parameters
    sample_sizes = [500, 1000, 2000, 5000]  # Start smaller for speed
    trial_counts = [5, 10, 20]  # Reasonable range
    imbalance_ratios = [4.0, 9.0, 19.0]  # 80/20, 90/10, 95/5
    
    print(f"\n1. Experimental Design:")
    print(f"   Sample sizes: {sample_sizes}")
    print(f"   Trial counts: {trial_counts}")
    print(f"   Imbalance ratios: {imbalance_ratios}")
    print(f"   Total experiments: {len(sample_sizes) * len(trial_counts) * len(imbalance_ratios)}")
    
    all_results = []
    
    for imbalance_ratio in imbalance_ratios:
        ratio_name = f"{imbalance_ratio:.0f}_to_1"
        print(f"\n{'='*40}")
        print(f"IMBALANCE RATIO: {imbalance_ratio:.0f}:1")
        print(f"{'='*40}")
        
        for sample_size in sample_sizes:
            print(f"\n  Sample size: {sample_size}")
            
            # Create dataset
            X, y, dataset_info = create_imbalanced_dataset(sample_size, imbalance_ratio)
            
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            print(f"    Actual class distribution: {dict(Counter(y))}")
            
            for n_trials in trial_counts:
                print(f"    Running {n_trials} trials...")
                
                trials = []
                for trial in range(n_trials):
                    trial_seed = np.random.randint(0, 2**31 - 1)
                    result = run_single_trial(X_scaled, y, trial_seed)
                    if result:
                        trials.append(result)
                
                if trials:
                    # Calculate statistics
                    accuracy_improvements = [t['improvements']['accuracy'] for t in trials]
                    f1_improvements = [t['improvements']['f1'] for t in trials]
                    
                    accuracy_wins = [imp for imp in accuracy_improvements if imp > 0]
                    accuracy_losses = [imp for imp in accuracy_improvements if imp < 0]
                    f1_wins = [imp for imp in f1_improvements if imp > 0]
                    f1_losses = [imp for imp in f1_improvements if imp < 0]
                    
                    # Expected value
                    accuracy_ev = np.mean(accuracy_improvements) * 100
                    f1_ev = np.mean(f1_improvements) * 100
                    
                    # Statistical significance
                    from scipy import stats
                    acc_t_stat, acc_p_value = stats.ttest_1samp(accuracy_improvements, 0)
                    f1_t_stat, f1_p_value = stats.ttest_1samp(f1_improvements, 0)
                    
                    result_summary = {
                        'imbalance_ratio': imbalance_ratio,
                        'sample_size': sample_size,
                        'n_trials': len(trials),
                        'dataset_info': dataset_info,
                        'accuracy_analysis': {
                            'expected_value_pct': accuracy_ev,
                            'std_pct': np.std(accuracy_improvements) * 100,
                            'win_rate_pct': len(accuracy_wins) / len(trials) * 100,
                            'avg_win_size_pct': np.mean(accuracy_wins) * 100 if accuracy_wins else 0,
                            'avg_loss_size_pct': np.mean(accuracy_losses) * 100 if accuracy_losses else 0,
                            'wins': len(accuracy_wins),
                            'losses': len(accuracy_losses),
                            't_statistic': acc_t_stat,
                            'p_value': acc_p_value,
                            'significant': acc_p_value < 0.05
                        },
                        'f1_analysis': {
                            'expected_value_pct': f1_ev,
                            'std_pct': np.std(f1_improvements) * 100,
                            'win_rate_pct': len(f1_wins) / len(trials) * 100,
                            'avg_win_size_pct': np.mean(f1_wins) * 100 if f1_wins else 0,
                            'avg_loss_size_pct': np.mean(f1_losses) * 100 if f1_losses else 0,
                            'wins': len(f1_wins),
                            'losses': len(f1_losses),
                            't_statistic': f1_t_stat,
                            'p_value': f1_p_value,
                            'significant': f1_p_value < 0.05
                        },
                        'trial_details': trials
                    }
                    
                    all_results.append(result_summary)
                    
                    # Print quick summary
                    acc_status = "✅" if accuracy_ev > 0 else "❌"
                    acc_sig = "***" if acc_p_value < 0.001 else "**" if acc_p_value < 0.01 else "*" if acc_p_value < 0.05 else ""
                    
                    print(f"      {acc_status} {n_trials} trials: Acc EV = {accuracy_ev:+.3f}% ± {np.std(accuracy_improvements)*100:.3f}% "
                          f"(win rate: {len(accuracy_wins)}/{len(trials)}) {acc_sig}")
    
    return all_results

def analyze_sample_size_effects(all_results: List[Dict[str, Any]]):
    """Analyze how sample size affects Davidian Regularization performance."""
    print(f"\n2. Sample Size & Trials Analysis:")
    print("="*60)
    
    # Create comprehensive table
    print(f"{'Ratio':<8} {'Samples':<8} {'Trials':<8} {'Acc EV':<10} {'±Std':<8} {'Win%':<8} {'Avg Win':<10} {'p-val':<8} {'Sig'}")
    print("-" * 85)
    
    # Group by imbalance ratio for analysis
    by_ratio = {}
    for result in all_results:
        ratio = result['imbalance_ratio']
        if ratio not in by_ratio:
            by_ratio[ratio] = []
        by_ratio[ratio].append(result)
    
    # Analyze trends
    for ratio in sorted(by_ratio.keys()):
        ratio_results = by_ratio[ratio]
        
        print(f"\nImbalance Ratio {ratio:.0f}:1:")
        
        for result in sorted(ratio_results, key=lambda x: (x['sample_size'], x['n_trials'])):
            acc = result['accuracy_analysis']
            
            samples = result['sample_size']
            trials = result['n_trials']
            ev = acc['expected_value_pct']
            std = acc['std_pct']
            win_rate = acc['win_rate_pct']
            avg_win = acc['avg_win_size_pct']
            p_val = acc['p_value']
            
            sig_marker = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            status = "✅" if ev > 0 else "❌"
            
            print(f"{ratio:<8.0f} {samples:<8} {trials:<8} {ev:+8.3f}% {std:<8.3f} {win_rate:<8.1f} {avg_win:<10.3f} {p_val:<8.3f} {sig_marker}")
    
    # Correlation analysis
    print(f"\nCorrelation Analysis:")
    print("-" * 30)
    
    sample_sizes = [r['sample_size'] for r in all_results]
    trial_counts = [r['n_trials'] for r in all_results]
    imbalance_ratios = [r['imbalance_ratio'] for r in all_results]
    accuracy_evs = [r['accuracy_analysis']['expected_value_pct'] for r in all_results]
    
    from scipy import stats
    
    # Sample size correlation
    sample_corr, sample_p = stats.pearsonr(sample_sizes, accuracy_evs)
    print(f"Sample Size vs Expected Value: r={sample_corr:.3f}, p={sample_p:.3f}")
    if sample_p < 0.05:
        direction = "increases" if sample_corr > 0 else "decreases"
        print(f"  📈 Expected value {direction} with larger sample sizes!")
    
    # Trial count correlation
    trial_corr, trial_p = stats.pearsonr(trial_counts, accuracy_evs)
    print(f"Trial Count vs Expected Value: r={trial_corr:.3f}, p={trial_p:.3f}")
    if trial_p < 0.05:
        direction = "increases" if trial_corr > 0 else "decreases"
        print(f"  📈 Expected value {direction} with more trials!")
    
    # Imbalance ratio correlation
    ratio_corr, ratio_p = stats.pearsonr(imbalance_ratios, accuracy_evs)
    print(f"Imbalance Ratio vs Expected Value: r={ratio_corr:.3f}, p={ratio_p:.3f}")
    if ratio_p < 0.05:
        direction = "increases" if ratio_corr > 0 else "decreases"
        print(f"  📈 Expected value {direction} with higher imbalance!")
    
    # Find optimal conditions
    print(f"\nOptimal Conditions Analysis:")
    print("-" * 35)
    
    positive_ev_results = [r for r in all_results if r['accuracy_analysis']['expected_value_pct'] > 0]
    significant_results = [r for r in all_results if r['accuracy_analysis']['significant'] and r['accuracy_analysis']['expected_value_pct'] > 0]
    
    if positive_ev_results:
        best_result = max(positive_ev_results, key=lambda x: x['accuracy_analysis']['expected_value_pct'])
        print(f"Best Expected Value:")
        print(f"  Ratio: {best_result['imbalance_ratio']:.0f}:1")
        print(f"  Sample size: {best_result['sample_size']}")
        print(f"  Trials: {best_result['n_trials']}")
        print(f"  Expected value: +{best_result['accuracy_analysis']['expected_value_pct']:.3f}%")
        print(f"  Win rate: {best_result['accuracy_analysis']['win_rate_pct']:.1f}%")
        print(f"  Avg win size: +{best_result['accuracy_analysis']['avg_win_size_pct']:.3f}%")
    
    if significant_results:
        print(f"\nStatistically Significant Results: {len(significant_results)}")
        for result in significant_results:
            print(f"  {result['imbalance_ratio']:.0f}:1, N={result['sample_size']}, trials={result['n_trials']}: "
                  f"{result['accuracy_analysis']['expected_value_pct']:+.3f}% (p={result['accuracy_analysis']['p_value']:.3f})")
    
    return all_results

def save_sample_size_results(all_results):
    """Save results with proper JSON serialization."""
    os.makedirs('results', exist_ok=True)
    
    # Convert numpy types to Python types for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    # Clean results for JSON
    clean_results = []
    for result in all_results:
        clean_result = {}
        for key, value in result.items():
            if isinstance(value, dict):
                clean_result[key] = {k: convert_for_json(v) for k, v in value.items()}
            elif isinstance(value, list):
                clean_result[key] = [convert_for_json(item) if not isinstance(item, dict) 
                                   else {k: convert_for_json(v) for k, v in item.items()} 
                                   for item in value]
            else:
                clean_result[key] = convert_for_json(value)
        clean_results.append(clean_result)
    
    with open('results/sample_size_trials_results.json', 'w') as f:
        json.dump({
            'experiment_type': 'sample_size_trials_analysis',
            'description': 'Analysis of how sample size and trial count affect Davidian Regularization',
            'results': clean_results,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }, f, indent=2)
    
    print(f"\n✅ Results saved to results/sample_size_trials_results.json")

def main():
    """Run the sample size and trials experiment."""
    try:
        start_time = time.time()
        
        print("🔬 SAMPLE SIZE & TRIALS HYPOTHESIS:")
        print("   Larger sample sizes and more trials may reveal stronger")
        print("   benefits of Davidian Regularization on imbalanced data")
        
        all_results = run_sample_size_trials_experiment()
        
        if all_results:
            analyze_sample_size_effects(all_results)
            save_sample_size_results(all_results)
        
        elapsed = time.time() - start_time
        
        print(f"\n{'='*60}")
        print("🎉 SAMPLE SIZE & TRIALS ANALYSIS COMPLETED!")
        print(f"{'='*60}")
        print(f"Time: {elapsed:.1f}s")
        print("✅ Expected value analysis complete")
        print("✅ Sample size effects measured")
        print("✅ Trial count effects measured")
        
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
