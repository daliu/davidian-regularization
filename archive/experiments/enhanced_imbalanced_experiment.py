#!/usr/bin/env python3
"""
Enhanced imbalanced data experiment with feature variation and balanced sampling.

This experiment implements:
1. Feature count variation up to sqrt(minority_class_size)
2. Balanced sampling (random subset of majority class)
3. F1-score focused evaluation
4. Organized output structure
"""

import sys
import os
import time
import json
import numpy as np
from collections import Counter
from typing import Dict, List, Tuple, Any
import math

# Force single-threaded execution
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

print("ENHANCED IMBALANCED DATA EXPERIMENT")
print("="*60)
print("Feature variation + Balanced sampling + F1-score focus")
print("="*60)

def create_imbalanced_dataset_with_balanced_sampling(n_total: int, imbalance_ratio: float, 
                                                   n_features: int, random_seed: int = 42):
    """
    Create imbalanced dataset, then create balanced version by subsampling majority class.
    
    Args:
        n_total: Total samples in original imbalanced dataset
        imbalance_ratio: Original imbalance ratio
        n_features: Number of features
        random_seed: Random seed
    
    Returns:
        (X_imbalanced, y_imbalanced, X_balanced, y_balanced, metadata)
    """
    from sklearn.datasets import make_classification
    
    # Calculate original class weights
    minority_weight = 1.0 / (imbalance_ratio + 1.0)
    majority_weight = 1.0 - minority_weight
    
    # Create original imbalanced dataset
    X_imbalanced, y_imbalanced = make_classification(
        n_samples=n_total,
        n_features=n_features,
        n_informative=min(n_features, max(2, n_features - 2)),
        n_redundant=min(2, n_features // 10),
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
    n_majority_to_keep = n_minority  # Balance the classes
    
    # Randomly select majority samples to keep
    np.random.seed(random_seed + 1)
    selected_majority_indices = np.random.choice(majority_indices, n_majority_to_keep, replace=False)
    
    # Combine for balanced dataset
    balanced_indices = np.concatenate([minority_indices, selected_majority_indices])
    np.random.shuffle(balanced_indices)
    
    X_balanced = X_imbalanced[balanced_indices]
    y_balanced = y_imbalanced[balanced_indices]
    
    metadata = {
        'n_features': n_features,
        'imbalance_ratio': imbalance_ratio,
        'original_samples': n_total,
        'original_minority_count': n_minority,
        'original_majority_count': len(majority_indices),
        'balanced_samples': len(X_balanced),
        'balanced_minority_count': n_minority,
        'balanced_majority_count': n_majority_to_keep,
        'feature_to_minority_ratio': n_features / n_minority if n_minority > 0 else float('inf')
    }
    
    return X_imbalanced, y_imbalanced, X_balanced, y_balanced, metadata

def create_efficient_models():
    """Create efficient models for quick testing."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.naive_bayes import GaussianNB
    
    return [
        {'name': 'LogReg_Balanced', 'class': LogisticRegression, 
         'params': {'class_weight': 'balanced', 'random_state': 42, 'max_iter': 1500, 'n_jobs': 1}},
        {'name': 'LogReg_Unbalanced', 'class': LogisticRegression,
         'params': {'class_weight': None, 'random_state': 43, 'max_iter': 1500, 'n_jobs': 1}},
        {'name': 'RF_Balanced', 'class': RandomForestClassifier,
         'params': {'class_weight': 'balanced', 'n_estimators': 50, 'random_state': 42, 'n_jobs': 1}},
        {'name': 'RF_Unbalanced', 'class': RandomForestClassifier,
         'params': {'class_weight': None, 'n_estimators': 50, 'random_state': 43, 'n_jobs': 1}},
        {'name': 'GradBoost', 'class': GradientBoostingClassifier,
         'params': {'n_estimators': 50, 'learning_rate': 0.1, 'random_state': 42}},
        {'name': 'NaiveBayes', 'class': GaussianNB, 'params': {}}
    ]

def run_single_comparison_trial(X_imbalanced, y_imbalanced, X_balanced, y_balanced, 
                               trial_seed: int, dataset_type: str):
    """
    Run a single trial comparing imbalanced vs balanced datasets.
    
    Args:
        dataset_type: 'imbalanced' or 'balanced'
    """
    from sklearn.model_selection import train_test_split, StratifiedKFold
    from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
    from sklearn.preprocessing import StandardScaler
    
    # Choose dataset
    X, y = (X_imbalanced, y_imbalanced) if dataset_type == 'imbalanced' else (X_balanced, y_balanced)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=trial_seed, stratify=y
    )
    
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = create_efficient_models()
    
    # K-fold CV (K=5) using F1-score
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
                
                # F1-score (weighted for multiclass, binary for binary)
                avg_method = 'binary' if len(np.unique(y)) == 2 else 'weighted'
                train_f1 = f1_score(y_fold_train, train_pred, average=avg_method)
                val_f1 = f1_score(y_fold_val, val_pred, average=avg_method)
                
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
            
            test_f1 = f1_score(y_test, test_pred, average=avg_method)
            test_precision = precision_score(y_test, test_pred, average=avg_method, zero_division=0)
            test_recall = recall_score(y_test, test_pred, average=avg_method)
            test_accuracy = accuracy_score(y_test, test_pred)
            
            model_results.append({
                'model_name': model_config['name'],
                'cv_train_f1': mean_train_f1,
                'cv_val_f1': mean_val_f1,
                'train_val_diff': abs(mean_train_f1 - mean_val_f1),
                'davidian_score': dr_score,
                'test_f1': test_f1,
                'test_precision': test_precision,
                'test_recall': test_recall,
                'test_accuracy': test_accuracy
            })
            
        except Exception as e:
            continue
    
    if len(model_results) < 2:
        return None
    
    # Select best models
    best_davidian = max(model_results, key=lambda x: x['davidian_score'])
    best_f1_validation = max(model_results, key=lambda x: x['cv_val_f1'])
    
    return {
        'trial_seed': trial_seed,
        'dataset_type': dataset_type,
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'train_class_counts': {int(k): int(v) for k, v in Counter(y_train).items()},
        'test_class_counts': {int(k): int(v) for k, v in Counter(y_test).items()},
        'davidian_selected': {
            'model': best_davidian['model_name'],
            'test_f1': best_davidian['test_f1'],
            'test_precision': best_davidian['test_precision'],
            'test_recall': best_davidian['test_recall'],
            'test_accuracy': best_davidian['test_accuracy'],
            'train_val_diff': best_davidian['train_val_diff']
        },
        'f1_validation_selected': {
            'model': best_f1_validation['model_name'],
            'test_f1': best_f1_validation['test_f1'],
            'test_precision': best_f1_validation['test_precision'],
            'test_recall': best_f1_validation['test_recall'],
            'test_accuracy': best_f1_validation['test_accuracy'],
            'train_val_diff': best_f1_validation['train_val_diff']
        },
        'improvements': {
            'f1_diff': best_davidian['test_f1'] - best_f1_validation['test_f1'],
            'precision_diff': best_davidian['test_precision'] - best_f1_validation['test_precision'],
            'recall_diff': best_davidian['test_recall'] - best_f1_validation['test_recall'],
            'accuracy_diff': best_davidian['test_accuracy'] - best_f1_validation['test_accuracy']
        }
    }

def run_enhanced_experiment():
    """Run enhanced experiment with feature variation and balanced sampling."""
    
    # Create output directory structure
    output_base = 'results/enhanced_imbalanced'
    os.makedirs(output_base, exist_ok=True)
    os.makedirs(f'{output_base}/imbalanced_data', exist_ok=True)
    os.makedirs(f'{output_base}/balanced_data', exist_ok=True)
    
    print(f"\n1. Enhanced Experimental Design:")
    
    # Test configurations focusing on promising scenarios
    configurations = [
        # Based on previous results, focus on 19:1 ratio with feature variation
        {'ratio': 19.0, 'n_total': 2000, 'n_features': 10, 'trials': 12},
        {'ratio': 19.0, 'n_total': 2000, 'n_features': 15, 'trials': 12},
        {'ratio': 19.0, 'n_total': 2000, 'n_features': 20, 'trials': 12},
        
        # Test sqrt(minority_class) features - for 19:1 with 2000 samples, minority ~100, sqrt ~10
        {'ratio': 19.0, 'n_total': 3000, 'n_features': 12, 'trials': 15},  # sqrt(~150) ≈ 12
        
        # Test 9:1 ratio with feature variation (showed some promise)
        {'ratio': 9.0, 'n_total': 1000, 'n_features': 10, 'trials': 10},   # sqrt(~100) = 10
        {'ratio': 9.0, 'n_total': 2000, 'n_features': 14, 'trials': 12},   # sqrt(~200) ≈ 14
    ]
    
    print(f"   Configurations: {len(configurations)}")
    print(f"   Output directories: {output_base}/{{imbalanced_data, balanced_data}}")
    
    all_results = []
    
    for i, config in enumerate(configurations):
        ratio = config['ratio']
        n_total = config['n_total']
        n_features = config['n_features']
        trials = config['trials']
        
        print(f"\n  [{i+1}/{len(configurations)}] Ratio {ratio:.0f}:1, N={n_total}, Features={n_features}, Trials={trials}")
        
        # Create datasets
        X_imbalanced, y_imbalanced, X_balanced, y_balanced, metadata = create_imbalanced_dataset_with_balanced_sampling(
            n_total, ratio, n_features, random_seed=42 + i
        )
        
        print(f"    Original: {dict(Counter(y_imbalanced))}")
        print(f"    Balanced: {dict(Counter(y_balanced))}")
        print(f"    Features/Minority ratio: {metadata['feature_to_minority_ratio']:.2f}")
        
        # Test both imbalanced and balanced versions
        for dataset_type, X, y in [('imbalanced', X_imbalanced, y_imbalanced), 
                                  ('balanced', X_balanced, y_balanced)]:
            
            print(f"      Testing {dataset_type} version...")
            
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Run trials
            trial_results = []
            for trial in range(trials):
                trial_seed = np.random.randint(0, 2**31 - 1)
                result = run_single_comparison_trial(X_imbalanced, y_imbalanced, X_balanced, y_balanced, 
                                                   trial_seed, dataset_type)
                if result:
                    trial_results.append(result)
            
            if trial_results:
                # F1-score analysis
                f1_improvements = [t['improvements']['f1_diff'] for t in trial_results]
                
                f1_wins = [imp for imp in f1_improvements if imp > 0]
                f1_losses = [imp for imp in f1_improvements if imp < 0]
                
                # Expected value
                f1_ev = np.mean(f1_improvements) * 100
                f1_std = np.std(f1_improvements) * 100
                
                # Statistical significance
                from scipy import stats
                f1_t_stat, f1_p_value = stats.ttest_1samp(f1_improvements, 0)
                
                experiment_result = {
                    'configuration': config,
                    'dataset_type': dataset_type,
                    'metadata': metadata,
                    'n_successful_trials': len(trial_results),
                    'f1_analysis': {
                        'expected_value_pct': float(f1_ev),
                        'std_pct': float(f1_std),
                        'win_rate_pct': float(len(f1_wins) / len(trial_results) * 100),
                        'avg_win_size_pct': float(np.mean(f1_wins) * 100) if f1_wins else 0.0,
                        'avg_loss_size_pct': float(np.mean(f1_losses) * 100) if f1_losses else 0.0,
                        'wins': int(len(f1_wins)),
                        'losses': int(len(f1_losses)),
                        't_statistic': float(f1_t_stat),
                        'p_value': float(f1_p_value) if not np.isnan(f1_p_value) else None,
                        'significant': bool(f1_p_value < 0.05) if not np.isnan(f1_p_value) else False
                    }
                }
                
                all_results.append(experiment_result)
                
                # Print results
                f1_status = "✅" if f1_ev > 0 else "❌"
                f1_sig = "***" if f1_p_value < 0.001 else "**" if f1_p_value < 0.01 else "*" if f1_p_value < 0.05 else ""
                
                print(f"        {f1_status} F1 EV: {f1_ev:+.3f}% ± {f1_std:.3f}% {f1_sig}")
                print(f"           Win rate: {len(f1_wins)}/{len(trial_results)} ({len(f1_wins)/len(trial_results)*100:.1f}%)")
                if f1_wins:
                    print(f"           Avg win: +{np.mean(f1_wins)*100:.3f}%")
                if f1_losses:
                    print(f"           Avg loss: {np.mean(f1_losses)*100:.3f}%")
    
    return all_results

def analyze_enhanced_results(all_results: List[Dict[str, Any]]):
    """Analyze enhanced experimental results."""
    print(f"\n2. Enhanced Results Analysis:")
    print("="*60)
    
    # Separate imbalanced vs balanced results
    imbalanced_results = [r for r in all_results if r['dataset_type'] == 'imbalanced']
    balanced_results = [r for r in all_results if r['dataset_type'] == 'balanced']
    
    print(f"Imbalanced vs Balanced Dataset Comparison:")
    print("-" * 50)
    
    print(f"{'Type':<12} {'Ratio':<6} {'N':<6} {'Feat':<6} {'F1 EV':<10} {'Win%':<8} {'Avg Win':<10} {'Sig'}")
    print("-" * 75)
    
    for result in all_results:
        config = result['configuration']
        f1_analysis = result['f1_analysis']
        dataset_type = result['dataset_type']
        
        ratio = config['ratio']
        n_total = config['n_total']
        n_features = config['n_features']
        ev = f1_analysis['expected_value_pct']
        win_rate = f1_analysis['win_rate_pct']
        avg_win = f1_analysis['avg_win_size_pct']
        p_val = f1_analysis['p_value']
        
        sig_marker = "***" if p_val and p_val < 0.001 else "**" if p_val and p_val < 0.01 else "*" if p_val and p_val < 0.05 else ""
        
        print(f"{dataset_type:<12} {ratio:<6.0f} {n_total:<6} {n_features:<6} {ev:+8.3f}% {win_rate:<8.1f} {avg_win:<10.3f} {sig_marker}")
    
    # Feature analysis
    print(f"\nFeature Count Analysis:")
    print("-" * 30)
    
    # Group by feature count
    by_features = {}
    for result in imbalanced_results:  # Focus on imbalanced data
        n_features = result['configuration']['n_features']
        if n_features not in by_features:
            by_features[n_features] = []
        by_features[n_features].append(result)
    
    for n_features in sorted(by_features.keys()):
        feature_results = by_features[n_features]
        feature_evs = [r['f1_analysis']['expected_value_pct'] for r in feature_results]
        
        print(f"  {n_features} features ({len(feature_results)} configs): {np.mean(feature_evs):+.3f}% avg EV")
    
    # Find best configurations
    print(f"\nBest Configurations:")
    print("-" * 25)
    
    # Best imbalanced
    best_imbalanced = max(imbalanced_results, key=lambda x: x['f1_analysis']['expected_value_pct'])
    best_config_imb = best_imbalanced['configuration']
    best_f1_imb = best_imbalanced['f1_analysis']
    
    print(f"Best Imbalanced:")
    print(f"  Config: {best_config_imb['ratio']:.0f}:1, N={best_config_imb['n_total']}, Features={best_config_imb['n_features']}")
    print(f"  F1 Expected Value: {best_f1_imb['expected_value_pct']:+.3f}%")
    print(f"  Win rate: {best_f1_imb['win_rate_pct']:.1f}%")
    print(f"  Significance: p={best_f1_imb['p_value']:.3f}" if best_f1_imb['p_value'] else "  Significance: N/A")
    
    # Best balanced
    if balanced_results:
        best_balanced = max(balanced_results, key=lambda x: x['f1_analysis']['expected_value_pct'])
        best_config_bal = best_balanced['configuration']
        best_f1_bal = best_balanced['f1_analysis']
        
        print(f"\nBest Balanced:")
        print(f"  Config: {best_config_bal['ratio']:.0f}:1 → balanced, N={best_config_bal['n_total']}, Features={best_config_bal['n_features']}")
        print(f"  F1 Expected Value: {best_f1_bal['expected_value_pct']:+.3f}%")
        print(f"  Win rate: {best_f1_bal['win_rate_pct']:.1f}%")

def save_enhanced_results(all_results: List[Dict[str, Any]]):
    """Save enhanced results in organized directory structure."""
    output_base = 'results/enhanced_imbalanced'
    
    # Save comprehensive results
    with open(f'{output_base}/comprehensive_results.json', 'w') as f:
        json.dump({
            'experiment_type': 'enhanced_imbalanced_with_features_and_balanced_sampling',
            'description': 'Feature variation + balanced sampling + F1-score focus',
            'methodology': {
                'feature_variation': 'Up to sqrt(minority_class_size)',
                'balanced_sampling': 'Random subset of majority class to match minority',
                'primary_metric': 'F1-score (binary for binary classification)',
                'k_folds': 5,
                'evaluation': 'Test data performance comparison'
            },
            'results': all_results,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }, f, indent=2)
    
    # Save summary table
    summary_table = []
    for result in all_results:
        config = result['configuration']
        f1_analysis = result['f1_analysis']
        
        summary_table.append({
            'dataset_type': result['dataset_type'],
            'imbalance_ratio': float(config['ratio']),
            'sample_size': int(config['n_total']),
            'n_features': int(config['n_features']),
            'feature_to_minority_ratio': float(result['metadata']['feature_to_minority_ratio']),
            'n_trials': int(result['n_successful_trials']),
            'f1_expected_value_pct': float(f1_analysis['expected_value_pct']),
            'f1_std_pct': float(f1_analysis['std_pct']),
            'f1_win_rate_pct': float(f1_analysis['win_rate_pct']),
            'f1_avg_win_size_pct': float(f1_analysis['avg_win_size_pct']),
            'f1_p_value': f1_analysis['p_value'],
            'f1_significant': bool(f1_analysis['significant'])
        })
    
    with open(f'{output_base}/summary_table.json', 'w') as f:
        json.dump(summary_table, f, indent=2)
    
    # Save separate files for imbalanced vs balanced
    imbalanced_results = [r for r in all_results if r['dataset_type'] == 'imbalanced']
    balanced_results = [r for r in all_results if r['dataset_type'] == 'balanced']
    
    if imbalanced_results:
        with open(f'{output_base}/imbalanced_data/results.json', 'w') as f:
            json.dump(imbalanced_results, f, indent=2)
    
    if balanced_results:
        with open(f'{output_base}/balanced_data/results.json', 'w') as f:
            json.dump(balanced_results, f, indent=2)
    
    print(f"\n✅ Enhanced results saved to:")
    print(f"   - {output_base}/comprehensive_results.json")
    print(f"   - {output_base}/summary_table.json")
    print(f"   - {output_base}/imbalanced_data/results.json")
    print(f"   - {output_base}/balanced_data/results.json")

def main():
    """Run the enhanced imbalanced data experiment."""
    try:
        start_time = time.time()
        
        print("🔬 ENHANCED HYPOTHESES:")
        print("   1. Feature count variation affects Davidian Regularization effectiveness")
        print("   2. Balanced sampling (subset majority) vs full imbalanced data comparison")
        print("   3. F1-score provides more meaningful evaluation than accuracy")
        
        all_results = run_enhanced_experiment()
        
        if all_results:
            analyze_enhanced_results(all_results)
            save_enhanced_results(all_results)
        
        elapsed = time.time() - start_time
        
        print(f"\n{'='*60}")
        print("🎉 ENHANCED EXPERIMENT COMPLETED!")
        print(f"{'='*60}")
        print(f"Time: {elapsed:.1f}s")
        print("✅ Feature variation tested")
        print("✅ Balanced sampling implemented")
        print("✅ F1-score focused evaluation")
        print("✅ Organized output structure")
        
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
