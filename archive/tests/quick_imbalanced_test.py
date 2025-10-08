#!/usr/bin/env python3
"""
Quick imbalanced data test for Davidian Regularization.

Focus on expected value analysis: win rate × average improvement when winning
vs loss rate × average loss when losing = expected value of using Davidian Regularization.
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

print("QUICK IMBALANCED DATA TEST")
print("="*50)
print("Expected Value Analysis of Davidian Regularization")
print("="*50)

def create_quick_imbalanced_datasets():
    """Create a few imbalanced datasets for quick testing."""
    from sklearn.datasets import make_classification
    
    datasets = {}
    
    # 1. Moderate imbalance (80/20) - realistic scenario
    X1, y1 = make_classification(
        n_samples=1000, n_features=15, n_informative=10, n_classes=2,
        weights=[0.8, 0.2], class_sep=0.8, flip_y=0.02, random_state=42
    )
    datasets['moderate_80_20'] = (X1, y1, {'name': 'Moderate 80/20', 'ratio': 4.0})
    
    # 2. High imbalance (90/10) - challenging scenario  
    X2, y2 = make_classification(
        n_samples=1000, n_features=15, n_informative=10, n_classes=2,
        weights=[0.9, 0.1], class_sep=0.7, flip_y=0.02, random_state=43
    )
    datasets['high_90_10'] = (X2, y2, {'name': 'High 90/10', 'ratio': 9.0})
    
    # 3. Severe imbalance (95/5) - very challenging
    X3, y3 = make_classification(
        n_samples=1000, n_features=15, n_informative=10, n_classes=2,
        weights=[0.95, 0.05], class_sep=0.6, flip_y=0.01, random_state=44
    )
    datasets['severe_95_5'] = (X3, y3, {'name': 'Severe 95/5', 'ratio': 19.0})
    
    return datasets

def create_quick_models():
    """Create a smaller set of diverse models for quick testing."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.naive_bayes import GaussianNB
    
    models = [
        # Balanced models
        {'name': 'LogReg_Balanced', 'class': LogisticRegression, 
         'params': {'class_weight': 'balanced', 'random_state': 42, 'max_iter': 1000, 'n_jobs': 1}},
        {'name': 'RF_Balanced', 'class': RandomForestClassifier,
         'params': {'class_weight': 'balanced', 'n_estimators': 100, 'random_state': 42, 'n_jobs': 1}},
        {'name': 'GB_Balanced', 'class': GradientBoostingClassifier,
         'params': {'n_estimators': 100, 'random_state': 42}},
        
        # Unbalanced models (for comparison)
        {'name': 'LogReg_Unbalanced', 'class': LogisticRegression,
         'params': {'class_weight': None, 'random_state': 43, 'max_iter': 1000, 'n_jobs': 1}},
        {'name': 'RF_Unbalanced', 'class': RandomForestClassifier,
         'params': {'class_weight': None, 'n_estimators': 100, 'random_state': 43, 'n_jobs': 1}},
        
        # Different hyperparameters
        {'name': 'LogReg_L1', 'class': LogisticRegression,
         'params': {'penalty': 'l1', 'solver': 'liblinear', 'class_weight': 'balanced', 'random_state': 44, 'n_jobs': 1}},
        {'name': 'RF_Deep', 'class': RandomForestClassifier,
         'params': {'max_depth': None, 'class_weight': 'balanced', 'n_estimators': 50, 'random_state': 44, 'n_jobs': 1}},
        {'name': 'RF_Shallow', 'class': RandomForestClassifier,
         'params': {'max_depth': 3, 'class_weight': 'balanced', 'n_estimators': 100, 'random_state': 45, 'n_jobs': 1}},
        {'name': 'NaiveBayes', 'class': GaussianNB, 'params': {}},
        {'name': 'GB_Conservative', 'class': GradientBoostingClassifier,
         'params': {'n_estimators': 50, 'learning_rate': 0.05, 'random_state': 45}}
    ]
    
    return models

def run_quick_trial(X, y, dataset_info, trial_seed):
    """Run a single quick trial with detailed metrics tracking."""
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
    
    # Get models
    models = create_quick_models()
    
    # K-fold CV on training data (K=5)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=trial_seed)
    
    model_results = []
    
    for model_config in models:
        try:
            # K-fold CV for Davidian score
            fold_train_scores = []
            fold_val_scores = []
            
            for train_idx, val_idx in cv.split(X_train_scaled, y_train):
                X_fold_train, X_fold_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
                y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
                
                model = model_config['class'](**model_config['params'])
                model.fit(X_fold_train, y_fold_train)
                
                train_pred = model.predict(X_fold_train)
                val_pred = model.predict(X_fold_val)
                
                # Use F1-score for imbalanced data
                train_f1 = f1_score(y_fold_train, train_pred, average='weighted')
                val_f1 = f1_score(y_fold_val, val_pred, average='weighted')
                
                fold_train_scores.append(train_f1)
                fold_val_scores.append(val_f1)
            
            mean_train = np.mean(fold_train_scores)
            mean_val = np.mean(fold_val_scores)
            
            # Davidian score
            dr_score = mean_val - abs(mean_train - mean_val)
            
            # Train final model and evaluate on test
            final_model = model_config['class'](**model_config['params'])
            final_model.fit(X_train_scaled, y_train)
            test_pred = final_model.predict(X_test_scaled)
            
            test_accuracy = accuracy_score(y_test, test_pred)
            test_f1 = f1_score(y_test, test_pred, average='weighted')
            test_precision = precision_score(y_test, test_pred, average='weighted', zero_division=0)
            test_recall = recall_score(y_test, test_pred, average='weighted')
            
            model_results.append({
                'model_name': model_config['name'],
                'cv_train_f1': mean_train,
                'cv_val_f1': mean_val,
                'train_val_diff': abs(mean_train - mean_val),
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
    
    # Select best model by each method
    best_davidian = max(model_results, key=lambda x: x['davidian_score'])
    best_validation = max(model_results, key=lambda x: x['cv_val_f1'])
    
    return {
        'trial_seed': trial_seed,
        'train_class_dist': dict(Counter(y_train)),
        'test_class_dist': dict(Counter(y_test)),
        'n_models': len(model_results),
        'davidian_selected': {
            'model': best_davidian['model_name'],
            'davidian_score': best_davidian['davidian_score'],
            'train_val_diff': best_davidian['train_val_diff'],
            'test_accuracy': best_davidian['test_accuracy'],
            'test_f1': best_davidian['test_f1'],
            'test_precision': best_davidian['test_precision'],
            'test_recall': best_davidian['test_recall']
        },
        'validation_selected': {
            'model': best_validation['model_name'],
            'validation_score': best_validation['cv_val_f1'],
            'train_val_diff': best_validation['train_val_diff'],
            'test_accuracy': best_validation['test_accuracy'],
            'test_f1': best_validation['test_f1'],
            'test_precision': best_validation['test_precision'],
            'test_recall': best_validation['test_recall']
        },
        'improvements': {
            'accuracy_diff': best_davidian['test_accuracy'] - best_validation['test_accuracy'],
            'f1_diff': best_davidian['test_f1'] - best_validation['test_f1'],
            'precision_diff': best_davidian['test_precision'] - best_validation['test_precision'],
            'recall_diff': best_davidian['test_recall'] - best_validation['test_recall']
        }
    }

def run_quick_imbalanced_experiment():
    """Run quick experiment with expected value analysis."""
    datasets = create_quick_imbalanced_datasets()
    
    print("\n1. Running quick imbalanced data experiments...")
    
    all_results = {}
    
    for dataset_name, (X, y, info) in datasets.items():
        print(f"\n  Testing {info['name']} (ratio {info['ratio']}:1)...")
        
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        trials = []
        
        # Run 8 quick trials
        for trial in range(8):
            trial_seed = np.random.randint(0, 2**31 - 1)
            result = run_quick_trial(X_scaled, y, info, trial_seed)
            
            if result:
                trials.append(result)
        
        if trials:
            # Calculate expected value analysis
            accuracy_improvements = [t['improvements']['accuracy_diff'] for t in trials]
            f1_improvements = [t['improvements']['f1_diff'] for t in trials]
            
            # Win/loss analysis
            accuracy_wins = [imp for imp in accuracy_improvements if imp > 0]
            accuracy_losses = [imp for imp in accuracy_improvements if imp < 0]
            f1_wins = [imp for imp in f1_improvements if imp > 0]
            f1_losses = [imp for imp in f1_improvements if imp < 0]
            
            # Expected value calculation
            accuracy_expected_value = np.mean(accuracy_improvements) * 100  # Convert to percentage
            f1_expected_value = np.mean(f1_improvements) * 100
            
            # Detailed statistics
            stats = {
                'dataset': dataset_name,
                'imbalance_ratio': info['ratio'],
                'n_trials': len(trials),
                'accuracy_analysis': {
                    'expected_value_pct': accuracy_expected_value,
                    'std_pct': np.std(accuracy_improvements) * 100,
                    'win_rate': len(accuracy_wins) / len(trials) * 100,
                    'avg_win_size_pct': np.mean(accuracy_wins) * 100 if accuracy_wins else 0,
                    'avg_loss_size_pct': np.mean(accuracy_losses) * 100 if accuracy_losses else 0,
                    'wins': len(accuracy_wins),
                    'losses': len(accuracy_losses),
                    'ties': len(trials) - len(accuracy_wins) - len(accuracy_losses)
                },
                'f1_analysis': {
                    'expected_value_pct': f1_expected_value,
                    'std_pct': np.std(f1_improvements) * 100,
                    'win_rate': len(f1_wins) / len(trials) * 100,
                    'avg_win_size_pct': np.mean(f1_wins) * 100 if f1_wins else 0,
                    'avg_loss_size_pct': np.mean(f1_losses) * 100 if f1_losses else 0,
                    'wins': len(f1_wins),
                    'losses': len(f1_losses),
                    'ties': len(trials) - len(f1_wins) - len(f1_losses)
                },
                'all_trials': trials
            }
            
            all_results[dataset_name] = stats
            
            # Print results
            print(f"    Results ({len(trials)} trials):")
            print(f"      Accuracy Expected Value: {accuracy_expected_value:+.3f}% ± {np.std(accuracy_improvements)*100:.3f}%")
            print(f"        Win rate: {len(accuracy_wins)}/{len(trials)} ({len(accuracy_wins)/len(trials)*100:.1f}%)")
            if accuracy_wins:
                print(f"        Avg win size: +{np.mean(accuracy_wins)*100:.3f}%")
            if accuracy_losses:
                print(f"        Avg loss size: {np.mean(accuracy_losses)*100:.3f}%")
            
            print(f"      F1-Score Expected Value: {f1_expected_value:+.3f}% ± {np.std(f1_improvements)*100:.3f}%")
            print(f"        Win rate: {len(f1_wins)}/{len(trials)} ({len(f1_wins)/len(trials)*100:.1f}%)")
            if f1_wins:
                print(f"        Avg win size: +{np.mean(f1_wins)*100:.3f}%")
            if f1_losses:
                print(f"        Avg loss size: {np.mean(f1_losses)*100:.3f}%")
    
    return all_results

def analyze_expected_value_results(all_results):
    """Analyze expected value across different imbalance levels."""
    print(f"\n2. Expected Value Analysis:")
    print("="*50)
    
    print(f"{'Dataset':<15} {'Ratio':<8} {'Acc EV':<10} {'F1 EV':<10} {'Acc Win%':<10} {'F1 Win%':<10}")
    print("-" * 65)
    
    imbalance_ratios = []
    accuracy_evs = []
    f1_evs = []
    
    for dataset_name, stats in all_results.items():
        ratio = stats['imbalance_ratio']
        acc_ev = stats['accuracy_analysis']['expected_value_pct']
        f1_ev = stats['f1_analysis']['expected_value_pct']
        acc_win_rate = stats['accuracy_analysis']['win_rate']
        f1_win_rate = stats['f1_analysis']['win_rate']
        
        imbalance_ratios.append(ratio)
        accuracy_evs.append(acc_ev)
        f1_evs.append(f1_ev)
        
        print(f"{dataset_name:<15} {ratio:<8.1f} {acc_ev:+8.3f}% {f1_ev:+8.3f}% {acc_win_rate:8.1f}% {f1_win_rate:8.1f}%")
    
    # Overall expected value
    overall_acc_ev = np.mean(accuracy_evs)
    overall_f1_ev = np.mean(f1_evs)
    
    print(f"\nOverall Expected Values:")
    print(f"  Accuracy: {overall_acc_ev:+.3f}%")
    print(f"  F1-Score: {overall_f1_ev:+.3f}%")
    
    # Correlation with imbalance
    if len(imbalance_ratios) > 2:
        from scipy import stats
        acc_corr, acc_p = stats.pearsonr(imbalance_ratios, accuracy_evs)
        f1_corr, f1_p = stats.pearsonr(imbalance_ratios, f1_evs)
        
        print(f"\nCorrelation with Imbalance Ratio:")
        print(f"  Accuracy EV vs Ratio: r={acc_corr:.3f}, p={acc_p:.3f}")
        print(f"  F1 EV vs Ratio: r={f1_corr:.3f}, p={f1_p:.3f}")
        
        if acc_p < 0.1:
            direction = "increases" if acc_corr > 0 else "decreases"
            print(f"  📈 Accuracy expected value {direction} with imbalance!")
        
        if f1_p < 0.1:
            direction = "increases" if f1_corr > 0 else "decreases"
            print(f"  📈 F1 expected value {direction} with imbalance!")
    
    # Decision recommendation
    print(f"\n3. Decision Analysis:")
    print("="*30)
    
    positive_ev_count = sum(1 for ev in accuracy_evs if ev > 0)
    
    if overall_acc_ev > 0.1:  # Threshold for meaningful improvement
        print("✅ RECOMMENDATION: Use Davidian Regularization")
        print(f"   Expected value: +{overall_acc_ev:.3f}% accuracy improvement")
        print(f"   Positive EV on {positive_ev_count}/{len(accuracy_evs)} imbalance levels")
    elif overall_acc_ev > 0:
        print("⚖️  MARGINAL: Davidian Regularization shows small positive expected value")
        print(f"   Expected value: +{overall_acc_ev:.3f}% accuracy improvement")
        print("   Consider for research/experimental use")
    else:
        print("❌ NOT RECOMMENDED: Negative expected value")
        print(f"   Expected value: {overall_acc_ev:.3f}% accuracy loss")
    
    return all_results

def save_quick_results(all_results):
    """Save quick experiment results."""
    os.makedirs('results', exist_ok=True)
    
    with open('results/quick_imbalanced_results.json', 'w') as f:
        json.dump({
            'experiment_type': 'quick_imbalanced_expected_value',
            'description': 'Expected value analysis of Davidian Regularization on imbalanced data',
            'methodology': 'K=5 CV, 8 trials per dataset, focus on expected value calculation',
            'results': all_results,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }, f, indent=2, default=str)
    
    print(f"\n✅ Results saved to results/quick_imbalanced_results.json")

def main():
    """Run quick imbalanced data experiment."""
    try:
        start_time = time.time()
        
        print("🔬 EXPECTED VALUE HYPOTHESIS:")
        print("   Even with <50% win rate, Davidian Regularization could have")
        print("   positive expected value if wins are larger than losses")
        
        all_results = run_quick_imbalanced_experiment()
        
        if all_results:
            analyze_expected_value_results(all_results)
            save_quick_results(all_results)
        
        elapsed = time.time() - start_time
        
        print(f"\n{'='*50}")
        print("🎉 QUICK TEST COMPLETED!")
        print(f"{'='*50}")
        print(f"Time: {elapsed:.1f}s")
        print("✅ Expected value analysis complete")
        print("✅ Imbalance correlation tested")
        
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
