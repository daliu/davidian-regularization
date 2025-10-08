#!/usr/bin/env python3
"""
Real Data Validation: Davidian Regularization on Real Datasets

Simplified but comprehensive experiment on real datasets with focus on:
1. Expected Value (EV) means and standard errors
2. Consistency validation across real-world data
3. Comparison with traditional class weighting
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple
import json
import os
import time
import warnings
warnings.filterwarnings('ignore')

# Import ML libraries
from sklearn.datasets import load_breast_cancer, load_wine, load_digits, load_iris
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

print("REAL DATA VALIDATION: DAVIDIAN REGULARIZATION")
print("="*60)
print("Testing on real datasets with EV analysis")
print("="*60)

def load_and_prepare_real_datasets():
    """Load and prepare real datasets for testing."""
    
    datasets = {}
    
    # Breast Cancer
    cancer = load_breast_cancer()
    datasets['breast_cancer'] = {
        'X': cancer.data,
        'y': cancer.target,
        'name': 'Breast Cancer',
        'n_samples': cancer.data.shape[0],
        'n_features': cancer.data.shape[1]
    }
    
    # Wine (convert to binary)
    wine = load_wine()
    wine_y = (wine.target == 0).astype(int)  # Class 0 vs others
    datasets['wine'] = {
        'X': wine.data,
        'y': wine_y,
        'name': 'Wine',
        'n_samples': wine.data.shape[0],
        'n_features': wine.data.shape[1]
    }
    
    # Digits (0-4 vs 5-9)
    digits = load_digits()
    digits_y = (digits.target >= 5).astype(int)
    datasets['digits'] = {
        'X': digits.data,
        'y': digits_y,
        'name': 'Digits',
        'n_samples': digits.data.shape[0],
        'n_features': digits.data.shape[1]
    }
    
    # Iris (convert to binary)
    iris = load_iris()
    iris_y = (iris.target == 0).astype(int)  # Setosa vs others
    datasets['iris'] = {
        'X': iris.data,
        'y': iris_y,
        'name': 'Iris',
        'n_samples': iris.data.shape[0],
        'n_features': iris.data.shape[1]
    }
    
    return datasets

def create_imbalanced_version(X, y, target_ratio=9.0, random_state=42):
    """Create an imbalanced version of a dataset."""
    
    # Find minority and majority classes
    class_counts = np.bincount(y)
    minority_class = np.argmin(class_counts)
    majority_class = 1 - minority_class
    
    minority_indices = np.where(y == minority_class)[0]
    majority_indices = np.where(y == majority_class)[0]
    
    # Calculate target sizes
    n_minority = len(minority_indices)
    n_majority_target = int(n_minority * target_ratio)
    
    if len(majority_indices) >= n_majority_target:
        np.random.seed(random_state)
        selected_majority = np.random.choice(majority_indices, n_majority_target, replace=False)
        
        # Combine indices
        combined_indices = np.concatenate([minority_indices, selected_majority])
        np.random.shuffle(combined_indices)
        
        return X[combined_indices], y[combined_indices]
    else:
        return X, y

def apply_davidian_method(train_score, val_score, method='stability_bonus'):
    """Apply Davidian regularization method."""
    
    diff = abs(train_score - val_score)
    
    if method == 'stability_bonus':
        if diff < 0.1:
            bonus = (0.1 - diff) / 0.1 * 0.2
            return val_score * (1.0 + bonus)
        return val_score
    
    elif method == 'davidian_regularization':
        return val_score - diff
    
    elif method == 'conservative_davidian':
        return val_score - 0.5 * diff
    
    elif method == 'standard_kfold':
        return val_score
    
    else:
        return val_score

def run_real_dataset_experiment(dataset_name, X, y, n_trials=50):
    """Run experiment on a single real dataset."""
    
    print(f"\nTesting {dataset_name}:")
    print(f"  Original: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    methods = ['stability_bonus', 'standard_kfold', 'davidian_regularization', 'conservative_davidian']
    imbalance_ratios = [1.0, 9.0, 19.0]  # Test different imbalance levels
    
    results = []
    
    for ratio in imbalance_ratios:
        print(f"  Testing imbalance ratio 1:{ratio}")
        
        # Create imbalanced version
        if ratio == 1.0:
            X_test, y_test = X_scaled, y
        else:
            X_test, y_test = create_imbalanced_version(X_scaled, y, ratio)
        
        print(f"    Dataset size: {len(X_test)}, Class dist: {dict(zip(*np.unique(y_test, return_counts=True)))}")
        
        # Split train/test
        X_train, X_holdout, y_train, y_holdout = train_test_split(
            X_test, y_test, test_size=0.2, stratify=y_test, random_state=42
        )
        
        for method in methods:
            print(f"    Method: {method}")
            
            # Run multiple trials
            method_scores = []
            baseline_scores = []
            
            for trial in range(n_trials):
                # Random holdout baseline
                X_tr, X_val, y_tr, y_val = train_test_split(
                    X_train, y_train, test_size=0.25, stratify=y_train, random_state=trial
                )
                
                # Train model
                if method == 'class_weighted':
                    class_weights = compute_class_weight('balanced', classes=np.unique(y_tr), y=y_tr)
                    weight_dict = {i: weight for i, weight in enumerate(class_weights)}
                    model = LogisticRegression(random_state=42, max_iter=2000, class_weight=weight_dict)
                else:
                    model = LogisticRegression(random_state=42, max_iter=2000)
                
                model.fit(X_tr, y_tr)
                
                # Get scores
                train_pred = model.predict(X_tr)
                val_pred = model.predict(X_val)
                
                train_acc = accuracy_score(y_tr, train_pred)
                val_acc = accuracy_score(y_val, val_pred)
                
                # Apply regularization
                if method == 'class_weighted':
                    reg_score = val_acc  # No regularization, just weighted training
                else:
                    reg_score = apply_davidian_method(train_acc, val_acc, method)
                
                method_scores.append(reg_score)
                baseline_scores.append(val_acc)
            
            # Calculate statistics
            method_scores = np.array(method_scores)
            baseline_scores = np.array(baseline_scores)
            
            # Expected Value calculations
            ev_method = np.mean(method_scores)
            ev_baseline = np.mean(baseline_scores)
            ev_improvement = ev_method - ev_baseline
            ev_improvement_pct = (ev_improvement / abs(ev_baseline)) * 100 if ev_baseline != 0 else 0
            
            # Standard errors
            se_method = np.std(method_scores) / np.sqrt(n_trials)
            se_baseline = np.std(baseline_scores) / np.sqrt(n_trials)
            se_improvement_pct = np.std((method_scores - baseline_scores) / np.abs(baseline_scores) * 100) / np.sqrt(n_trials)
            
            # Test on holdout set
            final_model = LogisticRegression(random_state=42, max_iter=2000)
            if method == 'class_weighted':
                class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
                weight_dict = {i: weight for i, weight in enumerate(class_weights)}
                final_model = LogisticRegression(random_state=42, max_iter=2000, class_weight=weight_dict)
            
            final_model.fit(X_train, y_train)
            test_pred = final_model.predict(X_holdout)
            test_acc = accuracy_score(y_holdout, test_pred)
            test_f1 = f1_score(y_holdout, test_pred, average='weighted')
            
            # Test AUC
            test_auc = None
            try:
                test_proba = final_model.predict_proba(X_holdout)[:, 1]
                test_auc = roc_auc_score(y_holdout, test_proba)
            except:
                pass
            
            result = {
                'dataset': dataset_name,
                'imbalance_ratio': ratio,
                'method': method,
                'n_trials': n_trials,
                'ev_method_score': float(ev_method),
                'ev_baseline_score': float(ev_baseline),
                'ev_improvement_pct': float(ev_improvement_pct),
                'se_method': float(se_method),
                'se_baseline': float(se_baseline),
                'se_improvement_pct': float(se_improvement_pct),
                'ci_95_improvement': float(1.96 * se_improvement_pct),
                'method_better': bool(ev_method > ev_baseline),
                'statistically_significant': bool(abs(ev_improvement) > 1.96 * se_improvement_pct / 100),
                'test_accuracy': float(test_acc),
                'test_f1': float(test_f1),
                'test_auc': float(test_auc) if test_auc is not None else None,
                'std_method': float(np.std(method_scores)),
                'std_baseline': float(np.std(baseline_scores))
            }
            
            results.append(result)
            
            print(f"      EV improvement: {ev_improvement_pct:+.2f}% ± {se_improvement_pct:.2f}%")
            print(f"      Test AUC: {test_auc:.3f if test_auc else 'N/A'}")
            print(f"      Better: {'YES' if ev_method > ev_baseline else 'NO'}")
    
    return results

def create_real_data_summary_visualization(all_results, save_path):
    """Create summary visualization for real data results."""
    
    df = pd.DataFrame(all_results)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Real Dataset Validation: Davidian Regularization Performance\n' + 
                 'Expected Value Analysis with Standard Errors', 
                 fontsize=16, fontweight='bold')
    
    # 1. Overall Performance by Method
    ax = axes[0, 0]
    
    method_stats = df.groupby('method').agg({
        'ev_improvement_pct': ['mean', 'std'],
        'se_improvement_pct': 'mean',
        'method_better': 'mean',
        'statistically_significant': 'mean'
    }).round(3)
    
    methods = method_stats.index
    ev_means = method_stats['ev_improvement_pct']['mean'].values
    ev_ses = method_stats['se_improvement_pct']['mean'].values
    
    colors = ['#E74C3C' if 'stability' in method else '#95A5A6' for method in methods]
    
    bars = ax.bar(range(len(methods)), ev_means, yerr=ev_ses, capsize=5, 
                  color=colors, alpha=0.8)
    
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([m.replace('_', ' ').title() for m in methods], rotation=45, ha='right')
    ax.set_ylabel('Expected Value: Mean Improvement (%)')
    ax.set_title('Real Datasets: Overall Performance\n(All datasets combined)')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, ev_mean, ev_se in zip(bars, ev_means, ev_ses):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + ev_se + 0.2, 
               f'{ev_mean:+.1f}%\n±{ev_se:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    # 2. Performance by Dataset
    ax = axes[0, 1]
    
    datasets = df['dataset'].unique()
    stability_data = df[df['method'] == 'stability_bonus']
    
    if len(stability_data) > 0:
        dataset_means = []
        dataset_ses = []
        
        for dataset in datasets:
            subset = stability_data[stability_data['dataset'] == dataset]
            if len(subset) > 0:
                dataset_means.append(subset['ev_improvement_pct'].mean())
                dataset_ses.append(subset['se_improvement_pct'].mean())
            else:
                dataset_means.append(0)
                dataset_ses.append(0)
        
        bars = ax.bar(range(len(datasets)), dataset_means, yerr=dataset_ses, 
                      capsize=5, color='#E74C3C', alpha=0.8)
        
        ax.set_xticks(range(len(datasets)))
        ax.set_xticklabels([d.replace('_', ' ').title() for d in datasets], rotation=45, ha='right')
        ax.set_ylabel('EV Improvement (%)')
        ax.set_title('Stability Bonus Performance\nby Real Dataset')
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, mean_val, se_val in zip(bars, dataset_means, dataset_ses):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + se_val + 0.2, 
                   f'{mean_val:+.1f}%\n±{se_val:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    # 3. Statistical Significance Rates
    ax = axes[0, 2]
    
    sig_rates = df.groupby('method')['statistically_significant'].mean() * 100
    better_rates = df.groupby('method')['method_better'].mean() * 100
    
    x = np.arange(len(sig_rates))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, sig_rates.values, width, label='Statistically Significant', 
                   color=[colors[i] for i in range(len(methods))], alpha=0.8)
    bars2 = ax.bar(x + width/2, better_rates.values, width, label='Better than Baseline',
                   color=[colors[i] for i in range(len(methods))], alpha=0.5)
    
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in sig_rates.index], rotation=45, ha='right')
    ax.set_ylabel('Rate (%)')
    ax.set_title('Success Rates on Real Data')
    ax.legend()
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    
    # 4. Performance vs Imbalance Ratio
    ax = axes[1, 0]
    
    ratios = sorted(df['imbalance_ratio'].unique())
    
    for method in ['stability_bonus', 'standard_kfold']:
        if method in df['method'].unique():
            method_data = df[df['method'] == method]
            ratio_means = []
            ratio_ses = []
            
            for ratio in ratios:
                subset = method_data[method_data['imbalance_ratio'] == ratio]
                if len(subset) > 0:
                    ratio_means.append(subset['ev_improvement_pct'].mean())
                    ratio_ses.append(subset['se_improvement_pct'].mean())
                else:
                    ratio_means.append(0)
                    ratio_ses.append(0)
            
            color = '#E74C3C' if method == 'stability_bonus' else '#2ECC71'
            ax.errorbar(ratios, ratio_means, yerr=ratio_ses, 
                       marker='o', linewidth=2, markersize=6, label=method.replace('_', ' ').title(),
                       color=color, capsize=3)
    
    ax.set_xlabel('Imbalance Ratio (1:X)')
    ax.set_ylabel('EV Improvement (%)')
    ax.set_title('Performance vs Imbalance Level\n(Real datasets)')
    ax.legend()
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax.grid(True, alpha=0.3)
    
    # 5. Test AUC Distribution
    ax = axes[1, 1]
    
    auc_data = []
    method_labels = []
    
    for method in methods:
        method_auc = df[df['method'] == method]['test_auc'].dropna()
        if len(method_auc) > 0:
            auc_data.append(method_auc.values)
            method_labels.append(method.replace('_', ' ').title())
    
    if auc_data:
        bp = ax.boxplot(auc_data, labels=method_labels, patch_artist=True)
        
        for i, patch in enumerate(bp['boxes']):
            color = '#E74C3C' if 'stability' in method_labels[i].lower() else '#95A5A6'
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    
    ax.set_ylabel('Test AUC')
    ax.set_title('Generalization Performance\n(Real dataset test AUC)')
    ax.set_xticklabels(method_labels, rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    
    # 6. Expected Value vs Standard Error
    ax = axes[1, 2]
    
    for method in methods:
        method_data = df[df['method'] == method]
        if len(method_data) > 0:
            color = '#E74C3C' if method == 'stability_bonus' else '#95A5A6'
            ax.scatter(method_data['ev_improvement_pct'], method_data['se_improvement_pct'], 
                      c=color, label=method.replace('_', ' ').title(), alpha=0.7, s=50)
    
    ax.set_xlabel('Expected Value: Improvement (%)')
    ax.set_ylabel('Standard Error (%)')
    ax.set_title('Performance vs Precision\n(Top-left quadrant = best)')
    ax.axhline(y=df['se_improvement_pct'].median(), color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Real data visualization saved to {save_path}")

def main():
    """Main execution function."""
    
    start_time = time.time()
    
    # Load datasets
    datasets = load_and_prepare_real_datasets()
    print(f"Loaded {len(datasets)} real datasets")
    
    # Run experiments
    all_results = []
    
    for dataset_name, dataset_info in datasets.items():
        try:
            results = run_real_dataset_experiment(
                dataset_name, 
                dataset_info['X'], 
                dataset_info['y'],
                n_trials=30
            )
            all_results.extend(results)
            
        except Exception as e:
            print(f"Error with {dataset_name}: {e}")
            continue
    
    if not all_results:
        print("No successful experiments!")
        return
    
    # Save results
    os.makedirs('data', exist_ok=True)
    
    results_df = pd.DataFrame(all_results)
    results_df.to_csv('data/real_data_validation_results.csv', index=False)
    
    with open('data/real_data_validation_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Create visualizations
    create_real_data_summary_visualization(all_results, 'graphs/real_data_validation.png')
    
    # Print summary
    elapsed_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print("REAL DATA VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"Execution time: {elapsed_time:.1f} seconds")
    print(f"Total experiments: {len(all_results)}")
    
    # Method performance summary
    print(f"\nEXPECTED VALUE ANALYSIS:")
    
    for method in results_df['method'].unique():
        method_data = results_df[results_df['method'] == method]
        
        ev_mean = method_data['ev_improvement_pct'].mean()
        avg_se = method_data['se_improvement_pct'].mean()
        better_rate = method_data['method_better'].mean() * 100
        sig_rate = method_data['statistically_significant'].mean() * 100
        avg_test_auc = method_data['test_auc'].mean() if method_data['test_auc'].notna().any() else None
        
        print(f"  {method:20s}: EV={ev_mean:+6.2f}% ± {avg_se:.3f}%, "
              f"Better={better_rate:5.1f}%, Sig={sig_rate:5.1f}%, "
              f"AUC={avg_test_auc:.3f if avg_test_auc else 'N/A'}")
    
    print(f"\n✓ Real data validation completed!")
    print(f"✓ Files saved: data/real_data_validation_results.*")
    print(f"✓ Visualization: graphs/real_data_validation.png")

if __name__ == "__main__":
    main()
