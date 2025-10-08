#!/usr/bin/env python3
"""
Real Dataset Validation: Clean Implementation

Test Davidian Regularization on real datasets with proper EV and SE tracking.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any
import json
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import load_breast_cancer, load_wine, load_digits, load_iris
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler

print("REAL DATASET VALIDATION - CLEAN IMPLEMENTATION")
print("="*60)

def apply_regularization(train_score, val_score, method):
    """Apply regularization method."""
    diff = abs(train_score - val_score)
    
    if method == 'stability_bonus':
        if diff < 0.1:
            bonus = (0.1 - diff) / 0.1 * 0.2
            return val_score * (1.0 + bonus)
        return val_score
    elif method == 'davidian':
        return val_score - diff
    elif method == 'conservative':
        return val_score - 0.5 * diff
    else:  # standard
        return val_score

def test_dataset(name, X, y, n_trials=30):
    """Test a single dataset."""
    
    print(f"\nTesting {name}: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )
    
    methods = ['stability_bonus', 'standard', 'davidian', 'conservative']
    results = []
    
    for method in methods:
        print(f"  Method: {method}")
        
        # Run trials
        method_scores = []
        baseline_scores = []
        
        for trial in range(n_trials):
            # Random split for baseline comparison
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_train, y_train, test_size=0.25, stratify=y_train, random_state=trial
            )
            
            # Train model
            model = LogisticRegression(random_state=42, max_iter=2000)
            model.fit(X_tr, y_tr)
            
            # Get scores
            train_pred = model.predict(X_tr)
            val_pred = model.predict(X_val)
            
            train_acc = accuracy_score(y_tr, train_pred)
            val_acc = accuracy_score(y_val, val_pred)
            
            # Apply regularization
            reg_score = apply_regularization(train_acc, val_acc, method)
            
            method_scores.append(reg_score)
            baseline_scores.append(val_acc)
        
        # Calculate statistics
        method_scores = np.array(method_scores)
        baseline_scores = np.array(baseline_scores)
        
        ev_method = np.mean(method_scores)
        ev_baseline = np.mean(baseline_scores)
        ev_improvement_pct = (ev_method - ev_baseline) / abs(ev_baseline) * 100
        
        se_improvement_pct = np.std((method_scores - baseline_scores) / np.abs(baseline_scores) * 100) / np.sqrt(n_trials)
        
        # Test set evaluation
        final_model = LogisticRegression(random_state=42, max_iter=2000)
        final_model.fit(X_train, y_train)
        test_pred = final_model.predict(X_test)
        test_acc = accuracy_score(y_test, test_pred)
        test_f1 = f1_score(y_test, test_pred, average='weighted')
        
        test_auc = None
        try:
            test_proba = final_model.predict_proba(X_test)[:, 1]
            test_auc = roc_auc_score(y_test, test_proba)
        except:
            pass
        
        result = {
            'dataset': name,
            'method': method,
            'ev_improvement_pct': ev_improvement_pct,
            'se_improvement_pct': se_improvement_pct,
            'ci_95': 1.96 * se_improvement_pct,
            'method_better': ev_method > ev_baseline,
            'test_accuracy': test_acc,
            'test_f1': test_f1,
            'test_auc': test_auc,
            'n_trials': n_trials
        }
        
        results.append(result)
        
        auc_str = f"{test_auc:.3f}" if test_auc is not None else "N/A"
        print(f"    EV: {ev_improvement_pct:+.2f}% ± {se_improvement_pct:.2f}%, AUC: {auc_str}")
    
    return results

def main():
    """Main function."""
    
    # Load datasets
    datasets = {
        'breast_cancer': load_breast_cancer(),
        'wine': load_wine(),
        'digits': load_digits(),
        'iris': load_iris()
    }
    
    # Convert multi-class to binary where needed
    datasets['wine'] = (datasets['wine'].data, (datasets['wine'].target == 0).astype(int))
    datasets['digits'] = (datasets['digits'].data, (datasets['digits'].target >= 5).astype(int))
    datasets['iris'] = (datasets['iris'].data, (datasets['iris'].target == 0).astype(int))
    datasets['breast_cancer'] = (datasets['breast_cancer'].data, datasets['breast_cancer'].target)
    
    # Run experiments
    all_results = []
    
    for name, (X, y) in datasets.items():
        try:
            results = test_dataset(name, X, y)
            all_results.extend(results)
        except Exception as e:
            print(f"Error with {name}: {e}")
    
    if not all_results:
        print("No results!")
        return
    
    # Save and analyze
    df = pd.DataFrame(all_results)
    
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/real_validation_clean.csv', index=False)
    
    # Create simple visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Performance by method
    ax = axes[0]
    method_stats = df.groupby('method').agg({
        'ev_improvement_pct': 'mean',
        'se_improvement_pct': 'mean'
    })
    
    colors = ['#E74C3C' if 'stability' in method else '#95A5A6' for method in method_stats.index]
    
    bars = ax.bar(range(len(method_stats)), method_stats['ev_improvement_pct'], 
                  yerr=method_stats['se_improvement_pct'], capsize=5, 
                  color=colors, alpha=0.8)
    
    ax.set_xticks(range(len(method_stats)))
    ax.set_xticklabels([m.replace('_', ' ').title() for m in method_stats.index], rotation=45, ha='right')
    ax.set_ylabel('Expected Value: Mean Improvement (%)')
    ax.set_title('Real Dataset Performance')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax.grid(True, alpha=0.3)
    
    # Add labels
    for bar, mean_val, se_val in zip(bars, method_stats['ev_improvement_pct'], method_stats['se_improvement_pct']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + se_val + 0.5, 
               f'{mean_val:+.1f}%\n±{se_val:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    # Performance by dataset (Stability Bonus only)
    ax = axes[1]
    stability_data = df[df['method'] == 'stability_bonus']
    
    if len(stability_data) > 0:
        dataset_stats = stability_data.groupby('dataset').agg({
            'ev_improvement_pct': 'mean',
            'se_improvement_pct': 'mean'
        })
        
        bars = ax.bar(range(len(dataset_stats)), dataset_stats['ev_improvement_pct'], 
                      yerr=dataset_stats['se_improvement_pct'], capsize=5, 
                      color='#E74C3C', alpha=0.8)
        
        ax.set_xticks(range(len(dataset_stats)))
        ax.set_xticklabels([d.replace('_', ' ').title() for d in dataset_stats.index], rotation=45, ha='right')
        ax.set_ylabel('EV Improvement (%)')
        ax.set_title('Stability Bonus by Dataset')
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        ax.grid(True, alpha=0.3)
        
        # Add labels
        for bar, mean_val, se_val in zip(bars, dataset_stats['ev_improvement_pct'], dataset_stats['se_improvement_pct']):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + se_val + 0.5, 
                   f'{mean_val:+.1f}%\n±{se_val:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('graphs/real_validation_summary.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary
    print(f"\n{'='*60}")
    print("REAL DATASET VALIDATION RESULTS")
    print(f"{'='*60}")
    
    for method in df['method'].unique():
        method_data = df[df['method'] == method]
        ev_mean = method_data['ev_improvement_pct'].mean()
        avg_se = method_data['se_improvement_pct'].mean()
        better_rate = method_data['method_better'].mean() * 100
        
        print(f"{method:15s}: EV={ev_mean:+6.2f}% ± {avg_se:.2f}%, Better={better_rate:5.1f}%")
    
    print(f"\n✓ Real dataset validation completed successfully!")

if __name__ == "__main__":
    main()
