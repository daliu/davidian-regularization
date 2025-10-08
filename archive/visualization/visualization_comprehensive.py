#!/usr/bin/env python3
"""
Comprehensive visualization module for Davidian Regularization experiments.

This module creates detailed charts and visualizations to demonstrate the effectiveness
of Davidian Regularization compared to random sampling across different parameters.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_results(filename: str = 'results/comprehensive_davidian_results.json') -> Dict[str, Any]:
    """Load experimental results from JSON file."""
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_master_results_chart(results_df: pd.DataFrame, save_path: str = 'plots/master_results_chart.png'):
    """
    Create the massive chart showing all relevant variables for each trial.
    
    This chart follows the user's requirement for a comprehensive view of:
    - K-folds, number of samples, models, Max test AUC, mean AUC within each trial
    - Regularization method, with confidence intervals
    """
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    
    # Define the grid layout
    gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1, 1], width_ratios=[1, 1, 1])
    
    # 1. Win Rate by Regularization Method (vs Random Holdout Baseline)
    ax1 = fig.add_subplot(gs[0, 0])
    method_wins = results_df.groupby('regularization_method')['method_wins'].agg(['mean', 'count']).reset_index()
    method_wins['win_rate'] = method_wins['mean'] * 100
    
    bars = ax1.bar(method_wins['regularization_method'], method_wins['win_rate'])
    ax1.set_title('Win Rate by Regularization Method', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Win Rate (%)')
    ax1.set_ylim(0, 100)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, win_rate in zip(bars, method_wins['win_rate']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{win_rate:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 2. Max Improvement by Sample Size (vs Random Holdout Baseline)
    ax2 = fig.add_subplot(gs[0, 1])
    sample_stats = results_df.groupby('sample_size')['max_improvement_pct'].agg(['mean', 'std']).reset_index()
    
    bars = ax2.bar(sample_stats['sample_size'].astype(str), sample_stats['mean'])
    ax2.errorbar(range(len(sample_stats)), sample_stats['mean'], yerr=sample_stats['std'], 
                fmt='none', color='black', capsize=5)
    ax2.set_title('Max Improvement by Sample Size vs Baseline', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Mean Improvement (%)')
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    
    # Add value labels
    for i, (mean_imp, std_imp) in enumerate(zip(sample_stats['mean'], sample_stats['std'])):
        ax2.text(i, mean_imp + std_imp + 0.5, f'{mean_imp:+.1f}±{std_imp:.1f}%', 
                ha='center', va='bottom', fontsize=10)
    
    # 3. Test AUC Distribution by Model Type
    ax3 = fig.add_subplot(gs[0, 2])
    auc_data = results_df[results_df['test_auc'].notna()]
    if not auc_data.empty:
        sns.boxplot(data=auc_data, x='model_type', y='test_auc', ax=ax3)
        ax3.set_title('Test AUC Distribution by Model Type', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Test AUC')
        plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
    else:
        ax3.text(0.5, 0.5, 'No AUC data available', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Test AUC Distribution by Model Type', fontsize=14, fontweight='bold')
    
    # 4. Max Improvement Heatmap by Imbalance Ratio and K-folds (vs Baseline)
    ax4 = fig.add_subplot(gs[1, :])
    pivot_data = results_df.pivot_table(
        values='max_improvement_pct', 
        index='imbalance_ratio', 
        columns='k_folds', 
        aggfunc='mean'
    )
    
    sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='RdYlGn', center=0, ax=ax4,
                cbar_kws={'label': 'Mean Improvement (%)'})
    ax4.set_title('Max Improvement Heatmap: Imbalance Ratio vs K-folds (vs Baseline)', fontsize=14, fontweight='bold')
    ax4.set_xlabel('K-folds')
    ax4.set_ylabel('Imbalance Ratio')
    
    # 5. Performance Comparison: Method vs Baseline by Regularization Method
    ax5 = fig.add_subplot(gs[2, 0])
    method_comparison = results_df.groupby('regularization_method')[['max_method_score', 'max_baseline_score']].mean()
    
    x = np.arange(len(method_comparison))
    width = 0.35
    
    bars1 = ax5.bar(x - width/2, method_comparison['max_method_score'], width, label='Method', alpha=0.8)
    bars2 = ax5.bar(x + width/2, method_comparison['max_baseline_score'], width, label='Baseline', alpha=0.8)
    
    ax5.set_title('Max Scores: Method vs Baseline', fontsize=14, fontweight='bold')
    ax5.set_ylabel('Mean Score')
    ax5.set_xticks(x)
    ax5.set_xticklabels(method_comparison.index, rotation=45, ha='right')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Confidence Intervals by Number of Trials (Max Improvement vs Baseline)
    ax6 = fig.add_subplot(gs[2, 1])
    trial_stats = results_df.groupby('n_trials')['max_improvement_pct'].agg(['mean', 'std', 'count']).reset_index()
    trial_stats['ci'] = 1.96 * trial_stats['std'] / np.sqrt(trial_stats['count'])  # 95% CI
    
    ax6.errorbar(trial_stats['n_trials'], trial_stats['mean'], yerr=trial_stats['ci'], 
                marker='o', capsize=5, capthick=2, linewidth=2)
    ax6.set_title('Max Improvement vs Baseline with 95% CI', fontsize=14, fontweight='bold')
    ax6.set_xlabel('Number of Trials')
    ax6.set_ylabel('Mean Improvement (%) ± 95% CI')
    ax6.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax6.grid(True, alpha=0.3)
    
    # 7. Model Performance by Minority Class Percentage (Max Improvement vs Baseline)
    ax7 = fig.add_subplot(gs[2, 2])
    minority_bins = pd.cut(results_df['minority_percentage'], bins=5)
    minority_stats = results_df.groupby(minority_bins)['max_improvement_pct'].agg(['mean', 'std']).reset_index()
    minority_stats['bin_center'] = minority_stats['minority_percentage'].apply(lambda x: x.mid)
    
    ax7.bar(range(len(minority_stats)), minority_stats['mean'])
    ax7.errorbar(range(len(minority_stats)), minority_stats['mean'], yerr=minority_stats['std'], 
                fmt='none', color='black', capsize=5)
    ax7.set_title('Max Improvement vs Baseline by Minority %', fontsize=14, fontweight='bold')
    ax7.set_ylabel('Mean Improvement (%)')
    ax7.set_xticks(range(len(minority_stats)))
    ax7.set_xticklabels([f'{x.left:.1f}-{x.right:.1f}%' for x in minority_stats['minority_percentage']], 
                       rotation=45, ha='right')
    ax7.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    
    # 8. Detailed Performance Table (bottom section)
    ax8 = fig.add_subplot(gs[3, :])
    ax8.axis('tight')
    ax8.axis('off')
    
    # Create summary table with PRIMARY and SECONDARY metrics
    summary_stats = []
    for method in results_df['regularization_method'].unique():
        method_data = results_df[results_df['regularization_method'] == method]
        summary_stats.append([
            method,
            f"{method_data['method_wins'].mean()*100:.1f}%",
            f"{method_data['max_improvement_pct'].mean():+.2f}%",
            f"±{method_data['max_improvement_pct'].std():.2f}%",
            f"{method_data['mean_improvement_pct'].mean():+.2f}%",
            f"±{method_data['mean_improvement_pct'].std():.2f}%",
            f"{method_data['test_accuracy'].mean():.3f}",
            f"{method_data['test_auc'].mean():.3f}" if method_data['test_auc'].notna().any() else "N/A",
            len(method_data)
        ])
    
    table_headers = ['Method', 'Win Rate', 'Max Δ', '±Std', 'Mean Δ', '±Std', 'Test Acc', 'Test AUC', 'N']
    table = ax8.table(cellText=summary_stats, colLabels=table_headers, 
                     cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style the table
    for i in range(len(table_headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax8.set_title('Performance Summary: All Methods vs Random Holdout Baseline', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Master results chart saved to {save_path}")

def create_method_comparison_chart(results_df: pd.DataFrame, save_path: str = 'plots/method_comparison.png'):
    """Create detailed comparison chart between different regularization methods."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Davidian Regularization Methods: Detailed Comparison', fontsize=16, fontweight='bold')
    
    # 1. Win Rate Comparison
    ax = axes[0, 0]
    method_wins = results_df.groupby('regularization_method')['davidian_wins'].mean() * 100
    colors = plt.cm.tab10(np.linspace(0, 1, len(method_wins)))
    
    bars = ax.bar(method_wins.index, method_wins.values, color=colors)
    ax.set_title('Win Rate by Method')
    ax.set_ylabel('Win Rate (%)')
    ax.set_ylim(0, 100)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Add horizontal line at 50%
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.7, label='Random chance')
    ax.legend()
    
    # Add value labels
    for bar, value in zip(bars, method_wins.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
               f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 2. Mean Improvement Distribution
    ax = axes[0, 1]
    sns.boxplot(data=results_df, x='regularization_method', y='improvement_pct', ax=ax)
    ax.set_title('Improvement Distribution by Method')
    ax.set_ylabel('Improvement (%)')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # 3. Performance by Sample Size
    ax = axes[0, 2]
    for method in results_df['regularization_method'].unique():
        method_data = results_df[results_df['regularization_method'] == method]
        sample_means = method_data.groupby('sample_size')['improvement_pct'].mean()
        ax.plot(sample_means.index, sample_means.values, marker='o', label=method, linewidth=2)
    
    ax.set_title('Performance by Sample Size')
    ax.set_xlabel('Sample Size')
    ax.set_ylabel('Mean Improvement (%)')
    ax.set_xscale('log')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # 4. Performance by Imbalance Ratio
    ax = axes[1, 0]
    for method in results_df['regularization_method'].unique():
        method_data = results_df[results_df['regularization_method'] == method]
        ratio_means = method_data.groupby('imbalance_ratio')['improvement_pct'].mean()
        ax.plot(ratio_means.index, ratio_means.values, marker='s', label=method, linewidth=2)
    
    ax.set_title('Performance by Imbalance Ratio')
    ax.set_xlabel('Imbalance Ratio')
    ax.set_ylabel('Mean Improvement (%)')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # 5. Test AUC Comparison
    ax = axes[1, 1]
    auc_data = results_df[results_df['test_auc'].notna()]
    if not auc_data.empty:
        sns.boxplot(data=auc_data, x='regularization_method', y='test_auc', ax=ax)
        ax.set_title('Test AUC by Method')
        ax.set_ylabel('Test AUC')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    else:
        ax.text(0.5, 0.5, 'No AUC data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Test AUC by Method')
    
    # 6. Stability Analysis (Standard Deviation of Improvements)
    ax = axes[1, 2]
    method_stability = results_df.groupby('regularization_method')['improvement_pct'].std()
    bars = ax.bar(method_stability.index, method_stability.values, color=colors)
    ax.set_title('Method Stability (Lower is Better)')
    ax.set_ylabel('Std Dev of Improvements (%)')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Add value labels
    for bar, value in zip(bars, method_stability.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
               f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Method comparison chart saved to {save_path}")

def create_parameter_analysis_charts(results_df: pd.DataFrame, save_path: str = 'plots/parameter_analysis.png'):
    """Create charts analyzing the effect of different parameters."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Parameter Analysis: Effect on Davidian Regularization Performance', fontsize=16, fontweight='bold')
    
    # 1. Sample Size Effect
    ax = axes[0, 0]
    sample_stats = results_df.groupby('sample_size').agg({
        'improvement_pct': ['mean', 'std', 'count'],
        'davidian_wins': 'mean'
    }).round(3)
    
    sample_stats.columns = ['_'.join(col).strip() for col in sample_stats.columns.values]
    sample_stats['ci'] = 1.96 * sample_stats['improvement_pct_std'] / np.sqrt(sample_stats['improvement_pct_count'])
    
    x_pos = range(len(sample_stats))
    ax.bar(x_pos, sample_stats['improvement_pct_mean'], yerr=sample_stats['ci'], 
           capsize=5, alpha=0.7, color='skyblue')
    ax.set_title('Effect of Sample Size')
    ax.set_xlabel('Sample Size')
    ax.set_ylabel('Mean Improvement (%) ± 95% CI')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(sample_stats.index)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax.grid(True, alpha=0.3)
    
    # Add win rate as secondary y-axis
    ax2 = ax.twinx()
    ax2.plot(x_pos, sample_stats['davidian_wins_mean'] * 100, 'ro-', linewidth=2, markersize=8)
    ax2.set_ylabel('Win Rate (%)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    # 2. K-folds Effect
    ax = axes[0, 1]
    k_stats = results_df.groupby('k_folds').agg({
        'improvement_pct': ['mean', 'std', 'count'],
        'davidian_wins': 'mean'
    }).round(3)
    
    k_stats.columns = ['_'.join(col).strip() for col in k_stats.columns.values]
    k_stats['ci'] = 1.96 * k_stats['improvement_pct_std'] / np.sqrt(k_stats['improvement_pct_count'])
    
    x_pos = range(len(k_stats))
    ax.bar(x_pos, k_stats['improvement_pct_mean'], yerr=k_stats['ci'], 
           capsize=5, alpha=0.7, color='lightgreen')
    ax.set_title('Effect of K-folds')
    ax.set_xlabel('Number of K-folds')
    ax.set_ylabel('Mean Improvement (%) ± 95% CI')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(k_stats.index)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax.grid(True, alpha=0.3)
    
    # Add win rate as secondary y-axis
    ax2 = ax.twinx()
    ax2.plot(x_pos, k_stats['davidian_wins_mean'] * 100, 'ro-', linewidth=2, markersize=8)
    ax2.set_ylabel('Win Rate (%)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    # 3. Number of Trials Effect
    ax = axes[1, 0]
    trial_stats = results_df.groupby('n_trials').agg({
        'improvement_pct': ['mean', 'std', 'count'],
        'davidian_wins': 'mean'
    }).round(3)
    
    trial_stats.columns = ['_'.join(col).strip() for col in trial_stats.columns.values]
    trial_stats['ci'] = 1.96 * trial_stats['improvement_pct_std'] / np.sqrt(trial_stats['improvement_pct_count'])
    
    x_pos = range(len(trial_stats))
    ax.bar(x_pos, trial_stats['improvement_pct_mean'], yerr=trial_stats['ci'], 
           capsize=5, alpha=0.7, color='orange')
    ax.set_title('Effect of Number of Trials')
    ax.set_xlabel('Number of Trials')
    ax.set_ylabel('Mean Improvement (%) ± 95% CI')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(trial_stats.index)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax.grid(True, alpha=0.3)
    
    # Add win rate as secondary y-axis
    ax2 = ax.twinx()
    ax2.plot(x_pos, trial_stats['davidian_wins_mean'] * 100, 'ro-', linewidth=2, markersize=8)
    ax2.set_ylabel('Win Rate (%)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    # 4. Model Type Effect
    ax = axes[1, 1]
    model_stats = results_df.groupby('model_type').agg({
        'improvement_pct': ['mean', 'std', 'count'],
        'davidian_wins': 'mean'
    }).round(3)
    
    model_stats.columns = ['_'.join(col).strip() for col in model_stats.columns.values]
    model_stats['ci'] = 1.96 * model_stats['improvement_pct_std'] / np.sqrt(model_stats['improvement_pct_count'])
    
    x_pos = range(len(model_stats))
    ax.bar(x_pos, model_stats['improvement_pct_mean'], yerr=model_stats['ci'], 
           capsize=5, alpha=0.7, color='lightcoral')
    ax.set_title('Effect of Model Type')
    ax.set_xlabel('Model Type')
    ax.set_ylabel('Mean Improvement (%) ± 95% CI')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_stats.index, rotation=45, ha='right')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax.grid(True, alpha=0.3)
    
    # Add win rate as secondary y-axis
    ax2 = ax.twinx()
    ax2.plot(x_pos, model_stats['davidian_wins_mean'] * 100, 'ro-', linewidth=2, markersize=8)
    ax2.set_ylabel('Win Rate (%)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Parameter analysis charts saved to {save_path}")

def create_imbalance_analysis_chart(results_df: pd.DataFrame, save_path: str = 'plots/imbalance_analysis.png'):
    """Create detailed analysis of performance across different imbalance ratios."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Class Imbalance Analysis: Davidian Regularization Performance', fontsize=16, fontweight='bold')
    
    # 1. Performance by Imbalance Ratio
    ax = axes[0, 0]
    imbalance_stats = results_df.groupby('imbalance_ratio').agg({
        'improvement_pct': ['mean', 'std', 'count'],
        'davidian_wins': 'mean',
        'test_auc': 'mean'
    }).round(3)
    
    imbalance_stats.columns = ['_'.join(col).strip() for col in imbalance_stats.columns.values]
    imbalance_stats['ci'] = 1.96 * imbalance_stats['improvement_pct_std'] / np.sqrt(imbalance_stats['improvement_pct_count'])
    
    x_pos = range(len(imbalance_stats))
    bars = ax.bar(x_pos, imbalance_stats['improvement_pct_mean'], yerr=imbalance_stats['ci'], 
                  capsize=5, alpha=0.7)
    ax.set_title('Performance by Imbalance Ratio')
    ax.set_xlabel('Imbalance Ratio')
    ax.set_ylabel('Mean Improvement (%) ± 95% CI')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'1:{ratio:.0f}' for ratio in imbalance_stats.index])
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax.grid(True, alpha=0.3)
    
    # Color bars based on performance
    for i, bar in enumerate(bars):
        if imbalance_stats['improvement_pct_mean'].iloc[i] > 0:
            bar.set_color('green')
            bar.set_alpha(0.7)
        else:
            bar.set_color('red')
            bar.set_alpha(0.7)
    
    # 2. Win Rate by Imbalance Ratio
    ax = axes[0, 1]
    win_rates = imbalance_stats['davidian_wins_mean'] * 100
    bars = ax.bar(x_pos, win_rates, alpha=0.7, color='skyblue')
    ax.set_title('Win Rate by Imbalance Ratio')
    ax.set_xlabel('Imbalance Ratio')
    ax.set_ylabel('Win Rate (%)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'1:{ratio:.0f}' for ratio in imbalance_stats.index])
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.7, label='Random chance')
    ax.set_ylim(0, 100)
    ax.legend()
    
    # Add value labels
    for bar, value in zip(bars, win_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
               f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 3. Heatmap: Method vs Imbalance Ratio
    ax = axes[1, 0]
    heatmap_data = results_df.pivot_table(
        values='improvement_pct', 
        index='regularization_method', 
        columns='imbalance_ratio', 
        aggfunc='mean'
    )
    
    sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlGn', center=0, ax=ax,
                cbar_kws={'label': 'Mean Improvement (%)'})
    ax.set_title('Method Performance Heatmap')
    ax.set_xlabel('Imbalance Ratio')
    ax.set_ylabel('Regularization Method')
    
    # Format x-axis labels
    ax.set_xticklabels([f'1:{ratio:.0f}' for ratio in heatmap_data.columns])
    
    # 4. Test AUC by Imbalance Ratio
    ax = axes[1, 1]
    auc_data = results_df[results_df['test_auc'].notna()]
    if not auc_data.empty:
        auc_by_ratio = auc_data.groupby('imbalance_ratio')['test_auc'].agg(['mean', 'std']).reset_index()
        
        x_pos = range(len(auc_by_ratio))
        ax.bar(x_pos, auc_by_ratio['mean'], yerr=auc_by_ratio['std'], 
               capsize=5, alpha=0.7, color='gold')
        ax.set_title('Test AUC by Imbalance Ratio')
        ax.set_xlabel('Imbalance Ratio')
        ax.set_ylabel('Mean Test AUC ± Std')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'1:{ratio:.0f}' for ratio in auc_by_ratio['imbalance_ratio']])
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (mean_auc, std_auc) in enumerate(zip(auc_by_ratio['mean'], auc_by_ratio['std'])):
            ax.text(i, mean_auc + std_auc + 0.02, f'{mean_auc:.3f}', 
                   ha='center', va='bottom', fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'No AUC data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Test AUC by Imbalance Ratio')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Imbalance analysis chart saved to {save_path}")

def generate_comprehensive_report(results_file: str = 'results/comprehensive_davidian_results.json'):
    """Generate comprehensive visualization report."""
    
    # Create plots directory if it doesn't exist
    import os
    os.makedirs('plots', exist_ok=True)
    
    print("Loading results and generating comprehensive visualization report...")
    
    # Load results
    results = load_results(results_file)
    
    # Create DataFrame from results with new metrics structure
    results_df = pd.DataFrame([
        {
            'experiment_id': r['experiment_id'],
            'sample_size': r['parameters']['sample_size'],
            'imbalance_ratio': r['parameters']['imbalance_ratio'],
            'model_type': r['parameters']['model_type'],
            'k_folds': r['parameters']['k'],
            'n_trials': r['parameters']['n_trials'],
            'regularization_method': r['parameters']['regularization_method'],
            # PRIMARY METRICS (Winner-takes-all)
            'max_method_score': r['comparison']['max_method_score'],
            'max_baseline_score': r['comparison']['max_baseline_score'],
            'max_improvement_pct': r['comparison']['max_improvement_pct'],
            'method_wins': r['comparison']['method_wins'],
            # SECONDARY METRICS (Mean ± std)
            'mean_method_score': r['comparison']['mean_method_score'],
            'mean_baseline_score': r['comparison']['mean_baseline_score'],
            'std_method_score': r['comparison']['std_method_score'],
            'std_baseline_score': r['comparison']['std_baseline_score'],
            'mean_improvement_pct': r['comparison']['mean_improvement_pct'],
            # TEST SET METRICS
            'test_accuracy': r['comparison']['test_accuracy'],
            'test_f1': r['comparison']['test_f1'],
            'test_auc': r['comparison']['test_auc'],
            # DATASET INFO
            'minority_percentage': r['dataset_metadata']['minority_percentage']
        }
        for r in results['results']
    ])
    
    print(f"Loaded {len(results_df)} experimental results")
    
    # Generate all visualization charts
    print("\n1. Creating master results chart...")
    create_master_results_chart(results_df)
    
    print("\n2. Creating method comparison chart...")
    create_method_comparison_chart(results_df)
    
    print("\n3. Creating parameter analysis charts...")
    create_parameter_analysis_charts(results_df)
    
    print("\n4. Creating imbalance analysis chart...")
    create_imbalance_analysis_chart(results_df)
    
    # Print summary statistics
    print("\n" + "="*80)
    print("COMPREHENSIVE VISUALIZATION REPORT SUMMARY")
    print("="*80)
    
    overall_stats = results['summary']['overall_statistics']
    print(f"Total experiments: {len(results_df)}")
    print(f"Overall Davidian win rate: {overall_stats['overall_win_rate']:.1f}%")
    print(f"Mean improvement: {overall_stats['mean_improvement_pct']:+.2f}%")
    print(f"Best performing method: {max(overall_stats['by_regularization_method'].items(), key=lambda x: x[1]['win_rate'])[0]}")
    
    print(f"\nAll visualization charts saved to 'plots/' directory:")
    print(f"  - plots/master_results_chart.png")
    print(f"  - plots/method_comparison.png") 
    print(f"  - plots/parameter_analysis.png")
    print(f"  - plots/imbalance_analysis.png")

if __name__ == "__main__":
    generate_comprehensive_report()
