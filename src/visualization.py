"""
Visualization functions for Davidian Regularization experiments.

This module provides functions to create plots and charts comparing
the performance of Davidian Regularization vs random sampling.
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def plot_performance_comparison(davidian_results: Dict[str, Any],
                               random_results: Dict[str, Any],
                               metric: str = 'accuracy',
                               title: Optional[str] = None,
                               save_path: Optional[str] = None) -> Figure:
    """
    Create a bar plot comparing Davidian vs Random performance.
    
    Args:
        davidian_results: Results from Davidian Regularization
        random_results: Results from random sampling
        metric: Metric to compare
        title: Plot title (optional)
        save_path: Path to save the plot (optional)
        
    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = ['Random Sampling', 'Davidian Regularization']
    scores = [
        random_results.get(metric, 0),
        davidian_results.get(metric, 0)
    ]
    
    bars = ax.bar(methods, scores, color=['#ff7f0e', '#2ca02c'], alpha=0.8)
    
    # Add value labels on bars
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{score:.4f}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(title or f'{metric.replace("_", " ").title()} Comparison')
    ax.grid(True, alpha=0.3)
    
    # Add improvement annotation
    if scores[1] != 0 and scores[0] != 0:
        improvement = ((scores[1] - scores[0]) / abs(scores[0])) * 100
        ax.text(0.5, max(scores) * 0.9, f'Improvement: {improvement:+.2f}%',
                ha='center', va='center', transform=ax.transData,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_trial_convergence(results: Dict[str, Any], 
                          metric: str = 'mean_best_4_score',
                          title: Optional[str] = None,
                          save_path: Optional[str] = None) -> Figure:
    """
    Plot how performance converges as number of trials increases.
    
    Args:
        results: Results dictionary containing trial data
        metric: Metric to plot
        title: Plot title (optional)
        save_path: Path to save the plot (optional)
        
    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    trial_counts = [1, 10, 100, 1000]
    
    # Plot Davidian results for different k values
    if 'davidian_results' in results:
        for k in [2, 3, 4, 5]:
            k_key = f'k_{k}'
            if k_key in results['davidian_results']:
                scores = []
                for n_trials in trial_counts:
                    trial_key = f'trials_{n_trials}'
                    if trial_key in results['davidian_results'][k_key]:
                        score = results['davidian_results'][k_key][trial_key].get(metric, 0)
                        scores.append(score)
                    else:
                        scores.append(None)
                
                # Filter out None values
                valid_trials = [t for t, s in zip(trial_counts, scores) if s is not None]
                valid_scores = [s for s in scores if s is not None]
                
                if valid_scores:
                    ax.plot(valid_trials, valid_scores, marker='o', linewidth=2,
                           label=f'Davidian k={k}', alpha=0.8)
    
    # Plot Random results
    if 'random_results' in results:
        scores = []
        for n_trials in trial_counts:
            trial_key = f'trials_{n_trials}'
            if trial_key in results['random_results']:
                score = results['random_results'][trial_key].get(metric, 0)
                scores.append(score)
            else:
                scores.append(None)
        
        valid_trials = [t for t, s in zip(trial_counts, scores) if s is not None]
        valid_scores = [s for s in scores if s is not None]
        
        if valid_scores:
            ax.plot(valid_trials, valid_scores, marker='s', linewidth=2,
                   label='Random Sampling', linestyle='--', alpha=0.8)
    
    ax.set_xlabel('Number of Trials')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(title or f'{metric.replace("_", " ").title()} vs Number of Trials')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_improvement_heatmap(comparison_summary: List[Dict[str, Any]],
                            title: Optional[str] = None,
                            save_path: Optional[str] = None) -> Figure:
    """
    Create a heatmap showing improvement percentages across datasets and models.
    
    Args:
        comparison_summary: List of comparison summaries
        title: Plot title (optional)
        save_path: Path to save the plot (optional)
        
    Returns:
        Matplotlib Figure object
    """
    # Create DataFrame for heatmap
    data_for_heatmap = []
    
    for summary in comparison_summary:
        dataset = summary['dataset']
        model = summary['model_type']
        
        # Get primary metric improvement
        primary_improvement = 0
        if summary['comparison']:
            # Use the first available metric
            first_metric = list(summary['comparison'].keys())[0]
            primary_improvement = summary['comparison'][first_metric]['improvement_pct']
        
        data_for_heatmap.append({
            'Dataset': dataset,
            'Model': model,
            'Improvement (%)': primary_improvement
        })
    
    if not data_for_heatmap:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, 'No data available for heatmap', 
                ha='center', va='center', transform=ax.transAxes)
        return fig
    
    df = pd.DataFrame(data_for_heatmap)
    pivot_df = df.pivot(index='Dataset', columns='Model', values='Improvement (%)')
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create heatmap
    sns.heatmap(pivot_df, annot=True, cmap='RdYlGn', center=0,
                fmt='.2f', ax=ax, cbar_kws={'label': 'Improvement (%)'})
    
    ax.set_title(title or 'Davidian Regularization Improvement by Dataset and Model')
    ax.set_xlabel('Model Type')
    ax.set_ylabel('Dataset')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_confusion_matrix(confusion_matrix: List[List[int]], 
                         class_names: Optional[List[str]] = None,
                         title: Optional[str] = None,
                         save_path: Optional[str] = None) -> Figure:
    """
    Plot confusion matrix as a heatmap.
    
    Args:
        confusion_matrix: Confusion matrix as nested list
        class_names: Names of classes (optional)
        title: Plot title (optional)
        save_path: Path to save the plot (optional)
        
    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    cm_array = np.array(confusion_matrix)
    
    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(cm_array))]
    
    sns.heatmap(cm_array, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    
    ax.set_title(title or 'Confusion Matrix')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_feature_importance(feature_importance: Dict[str, float],
                           top_n: int = 10,
                           title: Optional[str] = None,
                           save_path: Optional[str] = None) -> Figure:
    """
    Plot feature importance as a horizontal bar chart.
    
    Args:
        feature_importance: Dictionary mapping feature names to importance scores
        top_n: Number of top features to show
        title: Plot title (optional)
        save_path: Path to save the plot (optional)
        
    Returns:
        Matplotlib Figure object
    """
    if not feature_importance or 'note' in feature_importance:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, 'Feature importance not available for this model type',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title or 'Feature Importance')
        return fig
    
    # Sort features by importance
    sorted_features = sorted(feature_importance.items(), 
                           key=lambda x: abs(x[1]), reverse=True)[:top_n]
    
    features, importances = zip(*sorted_features)
    
    fig, ax = plt.subplots(figsize=(10, max(6, len(features) * 0.5)))
    
    # Create horizontal bar plot
    bars = ax.barh(range(len(features)), importances, 
                   color=['green' if imp >= 0 else 'red' for imp in importances],
                   alpha=0.7)
    
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features)
    ax.set_xlabel('Importance Score')
    ax.set_title(title or f'Top {len(features)} Feature Importances')
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (bar, importance) in enumerate(zip(bars, importances)):
        ax.text(importance + (max(importances) - min(importances)) * 0.01,
                i, f'{importance:.4f}', va='center', ha='left' if importance >= 0 else 'right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_metric_distribution(results_list: List[Dict[str, Any]], 
                            metric: str = 'accuracy',
                            title: Optional[str] = None,
                            save_path: Optional[str] = None) -> Figure:
    """
    Plot distribution of a metric across multiple experiments.
    
    Args:
        results_list: List of result dictionaries
        metric: Metric to plot distribution for
        title: Plot title (optional)
        save_path: Path to save the plot (optional)
        
    Returns:
        Matplotlib Figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Extract metric values
    davidian_values = []
    random_values = []
    
    for result in results_list:
        if 'comparison' in result and metric in result['comparison']:
            davidian_values.append(result['comparison'][metric]['davidian_score'])
            random_values.append(result['comparison'][metric]['random_score'])
    
    if davidian_values and random_values:
        # Histogram
        ax1.hist(random_values, alpha=0.7, label='Random Sampling', bins=10, color='orange')
        ax1.hist(davidian_values, alpha=0.7, label='Davidian Regularization', bins=10, color='green')
        ax1.set_xlabel(metric.replace('_', ' ').title())
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'{metric.replace("_", " ").title()} Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        ax2.boxplot([random_values, davidian_values], 
                   labels=['Random', 'Davidian'], patch_artist=True,
                   boxprops=dict(facecolor='lightblue', alpha=0.7))
        ax2.set_ylabel(metric.replace('_', ' ').title())
        ax2.set_title(f'{metric.replace("_", " ").title()} Box Plot')
        ax2.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax1.transAxes)
        ax2.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax2.transAxes)
    
    plt.suptitle(title or f'{metric.replace("_", " ").title()} Analysis')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_comprehensive_report(all_results: List[Dict[str, Any]],
                               output_dir: str = 'plots') -> None:
    """
    Create a comprehensive visualization report.
    
    Args:
        all_results: List of all experimental results
        output_dir: Directory to save plots
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Creating comprehensive visualization report in {output_dir}/")
    
    # 1. Overall performance comparison
    if all_results:
        # Aggregate metrics
        metrics_to_plot = ['accuracy', 'f1_score', 'auc', 'r2_score']
        
        for metric in metrics_to_plot:
            metric_results = [r for r in all_results 
                            if 'comparison' in r and metric in r['comparison']]
            if metric_results:
                fig = plot_metric_distribution(metric_results, metric,
                                             title=f'{metric.replace("_", " ").title()} Across All Experiments')
                plt.savefig(f'{output_dir}/{metric}_distribution.png', dpi=300, bbox_inches='tight')
                plt.close(fig)
        
        # 2. Improvement heatmap
        fig = plot_improvement_heatmap(all_results,
                                     title='Davidian Regularization Improvement Across Experiments')
        plt.savefig(f'{output_dir}/improvement_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Visualization report completed. Check {output_dir}/ for all plots.")
    else:
        print("No results available for visualization report.")


def plot_learning_curves(train_scores: List[float], val_scores: List[float],
                        epochs: List[int], title: Optional[str] = None,
                        save_path: Optional[str] = None) -> Figure:
    """
    Plot learning curves for training and validation scores.
    
    Args:
        train_scores: Training scores over epochs
        val_scores: Validation scores over epochs
        epochs: Epoch numbers
        title: Plot title (optional)
        save_path: Path to save the plot (optional)
        
    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(epochs, train_scores, 'o-', label='Training Score', alpha=0.8)
    ax.plot(epochs, val_scores, 's-', label='Validation Score', alpha=0.8)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Score')
    ax.set_title(title or 'Learning Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
