"""
Evaluation pipeline for Davidian Regularization experiments.

This module provides comprehensive evaluation functions including
metrics calculation, confusion matrices, and performance comparisons.
"""

from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, classification_report, roc_auc_score
)
import warnings
warnings.filterwarnings('ignore')


def calculate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                                   y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Calculate comprehensive classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (optional)
        
    Returns:
        Dictionary containing all classification metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
        'classification_report': classification_report(y_true, y_pred, output_dict=True)
    }
    
    # Add AUC if probabilities are provided and it's binary classification
    if y_pred_proba is not None and len(np.unique(y_true)) == 2:
        try:
            if y_pred_proba.ndim > 1:
                y_pred_proba = y_pred_proba[:, 1]  # Use positive class probabilities
            metrics['auc'] = roc_auc_score(y_true, y_pred_proba)
        except Exception as e:
            metrics['auc'] = None
            metrics['auc_error'] = str(e)
    
    return metrics


def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """
    Calculate comprehensive regression metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary containing all regression metrics
    """
    metrics = {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2_score': r2_score(y_true, y_pred)
    }
    
    # Add additional metrics
    metrics['mean_absolute_percentage_error'] = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))) * 100
    metrics['explained_variance'] = 1 - np.var(y_true - y_pred) / np.var(y_true)
    
    return metrics


def calculate_text_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    """
    Calculate text-based metrics for QA and text classification tasks.
    
    Args:
        y_true: True text/answers
        y_pred: Predicted text/answers
        
    Returns:
        Dictionary containing text metrics
    """
    # Exact match accuracy
    exact_matches = sum(1 for true, pred in zip(y_true, y_pred) 
                       if true.lower().strip() == pred.lower().strip())
    exact_match_accuracy = exact_matches / len(y_true)
    
    # Partial match accuracy (if predicted text contains true text or vice versa)
    partial_matches = sum(1 for true, pred in zip(y_true, y_pred)
                         if true.lower().strip() in pred.lower().strip() or 
                            pred.lower().strip() in true.lower().strip())
    partial_match_accuracy = partial_matches / len(y_true)
    
    # Average text length comparison
    true_lengths = [len(text.split()) for text in y_true]
    pred_lengths = [len(text.split()) for text in y_pred]
    
    metrics = {
        'exact_match_accuracy': exact_match_accuracy,
        'partial_match_accuracy': partial_match_accuracy,
        'avg_true_length': np.mean(true_lengths),
        'avg_pred_length': np.mean(pred_lengths),
        'length_difference': np.mean(pred_lengths) - np.mean(true_lengths)
    }
    
    # Try to calculate semantic similarity if possible
    try:
        from src.models import calculate_text_similarity
        metrics['semantic_similarity'] = calculate_text_similarity(y_pred, y_true)
    except:
        metrics['semantic_similarity'] = None
    
    return metrics


def evaluate_model_comprehensive(model: Any, X_test: np.ndarray, y_test: np.ndarray,
                                task_type: str, feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Perform comprehensive model evaluation.
    
    Args:
        model: Trained model instance
        X_test: Test features
        y_test: Test targets
        task_type: Type of task ('classification', 'regression', etc.)
        feature_names: Names of features (optional)
        
    Returns:
        Dictionary containing comprehensive evaluation results
    """
    results = {
        'task_type': task_type,
        'n_test_samples': len(y_test)
    }
    
    # Make predictions
    y_pred = model.predict(X_test)
    results['predictions'] = y_pred.tolist() if hasattr(y_pred, 'tolist') else list(y_pred)
    
    # Calculate metrics based on task type
    if task_type == 'classification':
        # Get probabilities if available
        y_pred_proba = None
        if hasattr(model, 'predict_proba'):
            try:
                y_pred_proba = model.predict_proba(X_test)
                results['predicted_probabilities'] = y_pred_proba.tolist()
            except:
                pass
        
        metrics = calculate_classification_metrics(y_test, y_pred, y_pred_proba)
        results.update(metrics)
        
    elif task_type in ['regression', 'time_series_regression']:
        metrics = calculate_regression_metrics(y_test, y_pred)
        results.update(metrics)
        
    elif task_type in ['text_classification', 'question_answering']:
        if isinstance(y_test[0], str):  # Text targets
            metrics = calculate_text_metrics(y_test, y_pred)
        else:  # Numeric targets for text classification
            y_pred_proba = None
            if hasattr(model, 'predict_proba'):
                try:
                    y_pred_proba = model.predict_proba(X_test)
                except:
                    pass
            metrics = calculate_classification_metrics(y_test, y_pred, y_pred_proba)
        results.update(metrics)
    
    # Get feature importance if available
    if hasattr(model, 'get_feature_importance'):
        try:
            feature_importance = model.get_feature_importance(feature_names)
            results['feature_importance'] = feature_importance
        except:
            results['feature_importance'] = {}
    
    return results


def compare_model_performance(results_dict: Dict[str, Dict[str, Any]], 
                             primary_metric: str = 'accuracy') -> pd.DataFrame:
    """
    Compare performance across different models/methods.
    
    Args:
        results_dict: Dictionary mapping method names to their results
        primary_metric: Primary metric to use for comparison
        
    Returns:
        DataFrame with comparison results
    """
    comparison_data = []
    
    for method_name, results in results_dict.items():
        row = {'method': method_name}
        
        # Extract key metrics
        if primary_metric in results:
            row[primary_metric] = results[primary_metric]
        
        # Add other common metrics
        for metric in ['accuracy', 'f1_score', 'auc', 'mse', 'r2_score', 
                      'exact_match_accuracy', 'semantic_similarity']:
            if metric in results:
                row[metric] = results[metric]
        
        # Add sample size
        if 'n_test_samples' in results:
            row['n_test_samples'] = results['n_test_samples']
            
        comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    
    # Sort by primary metric (descending for most metrics, ascending for error metrics)
    ascending = primary_metric in ['mse', 'rmse', 'mae']
    if primary_metric in df.columns:
        df = df.sort_values(primary_metric, ascending=ascending)
    
    return df


def create_performance_summary(davidian_results: Dict[str, Any],
                              random_results: Dict[str, Any],
                              dataset_name: str, model_type: str) -> Dict[str, Any]:
    """
    Create a comprehensive performance summary comparing methods.
    
    Args:
        davidian_results: Results from Davidian Regularization
        random_results: Results from random sampling
        dataset_name: Name of the dataset
        model_type: Type of model used
        
    Returns:
        Dictionary containing performance summary
    """
    summary = {
        'dataset': dataset_name,
        'model_type': model_type,
        'comparison': {}
    }
    
    # Extract key metrics for comparison
    metrics_to_compare = []
    
    # Determine which metrics to compare based on what's available
    for metric in ['accuracy', 'f1_score', 'auc', 'r2_score', 'mse', 
                   'exact_match_accuracy', 'semantic_similarity']:
        if metric in davidian_results and metric in random_results:
            metrics_to_compare.append(metric)
    
    # Compare each metric
    for metric in metrics_to_compare:
        davidian_score = davidian_results[metric]
        random_score = random_results[metric]
        
        # Calculate improvement (handle None values)
        if davidian_score is not None and random_score is not None:
            if metric in ['mse', 'rmse', 'mae']:  # Lower is better
                improvement = random_score - davidian_score
                improvement_pct = (improvement / abs(random_score)) * 100 if random_score != 0 else 0
            else:  # Higher is better
                improvement = davidian_score - random_score
                improvement_pct = (improvement / abs(random_score)) * 100 if random_score != 0 else 0
            
            summary['comparison'][metric] = {
                'davidian_score': davidian_score,
                'random_score': random_score,
                'improvement': improvement,
                'improvement_pct': improvement_pct,
                'davidian_better': improvement > 0
            }
    
    # Overall assessment
    better_count = sum(1 for comp in summary['comparison'].values() 
                      if comp['davidian_better'])
    total_metrics = len(summary['comparison'])
    
    summary['overall'] = {
        'metrics_compared': total_metrics,
        'davidian_better_count': better_count,
        'davidian_better_pct': (better_count / total_metrics) * 100 if total_metrics > 0 else 0,
        'overall_better': better_count > total_metrics / 2
    }
    
    return summary


def aggregate_results_across_datasets(all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate results across multiple datasets and models.
    
    Args:
        all_results: List of performance summaries from different experiments
        
    Returns:
        Dictionary containing aggregated results
    """
    aggregated = {
        'total_experiments': len(all_results),
        'datasets_tested': list(set(result['dataset'] for result in all_results)),
        'models_tested': list(set(result['model_type'] for result in all_results)),
        'overall_performance': {},
        'by_dataset': {},
        'by_model': {}
    }
    
    # Overall performance across all experiments
    total_better = sum(1 for result in all_results if result['overall']['overall_better'])
    aggregated['overall_performance'] = {
        'experiments_where_davidian_better': total_better,
        'success_rate_pct': (total_better / len(all_results)) * 100,
        'avg_improvement_pct': np.mean([
            result['overall']['davidian_better_pct'] for result in all_results
        ])
    }
    
    # Performance by dataset
    for dataset in aggregated['datasets_tested']:
        dataset_results = [r for r in all_results if r['dataset'] == dataset]
        dataset_better = sum(1 for r in dataset_results if r['overall']['overall_better'])
        
        aggregated['by_dataset'][dataset] = {
            'experiments': len(dataset_results),
            'davidian_better_count': dataset_better,
            'success_rate_pct': (dataset_better / len(dataset_results)) * 100
        }
    
    # Performance by model
    for model in aggregated['models_tested']:
        model_results = [r for r in all_results if r['model_type'] == model]
        model_better = sum(1 for r in model_results if r['overall']['overall_better'])
        
        aggregated['by_model'][model] = {
            'experiments': len(model_results),
            'davidian_better_count': model_better,
            'success_rate_pct': (model_better / len(model_results)) * 100
        }
    
    return aggregated


def save_results_to_csv(results: Dict[str, Any], filename: str) -> None:
    """
    Save results to CSV file for further analysis.
    
    Args:
        results: Results dictionary to save
        filename: Output filename
    """
    # Flatten the results for CSV export
    flattened_data = []
    
    if 'comparison' in results:
        for metric, data in results['comparison'].items():
            row = {
                'dataset': results.get('dataset', 'unknown'),
                'model_type': results.get('model_type', 'unknown'),
                'metric': metric,
                'davidian_score': data['davidian_score'],
                'random_score': data['random_score'],
                'improvement': data['improvement'],
                'improvement_pct': data['improvement_pct'],
                'davidian_better': data['davidian_better']
            }
            flattened_data.append(row)
    
    if flattened_data:
        df = pd.DataFrame(flattened_data)
        df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")
    else:
        print("No data to save to CSV")
