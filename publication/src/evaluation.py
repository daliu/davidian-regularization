"""
Evaluation Module: Comprehensive Statistical Analysis

This module provides functions for calculating comprehensive performance metrics,
statistical significance tests, and expected value analysis used in the research.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import logging

logger = logging.getLogger(__name__)


def calculate_comprehensive_metrics(true_labels: np.ndarray,
                                  predicted_labels: np.ndarray,
                                  predicted_probabilities: Optional[np.ndarray] = None,
                                  metric_prefix: str = '') -> Dict[str, Any]:
    """
    Calculate comprehensive performance metrics for model evaluation.
    
    This function calculates all metrics used in the research for consistent
    evaluation across experiments.
    
    Args:
        true_labels: Ground truth labels
        predicted_labels: Model predictions
        predicted_probabilities: Prediction probabilities (optional, for AUC)
        metric_prefix: Prefix for metric names (e.g., 'test_', 'validation_')
        
    Returns:
        Dictionary containing all calculated metrics
    """
    metrics = {}
    
    # Basic classification metrics
    metrics[f'{metric_prefix}accuracy'] = accuracy_score(true_labels, predicted_labels)
    metrics[f'{metric_prefix}precision'] = precision_score(true_labels, predicted_labels, 
                                                          average='weighted', zero_division=0)
    metrics[f'{metric_prefix}recall'] = recall_score(true_labels, predicted_labels, 
                                                    average='weighted', zero_division=0)
    metrics[f'{metric_prefix}f1_score'] = f1_score(true_labels, predicted_labels, 
                                                  average='weighted', zero_division=0)
    
    # AUC calculation (primary metric for research)
    if predicted_probabilities is not None and len(np.unique(true_labels)) == 2:
        try:
            if predicted_probabilities.ndim > 1:
                predicted_probabilities = predicted_probabilities[:, 1]  # Positive class
            metrics[f'{metric_prefix}auc'] = roc_auc_score(true_labels, predicted_probabilities)
        except Exception as e:
            logger.warning(f"AUC calculation failed: {e}")
            metrics[f'{metric_prefix}auc'] = None
    else:
        metrics[f'{metric_prefix}auc'] = None
    
    # Confusion matrix
    confusion_matrix_result = confusion_matrix(true_labels, predicted_labels)
    metrics[f'{metric_prefix}confusion_matrix'] = confusion_matrix_result.tolist()
    
    # Classification report
    classification_report_result = classification_report(true_labels, predicted_labels, 
                                                        output_dict=True, zero_division=0)
    metrics[f'{metric_prefix}classification_report'] = classification_report_result
    
    logger.debug(f"Calculated {len(metrics)} metrics with prefix '{metric_prefix}'")
    
    return metrics


def evaluate_statistical_significance(method_scores: np.ndarray,
                                     baseline_scores: np.ndarray,
                                     confidence_level: float = 0.95) -> Dict[str, Any]:
    """
    Evaluate statistical significance between method and baseline scores.
    
    Uses non-overlapping confidence intervals test as implemented in the research.
    
    Args:
        method_scores: Array of method performance scores
        baseline_scores: Array of baseline performance scores
        confidence_level: Confidence level for intervals (default: 0.95)
        
    Returns:
        Dictionary containing statistical significance analysis
    """
    # Calculate basic statistics
    method_mean = np.mean(method_scores)
    baseline_mean = np.mean(baseline_scores)
    
    method_std = np.std(method_scores)
    baseline_std = np.std(baseline_scores)
    
    n_trials = len(method_scores)
    
    # Standard errors
    method_se = method_std / np.sqrt(n_trials)
    baseline_se = baseline_std / np.sqrt(n_trials)
    
    # Confidence intervals
    z_score = 1.96 if confidence_level == 0.95 else 2.576  # 95% or 99%
    method_ci = z_score * method_se
    baseline_ci = z_score * baseline_se
    
    # Statistical significance test (non-overlapping confidence intervals)
    method_lower = method_mean - method_ci
    method_upper = method_mean + method_ci
    baseline_lower = baseline_mean - baseline_ci
    baseline_upper = baseline_mean + baseline_ci
    
    confidence_intervals_overlap = not (method_upper < baseline_lower or baseline_upper < method_lower)
    statistically_significant = not confidence_intervals_overlap
    
    # Effect size calculation
    pooled_std = np.sqrt(((n_trials - 1) * method_std**2 + (n_trials - 1) * baseline_std**2) / (2 * n_trials - 2))
    cohens_d = (method_mean - baseline_mean) / pooled_std if pooled_std > 0 else 0
    
    # Improvement statistics
    improvement = method_mean - baseline_mean
    improvement_percentage = (improvement / abs(baseline_mean)) * 100 if baseline_mean != 0 else 0
    
    return {
        'method_mean': method_mean,
        'baseline_mean': baseline_mean,
        'improvement': improvement,
        'improvement_percentage': improvement_percentage,
        
        'method_standard_error': method_se,
        'baseline_standard_error': baseline_se,
        'method_confidence_interval': method_ci,
        'baseline_confidence_interval': baseline_ci,
        
        'statistically_significant': statistically_significant,
        'confidence_intervals_overlap': confidence_intervals_overlap,
        'confidence_level': confidence_level,
        
        'effect_size_cohens_d': cohens_d,
        'effect_size_interpretation': interpret_effect_size(cohens_d),
        
        'method_better': improvement > 0,
        'practical_significance': abs(improvement_percentage) > 5.0  # 5% threshold
    }


def compute_expected_value_statistics(experimental_results: List[Dict[str, Any]],
                                    metric_name: str = 'expected_value_improvement_percentage') -> Dict[str, Any]:
    """
    Compute Expected Value statistics across multiple experiments.
    
    This function calculates the comprehensive statistics used throughout
    the research for aggregating results across experimental conditions.
    
    Args:
        experimental_results: List of experiment result dictionaries
        metric_name: Name of metric to analyze
        
    Returns:
        Dictionary containing Expected Value analysis
    """
    if not experimental_results:
        return {}
    
    # Extract metric values
    metric_values = []
    for result in experimental_results:
        if metric_name in result:
            metric_values.append(result[metric_name])
    
    if not metric_values:
        logger.warning(f"No values found for metric: {metric_name}")
        return {}
    
    metric_array = np.array(metric_values)
    n_experiments = len(metric_array)
    
    # Expected Value calculations
    expected_value_mean = np.mean(metric_array)
    expected_value_std = np.std(metric_array)
    expected_value_se = expected_value_std / np.sqrt(n_experiments)
    
    # Confidence intervals
    confidence_interval_95 = 1.96 * expected_value_se
    confidence_interval_99 = 2.576 * expected_value_se
    
    # Distribution statistics
    percentiles = np.percentile(metric_array, [25, 50, 75])
    
    # Success rate (positive improvements)
    success_rate = np.sum(metric_array > 0) / n_experiments * 100
    
    return {
        'metric_name': metric_name,
        'number_of_experiments': n_experiments,
        
        # Expected Value Statistics
        'expected_value_mean': expected_value_mean,
        'expected_value_standard_deviation': expected_value_std,
        'expected_value_standard_error': expected_value_se,
        
        # Confidence Intervals
        'confidence_interval_95': confidence_interval_95,
        'confidence_interval_99': confidence_interval_99,
        'confidence_interval_95_lower': expected_value_mean - confidence_interval_95,
        'confidence_interval_95_upper': expected_value_mean + confidence_interval_95,
        
        # Distribution Statistics
        'minimum_value': np.min(metric_array),
        'maximum_value': np.max(metric_array),
        'median_value': percentiles[1],
        'quartile_25': percentiles[0],
        'quartile_75': percentiles[2],
        'interquartile_range': percentiles[2] - percentiles[0],
        
        # Success Metrics
        'success_rate_percentage': success_rate,
        'positive_results_count': np.sum(metric_array > 0),
        'negative_results_count': np.sum(metric_array < 0),
        'neutral_results_count': np.sum(metric_array == 0),
        
        # Consistency Metrics
        'coefficient_of_variation': expected_value_std / (abs(expected_value_mean) + 0.001),
        'consistency_score': 1.0 / (expected_value_std + 0.001),
        
        # Raw data for further analysis
        'all_values': metric_values
    }


def interpret_effect_size(cohens_d: float) -> str:
    """
    Interpret Cohen's d effect size according to standard conventions.
    
    Args:
        cohens_d: Cohen's d effect size value
        
    Returns:
        String interpretation of effect size
    """
    abs_d = abs(cohens_d)
    
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def create_performance_summary_table(comparison_results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Create a comprehensive performance summary table for publication.
    
    Args:
        comparison_results: Results from method comparison
        
    Returns:
        DataFrame suitable for publication tables
    """
    summary_data = []
    
    for method_name, results in comparison_results.items():
        # Format method name for publication
        formatted_method_name = method_name.replace('_', ' ').title()
        if method_name == 'stability_bonus':
            formatted_method_name = '★ ' + formatted_method_name
        
        summary_data.append({
            'Method': formatted_method_name,
            'EV Mean (%)': f"{results['expected_value_improvement_percentage']:+.2f}",
            'Standard Error (%)': f"±{results['standard_error_improvement_percentage']:.3f}",
            '95% CI (%)': f"±{results['confidence_interval_regularized']:.3f}",
            'Statistically Significant': 'Yes' if results['statistically_significant'] else 'No',
            'Better than Baseline': 'Yes' if results['method_better_than_baseline'] else 'No',
            'Consistency Score': f"{1.0 / (results['standard_deviation_regularized'] + 0.001):.2f}",
            'Number of Trials': results['experimental_parameters']['number_of_trials']
        })
    
    # Sort by performance
    df = pd.DataFrame(summary_data)
    df['EV_numeric'] = df['EV Mean (%)'].str.replace('+', '').str.replace('%', '').astype(float)
    df = df.sort_values('EV_numeric', ascending=False).drop('EV_numeric', axis=1)
    
    return df
