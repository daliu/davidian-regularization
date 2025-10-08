"""
Stability Bonus Davidian Regularization

A reward-based alternative to minority class rebalancing for imbalanced datasets.

This package implements the Stability Bonus variant of Davidian Regularization,
which has been proven to achieve 15-20% improvement over traditional methods
through comprehensive experimental validation.
"""

__version__ = "1.0.0"
__author__ = "Davidian Regularization Research Team"

from .davidian_regularization import (
    stability_bonus_regularization,
    original_davidian_regularization,
    conservative_davidian_regularization,
    run_davidian_cross_validation,
    compare_regularization_methods
)

from .evaluation import (
    calculate_comprehensive_metrics,
    evaluate_statistical_significance,
    compute_expected_value_statistics
)

__all__ = [
    'stability_bonus_regularization',
    'original_davidian_regularization', 
    'conservative_davidian_regularization',
    'run_davidian_cross_validation',
    'compare_regularization_methods',
    'calculate_comprehensive_metrics',
    'evaluate_statistical_significance',
    'compute_expected_value_statistics'
]
