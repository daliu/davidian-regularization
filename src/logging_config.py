"""
Logging configuration for Davidian Regularization experiments.

This module provides comprehensive logging setup to track experiment progress,
debug issues, and maintain detailed records of all operations.
"""

import logging
import os
import sys
from datetime import datetime
from typing import Optional
import json


def setup_logging(log_level: str = 'INFO', 
                 log_file: Optional[str] = None,
                 console_output: bool = True) -> logging.Logger:
    """
    Set up comprehensive logging for the experiment.
    
    Args:
        log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        log_file: Path to log file (optional, defaults to timestamped file)
        console_output: Whether to also output to console
        
    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Generate default log file name if not provided
    if log_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = f'logs/davidian_experiment_{timestamp}.log'
    
    # Configure logging
    logger = logging.getLogger('davidian_regularization')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(getattr(logging, log_level.upper()))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Log initial setup
    logger.info("="*80)
    logger.info("DAVIDIAN REGULARIZATION EXPERIMENT LOGGING STARTED")
    logger.info("="*80)
    logger.info(f"Log level: {log_level}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Console output: {console_output}")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Working directory: {os.getcwd()}")
    
    return logger


def log_experiment_config(logger: logging.Logger, config: dict) -> None:
    """
    Log experiment configuration details.
    
    Args:
        logger: Logger instance
        config: Configuration dictionary
    """
    logger.info("EXPERIMENT CONFIGURATION:")
    logger.info("-" * 40)
    for key, value in config.items():
        logger.info(f"{key}: {value}")
    logger.info("-" * 40)


def log_dataset_info(logger: logging.Logger, dataset_name: str, metadata: dict) -> None:
    """
    Log dataset information.
    
    Args:
        logger: Logger instance
        dataset_name: Name of the dataset
        metadata: Dataset metadata
    """
    logger.info(f"LOADING DATASET: {dataset_name}")
    logger.info(f"  Name: {metadata.get('name', 'Unknown')}")
    logger.info(f"  Type: {metadata.get('type', 'Unknown')}")
    logger.info(f"  Samples: {metadata.get('n_samples', 'Unknown')}")
    logger.info(f"  Features: {metadata.get('n_features', 'Unknown')}")
    if 'n_classes' in metadata:
        logger.info(f"  Classes: {metadata['n_classes']}")
    logger.info(f"  Description: {metadata.get('description', 'No description')}")


def log_model_info(logger: logging.Logger, model_type: str, model_params: dict) -> None:
    """
    Log model information.
    
    Args:
        logger: Logger instance
        model_type: Type of model
        model_params: Model parameters
    """
    logger.info(f"INITIALIZING MODEL: {model_type}")
    logger.info(f"  Parameters: {model_params}")


def log_cross_validation_start(logger: logging.Logger, k: int, n_trials: int, 
                              alpha: float, method: str) -> None:
    """
    Log cross-validation start.
    
    Args:
        logger: Logger instance
        k: Number of folds
        n_trials: Number of trials
        alpha: Regularization parameter
        method: Method name (Davidian or Random)
    """
    logger.info(f"STARTING {method.upper()} CROSS-VALIDATION:")
    logger.info(f"  K-folds: {k}")
    logger.info(f"  Trials: {n_trials}")
    if method.lower() == 'davidian':
        logger.info(f"  Alpha (penalty weight): {alpha}")


def log_trial_progress(logger: logging.Logger, trial: int, total_trials: int, 
                      score: float, method: str) -> None:
    """
    Log progress of individual trials.
    
    Args:
        logger: Logger instance
        trial: Current trial number
        total_trials: Total number of trials
        score: Score achieved in this trial
        method: Method name
    """
    progress_pct = ((trial + 1) / total_trials) * 100
    logger.debug(f"{method} Trial {trial + 1}/{total_trials} ({progress_pct:.1f}%): Score = {score:.6f}")


def log_fold_results(logger: logging.Logger, fold: int, train_score: float, 
                    val_score: float, regularized_score: Optional[float] = None) -> None:
    """
    Log results from individual folds.
    
    Args:
        logger: Logger instance
        fold: Fold number
        train_score: Training score
        val_score: Validation score
        regularized_score: Davidian regularized score (optional)
    """
    logger.debug(f"  Fold {fold}: Train={train_score:.6f}, Val={val_score:.6f}")
    if regularized_score is not None:
        penalty = abs(train_score - val_score)
        logger.debug(f"    Penalty={penalty:.6f}, Regularized={regularized_score:.6f}")


def log_method_results(logger: logging.Logger, method: str, results: dict) -> None:
    """
    Log final results from a method.
    
    Args:
        logger: Logger instance
        method: Method name
        results: Results dictionary
    """
    logger.info(f"{method.upper()} RESULTS:")
    
    if 'mean_best_4_score' in results:
        logger.info(f"  Best 4 Mean Score: {results['mean_best_4_score']:.6f}")
    if 'overall_mean_score' in results:
        logger.info(f"  Overall Mean Score: {results['overall_mean_score']:.6f}")
        logger.info(f"  Overall Std Score: {results['overall_std_score']:.6f}")
    
    if 'best_4_indices' in results:
        logger.info(f"  Best 4 Trial Indices: {results['best_4_indices']}")
        logger.info(f"  Best 4 Scores: {[f'{s:.6f}' for s in results['best_4_scores']]}")


def log_comparison_results(logger: logging.Logger, davidian_results: dict, 
                          random_results: dict, dataset_name: str, model_type: str) -> None:
    """
    Log comparison between Davidian and Random methods.
    
    Args:
        logger: Logger instance
        davidian_results: Davidian regularization results
        random_results: Random sampling results
        dataset_name: Name of dataset
        model_type: Type of model
    """
    logger.info("="*60)
    logger.info(f"COMPARISON RESULTS: {dataset_name} + {model_type}")
    logger.info("="*60)
    
    # Extract key metrics for comparison
    for trial_key in ['trials_1', 'trials_10', 'trials_100', 'trials_1000']:
        if trial_key in random_results:
            random_score = random_results[trial_key]['mean_best_4_score']
            logger.info(f"{trial_key.replace('_', ' ').title()}:")
            logger.info(f"  Random Sampling: {random_score:.6f}")
            
            # Find best Davidian result for this trial count
            best_davidian_score = 0
            best_k = None
            for k in [2, 3, 4, 5]:
                k_key = f'k_{k}'
                if k_key in davidian_results and trial_key in davidian_results[k_key]:
                    davidian_score = davidian_results[k_key][trial_key]['mean_best_4_score']
                    if davidian_score > best_davidian_score:
                        best_davidian_score = davidian_score
                        best_k = k
            
            if best_k is not None:
                improvement = best_davidian_score - random_score
                improvement_pct = (improvement / abs(random_score)) * 100 if random_score != 0 else 0
                logger.info(f"  Davidian (k={best_k}): {best_davidian_score:.6f}")
                logger.info(f"  Improvement: {improvement:+.6f} ({improvement_pct:+.2f}%)")
                
                if improvement > 0:
                    logger.info("  ✓ DAVIDIAN BETTER")
                else:
                    logger.info("  ✗ Random better")
            logger.info("")


def log_error(logger: logging.Logger, error: Exception, context: str) -> None:
    """
    Log error with context information.
    
    Args:
        logger: Logger instance
        error: Exception that occurred
        context: Context where error occurred
    """
    logger.error(f"ERROR in {context}: {type(error).__name__}: {str(error)}")
    logger.debug(f"Error details:", exc_info=True)


def log_experiment_summary(logger: logging.Logger, total_experiments: int, 
                          successful_experiments: int, total_time: float,
                          success_rate: Optional[float] = None) -> None:
    """
    Log final experiment summary.
    
    Args:
        logger: Logger instance
        total_experiments: Total number of experiments planned
        successful_experiments: Number of successful experiments
        total_time: Total time taken in seconds
        success_rate: Overall success rate of Davidian method (optional)
    """
    logger.info("="*80)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("="*80)
    logger.info(f"Total experiments planned: {total_experiments}")
    logger.info(f"Successful experiments: {successful_experiments}")
    logger.info(f"Failed experiments: {total_experiments - successful_experiments}")
    logger.info(f"Success rate: {(successful_experiments/total_experiments)*100:.1f}%")
    logger.info(f"Total time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
    
    if success_rate is not None:
        logger.info(f"Davidian Regularization success rate: {success_rate:.1f}%")
    
    logger.info("="*80)
    logger.info("EXPERIMENT COMPLETED")
    logger.info("="*80)


def save_results_with_logging(logger: logging.Logger, results: dict, 
                             filename: str) -> None:
    """
    Save results to file with logging.
    
    Args:
        logger: Logger instance
        results: Results to save
        filename: Output filename
    """
    try:
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved successfully to: {filename}")
    except Exception as e:
        log_error(logger, e, f"saving results to {filename}")


class ExperimentTimer:
    """Context manager for timing operations with logging."""
    
    def __init__(self, logger: logging.Logger, operation_name: str):
        self.logger = logger
        self.operation_name = operation_name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.info(f"STARTING: {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        if exc_type is None:
            self.logger.info(f"COMPLETED: {self.operation_name} (took {duration:.2f}s)")
        else:
            self.logger.error(f"FAILED: {self.operation_name} (took {duration:.2f}s)")
            self.logger.error(f"Error: {exc_type.__name__}: {exc_val}")


# Convenience function to get logger
def get_logger() -> logging.Logger:
    """Get the configured logger instance."""
    return logging.getLogger('davidian_regularization')
