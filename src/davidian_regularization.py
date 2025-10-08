"""
Core implementation of Davidian Regularization algorithm.

This module contains the main algorithm for Davidian Regularization,
which penalizes validation scores based on the difference between
training and validation performance to reduce bias in dataset selection.
"""

from typing import List, Tuple, Dict, Any, Optional, Callable
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score
import warnings
import logging
warnings.filterwarnings('ignore')


def calculate_davidian_penalty(train_score: float, val_score: float, 
                              alpha: float = 1.0) -> float:
    """
    Calculate the Davidian Regularization penalty.
    
    Args:
        train_score: Training score
        val_score: Validation score
        alpha: Penalty weight (default: 1.0)
        
    Returns:
        Davidian regularization penalty
    """
    return alpha * abs(train_score - val_score)


def apply_davidian_regularization(train_score: float, val_score: float,
                                 alpha: float = 1.0) -> float:
    """
    Apply Davidian Regularization to validation score.
    
    Args:
        train_score: Training score
        val_score: Validation score
        alpha: Penalty weight (default: 1.0)
        
    Returns:
        Regularized validation score
    """
    penalty = calculate_davidian_penalty(train_score, val_score, alpha)
    return val_score - penalty


def get_scoring_function(task_type: str) -> Callable:
    """
    Get appropriate scoring function based on task type.
    
    Args:
        task_type: Type of task ('classification', 'regression', 'time_series_regression')
        
    Returns:
        Scoring function
    """
    if task_type == 'classification':
        return accuracy_score
    elif task_type in ['regression', 'time_series_regression']:
        return lambda y_true, y_pred: -mean_squared_error(y_true, y_pred)  # Negative MSE for maximization
    else:
        raise ValueError(f"Unsupported task type: {task_type}")


def davidian_cross_validation(X: np.ndarray, y: np.ndarray, 
                             model: Any, k: int = 5,
                             task_type: str = 'classification',
                             alpha: float = 1.0,
                             random_state: Optional[int] = None) -> Dict[str, Any]:
    """
    Perform k-fold cross-validation with Davidian Regularization.
    
    Args:
        X: Feature matrix
        y: Target vector
        model: Model instance with fit and predict methods
        k: Number of folds
        task_type: Type of task ('classification', 'regression', 'time_series_regression')
        alpha: Davidian regularization penalty weight
        random_state: Random state for reproducibility
        
    Returns:
        Dictionary containing cross-validation results
    """
    logger = logging.getLogger('davidian_regularization')
    logger.info(f"Starting {k}-fold cross-validation (task_type={task_type}, alpha={alpha}, random_state={random_state})")
    
    scoring_func = get_scoring_function(task_type)
    logger.debug(f"Using scoring function for {task_type}")
    
    # Choose appropriate cross-validation strategy
    if task_type == 'classification':
        cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
        logger.debug("Using StratifiedKFold for classification")
    else:
        cv = KFold(n_splits=k, shuffle=True, random_state=random_state)
        logger.debug("Using KFold for regression")
    
    fold_results = []
    train_scores = []
    val_scores = []
    regularized_scores = []
    
    logger.info(f"Processing {k} folds...")
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        logger.info(f"  Fold {fold_idx + 1}/{k}: train_size={len(train_idx)}, val_size={len(val_idx)}")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Train model
        logger.debug(f"    Training model on fold {fold_idx + 1}...")
        try:
            model.fit(X_train, y_train)
            logger.debug(f"    Model training completed for fold {fold_idx + 1}")
        except Exception as e:
            logger.error(f"    Model training failed on fold {fold_idx + 1}: {e}")
            raise
        
        # Get predictions
        logger.debug(f"    Making predictions for fold {fold_idx + 1}...")
        try:
            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)
            logger.debug(f"    Predictions completed for fold {fold_idx + 1}")
        except Exception as e:
            logger.error(f"    Prediction failed on fold {fold_idx + 1}: {e}")
            raise
        
        # Calculate scores
        logger.debug(f"    Calculating scores for fold {fold_idx + 1}...")
        try:
            train_score = scoring_func(y_train, train_pred)
            val_score = scoring_func(y_val, val_pred)
            logger.debug(f"    Scores calculated: train={train_score:.6f}, val={val_score:.6f}")
        except Exception as e:
            logger.error(f"    Score calculation failed on fold {fold_idx + 1}: {e}")
            raise
        
        # Apply Davidian Regularization
        regularized_score = apply_davidian_regularization(train_score, val_score, alpha)
        penalty = calculate_davidian_penalty(train_score, val_score, alpha)
        
        logger.info(f"    Fold {fold_idx + 1} results: train={train_score:.6f}, val={val_score:.6f}, "
                   f"penalty={penalty:.6f}, regularized={regularized_score:.6f}")
        
        fold_results.append({
            'fold': fold_idx,
            'train_score': train_score,
            'val_score': val_score,
            'regularized_score': regularized_score,
            'penalty': penalty
        })
        
        train_scores.append(train_score)
        val_scores.append(val_score)
        regularized_scores.append(regularized_score)
    
    # Calculate final statistics
    mean_train = np.mean(train_scores)
    mean_val = np.mean(val_scores)
    mean_regularized = np.mean(regularized_scores)
    
    logger.info(f"Cross-validation completed: mean_train={mean_train:.6f}, "
               f"mean_val={mean_val:.6f}, mean_regularized={mean_regularized:.6f}")
    
    return {
        'fold_results': fold_results,
        'mean_train_score': mean_train,
        'std_train_score': np.std(train_scores),
        'mean_val_score': mean_val,
        'std_val_score': np.std(val_scores),
        'mean_regularized_score': mean_regularized,
        'std_regularized_score': np.std(regularized_scores),
        'k': k,
        'alpha': alpha
    }


def multiple_trial_davidian_cv(X: np.ndarray, y: np.ndarray,
                              model_class: type, model_params: Dict[str, Any],
                              k: int = 5, n_trials: int = 10,
                              task_type: str = 'classification',
                              alpha: float = 1.0) -> Dict[str, Any]:
    """
    Perform multiple trials of Davidian cross-validation.
    
    Args:
        X: Feature matrix
        y: Target vector
        model_class: Model class to instantiate
        model_params: Parameters for model instantiation
        k: Number of folds
        n_trials: Number of trials to run
        task_type: Type of task
        alpha: Davidian regularization penalty weight
        
    Returns:
        Dictionary containing results from all trials
    """
    logger = logging.getLogger('davidian_regularization')
    logger.info(f"Starting multiple trial Davidian CV: {n_trials} trials, k={k}, alpha={alpha}")
    
    trial_results = []
    all_regularized_scores = []
    
    for trial in range(n_trials):
        logger.info(f"  Trial {trial + 1}/{n_trials}")
        
        # Create new model instance for each trial
        logger.debug(f"    Creating new {model_class.__name__} instance")
        try:
            model = model_class(**model_params)
        except Exception as e:
            logger.error(f"    Failed to create model instance: {e}")
            raise
        
        # Run cross-validation
        logger.debug(f"    Running cross-validation for trial {trial + 1}")
        try:
            cv_results = davidian_cross_validation(
                X, y, model, k=k, task_type=task_type, 
                alpha=alpha, random_state=trial
            )
            
            score = cv_results['mean_regularized_score']
            logger.info(f"    Trial {trial + 1} completed: regularized_score={score:.6f}")
            
            trial_results.append(cv_results)
            all_regularized_scores.append(score)
            
        except Exception as e:
            logger.error(f"    Trial {trial + 1} failed: {e}")
            raise
    
    # Find best trials
    logger.info("Finding best 4 trials...")
    sorted_indices = np.argsort(all_regularized_scores)[::-1]  # Descending order
    best_4_indices = sorted_indices[:4]
    best_4_scores = [all_regularized_scores[i] for i in best_4_indices]
    
    logger.info(f"Best 4 trial indices: {best_4_indices.tolist()}")
    logger.info(f"Best 4 scores: {[f'{s:.6f}' for s in best_4_scores]}")
    logger.info(f"Mean of best 4: {np.mean(best_4_scores):.6f}")
    
    return {
        'trial_results': trial_results,
        'all_regularized_scores': all_regularized_scores,
        'best_4_indices': best_4_indices.tolist(),
        'best_4_scores': best_4_scores,
        'mean_best_4_score': np.mean(best_4_scores),
        'overall_mean_score': np.mean(all_regularized_scores),
        'overall_std_score': np.std(all_regularized_scores),
        'n_trials': n_trials,
        'k': k,
        'alpha': alpha
    }


def random_split_validation(X: np.ndarray, y: np.ndarray,
                           model_class: type, model_params: Dict[str, Any],
                           n_trials: int = 10, test_size: float = 0.2,
                           task_type: str = 'classification') -> Dict[str, Any]:
    """
    Perform random train-validation splits for comparison.
    
    Args:
        X: Feature matrix
        y: Target vector
        model_class: Model class to instantiate
        model_params: Parameters for model instantiation
        n_trials: Number of random splits to perform
        test_size: Fraction of data to use for validation
        task_type: Type of task
        
    Returns:
        Dictionary containing results from all random splits
    """
    from sklearn.model_selection import train_test_split
    
    logger = logging.getLogger('davidian_regularization')
    logger.info(f"Starting random split validation: {n_trials} trials, test_size={test_size}")
    
    scoring_func = get_scoring_function(task_type)
    trial_results = []
    all_val_scores = []
    
    for trial in range(n_trials):
        logger.info(f"  Random split trial {trial + 1}/{n_trials}")
        
        # Random split
        logger.debug(f"    Performing train-test split for trial {trial + 1}")
        try:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=test_size, random_state=trial,
                stratify=y if task_type == 'classification' else None
            )
            logger.debug(f"    Split completed: train_size={len(X_train)}, val_size={len(X_val)}")
        except Exception as e:
            logger.error(f"    Train-test split failed for trial {trial + 1}: {e}")
            raise
        
        # Create and train model
        logger.debug(f"    Creating and training model for trial {trial + 1}")
        try:
            model = model_class(**model_params)
            model.fit(X_train, y_train)
            logger.debug(f"    Model training completed for trial {trial + 1}")
        except Exception as e:
            logger.error(f"    Model training failed for trial {trial + 1}: {e}")
            raise
        
        # Get predictions and scores
        logger.debug(f"    Making predictions and calculating scores for trial {trial + 1}")
        try:
            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)
            
            train_score = scoring_func(y_train, train_pred)
            val_score = scoring_func(y_val, val_pred)
            
            logger.info(f"    Trial {trial + 1} completed: train={train_score:.6f}, val={val_score:.6f}")
        except Exception as e:
            logger.error(f"    Prediction/scoring failed for trial {trial + 1}: {e}")
            raise
        
        trial_results.append({
            'trial': trial,
            'train_score': train_score,
            'val_score': val_score
        })
        
        all_val_scores.append(val_score)
    
    # Find best trials
    logger.info("Finding best 4 random split trials...")
    sorted_indices = np.argsort(all_val_scores)[::-1]  # Descending order
    best_4_indices = sorted_indices[:4]
    best_4_scores = [all_val_scores[i] for i in best_4_indices]
    
    logger.info(f"Best 4 random trial indices: {best_4_indices.tolist()}")
    logger.info(f"Best 4 random scores: {[f'{s:.6f}' for s in best_4_scores]}")
    logger.info(f"Mean of best 4 random: {np.mean(best_4_scores):.6f}")
    
    return {
        'trial_results': trial_results,
        'all_val_scores': all_val_scores,
        'best_4_indices': best_4_indices.tolist(),
        'best_4_scores': best_4_scores,
        'mean_best_4_score': np.mean(best_4_scores),
        'overall_mean_score': np.mean(all_val_scores),
        'overall_std_score': np.std(all_val_scores),
        'n_trials': n_trials
    }


def compare_methods(X: np.ndarray, y: np.ndarray,
                   model_class: type, model_params: Dict[str, Any],
                   k_values: List[int] = [2, 3, 4, 5],
                   trial_counts: List[int] = [1, 10, 100, 1000],
                   task_type: str = 'classification',
                   alpha: float = 1.0) -> Dict[str, Any]:
    """
    Compare Davidian Regularization with random sampling across different parameters.
    
    Args:
        X: Feature matrix
        y: Target vector
        model_class: Model class to instantiate
        model_params: Parameters for model instantiation
        k_values: List of k values for cross-validation
        trial_counts: List of trial counts to test
        task_type: Type of task
        alpha: Davidian regularization penalty weight
        
    Returns:
        Dictionary containing comprehensive comparison results
    """
    logger = logging.getLogger('davidian_regularization')
    
    results = {
        'davidian_results': {},
        'random_results': {},
        'comparison_summary': []
    }
    
    logger.info("Starting Davidian Regularization experiments...")
    for k in k_values:
        results['davidian_results'][f'k_{k}'] = {}
        
        for n_trials in trial_counts:
            logger.info(f"Running Davidian CV: k={k}, trials={n_trials}")
            
            # Davidian Regularization
            davidian_result = multiple_trial_davidian_cv(
                X, y, model_class, model_params,
                k=k, n_trials=n_trials, task_type=task_type, alpha=alpha
            )
            results['davidian_results'][f'k_{k}'][f'trials_{n_trials}'] = davidian_result
            
            logger.debug(f"Davidian k={k}, trials={n_trials}: "
                        f"mean_score={davidian_result['mean_best_4_score']:.6f}")
    
    logger.info("Starting Random sampling experiments...")
    # Random sampling comparison
    for n_trials in trial_counts:
        logger.info(f"Running Random splits: trials={n_trials}")
        
        random_result = random_split_validation(
            X, y, model_class, model_params,
            n_trials=n_trials, task_type=task_type
        )
        results['random_results'][f'trials_{n_trials}'] = random_result
        
        logger.debug(f"Random trials={n_trials}: "
                    f"mean_score={random_result['mean_best_4_score']:.6f}")
    
    logger.info("Creating comparison summary...")
    # Create comparison summary
    for k in k_values:
        for n_trials in trial_counts:
            davidian_score = results['davidian_results'][f'k_{k}'][f'trials_{n_trials}']['mean_best_4_score']
            random_score = results['random_results'][f'trials_{n_trials}']['mean_best_4_score']
            
            improvement = davidian_score - random_score
            improvement_pct = ((davidian_score - random_score) / abs(random_score)) * 100 if random_score != 0 else 0
            
            results['comparison_summary'].append({
                'k': k,
                'n_trials': n_trials,
                'davidian_score': davidian_score,
                'random_score': random_score,
                'improvement': improvement,
                'improvement_pct': improvement_pct
            })
            
            logger.debug(f"k={k}, trials={n_trials}: Davidian={davidian_score:.6f}, "
                        f"Random={random_score:.6f}, Improvement={improvement_pct:+.2f}%")
    
    logger.info("Method comparison completed!")
    return results
