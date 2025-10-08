"""
Thread-safe model implementations that avoid mutex blocking issues on M3 Macs.

This module provides simplified model wrappers that avoid problematic libraries
and use single-threaded execution to prevent concurrency issues.
"""

from typing import Any, Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
# Force single-threaded execution BEFORE any sklearn imports
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'

# Now safe to import sklearn with single threading
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')


class ThreadSafeLinearModel:
    """Thread-safe linear model wrapper."""
    
    def __init__(self, task_type: str = 'classification', **kwargs):
        """Initialize with explicit single-threading."""
        self.task_type = task_type
        
        if task_type == 'classification':
            # Explicitly disable threading
            self.model = LogisticRegression(
                random_state=42, 
                max_iter=1000, 
                n_jobs=1,  # Single thread
                **kwargs
            )
        else:
            self.model = LinearRegression(
                n_jobs=1,  # Single thread
                **kwargs
            )
        
        self.feature_names = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ThreadSafeLinearModel':
        """Fit the model with explicit thread safety."""
        # Ensure numpy uses single thread
        old_num_threads = os.environ.get('OMP_NUM_THREADS', '1')
        os.environ['OMP_NUM_THREADS'] = '1'
        
        try:
            self.model.fit(X, y)
        finally:
            os.environ['OMP_NUM_THREADS'] = old_num_threads
            
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with thread safety."""
        old_num_threads = os.environ.get('OMP_NUM_THREADS', '1')
        os.environ['OMP_NUM_THREADS'] = '1'
        
        try:
            return self.model.predict(X)
        finally:
            os.environ['OMP_NUM_THREADS'] = old_num_threads
            
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities (classification only)."""
        if self.task_type == 'classification' and hasattr(self.model, 'predict_proba'):
            old_num_threads = os.environ.get('OMP_NUM_THREADS', '1')
            os.environ['OMP_NUM_THREADS'] = '1'
            
            try:
                return self.model.predict_proba(X)
            finally:
                os.environ['OMP_NUM_THREADS'] = old_num_threads
        else:
            raise ValueError("predict_proba only available for classification tasks")
            
    def get_feature_importance(self, feature_names: Optional[List[str]] = None) -> Dict[str, float]:
        """Get feature importance/coefficients."""
        if hasattr(self.model, 'coef_'):
            coef = self.model.coef_
            if coef.ndim > 1:
                coef = coef[0]  # For binary classification
            
            if feature_names is None:
                feature_names = [f'feature_{i}' for i in range(len(coef))]
                
            importance_dict = dict(zip(feature_names, coef))
            return dict(sorted(importance_dict.items(), key=lambda x: abs(x[1]), reverse=True))
        else:
            return {}


class ThreadSafeGradientBoostingModel:
    """Thread-safe gradient boosting model wrapper."""
    
    def __init__(self, task_type: str = 'classification', **kwargs):
        """Initialize with explicit single-threading."""
        self.task_type = task_type
        
        # Default parameters that avoid threading issues
        default_params = {
            'random_state': 42, 
            'n_estimators': 50,  # Reduced for speed and stability
            'max_depth': 3,      # Reduced complexity
            'learning_rate': 0.1
        }
        default_params.update(kwargs)
        
        if task_type == 'classification':
            self.model = GradientBoostingClassifier(**default_params)
        else:
            self.model = GradientBoostingRegressor(**default_params)
            
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ThreadSafeGradientBoostingModel':
        """Fit the model with explicit thread safety."""
        old_num_threads = os.environ.get('OMP_NUM_THREADS', '1')
        os.environ['OMP_NUM_THREADS'] = '1'
        
        try:
            self.model.fit(X, y)
        finally:
            os.environ['OMP_NUM_THREADS'] = old_num_threads
            
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with thread safety."""
        old_num_threads = os.environ.get('OMP_NUM_THREADS', '1')
        os.environ['OMP_NUM_THREADS'] = '1'
        
        try:
            return self.model.predict(X)
        finally:
            os.environ['OMP_NUM_THREADS'] = old_num_threads
            
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities (classification only)."""
        if self.task_type == 'classification':
            old_num_threads = os.environ.get('OMP_NUM_THREADS', '1')
            os.environ['OMP_NUM_THREADS'] = '1'
            
            try:
                return self.model.predict_proba(X)
            finally:
                os.environ['OMP_NUM_THREADS'] = old_num_threads
        else:
            raise ValueError("predict_proba only available for classification tasks")
            
    def get_feature_importance(self, feature_names: Optional[List[str]] = None) -> Dict[str, float]:
        """Get feature importance."""
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            
            if feature_names is None:
                feature_names = [f'feature_{i}' for i in range(len(importances))]
                
            importance_dict = dict(zip(feature_names, importances))
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        else:
            return {}


class SimpleTimeSeriesModel:
    """Simple time series model that avoids deep learning libraries."""
    
    def __init__(self, sequence_length: int = 10, **kwargs):
        """Initialize simple time series model using linear regression."""
        self.sequence_length = sequence_length
        self.model = LinearRegression(n_jobs=1)
        self.is_fitted = False
        
    def _reshape_sequences(self, X: np.ndarray) -> np.ndarray:
        """Reshape 3D sequences to 2D for linear regression."""
        if X.ndim == 3:
            # Flatten sequences: (n_samples, seq_len, features) -> (n_samples, seq_len * features)
            return X.reshape(X.shape[0], -1)
        return X
        
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'SimpleTimeSeriesModel':
        """Fit the time series model."""
        old_num_threads = os.environ.get('OMP_NUM_THREADS', '1')
        os.environ['OMP_NUM_THREADS'] = '1'
        
        try:
            X_reshaped = self._reshape_sequences(X)
            self.model.fit(X_reshaped, y)
            self.is_fitted = True
        finally:
            os.environ['OMP_NUM_THREADS'] = old_num_threads
            
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on time series data."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        old_num_threads = os.environ.get('OMP_NUM_THREADS', '1')
        os.environ['OMP_NUM_THREADS'] = '1'
        
        try:
            X_reshaped = self._reshape_sequences(X)
            return self.model.predict(X_reshaped)
        finally:
            os.environ['OMP_NUM_THREADS'] = old_num_threads
        
    def get_feature_importance(self, feature_names: Optional[List[str]] = None) -> Dict[str, float]:
        """Get feature importance from linear model coefficients."""
        if hasattr(self.model, 'coef_') and self.is_fitted:
            coef = self.model.coef_
            
            if feature_names is None:
                feature_names = [f'feature_{i}' for i in range(len(coef))]
                
            importance_dict = dict(zip(feature_names, np.abs(coef)))
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        else:
            return {"note": "Model not fitted or no coefficients available"}


def get_thread_safe_model_class_and_params(model_type: str, task_type: str) -> Tuple[type, Dict[str, Any]]:
    """
    Get thread-safe model class and parameters.
    
    Args:
        model_type: Type of model ('linear', 'gbm', 'simple_ts')
        task_type: Type of task
        
    Returns:
        Tuple of (model_class, default_parameters)
    """
    if model_type == 'linear':
        return ThreadSafeLinearModel, {'task_type': task_type}
    elif model_type == 'gbm':
        return ThreadSafeGradientBoostingModel, {'task_type': task_type}
    elif model_type == 'simple_ts':
        return SimpleTimeSeriesModel, {'sequence_length': 10}
    else:
        raise ValueError(f"Unknown thread-safe model type: {model_type}")


def force_single_thread_environment():
    """Force single-threaded execution to avoid mutex issues."""
    thread_env_vars = {
        'OMP_NUM_THREADS': '1',
        'MKL_NUM_THREADS': '1',
        'NUMEXPR_NUM_THREADS': '1',
        'OPENBLAS_NUM_THREADS': '1',
        'VECLIB_MAXIMUM_THREADS': '1',
        'NUMBA_NUM_THREADS': '1'
    }
    
    for var, value in thread_env_vars.items():
        os.environ[var] = value
    
    print("Environment configured for single-threaded execution")


# Force single-threading on import
force_single_thread_environment()
