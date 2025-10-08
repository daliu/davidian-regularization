"""
Model implementations for Davidian Regularization experiments.

This module provides wrapper classes and functions for different types of models
including linear regression, gradient boosted trees, LSTM networks, and LLMs.
"""

from typing import Any, Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# Lazy import flags - will import when needed
TENSORFLOW_AVAILABLE = None
TRANSFORMERS_AVAILABLE = None

def _check_pytorch():
    """Lazy check for PyTorch availability."""
    global TENSORFLOW_AVAILABLE  # Reusing this flag for PyTorch
    if TENSORFLOW_AVAILABLE is None:
        try:
            import torch
            import torch.nn as nn
            TENSORFLOW_AVAILABLE = True
        except ImportError:
            TENSORFLOW_AVAILABLE = False
    return TENSORFLOW_AVAILABLE

def _check_transformers():
    """Lazy check for Transformers availability."""
    global TRANSFORMERS_AVAILABLE
    if TRANSFORMERS_AVAILABLE is None:
        try:
            from transformers import pipeline
            import torch
            TRANSFORMERS_AVAILABLE = True
        except ImportError:
            TRANSFORMERS_AVAILABLE = False
    return TRANSFORMERS_AVAILABLE


class LinearModel:
    """Wrapper for linear models (regression and classification)."""
    
    def __init__(self, task_type: str = 'classification', **kwargs):
        """
        Initialize linear model.
        
        Args:
            task_type: 'classification' or 'regression'
            **kwargs: Additional parameters for the model
        """
        self.task_type = task_type
        if task_type == 'classification':
            self.model = LogisticRegression(random_state=42, max_iter=1000, **kwargs)
        else:
            self.model = LinearRegression(**kwargs)
        self.feature_names = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearModel':
        """Fit the model to training data."""
        self.model.fit(X, y)
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data."""
        return self.model.predict(X)
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities (classification only)."""
        if self.task_type == 'classification' and hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
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


class GradientBoostingModel:
    """Wrapper for gradient boosting models."""
    
    def __init__(self, task_type: str = 'classification', **kwargs):
        """
        Initialize gradient boosting model.
        
        Args:
            task_type: 'classification' or 'regression'
            **kwargs: Additional parameters for the model
        """
        self.task_type = task_type
        default_params = {'random_state': 42, 'n_estimators': 100}
        default_params.update(kwargs)
        
        if task_type == 'classification':
            self.model = GradientBoostingClassifier(**default_params)
        else:
            self.model = GradientBoostingRegressor(**default_params)
            
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GradientBoostingModel':
        """Fit the model to training data."""
        self.model.fit(X, y)
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data."""
        return self.model.predict(X)
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities (classification only)."""
        if self.task_type == 'classification':
            return self.model.predict_proba(X)
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


class LSTMModel:
    """PyTorch-based LSTM model for time series regression."""
    
    def __init__(self, sequence_length: int = 10, lstm_units: int = 50, 
                 dropout_rate: float = 0.2, **kwargs):
        """
        Initialize LSTM model.
        
        Args:
            sequence_length: Length of input sequences
            lstm_units: Number of LSTM units
            dropout_rate: Dropout rate for regularization
            **kwargs: Additional parameters
        """
        if not _check_pytorch():
            raise ImportError("PyTorch is required for LSTM models")
            
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.model = None
        self.is_fitted = False
        self.device = 'cpu'  # Use CPU to avoid M3 GPU issues
        
    def _build_model(self, input_size: int) -> None:
        """Build the PyTorch LSTM model architecture."""
        import torch
        import torch.nn as nn
        
        class SimpleLSTM(nn.Module):
            def __init__(self, input_size, hidden_size, dropout_rate):
                super(SimpleLSTM, self).__init__()
                self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
                self.dropout1 = nn.Dropout(dropout_rate)
                self.lstm2 = nn.LSTM(hidden_size, hidden_size // 2, batch_first=True)
                self.dropout2 = nn.Dropout(dropout_rate)
                self.fc = nn.Linear(hidden_size // 2, 1)
                
            def forward(self, x):
                out, _ = self.lstm1(x)
                out = self.dropout1(out)
                out, _ = self.lstm2(out)
                out = self.dropout2(out)
                out = self.fc(out[:, -1, :])  # Take last time step
                return out
        
        self.model = SimpleLSTM(input_size, self.lstm_units, self.dropout_rate)
        self.model.to(self.device)
        
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 50, 
            batch_size: int = 32, verbose: int = 0) -> 'LSTMModel':
        """Fit the LSTM model to training data."""
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
        
        if self.model is None:
            self._build_model(X.shape[2])
            
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        # Create dataset and dataloader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Set up training
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if verbose > 0 and epoch % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}')
        
        self.is_fitted = True
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        import torch
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            predictions = self.model(X_tensor)
            return predictions.cpu().numpy().flatten()
        
    def get_feature_importance(self, feature_names: Optional[List[str]] = None) -> Dict[str, float]:
        """LSTM models don't have traditional feature importance."""
        return {"note": "LSTM models don't provide traditional feature importance"}


class TextClassificationModel:
    """Simple text classification model using pre-trained embeddings."""
    
    def __init__(self, model_name: str = "distilbert-base-uncased", **kwargs):
        """
        Initialize text classification model.
        
        Args:
            model_name: Name of the pre-trained model to use
            **kwargs: Additional parameters
        """
        if not _check_transformers():
            raise ImportError("Transformers library is required for text models")
            
        self.model_name = model_name
        self.classifier = None
        self.is_fitted = False
        
    def fit(self, X: List[str], y: np.ndarray) -> 'TextClassificationModel':
        """Fit the text classification model."""
        # For simplicity, we'll use a pre-trained sentiment classifier
        # In practice, you would fine-tune on your specific data
        self.classifier = pipeline("sentiment-analysis", 
                                 model="distilbert-base-uncased-finetuned-sst-2-english")
        self.is_fitted = True
        return self
        
    def predict(self, X: List[str]) -> np.ndarray:
        """Make predictions on text data."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        predictions = []
        for text in X:
            result = self.classifier(text)[0]
            # Convert to binary: POSITIVE=1, NEGATIVE=0
            pred = 1 if result['label'] == 'POSITIVE' else 0
            predictions.append(pred)
            
        return np.array(predictions)
        
    def get_feature_importance(self, feature_names: Optional[List[str]] = None) -> Dict[str, float]:
        """Text models don't have traditional feature importance."""
        return {"note": "Text models don't provide traditional feature importance"}


class QAModel:
    """Question-answering model using pre-trained transformers."""
    
    def __init__(self, model_name: str = "distilbert-base-cased-distilled-squad", **kwargs):
        """
        Initialize QA model.
        
        Args:
            model_name: Name of the pre-trained QA model
            **kwargs: Additional parameters
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library is required for QA models")
            
        self.model_name = model_name
        self.qa_pipeline = None
        self.is_fitted = False
        
    def fit(self, X: List[str], y: List[str]) -> 'QAModel':
        """Fit the QA model (using pre-trained model)."""
        self.qa_pipeline = pipeline("question-answering", model=self.model_name)
        self.training_qa_pairs = list(zip(X, y))
        self.is_fitted = True
        return self
        
    def predict(self, X: List[str]) -> List[str]:
        """Generate answers for questions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        answers = []
        for question in X:
            # For simplicity, use a generic context
            # In practice, you'd have specific contexts for each question
            context = "This is a general knowledge context containing information about various topics including geography, mathematics, science, and general facts."
            
            try:
                result = self.qa_pipeline(question=question, context=context)
                answers.append(result['answer'])
            except:
                answers.append("Unknown")
                
        return answers
        
    def get_feature_importance(self, feature_names: Optional[List[str]] = None) -> Dict[str, float]:
        """QA models don't have traditional feature importance."""
        return {"note": "QA models don't provide traditional feature importance"}


def calculate_text_similarity(predictions: List[str], targets: List[str]) -> float:
    """
    Calculate similarity between predicted and target text using cosine similarity.
    
    Args:
        predictions: List of predicted text
        targets: List of target text
        
    Returns:
        Average cosine similarity score
    """
    if not TRANSFORMERS_AVAILABLE:
        # Fallback to simple string matching
        matches = sum(1 for p, t in zip(predictions, targets) 
                     if p.lower().strip() == t.lower().strip())
        return matches / len(predictions)
    
    try:
        # Use sentence transformers for better embeddings
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        pred_embeddings = model.encode(predictions)
        target_embeddings = model.encode(targets)
        
        similarities = []
        for pred_emb, target_emb in zip(pred_embeddings, target_embeddings):
            sim = cosine_similarity([pred_emb], [target_emb])[0][0]
            similarities.append(sim)
            
        return np.mean(similarities)
        
    except ImportError:
        # Fallback to simple string matching
        matches = sum(1 for p, t in zip(predictions, targets) 
                     if p.lower().strip() in t.lower().strip() or 
                        t.lower().strip() in p.lower().strip())
        return matches / len(predictions)


def get_model_class_and_params(model_type: str, task_type: str, 
                               skip_deep_learning: bool = False) -> Tuple[type, Dict[str, Any]]:
    """
    Get model class and default parameters for a given model type and task.
    
    Args:
        model_type: Type of model ('linear', 'gbm', 'lstm', 'text_classification', 'qa')
        task_type: Type of task ('classification', 'regression', 'time_series_regression', etc.)
        skip_deep_learning: If True, skip models that require deep learning libraries
        
    Returns:
        Tuple of (model_class, default_parameters)
    """
    if model_type == 'linear':
        return LinearModel, {'task_type': task_type}
    elif model_type == 'gbm':
        return GradientBoostingModel, {'task_type': task_type}
    elif model_type == 'lstm':
        if skip_deep_learning:
            raise ValueError("LSTM models skipped due to skip_deep_learning=True")
        if not _check_pytorch():
            raise ValueError("PyTorch not available for LSTM models")
        return LSTMModel, {'sequence_length': 10}
    elif model_type == 'text_classification':
        if skip_deep_learning:
            raise ValueError("Text classification models skipped due to skip_deep_learning=True")
        if not _check_transformers():
            raise ValueError("Transformers not available for text classification")
        return TextClassificationModel, {}
    elif model_type == 'qa':
        if skip_deep_learning:
            raise ValueError("QA models skipped due to skip_deep_learning=True")
        if not _check_transformers():
            raise ValueError("Transformers not available for QA models")
        return QAModel, {}
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def evaluate_model_performance(y_true: np.ndarray, y_pred: np.ndarray, 
                              task_type: str) -> Dict[str, Any]:
    """
    Evaluate model performance with appropriate metrics.
    
    Args:
        y_true: True labels/values
        y_pred: Predicted labels/values
        task_type: Type of task
        
    Returns:
        Dictionary containing performance metrics
    """
    metrics = {}
    
    if task_type == 'classification':
        from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
        
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Classification report
        metrics['classification_report'] = classification_report(y_true, y_pred, output_dict=True)
        
    elif task_type in ['regression', 'time_series_regression']:
        from sklearn.metrics import mean_absolute_error, r2_score
        
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['r2_score'] = r2_score(y_true, y_pred)
        
    return metrics
