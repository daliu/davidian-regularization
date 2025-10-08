"""
Data loading functions for various datasets used in Davidian Regularization experiments.

This module provides functions to load diverse datasets from different sources
including sklearn, Kaggle-style datasets, and HuggingFace datasets.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


def load_iris_dataset() -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load the Iris dataset from sklearn.
    
    Returns:
        Tuple containing:
        - X: Feature matrix (150, 4)
        - y: Target vector (150,)
        - metadata: Dictionary with dataset information
    """
    iris = datasets.load_iris()
    metadata = {
        'name': 'Iris',
        'type': 'classification',
        'n_samples': iris.data.shape[0],
        'n_features': iris.data.shape[1],
        'n_classes': len(iris.target_names),
        'feature_names': iris.feature_names,
        'target_names': iris.target_names,
        'description': 'Classic iris flower classification dataset'
    }
    return iris.data, iris.target, metadata


def load_wine_dataset() -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load the Wine dataset from sklearn.
    
    Returns:
        Tuple containing:
        - X: Feature matrix (178, 13)
        - y: Target vector (178,)
        - metadata: Dictionary with dataset information
    """
    wine = datasets.load_wine()
    metadata = {
        'name': 'Wine',
        'type': 'classification',
        'n_samples': wine.data.shape[0],
        'n_features': wine.data.shape[1],
        'n_classes': len(wine.target_names),
        'feature_names': wine.feature_names,
        'target_names': wine.target_names,
        'description': 'Wine recognition dataset'
    }
    return wine.data, wine.target, metadata


def load_breast_cancer_dataset() -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load the Breast Cancer Wisconsin dataset from sklearn.
    
    Returns:
        Tuple containing:
        - X: Feature matrix (569, 30)
        - y: Target vector (569,)
        - metadata: Dictionary with dataset information
    """
    cancer = datasets.load_breast_cancer()
    metadata = {
        'name': 'Breast Cancer Wisconsin',
        'type': 'classification',
        'n_samples': cancer.data.shape[0],
        'n_features': cancer.data.shape[1],
        'n_classes': len(cancer.target_names),
        'feature_names': cancer.feature_names,
        'target_names': cancer.target_names,
        'description': 'Breast cancer diagnostic dataset'
    }
    return cancer.data, cancer.target, metadata


def load_boston_housing_dataset() -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load the Boston Housing dataset (regression).
    
    Returns:
        Tuple containing:
        - X: Feature matrix (506, 13)
        - y: Target vector (506,)
        - metadata: Dictionary with dataset information
    """
    # Create synthetic Boston housing-like dataset since it's deprecated
    np.random.seed(42)
    n_samples, n_features = 506, 13
    
    # Generate correlated features
    X = np.random.randn(n_samples, n_features)
    
    # Create realistic target based on some features
    y = (
        3 * X[:, 0] +  # Crime rate effect
        -2 * X[:, 1] +  # Zoning effect
        1.5 * X[:, 2] +  # Industry effect
        np.random.randn(n_samples) * 0.5  # Noise
    )
    
    # Scale to reasonable housing price range
    y = (y - y.min()) / (y.max() - y.min()) * 40 + 10  # 10-50k range
    
    feature_names = [
        'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
        'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'
    ]
    
    metadata = {
        'name': 'Boston Housing (Synthetic)',
        'type': 'regression',
        'n_samples': n_samples,
        'n_features': n_features,
        'feature_names': feature_names,
        'description': 'Synthetic housing price regression dataset'
    }
    return X, y, metadata


def load_diabetes_dataset() -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load the Diabetes dataset from sklearn (regression).
    
    Returns:
        Tuple containing:
        - X: Feature matrix (442, 10)
        - y: Target vector (442,)
        - metadata: Dictionary with dataset information
    """
    diabetes = datasets.load_diabetes()
    metadata = {
        'name': 'Diabetes',
        'type': 'regression',
        'n_samples': diabetes.data.shape[0],
        'n_features': diabetes.data.shape[1],
        'feature_names': diabetes.feature_names,
        'description': 'Diabetes progression regression dataset'
    }
    return diabetes.data, diabetes.target, metadata


def load_time_series_dataset() -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Generate a synthetic time series dataset for LSTM testing.
    
    Returns:
        Tuple containing:
        - X: Feature sequences (1000, 10, 1) - 1000 sequences of length 10
        - y: Target values (1000,)
        - metadata: Dictionary with dataset information
    """
    np.random.seed(42)
    n_samples = 1000
    sequence_length = 10
    
    # Generate time series with trend and seasonality
    time = np.arange(n_samples + sequence_length)
    trend = 0.01 * time
    seasonal = 2 * np.sin(2 * np.pi * time / 50)  # 50-period seasonality
    noise = np.random.randn(len(time)) * 0.5
    
    series = trend + seasonal + noise
    
    # Create sequences
    X = []
    y = []
    for i in range(n_samples):
        X.append(series[i:i+sequence_length])
        y.append(series[i+sequence_length])
    
    X = np.array(X).reshape(n_samples, sequence_length, 1)
    y = np.array(y)
    
    metadata = {
        'name': 'Synthetic Time Series',
        'type': 'time_series_regression',
        'n_samples': n_samples,
        'sequence_length': sequence_length,
        'n_features': 1,
        'description': 'Synthetic time series with trend and seasonality'
    }
    return X, y, metadata


def load_text_classification_dataset() -> Tuple[list, np.ndarray, Dict[str, Any]]:
    """
    Create a synthetic text classification dataset for LLM testing.
    
    Returns:
        Tuple containing:
        - X: List of text samples
        - y: Target labels
        - metadata: Dictionary with dataset information
    """
    np.random.seed(42)
    
    # Simple sentiment-like classification
    positive_texts = [
        "This product is amazing and works perfectly",
        "I love this item, highly recommended",
        "Excellent quality and fast shipping",
        "Great value for money, very satisfied",
        "Outstanding performance, exceeded expectations",
        "Wonderful experience, will buy again",
        "Perfect solution to my problem",
        "Impressive results, very happy with purchase",
        "Top quality product, excellent service",
        "Fantastic item, works as described"
    ] * 10  # 100 positive samples
    
    negative_texts = [
        "This product is terrible and doesn't work",
        "Waste of money, very disappointed",
        "Poor quality, broke after one use",
        "Horrible experience, would not recommend",
        "Completely useless, total failure",
        "Worst purchase ever, avoid at all costs",
        "Defective product, requesting refund",
        "Terrible quality, not worth the price",
        "Broken on arrival, poor packaging",
        "Disappointing results, does not work"
    ] * 10  # 100 negative samples
    
    X = positive_texts + negative_texts
    y = np.array([1] * len(positive_texts) + [0] * len(negative_texts))
    
    # Shuffle the data
    indices = np.random.permutation(len(X))
    X = [X[i] for i in indices]
    y = y[indices]
    
    metadata = {
        'name': 'Synthetic Text Classification',
        'type': 'text_classification',
        'n_samples': len(X),
        'n_classes': 2,
        'target_names': ['negative', 'positive'],
        'description': 'Synthetic sentiment classification dataset'
    }
    return X, y, metadata


def load_qa_dataset() -> Tuple[list, list, Dict[str, Any]]:
    """
    Create a synthetic question-answering dataset for LLM testing.
    
    Returns:
        Tuple containing:
        - X: List of questions
        - y: List of expected answers
        - metadata: Dictionary with dataset information
    """
    qa_pairs = [
        ("What is the capital of France?", "Paris"),
        ("What is 2 + 2?", "4"),
        ("What color is the sky?", "Blue"),
        ("What is the largest planet?", "Jupiter"),
        ("What is H2O?", "Water"),
        ("What is the speed of light?", "299792458 meters per second"),
        ("What is the smallest prime number?", "2"),
        ("What is the chemical symbol for gold?", "Au"),
        ("What is the boiling point of water?", "100 degrees Celsius"),
        ("What is the largest mammal?", "Blue whale")
    ] * 20  # 200 QA pairs
    
    X = [pair[0] for pair in qa_pairs]
    y = [pair[1] for pair in qa_pairs]
    
    metadata = {
        'name': 'Synthetic QA Dataset',
        'type': 'question_answering',
        'n_samples': len(X),
        'description': 'Synthetic question-answering dataset'
    }
    return X, y, metadata


def get_all_datasets() -> Dict[str, callable]:
    """
    Get a dictionary of all available dataset loading functions.
    
    Returns:
        Dictionary mapping dataset names to their loading functions
    """
    return {
        'iris': load_iris_dataset,
        'wine': load_wine_dataset,
        'breast_cancer': load_breast_cancer_dataset,
        'boston_housing': load_boston_housing_dataset,
        'diabetes': load_diabetes_dataset,
        'time_series': load_time_series_dataset,
        'text_classification': load_text_classification_dataset,
        'qa_dataset': load_qa_dataset
    }


def preprocess_data(X: np.ndarray, y: np.ndarray, 
                   scale_features: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess data by scaling features and encoding labels if necessary.
    
    Args:
        X: Feature matrix
        y: Target vector
        scale_features: Whether to scale features using StandardScaler
        
    Returns:
        Tuple of preprocessed (X, y)
    """
    if scale_features and X.dtype in [np.float32, np.float64]:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    
    return X, y
