"""
Data Loading Module

This module provides standardized data loading functions for both synthetic
and real-world datasets used in the Davidian Regularization research.

All functions follow consistent naming conventions and return standardized formats.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List
from sklearn.datasets import (
    make_classification, load_breast_cancer, load_wine, 
    load_digits, load_iris
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging

logger = logging.getLogger(__name__)


def load_synthetic_imbalanced_dataset(total_samples: int,
                                    imbalance_ratio: float,
                                    number_of_features: int = 20,
                                    informative_features: int = 15,
                                    redundant_features: int = 3,
                                    random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load synthetic imbalanced binary classification dataset.
    
    This function creates the controlled synthetic datasets used throughout
    the research to validate Davidian Regularization methods.
    
    Args:
        total_samples: Total number of samples to generate
        imbalance_ratio: Ratio of majority to minority class
        number_of_features: Total number of features
        informative_features: Number of informative features
        redundant_features: Number of redundant features
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (feature_matrix, target_vector, dataset_metadata)
    """
    # Calculate class weights from imbalance ratio
    minority_class_weight = 1.0 / (imbalance_ratio + 1.0)
    majority_class_weight = 1.0 - minority_class_weight
    
    # Generate synthetic dataset
    feature_matrix, target_vector = make_classification(
        n_samples=total_samples,
        n_features=number_of_features,
        n_informative=informative_features,
        n_redundant=redundant_features,
        n_clusters_per_class=1,
        weights=[majority_class_weight, minority_class_weight],
        flip_y=0.01,  # Small amount of label noise for realism
        random_state=random_state
    )
    
    # Standardize features
    feature_scaler = StandardScaler()
    standardized_feature_matrix = feature_scaler.fit_transform(feature_matrix)
    
    # Calculate actual class distribution
    unique_classes, class_counts = np.unique(target_vector, return_counts=True)
    class_distribution = dict(zip(unique_classes, class_counts))
    actual_imbalance_ratio = class_counts[0] / class_counts[1] if class_counts[1] > 0 else float('inf')
    
    # Create comprehensive metadata
    dataset_metadata = {
        'dataset_name': 'synthetic_imbalanced',
        'total_samples': total_samples,
        'number_of_features': number_of_features,
        'informative_features': informative_features,
        'redundant_features': redundant_features,
        'target_imbalance_ratio': imbalance_ratio,
        'actual_imbalance_ratio': actual_imbalance_ratio,
        'class_distribution': class_distribution,
        'minority_class_percentage': (class_counts[1] / total_samples) * 100,
        'majority_class_percentage': (class_counts[0] / total_samples) * 100,
        'feature_scaling_applied': True,
        'label_noise_rate': 0.01
    }
    
    logger.debug(f"Generated synthetic dataset: {total_samples} samples, "
                f"actual ratio 1:{actual_imbalance_ratio:.1f}")
    
    return standardized_feature_matrix, target_vector, dataset_metadata


def load_breast_cancer_dataset() -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load and preprocess Breast Cancer Wisconsin dataset.
    
    Returns:
        Tuple of (feature_matrix, target_vector, dataset_metadata)
    """
    breast_cancer_data = load_breast_cancer()
    
    # Standardize features
    feature_scaler = StandardScaler()
    standardized_features = feature_scaler.fit_transform(breast_cancer_data.data)
    
    # Calculate class distribution
    unique_classes, class_counts = np.unique(breast_cancer_data.target, return_counts=True)
    class_distribution = dict(zip(unique_classes, class_counts))
    imbalance_ratio = class_counts[0] / class_counts[1] if class_counts[1] > 0 else float('inf')
    
    dataset_metadata = {
        'dataset_name': 'breast_cancer_wisconsin',
        'source': 'sklearn.datasets',
        'total_samples': breast_cancer_data.data.shape[0],
        'number_of_features': breast_cancer_data.data.shape[1],
        'class_distribution': class_distribution,
        'imbalance_ratio': imbalance_ratio,
        'feature_names': breast_cancer_data.feature_names.tolist(),
        'target_names': breast_cancer_data.target_names.tolist(),
        'description': 'Breast cancer diagnostic dataset (malignant vs benign)',
        'feature_scaling_applied': True
    }
    
    return standardized_features, breast_cancer_data.target, dataset_metadata


def load_wine_dataset_binary() -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load Wine dataset converted to binary classification.
    
    Returns:
        Tuple of (feature_matrix, target_vector, dataset_metadata)
    """
    wine_data = load_wine()
    
    # Convert to binary: class 0 vs others
    binary_target = (wine_data.target == 0).astype(int)
    
    # Standardize features
    feature_scaler = StandardScaler()
    standardized_features = feature_scaler.fit_transform(wine_data.data)
    
    # Calculate class distribution
    unique_classes, class_counts = np.unique(binary_target, return_counts=True)
    class_distribution = dict(zip(unique_classes, class_counts))
    imbalance_ratio = class_counts[0] / class_counts[1] if class_counts[1] > 0 else float('inf')
    
    dataset_metadata = {
        'dataset_name': 'wine_recognition_binary',
        'source': 'sklearn.datasets',
        'total_samples': wine_data.data.shape[0],
        'number_of_features': wine_data.data.shape[1],
        'class_distribution': class_distribution,
        'imbalance_ratio': imbalance_ratio,
        'feature_names': wine_data.feature_names.tolist(),
        'original_classes': wine_data.target_names.tolist(),
        'binary_conversion': 'Class 0 vs Others',
        'description': 'Wine recognition dataset converted to binary classification',
        'feature_scaling_applied': True
    }
    
    return standardized_features, binary_target, dataset_metadata


def load_digits_dataset_binary() -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load Digits dataset converted to binary classification.
    
    Returns:
        Tuple of (feature_matrix, target_vector, dataset_metadata)
    """
    digits_data = load_digits()
    
    # Convert to binary: digits 0-4 vs 5-9
    binary_target = (digits_data.target >= 5).astype(int)
    
    # Standardize features
    feature_scaler = StandardScaler()
    standardized_features = feature_scaler.fit_transform(digits_data.data)
    
    # Calculate class distribution
    unique_classes, class_counts = np.unique(binary_target, return_counts=True)
    class_distribution = dict(zip(unique_classes, class_counts))
    imbalance_ratio = class_counts[0] / class_counts[1] if class_counts[1] > 0 else float('inf')
    
    dataset_metadata = {
        'dataset_name': 'digits_binary_classification',
        'source': 'sklearn.datasets',
        'total_samples': digits_data.data.shape[0],
        'number_of_features': digits_data.data.shape[1],
        'class_distribution': class_distribution,
        'imbalance_ratio': imbalance_ratio,
        'binary_conversion': 'Digits 0-4 vs 5-9',
        'description': 'Handwritten digits dataset converted to binary classification',
        'feature_scaling_applied': True
    }
    
    return standardized_features, binary_target, dataset_metadata


def load_iris_dataset_binary() -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load Iris dataset converted to binary classification.
    
    Returns:
        Tuple of (feature_matrix, target_vector, dataset_metadata)
    """
    iris_data = load_iris()
    
    # Convert to binary: Setosa vs others
    binary_target = (iris_data.target == 0).astype(int)
    
    # Standardize features
    feature_scaler = StandardScaler()
    standardized_features = feature_scaler.fit_transform(iris_data.data)
    
    # Calculate class distribution
    unique_classes, class_counts = np.unique(binary_target, return_counts=True)
    class_distribution = dict(zip(unique_classes, class_counts))
    imbalance_ratio = class_counts[0] / class_counts[1] if class_counts[1] > 0 else float('inf')
    
    dataset_metadata = {
        'dataset_name': 'iris_binary_classification',
        'source': 'sklearn.datasets',
        'total_samples': iris_data.data.shape[0],
        'number_of_features': iris_data.data.shape[1],
        'class_distribution': class_distribution,
        'imbalance_ratio': imbalance_ratio,
        'feature_names': iris_data.feature_names.tolist(),
        'original_classes': iris_data.target_names.tolist(),
        'binary_conversion': 'Setosa vs Others',
        'description': 'Iris flower dataset converted to binary classification',
        'feature_scaling_applied': True
    }
    
    return standardized_features, binary_target, dataset_metadata


def get_all_real_datasets() -> Dict[str, callable]:
    """
    Get dictionary of all available real dataset loading functions.
    
    Returns:
        Dictionary mapping dataset names to their loading functions
    """
    return {
        'breast_cancer_wisconsin': load_breast_cancer_dataset,
        'wine_recognition_binary': load_wine_dataset_binary,
        'digits_binary_classification': load_digits_dataset_binary,
        'iris_binary_classification': load_iris_dataset_binary
    }


def create_artificial_imbalance(feature_matrix: np.ndarray,
                              target_vector: np.ndarray,
                              target_imbalance_ratio: float,
                              random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Create artificially imbalanced version of a balanced dataset.
    
    Args:
        feature_matrix: Original feature matrix
        target_vector: Original target vector
        target_imbalance_ratio: Desired imbalance ratio (majority:minority)
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (imbalanced_features, imbalanced_targets, metadata)
    """
    # Identify minority and majority classes
    unique_classes, class_counts = np.unique(target_vector, return_counts=True)
    minority_class_index = np.argmin(class_counts)
    majority_class_index = 1 - minority_class_index
    
    minority_class_label = unique_classes[minority_class_index]
    majority_class_label = unique_classes[majority_class_index]
    
    # Get indices for each class
    minority_class_indices = np.where(target_vector == minority_class_label)[0]
    majority_class_indices = np.where(target_vector == majority_class_label)[0]
    
    # Calculate target sample sizes
    number_of_minority_samples = len(minority_class_indices)
    number_of_majority_samples_target = int(number_of_minority_samples * target_imbalance_ratio)
    
    # Sample majority class if we have enough samples
    if len(majority_class_indices) >= number_of_majority_samples_target:
        np.random.seed(random_state)
        selected_majority_indices = np.random.choice(
            majority_class_indices, 
            number_of_majority_samples_target, 
            replace=False
        )
        
        # Combine indices and shuffle
        combined_indices = np.concatenate([minority_class_indices, selected_majority_indices])
        np.random.shuffle(combined_indices)
        
        imbalanced_features = feature_matrix[combined_indices]
        imbalanced_targets = target_vector[combined_indices]
        
        # Calculate actual ratio
        final_class_counts = np.bincount(imbalanced_targets)
        actual_imbalance_ratio = final_class_counts[majority_class_label] / final_class_counts[minority_class_label]
        
        imbalance_metadata = {
            'target_imbalance_ratio': target_imbalance_ratio,
            'actual_imbalance_ratio': actual_imbalance_ratio,
            'original_size': len(feature_matrix),
            'imbalanced_size': len(imbalanced_features),
            'minority_class_samples': final_class_counts[minority_class_label],
            'majority_class_samples': final_class_counts[majority_class_label],
            'sampling_method': 'majority_class_undersampling'
        }
        
        return imbalanced_features, imbalanced_targets, imbalance_metadata
    
    else:
        # Not enough majority samples, return original
        logger.warning(f"Insufficient majority samples for ratio {target_imbalance_ratio}")
        return feature_matrix, target_vector, {'note': 'insufficient_samples_for_imbalance'}
