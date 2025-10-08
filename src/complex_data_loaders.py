"""
Complex dataset loaders for more challenging Davidian Regularization experiments.

This module provides functions to load more complex datasets from various sources
including sklearn's more challenging datasets, synthetic datasets with controlled
complexity, and datasets that simulate real-world ML challenges.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.datasets import make_classification, make_regression, make_blobs
import warnings
warnings.filterwarnings('ignore')


def load_digits_dataset() -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load the handwritten digits dataset (8x8 images).
    More complex than Iris/Wine with 64 features and 10 classes.
    """
    from sklearn.datasets import load_digits
    
    digits = load_digits()
    metadata = {
        'name': 'Handwritten Digits',
        'type': 'classification',
        'n_samples': digits.data.shape[0],
        'n_features': digits.data.shape[1],
        'n_classes': len(digits.target_names),
        'feature_names': [f'pixel_{i}' for i in range(digits.data.shape[1])],
        'target_names': digits.target_names,
        'description': '8x8 pixel handwritten digit recognition (0-9)',
        'complexity': 'high'
    }
    return digits.data, digits.target, metadata


def load_olivetti_faces_dataset() -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load the Olivetti faces dataset for face recognition.
    High-dimensional dataset with 400 samples, 4096 features, 40 classes.
    """
    from sklearn.datasets import fetch_olivetti_faces
    
    faces = fetch_olivetti_faces(shuffle=True, random_state=42)
    metadata = {
        'name': 'Olivetti Faces',
        'type': 'classification',
        'n_samples': faces.data.shape[0],
        'n_features': faces.data.shape[1],
        'n_classes': len(np.unique(faces.target)),
        'feature_names': [f'pixel_{i}' for i in range(faces.data.shape[1])],
        'description': '64x64 pixel face recognition dataset (40 people)',
        'complexity': 'very_high'
    }
    return faces.data, faces.target, metadata


def load_20newsgroups_vectorized() -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load a vectorized version of 20 newsgroups dataset.
    Text classification with TF-IDF features.
    """
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    # Load subset of categories to make it manageable
    categories = [
        'alt.atheism',
        'comp.graphics', 
        'comp.os.ms-windows.misc',
        'comp.sys.ibm.pc.hardware',
        'rec.autos',
        'rec.motorcycles',
        'sci.crypt',
        'sci.electronics'
    ]
    
    # Load training data
    newsgroups_train = fetch_20newsgroups(
        subset='train',
        categories=categories,
        shuffle=True,
        random_state=42,
        remove=('headers', 'footers', 'quotes')
    )
    
    # Vectorize text data
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', max_df=0.95, min_df=2)
    X = vectorizer.fit_transform(newsgroups_train.data).toarray()
    y = newsgroups_train.target
    
    metadata = {
        'name': '20 Newsgroups (Subset)',
        'type': 'classification',
        'n_samples': X.shape[0],
        'n_features': X.shape[1],
        'n_classes': len(categories),
        'feature_names': vectorizer.get_feature_names_out().tolist(),
        'target_names': [newsgroups_train.target_names[i] for i in range(len(categories))],
        'description': 'Text classification on newsgroup posts (TF-IDF features)',
        'complexity': 'high'
    }
    return X, y, metadata


def load_california_housing_dataset() -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load the California housing dataset for regression.
    More complex than diabetes with 20,640 samples and 8 features.
    """
    from sklearn.datasets import fetch_california_housing
    
    housing = fetch_california_housing()
    metadata = {
        'name': 'California Housing',
        'type': 'regression',
        'n_samples': housing.data.shape[0],
        'n_features': housing.data.shape[1],
        'feature_names': housing.feature_names,
        'description': 'California housing prices regression dataset',
        'complexity': 'medium'
    }
    return housing.data, housing.target, metadata


def load_synthetic_complex_classification() -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Create a synthetic complex classification dataset with controlled difficulty.
    """
    np.random.seed(42)
    
    # Create a challenging classification problem
    X, y = make_classification(
        n_samples=2000,
        n_features=20,
        n_informative=15,
        n_redundant=3,
        n_clusters_per_class=2,
        n_classes=5,
        class_sep=0.8,  # Moderate class separation
        flip_y=0.02,    # 2% label noise
        random_state=42
    )
    
    metadata = {
        'name': 'Synthetic Complex Classification',
        'type': 'classification',
        'n_samples': X.shape[0],
        'n_features': X.shape[1],
        'n_classes': 5,
        'feature_names': [f'feature_{i}' for i in range(X.shape[1])],
        'target_names': [f'class_{i}' for i in range(5)],
        'description': 'Synthetic classification with moderate difficulty and label noise',
        'complexity': 'medium'
    }
    return X, y, metadata


def load_synthetic_complex_regression() -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Create a synthetic complex regression dataset with controlled difficulty.
    """
    np.random.seed(42)
    
    # Create a challenging regression problem
    X, y = make_regression(
        n_samples=2000,
        n_features=15,
        n_informative=10,
        noise=0.2,      # 20% noise level
        bias=100.0,
        random_state=42
    )
    
    metadata = {
        'name': 'Synthetic Complex Regression',
        'type': 'regression',
        'n_samples': X.shape[0],
        'n_features': X.shape[1],
        'feature_names': [f'feature_{i}' for i in range(X.shape[1])],
        'description': 'Synthetic regression with moderate noise and complexity',
        'complexity': 'medium'
    }
    return X, y, metadata


def load_covtype_subset_dataset() -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load a subset of the Forest Cover Type dataset.
    Large-scale classification problem.
    """
    from sklearn.datasets import fetch_covtype
    
    # Load full dataset
    covtype = fetch_covtype()
    
    # Take a manageable subset (first 10,000 samples)
    n_subset = 10000
    indices = np.random.RandomState(42).choice(covtype.data.shape[0], n_subset, replace=False)
    
    X = covtype.data[indices]
    y = covtype.target[indices] - 1  # Convert to 0-based indexing
    
    metadata = {
        'name': 'Forest Cover Type (Subset)',
        'type': 'classification',
        'n_samples': X.shape[0],
        'n_features': X.shape[1],
        'n_classes': len(np.unique(y)),
        'feature_names': [f'feature_{i}' for i in range(X.shape[1])],
        'target_names': [f'cover_type_{i}' for i in range(len(np.unique(y)))],
        'description': 'Forest cover type prediction (subset of 10k samples)',
        'complexity': 'high'
    }
    return X, y, metadata


def load_kddcup99_subset_dataset() -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load a subset of the KDD Cup 99 network intrusion detection dataset.
    """
    from sklearn.datasets import fetch_kddcup99
    
    # Load subset for manageability
    kdd = fetch_kddcup99(subset='SA', percent10=True, shuffle=True, random_state=42)
    
    # Encode categorical features
    X = kdd.data
    y = kdd.target
    
    # Convert to numeric (simple approach)
    from sklearn.preprocessing import LabelEncoder
    
    # Handle mixed data types
    X_processed = []
    for i in range(X.shape[1]):
        column = X[:, i]
        if column.dtype == object:
            le = LabelEncoder()
            column_encoded = le.fit_transform(column.astype(str))
            X_processed.append(column_encoded)
        else:
            X_processed.append(column.astype(float))
    
    X_final = np.column_stack(X_processed)
    
    # Encode labels
    le_target = LabelEncoder()
    y_final = le_target.fit_transform(y.astype(str))
    
    metadata = {
        'name': 'KDD Cup 99 Network Intrusion (Subset)',
        'type': 'classification',
        'n_samples': X_final.shape[0],
        'n_features': X_final.shape[1],
        'n_classes': len(np.unique(y_final)),
        'feature_names': [f'network_feature_{i}' for i in range(X_final.shape[1])],
        'target_names': le_target.classes_.tolist(),
        'description': 'Network intrusion detection dataset',
        'complexity': 'high'
    }
    return X_final, y_final, metadata


def load_adult_income_dataset() -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load the Adult Income dataset (>50K income prediction).
    Classic dataset with mixed categorical and numerical features.
    """
    # Create a synthetic version similar to Adult dataset
    np.random.seed(42)
    
    n_samples = 5000
    
    # Simulate adult dataset features
    age = np.random.randint(17, 90, n_samples)
    education_num = np.random.randint(1, 17, n_samples)
    hours_per_week = np.random.randint(1, 100, n_samples)
    capital_gain = np.random.exponential(scale=100, size=n_samples)
    capital_loss = np.random.exponential(scale=50, size=n_samples)
    
    # Create some correlations
    income_score = (
        0.3 * (age - 17) / (90 - 17) +
        0.4 * (education_num - 1) / (17 - 1) +
        0.2 * (hours_per_week - 1) / (100 - 1) +
        0.1 * np.tanh(capital_gain / 1000) +
        np.random.normal(0, 0.2, n_samples)
    )
    
    # Convert to binary classification (>50K income)
    y = (income_score > 0.5).astype(int)
    
    X = np.column_stack([age, education_num, hours_per_week, capital_gain, capital_loss])
    
    metadata = {
        'name': 'Adult Income (Synthetic)',
        'type': 'classification',
        'n_samples': n_samples,
        'n_features': X.shape[1],
        'n_classes': 2,
        'feature_names': ['age', 'education_num', 'hours_per_week', 'capital_gain', 'capital_loss'],
        'target_names': ['<=50K', '>50K'],
        'description': 'Income prediction based on demographic features',
        'complexity': 'medium'
    }
    return X, y, metadata


def load_imbalanced_classification_dataset() -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Create an imbalanced classification dataset to test robustness.
    """
    np.random.seed(42)
    
    # Create imbalanced dataset
    X, y = make_classification(
        n_samples=3000,
        n_features=20,
        n_informative=15,
        n_redundant=2,
        n_clusters_per_class=1,
        n_classes=3,
        weights=[0.7, 0.2, 0.1],  # Imbalanced classes
        class_sep=0.6,
        flip_y=0.01,
        random_state=42
    )
    
    metadata = {
        'name': 'Imbalanced Classification',
        'type': 'classification',
        'n_samples': X.shape[0],
        'n_features': X.shape[1],
        'n_classes': 3,
        'feature_names': [f'feature_{i}' for i in range(X.shape[1])],
        'target_names': ['majority_class', 'minority_class_1', 'minority_class_2'],
        'description': 'Imbalanced classification with 70/20/10% class distribution',
        'complexity': 'high',
        'class_distribution': [0.7, 0.2, 0.1]
    }
    return X, y, metadata


def load_noisy_regression_dataset() -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Create a noisy regression dataset with outliers.
    """
    np.random.seed(42)
    
    # Base regression problem
    X, y = make_regression(
        n_samples=2500,
        n_features=12,
        n_informative=8,
        noise=0.3,
        bias=50.0,
        random_state=42
    )
    
    # Add outliers (5% of data)
    n_outliers = int(0.05 * len(y))
    outlier_indices = np.random.choice(len(y), n_outliers, replace=False)
    y[outlier_indices] += np.random.normal(0, 3 * np.std(y), n_outliers)
    
    metadata = {
        'name': 'Noisy Regression with Outliers',
        'type': 'regression',
        'n_samples': X.shape[0],
        'n_features': X.shape[1],
        'feature_names': [f'feature_{i}' for i in range(X.shape[1])],
        'description': 'Regression with 30% noise and 5% outliers',
        'complexity': 'high',
        'outlier_percentage': 5.0
    }
    return X, y, metadata


def load_multicollinear_regression_dataset() -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Create a regression dataset with multicollinearity issues.
    """
    np.random.seed(42)
    
    n_samples = 1500
    n_base_features = 5
    
    # Create base features
    X_base = np.random.randn(n_samples, n_base_features)
    
    # Add correlated features (multicollinearity)
    X_corr1 = X_base[:, 0:2] + 0.1 * np.random.randn(n_samples, 2)  # Correlated with first 2
    X_corr2 = X_base[:, 1:3] + 0.15 * np.random.randn(n_samples, 2)  # Correlated with 2nd and 3rd
    X_corr3 = 0.7 * X_base[:, 0] + 0.3 * X_base[:, 2] + 0.1 * np.random.randn(n_samples)  # Linear combination
    
    # Combine all features
    X = np.column_stack([X_base, X_corr1, X_corr2, X_corr3.reshape(-1, 1)])
    
    # Create target with complex relationships
    y = (
        2.0 * X[:, 0] +
        -1.5 * X[:, 1] +
        0.8 * X[:, 2] +
        0.5 * X[:, 0] * X[:, 1] +  # Interaction term
        0.3 * np.random.randn(n_samples)  # Noise
    )
    
    metadata = {
        'name': 'Multicollinear Regression',
        'type': 'regression',
        'n_samples': X.shape[0],
        'n_features': X.shape[1],
        'feature_names': [f'feature_{i}' for i in range(X.shape[1])],
        'description': 'Regression with multicollinearity and interaction effects',
        'complexity': 'high'
    }
    return X, y, metadata


def load_time_varying_dataset() -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Create a dataset where the relationship changes over time (concept drift).
    """
    np.random.seed(42)
    
    n_samples = 3000
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    
    # Create time-varying target
    time_points = np.linspace(0, 1, n_samples)
    
    # Coefficients that change over time
    coef_base = np.array([1.0, -0.5, 0.8, -0.3, 0.6, -0.2, 0.4, -0.1, 0.3, -0.4])
    
    y = np.zeros(n_samples)
    for i in range(n_samples):
        # Coefficients drift over time
        time_factor = time_points[i]
        coef_drift = coef_base * (1 + 0.5 * np.sin(2 * np.pi * time_factor))
        
        y[i] = np.dot(X[i], coef_drift) + 0.3 * np.random.randn()
    
    metadata = {
        'name': 'Time-Varying Regression (Concept Drift)',
        'type': 'regression',
        'n_samples': X.shape[0],
        'n_features': X.shape[1],
        'feature_names': [f'feature_{i}' for i in range(X.shape[1])],
        'description': 'Regression with concept drift - relationships change over time',
        'complexity': 'very_high'
    }
    return X, y, metadata


def load_high_dimensional_low_sample_dataset() -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Create a high-dimensional, low-sample dataset (curse of dimensionality).
    """
    np.random.seed(42)
    
    n_samples = 200  # Small sample size
    n_features = 500  # High dimensionality
    
    X = np.random.randn(n_samples, n_features)
    
    # Only first 10 features are informative
    true_coef = np.zeros(n_features)
    true_coef[:10] = np.random.randn(10)
    
    y = (np.dot(X, true_coef) > 0).astype(int)
    
    metadata = {
        'name': 'High-Dim Low-Sample Classification',
        'type': 'classification',
        'n_samples': X.shape[0],
        'n_features': X.shape[1],
        'n_classes': 2,
        'feature_names': [f'feature_{i}' for i in range(X.shape[1])],
        'target_names': ['class_0', 'class_1'],
        'description': 'High-dimensional (500 features) with low samples (200) - curse of dimensionality',
        'complexity': 'very_high'
    }
    return X, y, metadata


def load_clustered_dataset() -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Create a dataset with natural clustering structure.
    """
    np.random.seed(42)
    
    # Create clustered data
    X, y = make_blobs(
        n_samples=2000,
        centers=8,
        n_features=15,
        cluster_std=1.5,
        center_box=(-10.0, 10.0),
        random_state=42
    )
    
    # Convert to classification with fewer classes (merge some clusters)
    y_merged = y.copy()
    y_merged[y_merged >= 6] = y_merged[y_merged >= 6] - 2  # Merge last 2 clusters
    y_merged[y_merged >= 4] = y_merged[y_merged >= 4] - 1  # Merge middle clusters
    
    metadata = {
        'name': 'Clustered Classification',
        'type': 'classification',
        'n_samples': X.shape[0],
        'n_features': X.shape[1],
        'n_classes': len(np.unique(y_merged)),
        'feature_names': [f'feature_{i}' for i in range(X.shape[1])],
        'target_names': [f'cluster_group_{i}' for i in range(len(np.unique(y_merged)))],
        'description': 'Classification based on natural clustering structure',
        'complexity': 'medium'
    }
    return X, y_merged, metadata


def get_complex_datasets() -> Dict[str, callable]:
    """
    Get a dictionary of all complex dataset loading functions.
    
    Returns:
        Dictionary mapping dataset names to their loading functions
    """
    return {
        'digits': load_digits_dataset,
        'olivetti_faces': load_olivetti_faces_dataset,
        'newsgroups_tfidf': load_20newsgroups_vectorized,
        'california_housing': load_california_housing_dataset,
        'synthetic_complex_classification': load_synthetic_complex_classification,
        'synthetic_complex_regression': load_synthetic_complex_regression,
        'covtype_subset': load_covtype_subset_dataset,
        'kddcup99_subset': load_kddcup99_subset_dataset,
        'imbalanced_classification': load_imbalanced_classification_dataset,
        'noisy_regression': load_noisy_regression_dataset,
        'multicollinear_regression': load_multicollinear_regression_dataset,
        'time_varying_regression': load_time_varying_dataset,
        'high_dim_low_sample': load_high_dimensional_low_sample_dataset,
        'clustered_classification': load_clustered_dataset
    }


def get_dataset_by_complexity(complexity_level: str = 'medium') -> Dict[str, callable]:
    """
    Get datasets filtered by complexity level.
    
    Args:
        complexity_level: 'low', 'medium', 'high', 'very_high'
    
    Returns:
        Dictionary of datasets matching the complexity level
    """
    all_datasets = get_complex_datasets()
    
    complexity_mapping = {
        'low': [],  # Use simple datasets from original data_loaders.py
        'medium': [
            'synthetic_complex_classification',
            'synthetic_complex_regression', 
            'adult_income',
            'clustered_classification',
            'california_housing'
        ],
        'high': [
            'digits',
            'covtype_subset',
            'kddcup99_subset',
            'imbalanced_classification',
            'noisy_regression',
            'multicollinear_regression',
            'newsgroups_tfidf'
        ],
        'very_high': [
            'olivetti_faces',
            'time_varying_regression',
            'high_dim_low_sample'
        ]
    }
    
    if complexity_level not in complexity_mapping:
        return all_datasets
    
    return {name: all_datasets[name] for name in complexity_mapping[complexity_level] 
            if name in all_datasets}


def load_and_validate_dataset(dataset_name: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load and validate a dataset, with error handling.
    
    Args:
        dataset_name: Name of the dataset to load
        
    Returns:
        X, y, metadata tuple or None if loading fails
    """
    try:
        datasets = get_complex_datasets()
        if dataset_name not in datasets:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        X, y, metadata = datasets[dataset_name]()
        
        # Basic validation
        assert X.shape[0] == len(y), "Feature and target lengths don't match"
        assert X.shape[0] > 0, "Empty dataset"
        assert X.shape[1] > 0, "No features"
        
        print(f"✅ Loaded {metadata['name']}: {X.shape[0]} samples, {X.shape[1]} features")
        if metadata['type'] == 'classification':
            print(f"   Classes: {metadata['n_classes']}, Complexity: {metadata.get('complexity', 'unknown')}")
        else:
            print(f"   Target range: [{np.min(y):.2f}, {np.max(y):.2f}], Complexity: {metadata.get('complexity', 'unknown')}")
        
        return X, y, metadata
        
    except Exception as e:
        print(f"❌ Failed to load {dataset_name}: {e}")
        return None, None, None
