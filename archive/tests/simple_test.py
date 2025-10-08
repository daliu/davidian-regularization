#!/usr/bin/env python3
"""
Simple test without TensorFlow to avoid lock blocking issues.
"""

import sys
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Test basic imports
print("Testing basic imports...")
try:
    import numpy as np
    import pandas as pd
    import sklearn
    print("✓ Basic scientific libraries imported successfully")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

# Test our modules
print("Testing our modules...")
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.data_loaders import load_iris_dataset
    from src.logging_config import setup_logging
    print("✓ Our modules imported successfully")
except ImportError as e:
    print(f"✗ Module import error: {e}")
    sys.exit(1)

# Test basic functionality
print("Testing basic functionality...")
try:
    # Set up simple logging
    logger = setup_logging(log_level='INFO', log_file='simple_test.log')
    logger.info("Starting simple test")
    
    # Load data
    X, y, metadata = load_iris_dataset()
    logger.info(f"Loaded {metadata['name']}: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Simple sklearn test
    model = LogisticRegression(random_state=42, max_iter=1000)
    scores = cross_val_score(model, X, y, cv=3)
    logger.info(f"Cross-validation scores: {scores}")
    logger.info(f"Mean CV score: {scores.mean():.4f}")
    
    print("✓ Basic functionality test PASSED!")
    print(f"Mean accuracy: {scores.mean():.4f}")
    print("Check simple_test.log for detailed logs.")
    
except Exception as e:
    print(f"✗ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
