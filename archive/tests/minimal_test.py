#!/usr/bin/env python3
"""
Minimal test to isolate the hanging issue.
"""

import sys
import os

print("=== MINIMAL TEST START ===")

# Test 1: Basic Python
print("1. Basic Python: OK")

# Test 2: Basic imports
print("2. Testing basic imports...")
try:
    import numpy as np
    print("   ✓ NumPy imported")
    
    import pandas as pd
    print("   ✓ Pandas imported")
    
    import sklearn
    print("   ✓ Scikit-learn imported")
    
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import make_classification
    print("   ✓ Sklearn components imported")
    
except Exception as e:
    print(f"   ✗ Import failed: {e}")
    sys.exit(1)

# Test 3: Basic sklearn functionality
print("3. Testing basic sklearn...")
try:
    X, y = make_classification(n_samples=100, n_features=4, n_classes=3, 
                              n_informative=3, random_state=42)
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X, y)
    score = model.score(X, y)
    print(f"   ✓ Sklearn test completed: accuracy = {score:.4f}")
except Exception as e:
    print(f"   ✗ Sklearn test failed: {e}")
    sys.exit(1)

# Test 4: Add our src to path
print("4. Adding src to path...")
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
print("   ✓ Path added")

# Test 5: Try importing our simplest module first
print("5. Testing our data_loaders module...")
try:
    from src.data_loaders import load_iris_dataset
    print("   ✓ data_loaders imported")
    
    X, y, metadata = load_iris_dataset()
    print(f"   ✓ Iris dataset loaded: {X.shape}")
except Exception as e:
    print(f"   ✗ data_loaders failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Try logging module
print("6. Testing logging module...")
try:
    from src.logging_config import setup_logging
    print("   ✓ logging_config imported")
    
    logger = setup_logging(log_level='INFO', log_file='minimal_test.log', console_output=False)
    logger.info("Test log message")
    print("   ✓ Logger setup completed")
except Exception as e:
    print(f"   ✗ logging_config failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n=== MINIMAL TEST PASSED ===")
print("Core functionality works. The issue might be in specific model imports.")
print("Check minimal_test.log for log output.")
