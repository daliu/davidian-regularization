#!/usr/bin/env python3
"""
Verbose test with comprehensive logging to track progress.
"""

import sys
import os
import time
import traceback

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_step(step_name, func):
    """Test a step with detailed logging."""
    print(f"\n{'='*60}")
    print(f"STEP: {step_name}")
    print(f"{'='*60}")
    start_time = time.time()
    
    try:
        result = func()
        elapsed = time.time() - start_time
        print(f"✓ {step_name} COMPLETED in {elapsed:.2f}s")
        return result
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"✗ {step_name} FAILED after {elapsed:.2f}s")
        print(f"Error: {type(e).__name__}: {e}")
        traceback.print_exc()
        raise

def step1_imports():
    """Test basic imports."""
    print("Importing basic libraries...")
    import numpy as np
    import pandas as pd
    import sklearn
    print(f"  ✓ NumPy {np.__version__}")
    print(f"  ✓ Pandas {pd.__version__}")
    print(f"  ✓ Scikit-learn {sklearn.__version__}")
    
    print("Importing our modules...")
    from src.logging_config import setup_logging
    from src.data_loaders import load_iris_dataset
    print("  ✓ All modules imported successfully")
    
    return np, pd, sklearn, setup_logging, load_iris_dataset

def step2_logging_setup(setup_logging):
    """Test logging setup."""
    print("Setting up logging...")
    logger = setup_logging(log_level='INFO', log_file='verbose_test.log')
    print("  ✓ Logger configured")
    
    logger.info("This is a test log message")
    print("  ✓ Test log message written")
    
    return logger

def step3_data_loading(load_iris_dataset, logger):
    """Test data loading."""
    print("Loading Iris dataset...")
    X, y, metadata = load_iris_dataset()
    
    print(f"  ✓ Dataset loaded: {metadata['name']}")
    print(f"  ✓ Shape: {X.shape}")
    print(f"  ✓ Features: {metadata.get('n_features', 'N/A')}")
    print(f"  ✓ Classes: {metadata.get('n_classes', 'N/A')}")
    
    logger.info(f"Dataset loaded successfully: {X.shape[0]} samples, {X.shape[1]} features")
    
    return X, y, metadata

def step4_simple_model_test(X, y, logger):
    """Test simple model without our framework."""
    print("Testing simple sklearn model...")
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    
    model = LogisticRegression(random_state=42, max_iter=1000)
    print("  ✓ Model created")
    
    scores = cross_val_score(model, X, y, cv=3)
    print(f"  ✓ Cross-validation completed")
    print(f"  ✓ Scores: {scores}")
    print(f"  ✓ Mean score: {scores.mean():.4f}")
    
    logger.info(f"Simple model test completed: mean CV score = {scores.mean():.4f}")
    
    return scores

def step5_our_model_test(X, y, logger):
    """Test our model wrapper."""
    print("Testing our model wrapper...")
    from src.models import LinearModel
    breakpoint()
    model = LinearModel(task_type='classification')
    print("  ✓ Our LinearModel created")
    
    # Simple fit and predict test
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model.fit(X_train, y_train)
    print("  ✓ Model fitted")
    
    predictions = model.predict(X_test)
    print(f"  ✓ Predictions made: {len(predictions)} samples")
    
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_test, predictions)
    print(f"  ✓ Accuracy: {accuracy:.4f}")
    
    logger.info(f"Our model test completed: accuracy = {accuracy:.4f}")
    
    return model, accuracy

def step6_davidian_cv_test(X, y, logger):
    """Test Davidian cross-validation."""
    print("Testing Davidian cross-validation...")
    from src.davidian_regularization import davidian_cross_validation
    from src.models import LinearModel
    
    model = LinearModel(task_type='classification')
    print("  ✓ Model for CV created")
    
    print("  Running 3-fold cross-validation...")
    results = davidian_cross_validation(
        X, y, model, k=3, task_type='classification', alpha=1.0, random_state=42
    )
    
    print(f"  ✓ CV completed")
    print(f"  ✓ Mean train score: {results['mean_train_score']:.4f}")
    print(f"  ✓ Mean val score: {results['mean_val_score']:.4f}")
    print(f"  ✓ Mean regularized score: {results['mean_regularized_score']:.4f}")
    
    logger.info(f"Davidian CV test completed: regularized score = {results['mean_regularized_score']:.4f}")
    
    return results

def main():
    """Run all test steps."""
    print("VERBOSE DAVIDIAN REGULARIZATION TEST")
    print("="*60)
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Step 1: Imports
        np, pd, sklearn, setup_logging, load_iris_dataset = test_step(
            "Import Libraries", 
            lambda: step1_imports()
        )
        
        # Step 2: Logging
        logger = test_step(
            "Setup Logging",
            lambda: step2_logging_setup(setup_logging)
        )
        
        # Step 3: Data Loading
        X, y, metadata = test_step(
            "Load Dataset",
            lambda: step3_data_loading(load_iris_dataset, logger)
        )
        
        # Step 4: Simple Model Test
        scores = test_step(
            "Simple Model Test",
            lambda: step4_simple_model_test(X, y, logger)
        )
        
        # Step 5: Our Model Test
        model, accuracy = test_step(
            "Our Model Test",
            lambda: step5_our_model_test(X, y, logger)
        )
        
        # Step 6: Davidian CV Test
        results = test_step(
            "Davidian Cross-Validation Test",
            lambda: step6_davidian_cv_test(X, y, logger)
        )
        
        print(f"\n{'='*60}")
        print("ALL TESTS PASSED!")
        print(f"{'='*60}")
        print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("Check verbose_test.log for detailed logs.")
        
    except Exception as e:
        print(f"\n{'='*60}")
        print("TEST FAILED!")
        print(f"{'='*60}")
        print(f"Final error: {type(e).__name__}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
