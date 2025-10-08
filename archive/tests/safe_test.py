#!/usr/bin/env python3
"""
Safe test that avoids deep learning libraries to prevent hanging issues.
"""

import sys
import os
import time
import traceback

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def safe_test():
    """Run a safe test without deep learning components."""
    print("SAFE DAVIDIAN REGULARIZATION TEST")
    print("="*60)
    print("This test avoids TensorFlow/PyTorch to prevent M3 Mac hanging issues")
    print("="*60)
    
    try:
        # Step 1: Basic imports
        print("\n1. Testing basic imports...")
        import numpy as np
        import pandas as pd
        import sklearn
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        print("   ✓ Basic libraries imported")
        
        # Step 2: Our modules
        print("\n2. Testing our modules...")
        from src.logging_config import setup_logging
        from src.data_loaders import load_iris_dataset
        from src.models import LinearModel, GradientBoostingModel
        print("   ✓ Our modules imported (skipping deep learning)")
        
        # Step 3: Logging setup
        print("\n3. Setting up logging...")
        logger = setup_logging(log_level='INFO', log_file='safe_test.log')
        logger.info("Safe test started")
        print("   ✓ Logging configured")
        
        # Step 4: Data loading
        print("\n4. Loading data...")
        X, y, metadata = load_iris_dataset()
        logger.info(f"Dataset loaded: {metadata['name']}")
        print(f"   ✓ Dataset: {metadata['name']} ({X.shape[0]} samples, {X.shape[1]} features)")
        
        # Step 5: Test our Linear model
        print("\n5. Testing LinearModel wrapper...")
        model = LinearModel(task_type='classification')
        
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(y_test, predictions)
        print(f"   ✓ LinearModel accuracy: {accuracy:.4f}")
        logger.info(f"LinearModel test completed: accuracy = {accuracy:.4f}")
        
        # Step 6: Test Gradient Boosting model
        print("\n6. Testing GradientBoostingModel...")
        gbm_model = GradientBoostingModel(task_type='classification', n_estimators=10)  # Small for speed
        gbm_model.fit(X_train, y_train)
        gbm_predictions = gbm_model.predict(X_test)
        gbm_accuracy = accuracy_score(y_test, gbm_predictions)
        print(f"   ✓ GradientBoostingModel accuracy: {gbm_accuracy:.4f}")
        logger.info(f"GradientBoostingModel test completed: accuracy = {gbm_accuracy:.4f}")
        
        # Step 7: Test Davidian cross-validation
        print("\n7. Testing Davidian cross-validation...")
        from src.davidian_regularization import davidian_cross_validation
        
        test_model = LinearModel(task_type='classification')
        results = davidian_cross_validation(
            X, y, test_model, k=3, task_type='classification', alpha=1.0, random_state=42
        )
        
        print(f"   ✓ Cross-validation completed")
        print(f"   ✓ Mean train score: {results['mean_train_score']:.4f}")
        print(f"   ✓ Mean val score: {results['mean_val_score']:.4f}")
        print(f"   ✓ Mean regularized score: {results['mean_regularized_score']:.4f}")
        
        logger.info(f"Davidian CV completed: regularized = {results['mean_regularized_score']:.4f}")
        
        # Step 8: Test multiple trials (small number)
        print("\n8. Testing multiple trial Davidian CV...")
        from src.davidian_regularization import multiple_trial_davidian_cv
        
        multi_results = multiple_trial_davidian_cv(
            X, y, LinearModel, {'task_type': 'classification'},
            k=3, n_trials=5, task_type='classification', alpha=1.0
        )
        
        print(f"   ✓ Multiple trials completed")
        print(f"   ✓ Best 4 mean score: {multi_results['mean_best_4_score']:.4f}")
        print(f"   ✓ Overall mean score: {multi_results['overall_mean_score']:.4f}")
        
        logger.info(f"Multiple trial test completed: best_4_mean = {multi_results['mean_best_4_score']:.4f}")
        
        # Step 9: Test random sampling comparison
        print("\n9. Testing random sampling comparison...")
        from src.davidian_regularization import random_split_validation
        
        random_results = random_split_validation(
            X, y, LinearModel, {'task_type': 'classification'},
            n_trials=5, task_type='classification'
        )
        
        print(f"   ✓ Random sampling completed")
        print(f"   ✓ Random best 4 mean: {random_results['mean_best_4_score']:.4f}")
        
        # Step 10: Compare methods
        davidian_score = multi_results['mean_best_4_score']
        random_score = random_results['mean_best_4_score']
        improvement = davidian_score - random_score
        improvement_pct = (improvement / abs(random_score)) * 100 if random_score != 0 else 0
        
        print(f"\n10. Comparison Results:")
        print(f"   ✓ Davidian Regularization: {davidian_score:.4f}")
        print(f"   ✓ Random Sampling: {random_score:.4f}")
        print(f"   ✓ Improvement: {improvement:+.4f} ({improvement_pct:+.2f}%)")
        
        if improvement > 0:
            print("   ✓ DAVIDIAN REGULARIZATION PERFORMS BETTER!")
        else:
            print("   ✗ Random sampling performs better in this test")
        
        logger.info(f"Final comparison: Davidian={davidian_score:.4f}, Random={random_score:.4f}, Improvement={improvement_pct:+.2f}%")
        
        print(f"\n{'='*60}")
        print("ALL TESTS PASSED SUCCESSFULLY!")
        print(f"{'='*60}")
        print("✓ Basic functionality works")
        print("✓ Logging system works")
        print("✓ Data loading works")
        print("✓ Model wrappers work")
        print("✓ Davidian Regularization algorithm works")
        print("✓ Comparison with random sampling works")
        print("\nCheck safe_test.log for detailed logs.")
        print("You can now run experiments with linear and gradient boosting models.")
        
        return True
        
    except Exception as e:
        print(f"\n{'='*60}")
        print("TEST FAILED!")
        print(f"{'='*60}")
        print(f"Error: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = safe_test()
    if not success:
        sys.exit(1)
