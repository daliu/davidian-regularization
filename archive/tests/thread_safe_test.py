#!/usr/bin/env python3
"""
Thread-safe test that avoids mutex blocking issues on M3 Macs.
"""

import sys
import os
import time

# Force single-threaded execution before any imports
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'

print("THREAD-SAFE DAVIDIAN REGULARIZATION TEST")
print("="*60)
print("Configured for single-threaded execution to avoid M3 Mac mutex issues")
print("="*60)

def test_step(step_name, func):
    """Test a step with timing."""
    print(f"\n{step_name}...")
    start_time = time.time()
    
    try:
        result = func()
        elapsed = time.time() - start_time
        print(f"   ✓ {step_name} completed in {elapsed:.2f}s")
        return result
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"   ✗ {step_name} failed after {elapsed:.2f}s: {e}")
        raise

def step1_basic_imports():
    """Test basic imports with thread safety."""
    import numpy as np
    import pandas as pd
    import sklearn
    
    # Verify single-threading
    print(f"     NumPy threads: {os.environ.get('OMP_NUM_THREADS', 'default')}")
    print(f"     MKL threads: {os.environ.get('MKL_NUM_THREADS', 'default')}")
    
    return np, pd, sklearn

def step2_our_modules():
    """Import our modules."""
    sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
    
    from src.thread_safe_models import (
        ThreadSafeLinearModel, 
        ThreadSafeGradientBoostingModel,
        force_single_thread_environment
    )
    from src.data_loaders import load_iris_dataset
    
    # Double-check threading environment
    force_single_thread_environment()
    
    return ThreadSafeLinearModel, ThreadSafeGradientBoostingModel, load_iris_dataset

def step3_load_data(load_iris_dataset):
    """Load test data."""
    X, y, metadata = load_iris_dataset()
    print(f"     Dataset: {metadata['name']}")
    print(f"     Shape: {X.shape}")
    print(f"     Classes: {len(np.unique(y))}")
    
    return X, y, metadata

def step4_test_linear_model(X, y, ThreadSafeLinearModel):
    """Test thread-safe linear model."""
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    model = ThreadSafeLinearModel(task_type='classification')
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    print(f"     Accuracy: {accuracy:.4f}")
    return model, accuracy

def step5_test_gbm_model(X, y, ThreadSafeGradientBoostingModel):
    """Test thread-safe gradient boosting model."""
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    model = ThreadSafeGradientBoostingModel(
        task_type='classification',
        n_estimators=20  # Small for speed
    )
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    print(f"     Accuracy: {accuracy:.4f}")
    
    # Test feature importance
    importance = model.get_feature_importance()
    print(f"     Feature importance keys: {list(importance.keys())[:3]}...")
    
    return model, accuracy

def step6_test_davidian_cv(X, y, ThreadSafeLinearModel):
    """Test Davidian cross-validation with thread-safe models."""
    # Import our CV function
    from src.davidian_regularization import davidian_cross_validation
    
    model = ThreadSafeLinearModel(task_type='classification')
    
    results = davidian_cross_validation(
        X, y, model, k=3, task_type='classification', 
        alpha=1.0, random_state=42
    )
    
    print(f"     Mean train score: {results['mean_train_score']:.4f}")
    print(f"     Mean val score: {results['mean_val_score']:.4f}")
    print(f"     Mean regularized score: {results['mean_regularized_score']:.4f}")
    
    return results

def step7_test_multiple_trials(X, y, ThreadSafeLinearModel):
    """Test multiple trials with thread safety."""
    from src.davidian_regularization import multiple_trial_davidian_cv
    
    results = multiple_trial_davidian_cv(
        X, y, ThreadSafeLinearModel, {'task_type': 'classification'},
        k=3, n_trials=5, task_type='classification', alpha=1.0
    )
    
    print(f"     Best 4 mean: {results['mean_best_4_score']:.4f}")
    print(f"     Overall mean: {results['overall_mean_score']:.4f}")
    
    return results

def step8_test_comparison(X, y, ThreadSafeLinearModel):
    """Test Davidian vs Random comparison."""
    from src.davidian_regularization import (
        multiple_trial_davidian_cv, 
        random_split_validation
    )
    
    # Davidian results
    davidian_results = multiple_trial_davidian_cv(
        X, y, ThreadSafeLinearModel, {'task_type': 'classification'},
        k=3, n_trials=5, task_type='classification', alpha=1.0
    )
    
    # Random results
    random_results = random_split_validation(
        X, y, ThreadSafeLinearModel, {'task_type': 'classification'},
        n_trials=5, task_type='classification'
    )
    
    davidian_score = davidian_results['mean_best_4_score']
    random_score = random_results['mean_best_4_score']
    improvement = davidian_score - random_score
    improvement_pct = (improvement / abs(random_score)) * 100 if random_score != 0 else 0
    
    print(f"     Davidian: {davidian_score:.4f}")
    print(f"     Random: {random_score:.4f}")
    print(f"     Improvement: {improvement_pct:+.2f}%")
    
    return davidian_results, random_results, improvement_pct

def main():
    """Run all tests."""
    try:
        # Step 1: Basic imports
        np, pd, sklearn = test_step(
            "1. Basic imports", 
            step1_basic_imports
        )
        
        # Step 2: Our modules
        ThreadSafeLinearModel, ThreadSafeGradientBoostingModel, load_iris_dataset = test_step(
            "2. Thread-safe modules",
            lambda: step2_our_modules()
        )
        
        # Step 3: Load data
        X, y, metadata = test_step(
            "3. Load data",
            lambda: step3_load_data(load_iris_dataset)
        )
        
        # Step 4: Linear model
        linear_model, linear_acc = test_step(
            "4. Thread-safe Linear Model",
            lambda: step4_test_linear_model(X, y, ThreadSafeLinearModel)
        )
        
        # Step 5: GBM model
        gbm_model, gbm_acc = test_step(
            "5. Thread-safe Gradient Boosting",
            lambda: step5_test_gbm_model(X, y, ThreadSafeGradientBoostingModel)
        )
        
        # Step 6: Davidian CV
        cv_results = test_step(
            "6. Davidian Cross-Validation",
            lambda: step6_test_davidian_cv(X, y, ThreadSafeLinearModel)
        )
        
        # Step 7: Multiple trials
        multi_results = test_step(
            "7. Multiple Trials",
            lambda: step7_test_multiple_trials(X, y, ThreadSafeLinearModel)
        )
        
        # Step 8: Full comparison
        davidian_results, random_results, improvement = test_step(
            "8. Davidian vs Random Comparison",
            lambda: step8_test_comparison(X, y, ThreadSafeLinearModel)
        )
        
        print(f"\n{'='*60}")
        print("🎉 ALL TESTS PASSED! 🎉")
        print(f"{'='*60}")
        print("✅ Thread-safe implementation works")
        print("✅ No mutex blocking issues")
        print("✅ Davidian Regularization algorithm functional")
        print(f"✅ Performance improvement: {improvement:+.2f}%")
        print("\nYou can now run full experiments with thread-safe models!")
        
        return True
        
    except Exception as e:
        print(f"\n{'='*60}")
        print("❌ TEST FAILED")
        print(f"{'='*60}")
        print(f"Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
