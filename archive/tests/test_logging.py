#!/usr/bin/env python3
"""
Simple test script to verify the Davidian Regularization implementation with logging.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.logging_config import setup_logging, ExperimentTimer
from src.data_loaders import load_iris_dataset
from src.davidian_regularization import davidian_cross_validation
from src.models import LinearModel

def test_basic_functionality():
    """Test basic functionality with logging."""
    # Set up logging
    logger = setup_logging(log_level='INFO', log_file='test_log.log')
    
    logger.info("Starting basic functionality test...")
    
    try:
        with ExperimentTimer(logger, "Basic test"):
            # Load data
            logger.info("Loading Iris dataset...")
            X, y, metadata = load_iris_dataset()
            logger.info(f"Loaded {metadata['name']}: {X.shape[0]} samples, {X.shape[1]} features")
            
            # Create model
            logger.info("Creating Linear model...")
            model = LinearModel(task_type='classification')
            
            # Run Davidian cross-validation
            logger.info("Running Davidian cross-validation...")
            results = davidian_cross_validation(
                X, y, model, k=3, task_type='classification', alpha=1.0
            )
            
            logger.info("Results:")
            logger.info(f"  Mean train score: {results['mean_train_score']:.4f}")
            logger.info(f"  Mean val score: {results['mean_val_score']:.4f}")
            logger.info(f"  Mean regularized score: {results['mean_regularized_score']:.4f}")
            
            logger.info("✓ Basic functionality test PASSED!")
            return True
            
    except Exception as e:
        logger.error(f"✗ Basic functionality test FAILED: {e}")
        logger.debug("Error details:", exc_info=True)
        return False

if __name__ == "__main__":
    success = test_basic_functionality()
    if success:
        print("Test completed successfully! Check test_log.log for detailed logs.")
    else:
        print("Test failed! Check test_log.log for error details.")
        sys.exit(1)
