#!/usr/bin/env python3
"""
Reproducibility Validation Script

This script validates that all experimental results can be reproduced exactly,
ensuring the research meets publication standards for reproducibility.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import logging
from pathlib import Path
import hashlib
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from davidian_regularization import (
    stability_bonus_regularization,
    original_davidian_regularization,
    compare_regularization_methods
)
from evaluation import (
    calculate_comprehensive_metrics,
    evaluate_statistical_significance,
    compute_expected_value_statistics
)

logger = logging.getLogger(__name__)


def validate_core_functionality():
    """Validate that core functions work as expected."""
    
    logger.info("Validating core functionality...")
    
    test_cases = [
        {
            'name': 'Perfect generalization (no gap)',
            'train_score': 0.85,
            'validation_score': 0.85,
            'expected_stability_bonus': 0.85 * (1.0 + 0.2),  # Full bonus
            'expected_original': 0.85 - 0.0  # No penalty
        },
        {
            'name': 'Good generalization (small gap)',
            'train_score': 0.85,
            'validation_score': 0.83,
            'expected_stability_bonus': 0.83 * (1.0 + (0.1 - 0.02) / 0.1 * 0.2),  # Partial bonus
            'expected_original': 0.83 - 0.02  # Small penalty
        },
        {
            'name': 'Moderate overfitting (large gap)',
            'train_score': 0.95,
            'validation_score': 0.80,
            'expected_stability_bonus': 0.80,  # No bonus
            'expected_original': 0.80 - 0.15  # Large penalty
        }
    ]
    
    validation_results = []
    
    for test_case in test_cases:
        logger.info(f"  Testing: {test_case['name']}")
        
        # Test Stability Bonus
        stability_result = stability_bonus_regularization(
            test_case['train_score'], 
            test_case['validation_score']
        )
        
        # Test Original Davidian
        original_result = original_davidian_regularization(
            test_case['train_score'],
            test_case['validation_score']
        )
        
        # Validate results
        stability_correct = abs(stability_result - test_case['expected_stability_bonus']) < 0.001
        original_correct = abs(original_result - test_case['expected_original']) < 0.001
        
        validation_results.append({
            'test_case': test_case['name'],
            'stability_bonus_correct': stability_correct,
            'original_davidian_correct': original_correct,
            'stability_bonus_result': stability_result,
            'original_davidian_result': original_result
        })
        
        logger.info(f"    Stability Bonus: {stability_result:.4f} ({'✓' if stability_correct else '✗'})")
        logger.info(f"    Original Davidian: {original_result:.4f} ({'✓' if original_correct else '✗'})")
    
    # Overall validation
    all_tests_passed = all(r['stability_bonus_correct'] and r['original_davidian_correct'] 
                          for r in validation_results)
    
    if all_tests_passed:
        logger.info("✅ All core functionality tests passed")
    else:
        logger.error("❌ Some core functionality tests failed")
    
    return all_tests_passed, validation_results


def validate_experimental_reproducibility():
    """Validate that experimental results can be reproduced."""
    
    logger.info("Validating experimental reproducibility...")
    
    # Test with small controlled experiment
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    
    # Create test dataset
    X, y = make_classification(
        n_samples=500,
        n_features=10,
        n_informative=8,
        n_redundant=1,
        weights=[0.9, 0.1],  # 1:9 imbalance
        random_state=42
    )
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Test reproducibility with fixed random state
    logger.info("  Testing reproducibility with fixed random seeds...")
    
    model_config = {
        'class': LogisticRegression,
        'parameters': {'random_state': 42, 'max_iter': 1000}
    }
    
    # Run same experiment twice
    results_1 = compare_regularization_methods(
        feature_matrix=X_scaled,
        target_vector=y,
        model_class=model_config['class'],
        model_parameters=model_config['parameters'],
        methods_to_compare=['stability_bonus', 'original_davidian'],
        k_folds=5,
        number_of_trials=10,
        random_state=42
    )
    
    results_2 = compare_regularization_methods(
        feature_matrix=X_scaled,
        target_vector=y,
        model_class=model_config['class'],
        model_parameters=model_config['parameters'],
        methods_to_compare=['stability_bonus', 'original_davidian'],
        k_folds=5,
        number_of_trials=10,
        random_state=42
    )
    
    # Compare results
    stability_1 = results_1['individual_method_results']['stability_bonus']['expected_value_improvement_percentage']
    stability_2 = results_2['individual_method_results']['stability_bonus']['expected_value_improvement_percentage']
    
    original_1 = results_1['individual_method_results']['original_davidian']['expected_value_improvement_percentage']
    original_2 = results_2['individual_method_results']['original_davidian']['expected_value_improvement_percentage']
    
    # Check reproducibility (should be identical with same random seed)
    stability_reproducible = abs(stability_1 - stability_2) < 0.001
    original_reproducible = abs(original_1 - original_2) < 0.001
    
    logger.info(f"    Stability Bonus: Run 1 = {stability_1:.4f}, Run 2 = {stability_2:.4f} "
               f"({'✓' if stability_reproducible else '✗'})")
    logger.info(f"    Original Davidian: Run 1 = {original_1:.4f}, Run 2 = {original_2:.4f} "
               f"({'✓' if original_reproducible else '✗'})")
    
    reproducibility_validated = stability_reproducible and original_reproducible
    
    if reproducibility_validated:
        logger.info("✅ Experimental reproducibility validated")
    else:
        logger.error("❌ Reproducibility validation failed")
    
    return reproducibility_validated


def validate_file_integrity():
    """Validate that all required files exist and have expected content."""
    
    logger.info("Validating file integrity...")
    
    required_files = {
        'src/davidian_regularization.py': ['stability_bonus_regularization', 'original_davidian_regularization'],
        'src/evaluation.py': ['calculate_comprehensive_metrics', 'evaluate_statistical_significance'],
        'src/data_loaders.py': ['load_synthetic_imbalanced_dataset', 'load_breast_cancer_dataset'],
        'paper.tex': ['\\title{', '\\section{Introduction}', 'Stability Bonus'],
        'references.bib': ['@article{', 'chawla2002smote'],
        'scripts/run_full_pipeline.py': ['def main()', 'run_synthetic_experiments'],
        'README.md': ['Publication Package', 'Quick Start']
    }
    
    validation_results = {}
    
    for filepath, expected_content in required_files.items():
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as file:
                    file_content = file.read()
                
                content_checks = [content in file_content for content in expected_content]
                all_content_present = all(content_checks)
                
                validation_results[filepath] = {
                    'exists': True,
                    'content_valid': all_content_present,
                    'missing_content': [expected_content[i] for i, check in enumerate(content_checks) if not check]
                }
                
                status = "✅" if all_content_present else "⚠️"
                logger.info(f"  {status} {filepath}")
                
                if not all_content_present:
                    logger.warning(f"    Missing content: {validation_results[filepath]['missing_content']}")
                
            except Exception as e:
                validation_results[filepath] = {'exists': True, 'content_valid': False, 'error': str(e)}
                logger.error(f"  ❌ {filepath}: {e}")
        else:
            validation_results[filepath] = {'exists': False, 'content_valid': False}
            logger.error(f"  ❌ Missing file: {filepath}")
    
    # Overall validation
    files_valid = all(result.get('content_valid', False) for result in validation_results.values())
    
    if files_valid:
        logger.info("✅ All file integrity checks passed")
    else:
        logger.error("❌ Some file integrity checks failed")
    
    return files_valid, validation_results


def create_reproducibility_report():
    """Create comprehensive reproducibility report."""
    
    logger.info("Creating reproducibility report...")
    
    # Run all validations
    core_functionality_valid, core_tests = validate_core_functionality()
    reproducibility_valid = validate_experimental_reproducibility()
    file_integrity_valid, file_checks = validate_file_integrity()
    
    # Create comprehensive report
    report = {
        'validation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'overall_status': 'PASSED' if all([core_functionality_valid, reproducibility_valid, file_integrity_valid]) else 'FAILED',
        
        'core_functionality': {
            'status': 'PASSED' if core_functionality_valid else 'FAILED',
            'test_results': core_tests
        },
        
        'experimental_reproducibility': {
            'status': 'PASSED' if reproducibility_valid else 'FAILED',
            'description': 'Experiments produce identical results with same random seeds'
        },
        
        'file_integrity': {
            'status': 'PASSED' if file_integrity_valid else 'FAILED',
            'file_checks': file_checks
        },
        
        'publication_readiness': {
            'latex_paper': os.path.exists('paper.tex'),
            'bibliography': os.path.exists('references.bib'),
            'figures_available': len([f for f in os.listdir('figures') if f.endswith('.png')]) >= 3,
            'source_code': os.path.exists('src/davidian_regularization.py'),
            'pipeline_script': os.path.exists('scripts/run_full_pipeline.py')
        }
    }
    
    # Save report
    with open('reproducibility_validation_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    logger.info("="*50)
    logger.info("REPRODUCIBILITY VALIDATION SUMMARY")
    logger.info("="*50)
    logger.info(f"Overall Status: {report['overall_status']}")
    logger.info(f"Core Functionality: {report['core_functionality']['status']}")
    logger.info(f"Experimental Reproducibility: {report['experimental_reproducibility']['status']}")
    logger.info(f"File Integrity: {report['file_integrity']['status']}")
    
    logger.info("\nPublication Readiness:")
    for component, status in report['publication_readiness'].items():
        status_icon = "✅" if status else "❌"
        logger.info(f"  {status_icon} {component.replace('_', ' ').title()}")
    
    if report['overall_status'] == 'PASSED':
        logger.info("\n🎉 REPRODUCIBILITY VALIDATION PASSED!")
        logger.info("🎉 Research package ready for publication submission")
    else:
        logger.error("\n❌ Reproducibility validation failed")
        logger.error("❌ Address issues before publication submission")
    
    return report


def main():
    """Main validation function."""
    
    logger.info("REPRODUCIBILITY VALIDATION FOR PUBLICATION")
    logger.info("="*60)
    logger.info("Validating all components for journal submission")
    
    # Create validation report
    validation_report = create_reproducibility_report()
    
    return validation_report['overall_status'] == 'PASSED'


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    success = main()
    sys.exit(0 if success else 1)
