#!/usr/bin/env python3
"""
Main script to run the comprehensive Davidian Regularization experiment.

This script orchestrates the full experimental pipeline including:
1. Running the comprehensive experiment across all parameter combinations
2. Generating detailed visualizations and charts
3. Creating summary reports and analysis
"""

import os
import time
import json

# Ensure results directory exists
os.makedirs('results', exist_ok=True)
os.makedirs('plots', exist_ok=True)

def run_full_experiment():
    """Run the complete experimental pipeline."""
    
    print("COMPREHENSIVE DAVIDIAN REGULARIZATION EXPERIMENT")
    print("="*80)
    print("This experiment will test Davidian Regularization in lieu of minority class rebalancing")
    print("across multiple dimensions:")
    print("- Sample sizes: 50, 500, 5000, 50000")
    print("- Class imbalance ratios: 1:1, 1:9, 1:19, 1:29, 1:49") 
    print("- K-fold values: 3, 4, 5, 10")
    print("- Trial counts: 5, 10, 15, 25")
    print("- Models: Linear Regression, Naive Bayes, Gradient Boosted Trees")
    print("- Regularization methods: Original, Conservative, Inverse_diff, Exponential_decay, Stability_bonus, None")
    print("="*80)
    
    start_time = time.time()
    
    # Step 1: Run the comprehensive experiment
    print("\nSTEP 1: Running comprehensive experiment...")
    print("-" * 50)
    
    try:
        from comprehensive_davidian_experiment import main as run_experiment
        run_experiment()
        print("✓ Comprehensive experiment completed successfully")
    except Exception as e:
        print(f"✗ Error running comprehensive experiment: {e}")
        return False
    
    # Step 2: Generate visualizations
    print("\nSTEP 2: Generating comprehensive visualizations...")
    print("-" * 50)
    
    try:
        from visualization_comprehensive import generate_comprehensive_report
        generate_comprehensive_report()
        print("✓ Visualization report generated successfully")
    except Exception as e:
        print(f"✗ Error generating visualizations: {e}")
        return False
    
    # Step 3: Create summary report
    print("\nSTEP 3: Creating summary report...")
    print("-" * 50)
    
    try:
        create_summary_report()
        print("✓ Summary report created successfully")
    except Exception as e:
        print(f"✗ Error creating summary report: {e}")
        return False
    
    total_time = time.time() - start_time
    
    print("\n" + "="*80)
    print("EXPERIMENT PIPELINE COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"Total execution time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print("\nGenerated files:")
    print("  Results:")
    print("    - results/comprehensive_davidian_results.json")
    print("    - results/comprehensive_davidian_results.csv")
    print("  Visualizations:")
    print("    - plots/master_results_chart.png")
    print("    - plots/method_comparison.png")
    print("    - plots/parameter_analysis.png") 
    print("    - plots/imbalance_analysis.png")
    print("  Reports:")
    print("    - EXPERIMENT_SUMMARY_REPORT.md")
    print("="*80)
    
    return True

def create_summary_report():
    """Create a comprehensive markdown summary report."""
    
    # Load results
    with open('results/comprehensive_davidian_results.json', 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    summary = results['summary']
    overall_stats = summary['overall_statistics']
    
    # Create markdown report
    report = f"""# Comprehensive Davidian Regularization Experiment Report

## Executive Summary

This report presents the results of a comprehensive experiment testing **Davidian Regularization** as an alternative to minority class rebalancing techniques for imbalanced datasets.

### Key Findings

- **Total Experiments Conducted**: {summary['total_experiments']}
- **Overall Success Rate**: {overall_stats['overall_win_rate']:.1f}% (Davidian outperformed random sampling)
- **Mean Performance Improvement**: {overall_stats['mean_improvement_pct']:+.2f}%
- **Median Performance Improvement**: {overall_stats['median_improvement_pct']:+.2f}%
- **Mean Test Accuracy**: {overall_stats['mean_test_accuracy']:.4f}
- **Mean Test F1-Score**: {overall_stats['mean_test_f1']:.4f}

## Experimental Design

### Parameters Tested

**Sample Sizes**: {', '.join(map(str, summary['parameters_tested']['sample_sizes']))}

**Class Imbalance Ratios**: {', '.join([f'1:{ratio:.0f}' for ratio in summary['parameters_tested']['imbalance_ratios']])}

**K-fold Values**: {', '.join(map(str, summary['parameters_tested']['k_values']))}

**Trial Counts**: {', '.join(map(str, summary['parameters_tested']['trial_counts']))}

**Models Tested**: {', '.join(summary['parameters_tested']['model_types'])}

**Regularization Methods**: {', '.join(summary['parameters_tested']['regularization_methods'])}

### Methodology

1. **Dataset Generation**: Created imbalanced binary classification datasets using sklearn's `make_classification`
2. **Cross-Validation**: Applied stratified k-fold cross-validation with multiple trials
3. **Regularization**: Tested six different Davidian regularization variants
4. **Baseline Comparison**: Compared against random train-validation splits
5. **Evaluation**: Measured performance using accuracy, precision, recall, F1-score, and AUC

## Results by Regularization Method

"""
    
    # Add method-specific results
    for method, stats in overall_stats['by_regularization_method'].items():
        report += f"""### {method.replace('_', ' ').title()}

- **Experiments**: {stats['count']}
- **Win Rate**: {stats['win_rate']:.1f}%
- **Mean Improvement**: {stats['mean_improvement']:+.2f}%
- **Standard Deviation**: ±{stats['std_improvement']:.2f}%
- **Median Improvement**: {stats['median_improvement']:+.2f}%

"""
    
    # Add best performing method
    best_method = max(overall_stats['by_regularization_method'].items(), key=lambda x: x[1]['win_rate'])
    report += f"""## Best Performing Method

**{best_method[0].replace('_', ' ').title()}** achieved the highest win rate of **{best_method[1]['win_rate']:.1f}%** with a mean improvement of **{best_method[1]['mean_improvement']:+.2f}%**.

## Mathematical Formulations

The different Davidian regularization variants tested:

1. **Original**: `regularized_score = val_score - α * |train_score - val_score|`
2. **Conservative**: `regularized_score = val_score - 0.5 * α * |train_score - val_score|`
3. **Inverse Diff**: `regularized_score = val_score * (1 / (1 + |train_score - val_score|))`
4. **Exponential Decay**: `regularized_score = val_score * exp(-|train_score - val_score|)`
5. **Stability Bonus**: `regularized_score = val_score * (1 + bonus)` if difference < threshold
6. **None**: `regularized_score = val_score` (baseline stratified k-fold)

## Conclusions

"""
    
    if overall_stats['overall_win_rate'] > 50:
        report += f"""✅ **Davidian Regularization shows promise** as an alternative to minority class rebalancing:

- Outperformed random sampling in {overall_stats['overall_win_rate']:.1f}% of experiments
- Achieved an average improvement of {overall_stats['mean_improvement_pct']:+.2f}%
- The {best_method[0].replace('_', ' ').title()} method was most effective

"""
    else:
        report += f"""⚠️ **Mixed results** for Davidian Regularization:

- Outperformed random sampling in only {overall_stats['overall_win_rate']:.1f}% of experiments
- Average improvement was {overall_stats['mean_improvement_pct']:+.2f}%
- Results suggest method may be dataset or parameter dependent

"""
    
    report += f"""## Recommendations

Based on these results:

1. **Method Selection**: Use {best_method[0].replace('_', ' ').title()} for best performance
2. **Use Cases**: Most effective on datasets with moderate to high imbalance
3. **Implementation**: Consider as complement to, rather than replacement for, traditional rebalancing
4. **Further Research**: Investigate optimal hyperparameters and dataset characteristics

## Technical Details

- **Execution Time**: {summary['elapsed_time_seconds']:.1f} seconds ({summary['elapsed_time_seconds']/60:.1f} minutes)
- **Reproducibility**: All experiments used fixed random seeds
- **Evaluation**: Stratified splits maintained class proportions
- **Models**: Tested on linear, probabilistic, and ensemble methods

---

*Report generated automatically from experimental results*
*Experiment conducted: {time.strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    # Save report
    with open('EXPERIMENT_SUMMARY_REPORT.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("Summary report saved to EXPERIMENT_SUMMARY_REPORT.md")

def main():
    """Main execution function."""
    
    print("Starting comprehensive Davidian Regularization experiment pipeline...")
    
    # Check if results already exist
    if os.path.exists('results/comprehensive_davidian_results.json'):
        response = input("Results file already exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Using existing results for visualization and reporting...")
            
            # Skip to visualization and reporting
            try:
                from visualization_comprehensive import generate_comprehensive_report
                generate_comprehensive_report()
                create_summary_report()
                print("✓ Visualization and reporting completed successfully")
                return True
            except Exception as e:
                print(f"✗ Error in visualization/reporting: {e}")
                return False
    
    # Run full experiment
    success = run_full_experiment()
    
    if success:
        print("\n🎉 Comprehensive experiment pipeline completed successfully!")
        print("Check the generated files for detailed results and analysis.")
    else:
        print("\n❌ Experiment pipeline failed. Check error messages above.")
        return False
    
    return True

if __name__ == "__main__":
    main()
