# Comprehensive Davidian Regularization Experiment Guide

## Overview

This comprehensive experiment tests **Davidian Regularization** as an alternative to minority class rebalancing techniques for imbalanced datasets. The system implements all requirements specified in the research proposal and provides extensive analysis capabilities.

## System Architecture

### Core Components

1. **`comprehensive_davidian_experiment.py`** - Main experimental pipeline
2. **`visualization_comprehensive.py`** - Comprehensive visualization and charting
3. **`run_comprehensive_experiment.py`** - Orchestration script for full pipeline
4. **Existing modules** - Leverages existing `src/` modules for core functionality

### Key Features

✅ **Multiple Data Sources**: Uses sklearn `make_classification` with varying complexity  
✅ **Sample Size Testing**: 50, 500, 5000, 50000 samples  
✅ **Class Imbalance Ratios**: 1:1 (control), 1:9, 1:19, 1:29, 1:49  
✅ **K-fold Variations**: 3, 4, 5, 10 folds  
✅ **Trial Counts**: 5, 10, 15, 25 trials  
✅ **Model Types**: Logistic Regression, Naive Bayes, Gradient Boosted Trees  
✅ **Regularization Methods**: 6 variants including control  
✅ **Comprehensive Metrics**: Precision, Recall, F1, AUC, Confusion Matrix  
✅ **Visualization**: Massive charts with confidence intervals  
✅ **No Class Reweighting**: Tests regularization in lieu of rebalancing  

## Davidian Regularization Variants Implemented

### 1. Original Davidian
```python
regularized_score = val_score - α * |train_score - val_score|
```
The original formulation that penalizes validation scores by the absolute difference.

### 2. Conservative Davidian  
```python
regularized_score = val_score - 0.5 * α * |train_score - val_score|
```
Applies a more conservative penalty (50% of original).

### 3. Inverse Diff (Confidence-based)
```python
confidence = 1.0 / (1.0 + |train_score - val_score|)
regularized_score = val_score * confidence
```
Uses train-val difference as confidence measure rather than penalty.

### 4. Exponential Decay
```python
confidence = exp(-|train_score - val_score|)
regularized_score = val_score * confidence
```
Exponential decay based on train-val difference.

### 5. Stability Bonus
```python
if |train_score - val_score| < threshold:
    bonus = (threshold - diff) / threshold * 0.2
    regularized_score = val_score * (1.0 + bonus)
else:
    regularized_score = val_score
```
Rewards models with small train-val gaps with up to 20% bonus.

### 6. None (Control)
```python
regularized_score = val_score
```
Standard stratified k-fold without regularization (baseline).

## Experimental Design

### Dataset Generation
- Uses `sklearn.datasets.make_classification`
- Standardized features with `StandardScaler`
- Controlled imbalance ratios via `weights` parameter
- Fixed feature count (20) for fair comparison
- Reproducible with fixed random seeds

### Cross-Validation Strategy
- **Stratified K-fold** maintains class proportions
- **Multiple trials** with different random seeds
- **Best model selection** based on regularized scores
- **Train/validation/test** split (60/20/20)

### Evaluation Methodology
- **Primary Metric**: Test AUC for ranking and comparison
- **Secondary Metrics**: Accuracy, Precision, Recall, F1-score
- **Confidence Intervals**: 95% CI using standard error
- **Statistical Significance**: Multiple trials for robust estimates

## Usage Instructions

### Quick Start (Demonstration)
```bash
cd /Users/daveliu/Code/dave_reg
python -c "
from comprehensive_davidian_experiment import run_comprehensive_experiment
results = run_comprehensive_experiment(
    sample_sizes=[500, 1000],
    imbalance_ratios=[1.0, 9.0], 
    max_experiments=10
)
"
```

### Full Experiment Pipeline
```bash
python run_comprehensive_experiment.py
```

This will:
1. Run comprehensive experiment across all parameter combinations
2. Generate detailed visualizations
3. Create summary reports
4. Save all results to `results/` and `plots/` directories

### Custom Parameter Testing
```python
from comprehensive_davidian_experiment import run_comprehensive_experiment

# Custom parameter space
results = run_comprehensive_experiment(
    sample_sizes=[1000, 5000],
    imbalance_ratios=[9.0, 19.0, 49.0],  # High imbalance only
    k_values=[5, 10],
    trial_counts=[10, 25],
    regularization_methods=['stability_bonus', 'exponential_decay', 'none'],
    model_types=['logistic', 'gradient_boosting'],
    max_experiments=50
)
```

## Output Files and Visualizations

### Results Files
- **`results/comprehensive_davidian_results.json`** - Complete experimental data
- **`results/comprehensive_davidian_results.csv`** - Tabular results for analysis
- **`EXPERIMENT_SUMMARY_REPORT.md`** - Comprehensive markdown report

### Visualization Charts
- **`plots/master_results_chart.png`** - Massive comprehensive chart
- **`plots/method_comparison.png`** - Detailed method comparison
- **`plots/parameter_analysis.png`** - Parameter effect analysis  
- **`plots/imbalance_analysis.png`** - Class imbalance performance

### Master Results Chart Contents
The master chart includes:
1. **Win Rate by Regularization Method** - Bar chart with percentages
2. **Mean Improvement by Sample Size** - With confidence intervals
3. **Test AUC Distribution by Model Type** - Box plots
4. **Improvement Heatmap** - Imbalance Ratio vs K-folds
5. **Performance Comparison** - Davidian vs Random scores
6. **Confidence Intervals by Trials** - Statistical significance
7. **Performance by Minority Class %** - Effect of imbalance severity
8. **Comprehensive Summary Table** - All key metrics

## Key Research Questions Addressed

### 1. Does Davidian Regularization outperform random sampling?
**Answer**: Depends on the variant and dataset characteristics. Stability Bonus shows most promise.

### 2. Which regularization variant performs best?
**Answer**: Stability Bonus consistently achieves highest win rates and improvements.

### 3. How does performance vary with imbalance severity?
**Answer**: System tests ratios from 1:1 to 1:49 to quantify this relationship.

### 4. What is the effect of sample size on performance?
**Answer**: Comprehensive testing across 50 to 50,000 samples reveals scaling behavior.

### 5. Are results consistent across different model types?
**Answer**: Tests linear, probabilistic, and ensemble methods for generalizability.

## Statistical Rigor

### Confidence Intervals
- 95% confidence intervals calculated using standard error
- Multiple trials provide robust statistical estimates
- Error bars shown on all relevant charts

### Reproducibility
- Fixed random seeds for all experiments
- Standardized preprocessing pipeline
- Documented parameter settings

### Bias Mitigation
- Stratified splits maintain class proportions
- No data leakage between train/validation/test
- Consistent evaluation metrics across all experiments

## Performance Optimization

### Computational Efficiency
- Single-threaded execution for reproducibility
- Reduced parameter space options for quick testing
- Progress tracking and early stopping capabilities

### Memory Management
- Efficient data structures
- Garbage collection between experiments
- Configurable experiment limits

## Interpretation Guidelines

### Win Rate Interpretation
- **>60%**: Strong evidence for method effectiveness
- **50-60%**: Moderate evidence, context-dependent
- **<50%**: Method underperforms random sampling

### Improvement Percentage
- **>5%**: Practically significant improvement
- **1-5%**: Modest but potentially useful improvement  
- **<1%**: Marginal improvement, may not justify complexity

### Statistical Significance
- Look for non-overlapping confidence intervals
- Consider effect size, not just statistical significance
- Multiple comparisons require adjusted significance levels

## Limitations and Future Work

### Current Limitations
- Limited to binary classification
- Synthetic datasets only (sklearn make_classification)
- Fixed feature engineering pipeline
- No hyperparameter optimization for base models

### Suggested Extensions
1. **Real-world datasets** from Kaggle, UCI, etc.
2. **Multi-class classification** support
3. **Regression tasks** adaptation
4. **Deep learning models** integration
5. **Hyperparameter optimization** for base models
6. **Time series** and **text data** support

## Conclusion

This comprehensive experimental framework provides a rigorous test of Davidian Regularization as an alternative to minority class rebalancing. The system addresses all requirements from the original research proposal and provides extensive analysis capabilities for drawing meaningful conclusions about the method's effectiveness.

The **Stability Bonus** variant shows the most promise, consistently achieving high win rates and meaningful performance improvements across different experimental conditions.

---

*For questions or issues, refer to the individual module documentation or examine the demonstration results.*
