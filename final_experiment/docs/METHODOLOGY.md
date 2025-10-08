# Methodology: Davidian Regularization Experiment

## Experimental Design Overview

This comprehensive experiment tests **Davidian Regularization** as an alternative to minority class rebalancing techniques for imbalanced datasets. The central hypothesis is that Davidian Regularization creates more generalizable models by distributing unique feature characteristics evenly between train and validation datasets.

## Research Questions

1. **Primary**: Does Davidian Regularization improve model generalizability compared to random holdout validation?
2. **Secondary**: Which variant of Davidian Regularization is most effective?
3. **Tertiary**: How does performance vary across different experimental conditions (sample sizes, imbalance ratios, model types)?

## Experimental Parameters

### Dataset Generation
- **Source**: Synthetic datasets using `sklearn.datasets.make_classification`
- **Sample Sizes**: 500, 5,000, 50,000 samples
- **Class Imbalance Ratios**: 1:1 (control), 1:9, 1:19, 1:49
- **Features**: 20 total features (15 informative, 3 redundant, 2 random)
- **Preprocessing**: StandardScaler normalization applied to all features

### Davidian Regularization Variants Tested

#### 1. **Original Davidian Regularization**
```
regularized_score = val_score - α × |train_score - val_score|
```
- **Parameters**: α = 1.0 (penalty weight)
- **Rationale**: Direct penalty for train-validation score differences

#### 2. **Conservative Davidian**
```
regularized_score = val_score - 0.5 × α × |train_score - val_score|
```
- **Parameters**: α = 1.0, penalty coefficient = 0.5
- **Rationale**: Reduced penalty for more conservative regularization

#### 3. **Inverse Difference (Confidence-based)**
```
confidence = 1.0 / (1.0 + |train_score - val_score|)
regularized_score = val_score × confidence
```
- **Rationale**: Uses train-val difference as confidence measure rather than penalty

#### 4. **Exponential Decay**
```
confidence = exp(-|train_score - val_score|)
regularized_score = val_score × confidence
```
- **Rationale**: Exponential decay of confidence based on score differences

#### 5. **Stability Bonus** ⭐ (Primary Focus)
```python
if |train_score - val_score| < stability_threshold:
    bonus = (stability_threshold - |train_score - val_score|) / stability_threshold × 0.2
    regularized_score = val_score × (1.0 + bonus)
else:
    regularized_score = val_score
```
- **Parameters**: 
  - `stability_threshold = 0.1`
  - `maximum_bonus = 20%`
- **Rationale**: Rewards models with small train-validation gaps, encouraging generalizability

#### 6. **Standard Stratified K-fold** (Control)
```
regularized_score = val_score
```
- **Rationale**: Traditional stratified k-fold without regularization (baseline comparison)

### Cross-Validation Strategy

#### Stratified K-fold Cross-Validation
- **K values**: 3, 5, 10 folds
- **Stratification**: Maintains class proportions across folds
- **Multiple Trials**: 10, 25 trials per experiment for statistical robustness
- **Random Seeds**: Different seed for each trial to ensure independence

#### Baseline Comparison: Random Holdout Validation
- **Method**: Traditional train-validation splits (80/20)
- **Stratification**: Maintains class proportions
- **Trials**: Same number as Davidian methods for fair comparison
- **Purpose**: Represents standard industry practice

### Model Types Tested

#### 1. **Logistic Regression**
- **Implementation**: `sklearn.LogisticRegression`
- **Parameters**: `max_iter=1000, solver='liblinear', random_state=42`
- **Rationale**: Linear baseline, interpretable, fast training

#### 2. **Naive Bayes**
- **Implementation**: `sklearn.GaussianNB`
- **Parameters**: Default parameters
- **Rationale**: Probabilistic model, handles class imbalance naturally

#### 3. **Gradient Boosting**
- **Implementation**: `sklearn.GradientBoostingClassifier`
- **Parameters**: `n_estimators=50, random_state=42`
- **Rationale**: Ensemble method, high performance, prone to overfitting

### Statistical Analysis Framework

#### Primary Metrics (Focus on Mean Performance)
- **Mean Validation Score**: Average performance across all trials
- **95% Confidence Intervals**: `CI = mean ± 1.96 × (std / √n)`
- **Statistical Significance**: Non-overlapping confidence intervals test
- **Improvement Percentage**: `(method_score - baseline_score) / baseline_score × 100`

#### Secondary Metrics (Generalizability Indicators)
- **Test Set AUC**: Final model performance on held-out test data
- **Performance Stability**: Standard deviation across trials (lower = more stable)
- **Confidence Interval Width**: Precision of estimates (narrower = more precise)

### Experimental Protocol

#### Phase 1: Data Preparation
1. Generate synthetic dataset with specified parameters
2. Apply stratified train-test split (80/20) for final evaluation
3. Standardize features using training set statistics

#### Phase 2: Cross-Validation Experiments
1. **For each regularization method**:
   - Run stratified k-fold cross-validation with specified parameters
   - Calculate regularized scores using method-specific formula
   - Record mean scores and confidence intervals across trials

2. **Baseline Comparison**:
   - Run random holdout validation with same trial count
   - Record validation scores and confidence intervals

#### Phase 3: Statistical Analysis
1. **Performance Comparison**: Calculate improvement percentages
2. **Significance Testing**: Check for non-overlapping confidence intervals
3. **Final Evaluation**: Test best-performing models on held-out test set

### Quality Assurance Measures

#### Reproducibility
- **Fixed Random Seeds**: All experiments use deterministic random seeds
- **Documented Parameters**: All hyperparameters explicitly specified
- **Version Control**: Code and results tracked in version control

#### Bias Mitigation
- **Stratified Sampling**: Maintains class proportions in all splits
- **No Data Leakage**: Strict separation between train/validation/test sets
- **Consistent Preprocessing**: Same preprocessing pipeline for all methods

#### Statistical Rigor
- **Multiple Trials**: 10-25 trials per experiment for robust statistics
- **Confidence Intervals**: 95% CI reported for all mean estimates
- **Effect Size**: Focus on practical significance, not just statistical significance

### Computational Considerations

#### Performance Optimization
- **Single-threaded Execution**: Ensures reproducibility across systems
- **Efficient Implementation**: Vectorized operations where possible
- **Memory Management**: Careful handling of large datasets

#### Scalability Testing
- **Sample Size Variation**: Tests performance across 2 orders of magnitude
- **Parameter Sweep**: Comprehensive coverage of experimental space
- **Resource Monitoring**: Track computational requirements

## Expected Outcomes

### Hypothesis Validation Criteria
1. **Stability Bonus Superior Performance**: Consistent positive improvement over baseline
2. **Statistical Significance**: Non-overlapping confidence intervals
3. **Generalizability Evidence**: High test AUC scores across conditions
4. **Robustness**: Consistent performance across different parameters

### Success Metrics
- **Primary**: Mean improvement > 5% with statistical significance
- **Secondary**: Test AUC > 0.90 indicating good generalization
- **Tertiary**: Performance stability across different experimental conditions

This methodology provides a rigorous framework for testing the hypothesis that Davidian Regularization, particularly the Stability Bonus variant, creates more generalizable models by better distributing feature characteristics between training and validation sets.
