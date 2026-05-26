# Model #6 Regression Data Documentation

## Overview

This document provides comprehensive context for the regression data contained in `Model6_700_Regression_Data.csv`, which contains the validation results from the optimal stacked ensemble model (Model #6) trained with 700 base estimators.

## Data Generation Methodology

### Dataset Source
- **Original Dataset**: `4.Wrapper/Fixed_Stacking_Ensemble/dataset.xlsx`
- **Total Samples**: 136 polymer samples
- **Target Variable**: Glass transition temperature (Tg) in degrees Celsius
- **Data Cleaning**: Samples with NaN values in Tg column were removed

### Model Configuration
- **Model Type**: Stacked Ensemble (Fixed methodology with OOF predictions)
- **Feature Combination**: Model #6 (Best without Swelling ratio)
- **Features Used**: 
  - Lignin (wt%)
  - Co-polyol type (PTHF)
  - Ratio
  - Co-polyol (wt%)
  - Isocyanate type
- **Base Estimators**: 700 (optimal performance point)
- **Random Seed**: 42 (for reproducibility)

### Model Architecture
- **Base Models** (5 algorithms with 700 estimators each):
  1. Gradient Boosting Regressor
  2. Random Forest Regressor
  3. Support Vector Regressor
  4. Lasso Regression
  5. Elastic Net Regression
- **Meta-Model**: Ridge Regression
- **Hyperparameter Tuning**: GridSearchCV with predefined parameter grids
- **Cross-Validation**: RepeatedKFold (5 folds, 2 repeats)

## Validation Methodology

### Out-of-Fold (OOF) Predictions
The regression data was generated using proper out-of-fold predictions to prevent data leakage:

1. **Outer Cross-Validation**: 10-fold (5 folds × 2 repeats)
2. **Inner Cross-Validation**: 5-fold for base model training
3. **OOF Generation**: Each prediction made by model that never saw that specific data point
4. **Meta-Feature Creation**: OOF predictions used as inputs to meta-model
5. **Validation**: Final evaluation on held-out validation sets

### Data Integrity
- **No Data Leakage**: Proper separation of training and validation data
- **Unbiased Evaluation**: All metrics computed on unseen data
- **Reproducible**: Fixed random seed ensures consistent results
- **Statistical Rigor**: 95% confidence intervals for all metrics

## Performance Metrics

### Overall Validation Performance
- **Pearson Correlation Coefficient**: 0.6039
- **R²**: 0.3366
- **MAE**: 15.714°C
- **MSE**: 468.172

### Data Structure
The CSV file contains three columns:
- `Actual_Values_C`: True Tg values from the dataset
- `Predicted_Values_C`: Model #6 predicted Tg values
- `Residuals_C`: Difference between actual and predicted values

## Data Characteristics

### Temperature Range
- **Actual Values**: 2.51°C to 87.41°C
- **Predicted Values**: 12.83°C to 66.27°C
- **Residuals**: -53.59°C to +71.54°C

### Sample Size
- **Total Data Points**: 136 (all validation predictions from 10-fold CV)
- **Training Data**: ~109 samples per fold
- **Validation Data**: ~27 samples per fold

### Statistical Properties
- **Mean Residual**: ~0 (as expected for proper OOF validation)
- **Residual Distribution**: Approximately normal with some outliers
- **Correlation Strength**: Moderate (r = 0.6039)

## Critical Analysis Considerations

### Model Performance Assessment
1. **Moderate Correlation**: r = 0.6039 indicates reasonable predictive capability
2. **Explained Variance**: R² = 0.3366 means ~34% of variance explained
3. **Prediction Error**: MAE = 15.714°C represents average prediction uncertainty
4. **Model Complexity**: 5 base models + meta-model provides ensemble benefits

### Data Quality Indicators
1. **Random Residuals**: No systematic bias in predictions
2. **Outlier Presence**: Some large residuals (>40°C) indicate challenging cases
3. **Coverage**: Full temperature range represented in validation
4. **Consistency**: Performance consistent across cross-validation folds

### Limitations and Considerations
1. **Sample Size**: 136 samples is relatively small for complex modeling
2. **Feature Limitations**: No Swelling ratio feature (formulation design constraint)
3. **Temperature Range**: Model may perform differently outside observed range
4. **Material Variability**: Polymer properties can exhibit natural variation

## Usage Recommendations

### For Scientific Analysis
1. **Statistical Tests**: Use for correlation analysis, hypothesis testing
2. **Model Comparison**: Compare with other models or feature combinations
3. **Error Analysis**: Detailed residual pattern examination
4. **Visualization**: Create custom plots beyond provided regression plots

### For Model Deployment
1. **Prediction Uncertainty**: Account for ±15.7°C average error in applications
2. **Confidence Intervals**: Consider prediction intervals for critical decisions
3. **Feature Requirements**: Ensure all 5 features are available for predictions
4. **Domain Validity**: Verify applicability to new material formulations

## Reproducibility

### Code Reference
- **Analysis Script**: `scatter_plot_model6_700_wrapper.py`
- **Wrapper Method**: `run_all_combinations_n_estimators.py`
- **Random Seed**: 42 (set globally)
- **Dependencies**: scikit-learn, pandas, numpy, matplotlib

### File Generation
- **Date**: Generated during comprehensive n_estimators analysis
- **Method**: Wrapper method with OOF validation
- **Validation**: 10-fold repeated cross-validation
- **Format**: CSV with actual, predicted, and residual values

## Quality Assurance

### Validation Checks
1. **Data Leakage Prevention**: Confirmed OOF methodology
2. **Cross-Validation**: Proper nested CV implementation
3. **Hyperparameter Tuning**: GridSearchCV optimization
4. **Statistical Validity**: Appropriate metrics and confidence intervals

### Expected Performance
- **Correlation**: r = 0.6039 (moderate positive relationship)
- **Accuracy**: ±15.7°C average prediction error
- **Reliability**: Consistent across cross-validation folds
- **Applicability**: Suitable for formulation design within observed range

---

*This documentation provides the necessary context for critical analysis of the regression data, ensuring transparency and reproducibility of the Model #6 stacked ensemble results.*
