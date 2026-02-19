# Corrected Stacking Ensemble Implementation

## Overview

This document describes the corrected stacking ensemble implementation that addresses critical data leakage issues found in the original DigiLignin codebase. The new implementation provides unbiased performance estimation through proper out-of-fold (OOF) predictions and nested cross-validation.

## Critical Issues Addressed

### 1. Data Leakage in Original Implementation

**Problem**: The original stacking method had severe data leakage:
- Base models were trained on the full dataset
- Meta-model was trained on predictions from the same data
- Validation metrics were calculated on in-sample predictions
- Scatter plots showed fitted values rather than true predictions

**Impact**: Overly optimistic performance metrics that don't reflect true generalization performance.

### 2. Methodological Corrections

**Solution**: Implemented proper stacking with:
- Out-of-fold (OOF) predictions for meta-features
- Nested cross-validation for unbiased performance estimation
- Strict held-out test set evaluation
- Transparent visualization of true test predictions

## Implementation Details

### File Structure

```
5.Model/
├── Stacked Ensembles.py              # Original implementation (WITH DATA LEAKAGE)
├── Corrected_Stacked_Ensembles.py    # Corrected implementation (UNBIASED)
├── Comparison_Analysis.py            # Comparison between methods
└── README_Corrected_Implementation.md # This documentation
```

### Key Functions

#### 1. `generate_oof_predictions()`
- Generates out-of-fold predictions for meta-features
- Prevents data leakage by ensuring each prediction comes from a model not trained on that sample
- Uses cross-validation to create unbiased meta-features

#### 2. `nested_cross_validation()`
- Performs nested cross-validation for unbiased performance estimation
- Outer CV: Performance estimation
- Inner CV: Hyperparameter tuning
- Provides confidence intervals for metrics

#### 3. `train_final_model_with_held_out_test()`
- Trains final model with strict held-out test set
- 20% of data never touched during training
- Provides true generalization performance estimate

#### 4. `plot_unbiased_results()`
- Creates plots using only held-out test predictions
- No data leakage in visualizations
- Clear labeling of prediction types

## Usage Guide

### Basic Usage

```python
import pandas as pd
from Corrected_Stacked_Ensembles import main_analysis

# Load your data
df = pd.read_csv('your_data.csv')

# Define features and target
X = df[['Lignin (wt%)', 'Ratio', 'Co-polyol type (PTHF)', 
        'Isocyanate (mmol NCO)', 'Isocyanate type', 
        'Tin(II) octoate', 'Swelling ratio (%)']]
y = df[['Tg (°C)']]

# Run complete analysis
nested_results, test_results, final_models = main_analysis(X, y)
```

### Advanced Usage

```python
from Corrected_Stacked_Ensembles import (
    nested_cross_validation,
    train_final_model_with_held_out_test,
    plot_unbiased_results
)

# 1. Nested cross-validation only
nested_results = nested_cross_validation(X, y)

# 2. Train final model with held-out test set
final_models, test_results = train_final_model_with_held_out_test(X, y)

# 3. Plot results
plot_unbiased_results(test_results, 'my_analysis')
```

## Performance Metrics

### Nested Cross-Validation Results
```python
# Access nested CV results
nested_results['r2']['mean']        # Mean R² with 95% CI
nested_results['r2']['ci_lower']    # Lower confidence bound
nested_results['r2']['ci_upper']    # Upper confidence bound
nested_results['r2']['raw_scores']  # Individual fold scores
```

### Held-Out Test Set Results
```python
# Access test set results
test_results['r2']          # Test R²
test_results['mse']         # Test MSE
test_results['mae']         # Test MAE
test_results['y_true']      # True values
test_results['y_pred']      # Predicted values
test_results['X_test']      # Test features
```

## Comparison with Original Method

### Key Differences

| Aspect | Original Method | Corrected Method |
|--------|----------------|------------------|
| Meta-features | Full dataset predictions | OOF predictions |
| Validation | In-sample predictions | Nested CV |
| Test evaluation | No strict test set | Held-out test set |
| Visualizations | Fitted values | True test predictions |
| Performance metrics | Overly optimistic | Unbiased estimates |

### Expected Performance Differences

- **R²**: Original method typically shows 10-30% higher R² due to leakage
- **MAE**: Original method typically underestimates error by 20-50%
- **Confidence**: Original method provides false confidence in model performance

## Model Persistence

### Saving Models
```python
import joblib

# Models are automatically saved during analysis
# Final models saved to: 'corrected_stacked_models.joblib'

# Manual saving
joblib.dump(final_models, 'my_models.joblib')
```

### Loading Models
```python
import joblib

# Load saved models
models = joblib.load('corrected_stacked_models.joblib')

# Access components
base_models = models['base_models']
meta_model = models['meta_model']
oof_scaler = models['oof_scaler']
y_scaler = models['y_scaler']
```

### Making New Predictions
```python
def predict_new_data(new_X, models):
    """Make predictions on new data using trained models."""
    
    # Extract model components
    base_models = models['base_models']
    meta_model = models['meta_model']
    oof_scaler = models['oof_scaler']
    y_scaler = models['y_scaler']
    
    # Generate meta-features
    meta_features = []
    for name, model, X_scaler, y_scaler_model in base_models:
        X_scaled = X_scaler.transform(new_X)
        pred_scaled = model.predict(X_scaled)
        pred = y_scaler_model.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
        meta_features.append(pred)
    
    meta_features = np.column_stack(meta_features)
    meta_scaled = oof_scaler.transform(meta_features)
    
    # Final prediction
    pred_scaled = meta_model.predict(meta_scaled)
    pred = y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
    
    return pred
```

## Validation and Quality Assurance

### Statistical Validation
- **Confidence Intervals**: 95% CI for all performance metrics
- **Nested CV**: Unbiased performance estimation
- **Held-out Test**: True generalization performance

### Visual Validation
- **Test Set Plots**: Only true test predictions shown
- **Residual Analysis**: Proper residual plots
- **Clear Labeling**: All plots clearly indicate prediction types

### Reproducibility
- **Random Seeds**: Fixed seeds for reproducible results
- **Consistent Splits**: Same CV splits across models
- **Version Control**: Clear documentation of method changes

## Recommendations for Deployment

### 1. Model Selection
- Use nested CV results for model comparison
- Select models based on unbiased test performance
- Consider confidence intervals in decision making

### 2. Performance Reporting
- Always report both nested CV and held-out test results
- Include confidence intervals
- Clearly state evaluation methodology

### 3. Deployment Monitoring
- Monitor model performance on new data
- Compare to expected performance from test set
- Retrain if performance degrades significantly

## Common Pitfalls to Avoid

### 1. Data Leakage
- Never use test data for training
- Always use OOF predictions for meta-features
- Separate training and validation strictly

### 2. Overfitting
- Use nested CV for hyperparameter tuning
- Monitor train vs validation performance
- Consider model complexity

### 3. Misleading Metrics
- Don't report in-sample performance
- Always use test set metrics for final reporting
- Include uncertainty estimates

## Troubleshooting

### Common Issues

1. **Memory Issues with Large Datasets**
   - Reduce CV folds or use smaller batch sizes
   - Consider incremental learning methods

2. **Long Training Times**
   - Reduce hyperparameter search space
   - Use parallel processing if available
   - Consider feature selection

3. **Poor Performance**
   - Check data quality and preprocessing
   - Verify feature engineering
   - Consider alternative models

### Debug Mode
```python
# Enable debug mode for detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Run with reduced data for testing
X_small = X.iloc[:50]  # Use small subset
y_small = y.iloc[:50]
nested_results, test_results, final_models = main_analysis(X_small, y_small)
```

## References

1. **Stacking Generalization**: Wolpert, D.H. (1992). "Stacked generalization"
2. **Cross-Validation**: Kohavi, R. (1995). "A study of cross-validation"
3. **Nested Cross-Validation**: Varma, S., Simon, R. (2006). "Bias in error estimation"
4. **Out-of-Fold Predictions**: Sill, J., et al. (2009). "Feature-selected models"

## Conclusion

The corrected stacking implementation provides:
- ✅ Unbiased performance estimation
- ✅ Proper data leakage prevention
- ✅ Reliable model evaluation
- ✅ Transparent reporting
- ✅ Reproducible results

This ensures that reported performance metrics truly reflect the model's ability to generalize to new data, which is critical for reliable scientific conclusions and model deployment.
