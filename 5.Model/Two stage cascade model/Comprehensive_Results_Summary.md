# Comprehensive Model Comparison Results

## Performance Metrics Summary

| Rank | Model | MAE Validation (°C) | MAE Train (°C) | Improvement vs Baseline | R² Validation | R² Train | Generalizability | Key Features | Data Leakage | Practical Use |
|------|-------|---------------------|----------------|-------------------------|---------------|----------|------------------|--------------|---------------|---------------|
| 1 | Original Stacked Ensemble | 11.31 | 11.33 | +33.74% | 0.687 | 0.683 | -0.02 | All formulation + swelling | Yes (original) | Limited (requires synthesis) |
| 2 | Fixed Stacked Ensemble | 16.38 | 16.00 | +4.04% | 0.295 | 0.392 | 0.38 | Reduced formulation + swelling | No (fixed) | Limited (requires synthesis) |
| 3 | Stage 2: Tg Prediction (Cascade) | 16.67 | 16.56 | +2.34% | 0.296 | 0.373 | 0.11 | Formulation + predicted swelling | No | High (fully predictive) |
| 4 | Baseline: Formulation Only | 17.07 | 16.93 | +0.00% | 0.286 | 0.341 | 0.13 | Formulation only | No | High (fully predictive) |
| 5 | Stage 1: Swelling Prediction | 24.83* | 23.10* | -45.46% | 0.669 | 0.742 | 1.73 | Formulation only | No | High (fully predictive) |

*Note: Stage 1 predicts swelling ratio (%), not temperature (°C)

## Key Insights

### 🏆 **Performance Ranking**
1. **Original Stacked Ensemble** (11.31°C) - Best accuracy but has data leakage
2. **Fixed Stacked Ensemble** (16.38°C) - Good accuracy, no leakage
3. **Cascade Model** (16.67°C) - Competitive accuracy, fully predictive
4. **Baseline** (17.07°C) - Reasonable accuracy, simplest approach
5. **Stage 1** (24.83%) - Different target (swelling ratio)

### 🔍 **Critical Findings**

#### Data Leakage Impact
- **Original Model**: 33.74% better than baseline due to data leakage
- **Fixed Model**: Only 4.04% improvement over baseline when leakage is fixed
- This shows the original results were artificially inflated

#### Practical vs Research Trade-offs
- **Research Use**: Fixed Stacked Ensemble (most accurate without leakage)
- **Practical Use**: Cascade Model (best balance of accuracy and practicality)
- **Quick Predictions**: Baseline (simplest implementation)

#### Generalizability Analysis
- **Negative values** (Original): Potential overfitting
- **Low values** (Cascade, Baseline): Good generalization
- **High values** (Stage 1): Poor generalization for swelling prediction

## Recommendations by Use Case

### 🎯 **For Research Publications**
- **Model**: Fixed Stacked Ensemble
- **Reason**: Highest accuracy without data leakage issues
- **MAE**: 16.38°C
- **Trade-off**: Requires swelling ratio measurement (synthesis needed)

### 🏭 **For Industrial Application**
- **Model**: Cascade Model (Stage 2)
- **Reason**: Fully predictive with good accuracy
- **MAE**: 16.67°C (only 0.59°C worse than fixed ensemble)
- **Advantage**: No synthesis required, formulation only

### ⚡ **For Quick Screening**
- **Model**: Baseline (Formulation Only)
- **Reason**: Simplest implementation
- **MAE**: 17.07°C (only 0.40°C worse than cascade)
- **Advantage**: Minimal computational requirements

## Technical Details

### Feature Sets Used
- **Original**: All 8 formulation features + swelling ratio
- **Fixed**: 6 reduced features (VIF < 10) + swelling ratio  
- **Baseline**: 6 reduced features only
- **Cascade**: 6 reduced features + predicted swelling ratio

### Model Architecture
- **Stacking**: Gradient Boosting + Random Forest + SVR + Lasso + ElasticNet → Meta-model
- **Cascade**: Stage 1 (Formulation → Swelling) → Stage 2 (Formulation + Predicted Swelling → Tg)

### Validation Method
- **10-fold cross-validation** with proper out-of-fold predictions
- **No data leakage** between training and validation sets
- **Generalizability metric**: Validation MAE - Training MAE

## Files Generated
- `Comprehensive_Model_Comparison.csv` - Raw comparison data
- `VIF_Analysis_Files/` - Feature reduction results
- `Stacking_Ensemble_Files/` - Fixed ensemble models and results
- `Cascade_Model_Files/` - Two-stage cascade models and results
