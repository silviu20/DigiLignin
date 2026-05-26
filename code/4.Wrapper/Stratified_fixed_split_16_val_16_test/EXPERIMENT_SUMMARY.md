# Experiment Summary: Fixed Stratified Split vs OOF Method

## Experiment Overview

**Objective:** Test a modified data splitting strategy as an alternative to the out-of-fold (OOF) cross-validation approach.

**Location:** `C:\Users\sacaru\digilignin\DigiLignin\4.Wrapper\Stratified_fixed_split_16_val_16_test`

**Date Created:** February 24, 2026

## Key Methodology Changes

### Data Splitting

| Aspect | OOF Method | Fixed Split Method |
|--------|------------|-------------------|
| **Validation Set** | Rotating folds (5×2=10) | Fixed 16 samples |
| **Test Set** | None (validation only) | Fixed 16 samples |
| **Training Set** | Varies by fold | Fixed remaining samples |
| **Stratification** | Random K-fold | Systematic sampling by target |
| **Reproducibility** | Fold-dependent | Fully reproducible |

### Stratification Strategy

The fixed split uses **systematic sampling** to ensure representative distribution:

1. Sort all samples by target variable (`Tg(deg C)`)
2. Select validation samples at regular intervals across the sorted range
3. Select test samples from remaining data using same strategy
4. Use remaining samples for training

**Benefits:**
- Even coverage of target variable range
- Maintains diversity in all splits
- Reproducible with fixed random seed
- Separate validation and test sets for unbiased evaluation

## Model Saving Architecture

### Comprehensive Tracking System

Every trained model is saved with:

**Filename Format:**
```
model_{type}_combo{id}_n{estimators}_{timestamp}.joblib
```

**Model Types:**
- `base_gb`: Gradient Boosting base learner
- `base_rf`: Random Forest base learner
- `base_svr`: Support Vector Regression base learner
- `base_lasso`: Lasso base learner
- `base_elasticnet`: ElasticNet base learner
- `meta_ridge`: Ridge meta-learner

**Example:**
```
model_meta_ridge_combo5_n700_20260224_200530.joblib
```

### Model Registry

CSV file (`model_registry.csv`) tracks all models with:
- Model filename and type
- Feature combination details
- Hyperparameters used
- Performance metrics (train, validation, test)
- Split information
- Timestamp

**Total Models Saved per Run:**
- 511 feature combinations × 13 n_estimators values × 6 models per combination
- **≈ 39,858 individual models**
- Plus 6,643 ensemble bundles

## Experimental Design

### Feature Combinations

**Mandatory Features (always included):**
- `Lignin (wt%)`
- `Co-polyol type (PTHF)`
- `r`

**Optional Features (all combinations tested):**
- `Copolyol (wt%)`
- `Isocyanate (wt%)`
- `Isocyanate (mmol NCO)`
- `Isocyonate type`
- `tin(II) octoate`
- `Sratio(%)`

**Total Combinations:** 511 (2^9 - 1)

### N_Estimators Values

```python
[1, 10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
```

### Total Experiments

511 combinations × 13 n_estimators = **6,643 experiments**

## Expected Outputs

### Primary Results

1. **fixed_split_results.csv**
   - All experimental results
   - Train, validation, and test metrics
   - Feature combination details
   - 6,643 rows (one per experiment)

2. **model_registry.csv**
   - Complete model tracking database
   - ~39,858 rows (one per model)
   - Includes all hyperparameters and metrics

3. **split_statistics.json**
   - Data split characteristics
   - Distribution statistics for each split
   - Verification of stratification quality

### Visualizations

**fixed_split_analysis.{png,tiff,pdf,svg}**
- 6 comprehensive plots:
  - A: Average MAE vs N_Estimators (Val & Test)
  - B: Average R² vs N_Estimators (Val & Test)
  - C: Best Performance vs N_Estimators
  - D: Performance by Feature Count
  - E: MAE Heatmap (Features vs Estimators)
  - F: Top 10 Best Combinations

**method_comparison.{png,tiff,pdf,svg}** (after running compare_with_oof.py)
- 6 comparison plots between fixed split and OOF methods

### Model Files

**models/** directory containing:
- All trained base learners
- All trained meta-learners
- Complete ensemble bundles (scalers + models)
- Estimated size: 500MB - 1GB

## Performance Metrics

Each experiment reports:

**Training Metrics:**
- R² (coefficient of determination)
- MSE (mean squared error)
- MAE (mean absolute error)

**Validation Metrics:**
- R², MSE, MAE on validation set

**Test Metrics:**
- R², MSE, MAE on test set

All metrics calculated on **unscaled** (original) target values for interpretability.

## Comparison Analysis

The `compare_with_oof.py` script provides:

1. **Best Model Comparison**
   - Fixed split best (by test MAE)
   - OOF best (by validation MAE)
   - Performance differences

2. **Trend Analysis**
   - Performance by n_estimators
   - Correlation between methods
   - Generalization gap analysis

3. **Statistical Comparison**
   - Distribution differences
   - Confidence in results
   - Method agreement

## Advantages of This Approach

### Scientific Rigor
- True held-out test set for unbiased evaluation
- Clear separation between validation (tuning) and test (evaluation)
- Reproducible results with fixed random seed

### Practical Benefits
- Faster training (single split vs. multiple folds)
- Models ready for deployment (trained on fixed training set)
- Easy to track and compare specific models
- Comprehensive model registry for future reference

### Methodological Insights
- Understand generalization gap (validation vs. test)
- Compare with OOF to validate approach
- Assess stability of model selection

## Limitations

1. **Single Split Dependency**
   - Results depend on specific train/val/test split
   - No confidence intervals like OOF method
   - Potential for lucky/unlucky splits

2. **Smaller Training Set**
   - 32 samples held out (vs. 20% in OOF)
   - May impact performance on small datasets

3. **No Variance Estimation**
   - Single performance estimate per configuration
   - Cannot assess model stability across folds

## Recommendations

### When to Use Fixed Split
- Dataset is sufficiently large (>100 samples)
- Need unbiased test set evaluation
- Preparing models for production
- Want faster experimentation

### When to Use OOF
- Dataset is small (<100 samples)
- Need robust performance estimates
- Want to assess model stability
- Cross-validation is domain standard

### Best Practice
- Run both methods and compare
- Use OOF for model selection
- Use fixed split for final evaluation
- Report both sets of results

## Files Created

### Scripts
- `run_fixed_split_experiments.py` - Main experiment script
- `compare_with_oof.py` - Comparison analysis script
- `verify_setup.py` - Setup verification script

### Documentation
- `README.md` - Comprehensive documentation
- `QUICKSTART.md` - Quick start guide
- `EXPERIMENT_SUMMARY.md` - This file

### Directories
- `models/` - Model storage directory

## Next Steps

1. **Verify Setup**
   ```bash
   python verify_setup.py
   ```

2. **Run Experiment**
   ```bash
   python run_fixed_split_experiments.py
   ```

3. **Compare Methods**
   ```bash
   python compare_with_oof.py
   ```

4. **Analyze Results**
   - Review CSV files
   - Examine visualizations
   - Select best model
   - Deploy to production

## Citation

If using this methodology in publications, please document:
- Fixed stratified split strategy (16 val, 16 test)
- Systematic sampling for stratification
- Model tracking and registry system
- Comparison with OOF cross-validation

## Version History

- **v1.0** (2026-02-24): Initial implementation
  - Fixed stratified splitting
  - Comprehensive model saving
  - Model registry system
  - Comparison with OOF method

## Contact

For questions or issues, refer to the main project documentation or contact the project maintainer.
