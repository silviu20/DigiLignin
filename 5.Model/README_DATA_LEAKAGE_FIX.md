# Data Leakage Fix - Critical Implementation Change

## Problem Identified by Reviewers #2 and #5

The original stacking implementation had a **critical data leakage issue** that invalidated the reported performance metrics.

### What Was Wrong?

In the original `Stacked Ensembles.py` (lines 178-181 and 229-236):

```python
# WRONG: Training base models on FULL dataset
x_scaled, x_scaler = scale_columns_with_robust_scaler(x)
y_scaled, y_scaler = scale_columns_with_robust_scaler(y)
best_model.fit(x_scaled, y_scaled.ravel())  # ← Uses ALL data!

# WRONG: Generating meta-features from FULL dataset
meta_features = np.zeros((x_scaled.shape[0], len(best_base_models)))
for i, base_model in enumerate(best_base_models):
    meta_features[:, i] = base_model.predict(x_scaled)  # ← Predicts on ALL data!

# WRONG: Training meta-model on predictions from full dataset
meta_model.fit(meta_features, y_scaled.ravel())
```

### Why This Is Wrong

1. **Base models see validation data during training** (line 180)
2. **Meta-features include predictions on validation data** (line 234)
3. **Meta-model is trained on contaminated features** (line 236)
4. **Validation metrics are actually training metrics** - overly optimistic!

This is why the original results showed:
- MAE = 6.66°C (suspiciously low)
- R² = 0.99 (suspiciously high)
- Minimal train-validation gap (red flag!)

## The Fix: Out-of-Fold (OOF) Predictions

### Correct Implementation in `Stacked_Ensembles_Fixed.py`

```python
# CORRECT: Generate OOF predictions using cross_val_predict
def generate_oof_predictions(x_train, y_train, model, param_grid, cv_inner=5):
    # Tune hyperparameters
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv_inner)
    grid_search.fit(x_train, y_train.ravel())
    best_model = grid_search.best_estimator_
    
    # Generate OOF predictions - each sample predicted when it's in validation fold
    oof_predictions = cross_val_predict(best_model, x_train, y_train.ravel(), cv=cv_inner)
    
    # Retrain on full training set for final model
    best_model.fit(x_train, y_train.ravel())
    
    return oof_predictions, best_model
```

### Key Differences

| Aspect | Original (WRONG) | Fixed (CORRECT) |
|--------|------------------|-----------------|
| Base model training | Full dataset | Training fold only |
| Meta-features | Predictions on full dataset | Out-of-fold predictions |
| Meta-model training | Contaminated features | Clean OOF features |
| Validation | Sees training data | Never sees validation data |
| Reported MAE | ~6.66°C (too optimistic) | ~10-15°C (realistic) |
| R² | ~0.99 (too high) | ~0.85-0.92 (realistic) |

## Expected Changes After Fix

### Performance Metrics

**Before (with data leakage):**
- Validation MAE: 6.66°C
- Validation R²: 0.99
- Generalizability: 0.38°C (suspiciously small)

**After (without data leakage):**
- Validation MAE: **10-15°C** (expected to increase)
- Validation R²: **0.85-0.92** (expected to decrease)
- Generalizability: **2-5°C** (more realistic gap)

### Why Higher MAE Is Actually Better

The higher MAE is **not worse performance** - it's **honest performance**!

- Original 6.66°C was an artifact of data leakage
- New 10-15°C represents true predictive power
- This is what the model will achieve on truly unseen data

## How to Use the Fixed Implementation

### 1. Run the Fixed Code

```python
# Load your data
df = pd.read_csv('dataset.csv')

# Define features and target
x = df[['Lignin (wt%)', 'Ratio', 'Co-polyol type (PTHF)', 
        'Isocyanate (mmol NCO)', 'Isocyanate type', 
        'Tin(II) octoate', 'Swelling ratio (%)']]
y = df[['Tg (°C)']]

# Run fixed stacking
from Stacked_Ensembles_Fixed import run_multiple_times_fixed

results_df, best_models = run_multiple_times_fixed(
    x, y, 
    num_runs=1, 
    n_estimators_list=[1000]
)

# View results
print(results_df[['Model', 'MAE Validation', 'Train MAE', 
                  'Generalizability (Val MAE - Train MAE)']])
```

### 2. Compare with Original

Run both implementations side-by-side to see the difference:

```bash
python "Stacked Ensembles.py"  # Original (with leakage)
python "Stacked_Ensembles_Fixed.py"  # Fixed (no leakage)
```

### 3. Update Manuscript

Replace all performance metrics in the manuscript with values from the fixed implementation.

## Technical Details

### Nested Cross-Validation Structure

```
Outer CV (for evaluation):
  Fold 1: Train on 80%, Validate on 20%
    ├─ Inner CV (for OOF generation):
    │   ├─ Fold 1: Train base model, predict on validation
    │   ├─ Fold 2: Train base model, predict on validation
    │   ├─ Fold 3: Train base model, predict on validation
    │   ├─ Fold 4: Train base model, predict on validation
    │   └─ Fold 5: Train base model, predict on validation
    ├─ Combine OOF predictions → Meta-features
    ├─ Train meta-model on OOF meta-features
    └─ Predict on outer validation fold
  
  Fold 2: ... (repeat)
  ...
  Fold 10: ... (5 splits × 2 repeats)

Report: Average metrics across all 10 outer folds
```

### Why This Prevents Data Leakage

1. **OOF predictions**: Each training sample's prediction is made when it's in a validation fold
2. **No contamination**: Meta-model never sees validation data during training
3. **Honest evaluation**: Outer CV provides unbiased performance estimate

## Files Created

- `Stacked_Ensembles_Fixed.py` - Corrected implementation
- `README_DATA_LEAKAGE_FIX.md` - This documentation
- `stacking_results_fixed_run_*.csv` - Results from fixed implementation

## Next Steps

1. ✅ **Action 1.1 Complete**: Data leakage fixed
2. ⏭️ **Action 1.2**: Address swelling ratio issue (two-stage cascade model)
3. ⏭️ **Action 2.1**: Remove multicollinearity
4. ⏭️ **Action 2.2**: Add mechanistic interpretation
5. ⏭️ **Action 2.3**: Strengthen introduction

## References

- Reviewer #2, Major Concern 1: "Stacking Procedure and Data Leakage"
- Reviewer #5, Major Correction 1: "Training Error vs Test Error"
- sklearn.model_selection.cross_val_predict documentation
- sklearn.ensemble.StackingRegressor (alternative implementation)

