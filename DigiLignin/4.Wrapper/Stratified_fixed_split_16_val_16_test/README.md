# Fixed Stratified Split Ensemble Experiment

## Overview

This experimental folder implements a **fixed stratified data splitting strategy** as an alternative to the out-of-fold (OOF) cross-validation approach used in the parent directory.

## Key Differences from OOF Method

### Current OOF Approach (`Fixed_stacking_ensemble_with_n_estimators/`)
- Uses **RepeatedKFold** cross-validation (5 splits × 2 repeats = 10 folds)
- Generates out-of-fold predictions for meta-model training
- Provides robust performance estimates through multiple validation folds
- No fixed test set - validation performance averaged across folds

### New Fixed Split Approach (This Folder)
- Uses **fixed stratified splitting**:
  - **16 samples** for validation
  - **16 samples** for test
  - **Remaining samples** for training
- Single, reproducible split with `random_state=42`
- Separate validation and test sets for unbiased evaluation
- Stratified sampling ensures representative distribution across target range

## Data Splitting Strategy

### Stratification Method

The splitting strategy ensures even distribution across the target variable range:

1. **Sort by Target**: Data sorted by `Tg(deg C)` values
2. **Systematic Sampling**: 
   - Validation samples selected at regular intervals across sorted data
   - Test samples selected from remaining data using same strategy
   - Training set contains all remaining samples
3. **Diversity Preservation**: Ensures all splits contain representative samples from low, medium, and high target values

### Implementation Details

```python
def stratified_split(X, y, val_size=16, test_size=16, random_state=42):
    """
    Systematic sampling for stratification:
    - Step size = n_samples / split_size
    - Samples selected at regular intervals
    - Maintains target distribution across splits
    """
```

### Split Statistics

The script automatically generates `split_statistics.json` containing:
- Sample counts for each split
- Target variable statistics (mean, std, min, max) for each split
- Verification of distribution balance

## Model Saving and Tracking

### Model Naming Convention

All models saved with comprehensive naming:
```
model_{model_type}_combo{combination_id}_n{n_estimators}_{timestamp}.joblib
```

**Components:**
- `model_type`: Type of model (e.g., `base_gb`, `base_rf`, `base_svr`, `base_lasso`, `base_elasticnet`, `meta_ridge`)
- `combination_id`: Unique identifier for feature combination (1 to N)
- `n_estimators`: Number of estimators used (1, 10, 50, ..., 1000)
- `timestamp`: Training timestamp in format `YYYYMMDD_HHMMSS`

**Example:**
```
model_meta_ridge_combo5_n700_20260224_200530.joblib
```

### Model Registry

Comprehensive tracking system in `model_registry.csv`:

| Field | Description |
|-------|-------------|
| `model_filename` | Full filename of saved model |
| `model_type` | Type of model (base learner or meta-learner) |
| `combination_id` | Feature combination identifier |
| `n_estimators` | Number of estimators |
| `feature_combination` | List of features used |
| `num_features` | Count of features |
| `hyperparameters` | Model hyperparameters |
| `train_r2`, `train_mse`, `train_mae` | Training metrics |
| `val_r2`, `val_mse`, `val_mae` | Validation metrics |
| `test_r2`, `test_mse`, `test_mae` | Test metrics |
| `split_seed` | Random seed used for splitting |
| `train_size`, `val_size`, `test_size` | Split sizes |
| `timestamp` | Training timestamp |

### Saved Artifacts

Each experiment run saves:
1. **Individual Models**: All base learners and meta-learners in `models/` directory
2. **Ensemble Bundles**: Complete ensemble (scalers + all models) for easy deployment
3. **Model Registry**: CSV file tracking all models and their performance
4. **Results**: Comprehensive results CSV with all metrics
5. **Split Statistics**: JSON file documenting data split characteristics

## Feature Combinations

Tests all combinations of:
- **Mandatory features**: `Lignin (wt%)`, `Co-polyol type (PTHF)`, `r`
- **Optional features**: `Copolyol (wt%)`, `Isocyanate (wt%)`, `Isocyanate (mmol NCO)`, `Isocyonate type`, `tin(II) octoate`, `Sratio(%)`

Total combinations: **511** (all subsets of optional features with mandatory features)

## N_Estimators Values Tested

```python
[1, 10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
```

## Usage

### Running the Experiment

```bash
cd "C:\Users\sacaru\digilignin\DigiLignin\4.Wrapper\Stratified_fixed_split_16_val_16_test"
python run_fixed_split_experiments.py
```

### Expected Runtime

- **Per combination**: ~2-5 minutes (depending on n_estimators)
- **Total runtime**: ~15-40 hours for all 511 combinations × 13 n_estimators values

### Output Files

1. **fixed_split_results.csv**: Main results file with all metrics
2. **model_registry.csv**: Complete model tracking database
3. **split_statistics.json**: Data split characteristics
4. **fixed_split_analysis.png/tiff/pdf/svg**: Comprehensive visualization
5. **models/**: Directory containing all saved models

## Comparison with OOF Method

### Advantages of Fixed Split

1. **True Test Set**: Unbiased evaluation on held-out test data
2. **Faster Training**: Single split vs. multiple folds
3. **Reproducibility**: Exact same split every run
4. **Deployment Ready**: Models trained on fixed training set
5. **Clear Separation**: Validation for tuning, test for final evaluation

### Advantages of OOF Method

1. **Robust Estimates**: Multiple folds provide confidence intervals
2. **Better Data Utilization**: All samples used for both training and validation
3. **Variance Estimation**: Can assess model stability across folds
4. **Less Sensitive**: To specific train/val/test split

### When to Use Each

**Use Fixed Split When:**
- Need unbiased test set evaluation
- Preparing for production deployment
- Want faster experimentation
- Have sufficient data for separate test set

**Use OOF When:**
- Need robust performance estimates
- Limited data available
- Want to assess model stability
- Cross-validation is standard in your domain

## Visualization

The script generates comprehensive plots:

1. **Plot A**: Average MAE vs N_Estimators (Validation & Test)
2. **Plot B**: Average R² vs N_Estimators (Validation & Test)
3. **Plot C**: Best Performance vs N_Estimators
4. **Plot D**: Performance by Feature Count
5. **Plot E**: MAE Heatmap (Features vs Estimators)
6. **Plot F**: Top 10 Best Combinations (by Test MAE)

## Model Loading Example

```python
import joblib

# Load a specific model
model = joblib.load('models/model_meta_ridge_combo5_n700_20260224_200530.joblib')

# Load complete ensemble bundle
ensemble = joblib.load('models/ensemble_combo5_n700_20260224_200530.joblib')
x_scaler = ensemble['x_scaler']
y_scaler = ensemble['y_scaler']
base_models = ensemble['base_models']
meta_model = ensemble['meta_model']

# Make predictions
x_new_scaled = x_scaler.transform(x_new)
meta_features = np.column_stack([model.predict(x_new_scaled) for model in base_models])
y_pred_scaled = meta_model.predict(meta_features)
y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))
```

## Reproducibility

All random operations use `RANDOM_SEED = 42`:
- Data splitting
- Model initialization
- Hyperparameter tuning

## Notes

- Ensure dataset is available at `../Fixed_Stacking_Ensemble/dataset.xlsx`
- Requires sufficient disk space for model storage (~500MB-1GB for all models)
- Progress is printed to console during execution
- Models can be loaded individually or as complete ensembles

## Future Enhancements

Potential extensions:
1. Test multiple random seeds for split robustness
2. Implement k-fold stratified splitting with fixed test set
3. Add feature importance analysis
4. Create automated model selection based on validation performance
5. Implement ensemble pruning to reduce model count

## Contact

For questions or issues, refer to the main project documentation.
