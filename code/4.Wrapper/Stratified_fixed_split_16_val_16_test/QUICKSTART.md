# Quick Start Guide

## Prerequisites

Ensure you have the following installed:
- Python 3.7+
- Required packages: numpy, pandas, matplotlib, seaborn, scikit-learn, scipy, joblib, openpyxl

## Installation

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy joblib openpyxl
```

## Running the Experiment

### Step 1: Verify Setup

```bash
cd "C:\Users\sacaru\digilignin\DigiLignin\4.Wrapper\Stratified_fixed_split_16_val_16_test"
python verify_setup.py
```

This will check:
- Dataset availability
- Required packages
- Directory structure
- Preprocessing module access

### Step 2: Run the Experiment

```bash
python run_fixed_split_experiments.py
```

**Expected Output:**
- Console progress updates for each feature combination
- Split statistics printed at start
- Model training progress
- Final summary statistics

**Runtime:** 15-40 hours for all combinations

### Step 3: Compare with OOF Method

After the experiment completes:

```bash
python compare_with_oof.py
```

This generates comparison plots and statistics.

## Output Files

After running, you'll have:

| File | Description |
|------|-------------|
| `fixed_split_results.csv` | Main results with all metrics |
| `model_registry.csv` | Complete model tracking database |
| `split_statistics.json` | Data split characteristics |
| `fixed_split_analysis.png/tiff/pdf/svg` | Visualization plots |
| `models/*.joblib` | All trained models |

## Quick Analysis

### View Best Model

```python
import pandas as pd

# Load results
df = pd.read_csv('fixed_split_results.csv')

# Find best model by test MAE
best = df.nsmallest(1, 'Test MAE').iloc[0]
print(f"Best Model:")
print(f"  Combination ID: {best['Combination ID']}")
print(f"  N_Estimators: {best['n_estimators']}")
print(f"  Test MAE: {best['Test MAE']:.3f}°C")
print(f"  Test R²: {best['Test R2']:.3f}")
```

### Load and Use a Model

```python
import joblib
import numpy as np

# Load ensemble bundle (replace with actual filename)
ensemble = joblib.load('models/ensemble_combo5_n700_TIMESTAMP.joblib')

# Extract components
x_scaler = ensemble['x_scaler']
y_scaler = ensemble['y_scaler']
base_models = ensemble['base_models']
meta_model = ensemble['meta_model']

# Make predictions on new data
# x_new should have the same features as used in training
x_new_scaled = x_scaler.transform(x_new)

# Generate meta-features from base models
meta_features = np.column_stack([
    model.predict(x_new_scaled) for model in base_models
])

# Get final prediction
y_pred_scaled = meta_model.predict(meta_features)
y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))

print(f"Predicted Tg: {y_pred[0][0]:.2f}°C")
```

## Troubleshooting

### Dataset Not Found

**Error:** `FileNotFoundError: dataset.xlsx not found`

**Solution:** Ensure dataset is at:
```
C:\Users\sacaru\digilignin\DigiLignin\4.Wrapper\Fixed_Stacking_Ensemble\dataset.xlsx
```

### Out of Memory

**Error:** `MemoryError` during training

**Solution:** 
1. Reduce number of combinations tested
2. Test fewer n_estimators values
3. Close other applications
4. Run on a machine with more RAM

### Import Error

**Error:** `ModuleNotFoundError: No module named 'sklearn'`

**Solution:**
```bash
pip install scikit-learn
```

### Preprocessing Module Not Found

**Error:** Cannot import preprocessing module

**Solution:** Verify the path to preprocessing module:
```
C:\Users\sacaru\digilignin\DigiLignin\1.Loading and Preprocessing\Loading and preprocessing.py
```

## Partial Runs

To test a subset of combinations:

Edit `run_fixed_split_experiments.py`:

```python
# Original (all combinations)
estimator_values = [1, 10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

# Test run (fewer values)
estimator_values = [100, 500, 1000]

# Or limit combinations
all_combinations = all_combinations[:10]  # Test first 10 only
```

## Monitoring Progress

The script prints progress to console:
```
==============================================================
Feature Combination 5/511
Features: ['Lignin (wt%)', 'Co-polyol type (PTHF)', 'r', 'Copolyol (wt%)']
Number of Features: 4
==============================================================
    Processing n_estimators = 100...
    Processing n_estimators = 200...
    ...
```

## Next Steps

1. **Analyze Results:** Use `compare_with_oof.py` to compare methods
2. **Select Best Model:** Identify best performing combination
3. **Deploy Model:** Load best model for production use
4. **Feature Analysis:** Investigate which features contribute most
5. **Hyperparameter Tuning:** Further optimize best combinations

## Support

For issues or questions:
1. Check README.md for detailed documentation
2. Review error messages carefully
3. Verify all prerequisites are met
4. Check dataset format and location
