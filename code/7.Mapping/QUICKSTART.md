# Quick Start Guide - Best Model Mapping

## TL;DR - Run Everything at Once

```bash
cd C:\Users\sacaru\digilignin\DigiLignin\7.Mapping
python run_complete_workflow.py
```

This will automatically run all 4 steps in sequence. Estimated time: 20-50 minutes.

---

## Manual Step-by-Step (Recommended for First Time)

### Prerequisites

Ensure you have the dataset file:
- `C:\Users\sacaru\digilignin\DigiLignin\4.Wrapper\Fixed_Stacking_Ensemble\dataset.xlsx`

### Step 1: Retrain Model (5-15 min)

```bash
cd C:\Users\sacaru\digilignin\DigiLignin\7.Mapping
python retrain_best_model.py
```

**Expected output:**
```
================================================================================
RETRAINING BEST MODEL FOR MAPPING
================================================================================
Model: 10 base estimators
Features: Lignin (wt%), Co-polyol type (PTHF), r, Copolyol (wt%), Isocyanate (wt%)
...
MODEL PERFORMANCE
...
Validation Set:
  R² = 0.4940
  MSE = 391.95
  MAE = 13.41
...
MODEL SAVED SUCCESSFULLY!
```

**Files created:**
- `best_model_base_models.joblib`
- `best_model_meta_model.joblib`
- `best_model_x_scaler.joblib`
- `best_model_y_scaler.joblib`
- `best_model_metadata.json`

### Step 2: Generate Mapping (10-30 min)

```bash
python mapping_best_model.py
```

**Expected output:**
```
================================================================================
MAPPING WITH BEST MODEL
================================================================================
...
Total combinations to process: 2,937,780
...
Progress: 1,000,000/2,937,780 (34.0%) | Rate: 1500 combo/s | ETA: 21.5 min
...
MAPPING COMPLETED SUCCESSFULLY!
```

**Files created:**
- `mapped_results_tg_best_model.csv` (~300 MB)
- `mapping_summary.json`
- `mapped_results_sample.csv`

### Step 3: Visualize Distribution (< 1 min)

```bash
python visualize_distribution_best_model.py
```

**Files created:**
- `distribution_tg_best_model.png`
- `distribution_tg_best_model.svg`
- `distribution_tg_best_model.pdf`

### Step 4: Create Density Plots (2-5 min)

```bash
python visualize_density_plots_best_model.py
```

**Files created:**
- `density_plots_best_model/` directory with 5 PNG files
- `density_plots_best_model/plot_data/` with 5 CSV files

---

## What You Get

### 1. Model Files
Trained ensemble model ready for predictions:
- 5 base models (GB, RF, SVR, Lasso, ElasticNet)
- 1 meta-model (Ridge regression)
- Feature and target scalers

### 2. Mapping Results
Complete prediction space (~2.9M combinations):
- All combinations of 5 input features
- Predicted Tg for each combination
- Summary statistics

### 3. Visualizations
- **Distribution plot:** Shows frequency of Tg predictions
- **Density plots (5):** Shows how each feature affects Tg

---

## Quick Checks

### Verify Model Training Worked
```bash
python -c "import joblib; print('Models loaded successfully')" && joblib.load('best_model_base_models.joblib')
```

### Check Mapping File Size
```bash
python -c "import os; size=os.path.getsize('mapped_results_tg_best_model.csv'); print(f'Mapping file: {size/1e6:.1f} MB')"
```

### View Mapping Summary
```bash
python -c "import json; print(json.dumps(json.load(open('mapping_summary.json')), indent=2))"
```

---

## Common Issues

### "Could not find dataset.xlsx"
**Fix:** Check that the dataset exists at:
`C:\Users\sacaru\digilignin\DigiLignin\4.Wrapper\Fixed_Stacking_Ensemble\dataset.xlsx`

### "Could not find model files"
**Fix:** Run Step 1 first (`retrain_best_model.py`)

### Out of Memory
**Fix:** Edit `mapping_best_model.py` and reduce feature ranges:
```python
# Change line ~30-35 to use larger steps
feature_values = [
    np.arange(0, 70, 2),      # 2% steps instead of 1%
    [250, 650, 1000],
    np.arange(0.6, 1.4, 0.1), # 0.1 steps instead of 0.05
    np.arange(0, 66, 5),      # 5% steps instead of 2%
    np.arange(0, 20, 1),      # 1% steps instead of 0.5%
]
```

---

## File Structure After Completion

```
7.Mapping/
├── retrain_best_model.py
├── mapping_best_model.py
├── visualize_distribution_best_model.py
├── visualize_density_plots_best_model.py
├── run_complete_workflow.py
├── README_BEST_MODEL_MAPPING.md
├── QUICKSTART.md (this file)
│
├── best_model_base_models.joblib
├── best_model_meta_model.joblib
├── best_model_x_scaler.joblib
├── best_model_y_scaler.joblib
├── best_model_metadata.json
│
├── mapped_results_tg_best_model.csv
├── mapping_summary.json
├── mapped_results_sample.csv
│
├── distribution_tg_best_model.png
├── distribution_tg_best_model.svg
├── distribution_tg_best_model.pdf
│
└── density_plots_best_model/
    ├── density_plot_Lignin_wtpct_Tg_degC.png
    ├── density_plot_Co-polyol_type_PTHF_Tg_degC.png
    ├── density_plot_r_Tg_degC.png
    ├── density_plot_Copolyol_wtpct_Tg_degC.png
    ├── density_plot_Isocyanate_wtpct_Tg_degC.png
    └── plot_data/
        ├── density_data_Lignin_wtpct_Tg_degC.csv
        ├── density_data_Co-polyol_type_PTHF_Tg_degC.csv
        ├── density_data_r_Tg_degC.csv
        ├── density_data_Copolyol_wtpct_Tg_degC.csv
        └── density_data_Isocyanate_wtpct_Tg_degC.csv
```

---

## Next Steps After Mapping

1. **Analyze Summary Statistics**
   ```bash
   cat mapping_summary.json
   ```

2. **Find Optimal Formulations**
   Use the CSV to find combinations that achieve target Tg values

3. **Validate Key Predictions**
   Compare predictions with experimental data

4. **Optimize Formulations**
   Use insights from density plots to guide formulation design

---

## Need Help?

- **Full documentation:** See `README_BEST_MODEL_MAPPING.md`
- **Model details:** See `../4.Wrapper/Stratified_fixed_split_16_val_16_test/top_10_models_validation_with_test.csv`
- **Original experiments:** See `../4.Wrapper/Stratified_fixed_split_16_val_16_test/README.md`
