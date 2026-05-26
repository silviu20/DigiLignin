# Best Model Mapping - README

## Overview

This directory contains scripts to retrain, map, and visualize results from the best model identified in the wrapper experiments:

**Best Model Configuration:**
- **Number of base estimators:** 10
- **Features (5):** 
  1. Lignin (wt%)
  2. Co-polyol type (PTHF)
  3. r (ratio)
  4. Copolyol (wt%)
  5. Isocyanate (wt%)
- **Validation R²:** 0.4940
- **Test R²:** 0.5473

## Files Created

### Core Scripts

1. **`retrain_best_model.py`**
   - Retrains the best model from scratch
   - Saves all necessary artifacts for mapping
   - Outputs: model files, scalers, metadata

2. **`mapping_best_model.py`**
   - Performs comprehensive mapping across feature space
   - Generates predictions for all feature combinations
   - Creates summary statistics

3. **`visualize_distribution_best_model.py`**
   - Creates distribution plots of predicted Tg values
   - Shows frequency of predictions across temperature range

4. **`visualize_density_plots_best_model.py`**
   - Creates 2D density plots for each feature vs Tg
   - Shows relationships between inputs and predictions

### Legacy Scripts (Original)

- `Mapping.py` - Original mapping script (7 features)
- `Distribution of Predicted Tg Values_mapped_results.py` - Original distribution plot
- `Density plot for mapping data.py` - Original density plots
- `Merging and labelling plots.py` - Plot merging utility

## Workflow

### Step 1: Retrain the Best Model

```bash
cd C:\Users\sacaru\digilignin\DigiLignin\7.Mapping
python retrain_best_model.py
```

**What it does:**
- Loads the dataset from `../4.Wrapper/Fixed_Stacking_Ensemble/dataset.xlsx`
- Creates stratified splits (16 validation, 16 test, rest training)
- Trains 5 base models (GradientBoosting, RandomForest, SVR, Lasso, ElasticNet)
- Trains Ridge meta-model
- Saves all artifacts

**Output files:**
- `best_model_base_models.joblib` - Trained base models
- `best_model_meta_model.joblib` - Trained meta-model
- `best_model_x_scaler.joblib` - Feature scaler
- `best_model_y_scaler.joblib` - Target scaler
- `best_model_features.txt` - List of features
- `best_model_metadata.json` - Model performance metrics

**Expected runtime:** 5-15 minutes (depends on hardware)

### Step 2: Generate Mapping

```bash
python mapping_best_model.py
```

**What it does:**
- Loads trained models and scalers
- Generates predictions for all feature combinations:
  - Lignin: 0-70% (1% steps) = 70 values
  - Co-polyol type: [250, 650, 1000] = 3 values
  - r: 0.6-1.4 (0.05 steps) = 17 values
  - Copolyol: 0-66% (2% steps) = 34 values
  - Isocyanate: 0-20% (0.5% steps) = 41 values
- **Total combinations:** 70 × 3 × 17 × 34 × 41 = ~2,937,780 predictions

**Output files:**
- `mapped_results_tg_best_model.csv` - Full mapping results (~300 MB)
- `mapping_summary.json` - Summary statistics
- `mapped_results_sample.csv` - Random sample of 1000 predictions

**Expected runtime:** 10-30 minutes (depends on hardware)

### Step 3: Visualize Distribution

```bash
python visualize_distribution_best_model.py
```

**What it does:**
- Creates histogram of predicted Tg values
- Annotates highest, lowest, and median frequency bars
- Shows distribution across -10°C to 80°C range

**Output files:**
- `distribution_tg_best_model.png`
- `distribution_tg_best_model.svg`
- `distribution_tg_best_model.pdf`

**Expected runtime:** < 1 minute

### Step 4: Create Density Plots

```bash
python visualize_density_plots_best_model.py
```

**What it does:**
- Creates 2D density plots for each feature vs Tg
- Shows how each input feature affects predicted Tg
- Saves both plots and underlying data

**Output files:**
- `density_plots_best_model/` directory containing:
  - 5 density plot PNG files (one per feature)
  - `plot_data/` subdirectory with CSV data for each plot

**Expected runtime:** 2-5 minutes

## Feature Ranges in Mapping

| Feature | Range | Step Size | # Values |
|---------|-------|-----------|----------|
| Lignin (wt%) | 0-70% | 1% | 70 |
| Co-polyol type (PTHF) | [250, 650, 1000] | discrete | 3 |
| r (ratio) | 0.6-1.4 | 0.05 | 17 |
| Copolyol (wt%) | 0-66% | 2% | 34 |
| Isocyanate (wt%) | 0-20% | 0.5% | 41 |

## Customization

### Adjusting Feature Ranges

Edit the `feature_values` list in `mapping_best_model.py`:

```python
feature_values = [
    np.arange(0, 70, 1),              # Lignin (wt%)
    [250, 650, 1000],                 # Co-polyol type (PTHF)
    np.arange(0.6, 1.4 + 0.05, 0.05), # r
    np.arange(0, 66 + 2, 2),          # Copolyol (wt%)
    np.arange(0, 20 + 0.5, 0.5),      # Isocyanate (wt%)
]
```

### Reducing Mapping Size

To reduce computation time and file size, increase step sizes:

```python
feature_values = [
    np.arange(0, 70, 2),              # Lignin: 2% steps instead of 1%
    [250, 650, 1000],                 # Co-polyol type: unchanged
    np.arange(0.6, 1.4 + 0.1, 0.1),   # r: 0.1 steps instead of 0.05
    np.arange(0, 66 + 5, 5),          # Copolyol: 5% steps instead of 2%
    np.arange(0, 20 + 1, 1),          # Isocyanate: 1% steps instead of 0.5%
]
```

This would reduce total combinations from ~2.9M to ~147k (20× reduction).

### Adjusting Batch Size

In `mapping_best_model.py`, change the `batch_size` parameter:

```python
mapped_results = map_target_batch(base_models, meta_model, x_scaler, y_scaler, 
                                   batch_size=5000)  # Default is 10000
```

Smaller batch sizes provide more frequent progress updates but may be slightly slower.

## Troubleshooting

### Issue: "Could not find model files"

**Solution:** Run `retrain_best_model.py` first to generate the model files.

### Issue: Out of memory during mapping

**Solutions:**
1. Reduce feature ranges (see Customization section)
2. Reduce batch size in `mapping_best_model.py`
3. Close other applications to free up RAM

### Issue: Mapping takes too long

**Solutions:**
1. Reduce feature ranges to decrease total combinations
2. Use the sample file for quick visualization tests
3. Run on a machine with more CPU cores (uses parallel processing)

### Issue: Plots look different from expected

**Solution:** Ensure you're using the correct CSV file (`mapped_results_tg_best_model.csv`, not the old `mapped_results_tg.csv`)

## Model Performance Summary

From the wrapper experiments (Rank 1 model):

| Metric | Validation | Test |
|--------|-----------|------|
| R² | 0.4940 | 0.5473 |
| MSE | 391.95 | 338.37 |
| MAE | 13.41°C | 15.17°C |

The model shows good generalization with test R² higher than validation R², indicating robust performance.

## Next Steps

After completing the mapping:

1. **Analyze Results:** Examine `mapping_summary.json` for Tg range and distribution
2. **Identify Patterns:** Review density plots to understand feature-Tg relationships
3. **Optimize Formulations:** Use mapping results to find combinations that achieve target Tg values
4. **Validate Predictions:** Compare predictions with experimental data for key formulations

## Contact & Support

For questions about the model or mapping process, refer to:
- Original wrapper experiments: `../4.Wrapper/Stratified_fixed_split_16_val_16_test/`
- Model registry: `../4.Wrapper/Stratified_fixed_split_16_val_16_test/model_registry.csv`
- Top models analysis: `../4.Wrapper/Stratified_fixed_split_16_val_16_test/top_10_models_validation_with_test.csv`
