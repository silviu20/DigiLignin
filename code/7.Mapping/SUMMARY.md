# Best Model Mapping - Summary

## What Was Done

I've adapted the mapping code from `C:\Users\sacaru\digilignin\DigiLignin\7.Mapping` to work with your best model identified in the wrapper experiments.

## Best Model Configuration

**From:** `C:\Users\sacaru\digilignin\DigiLignin\4.Wrapper\Stratified_fixed_split_16_val_16_test`

- **Rank:** #1 (best validation performance)
- **Base Estimators:** 10
- **Features (5):**
  1. Lignin (wt%)
  2. Co-polyol type (PTHF)
  3. r (ratio)
  4. Copolyol (wt%)
  5. Isocyanate (wt%)

**Performance:**
- Validation R² = 0.4940
- Test R² = 0.5473
- Test MAE = 15.17°C

## New Files Created

### Core Scripts
1. **`retrain_best_model.py`** - Retrains the model and saves all artifacts
2. **`mapping_best_model.py`** - Generates ~2.9M predictions across feature space
3. **`visualize_distribution_best_model.py`** - Creates Tg distribution plots
4. **`visualize_density_plots_best_model.py`** - Creates 2D density plots
5. **`run_complete_workflow.py`** - Runs all steps automatically

### Documentation
6. **`README_BEST_MODEL_MAPPING.md`** - Comprehensive documentation
7. **`QUICKSTART.md`** - Quick start guide
8. **`SUMMARY.md`** - This file

## Key Differences from Original Code

| Aspect | Original (`Mapping.py`) | New (`mapping_best_model.py`) |
|--------|------------------------|-------------------------------|
| Features | 7 features | 5 features (best model) |
| Model Source | Hardcoded filenames | Retrained from scratch |
| Feature Ranges | Fixed ranges | Optimized for 5 features |
| Documentation | Minimal | Comprehensive |
| Progress Tracking | Basic | Detailed with ETA |
| Output Files | 1 CSV | CSV + JSON summary + sample |

## How to Use

### Option 1: Run Everything (Recommended)
```bash
cd C:\Users\sacaru\digilignin\DigiLignin\7.Mapping
python run_complete_workflow.py
```

### Option 2: Run Steps Individually
```bash
# Step 1: Retrain model
python retrain_best_model.py

# Step 2: Generate mapping
python mapping_best_model.py

# Step 3: Visualize distribution
python visualize_distribution_best_model.py

# Step 4: Create density plots
python visualize_density_plots_best_model.py
```

## Expected Outputs

### Model Files (~50 MB total)
- `best_model_base_models.joblib`
- `best_model_meta_model.joblib`
- `best_model_x_scaler.joblib`
- `best_model_y_scaler.joblib`
- `best_model_metadata.json`

### Mapping Results (~300 MB)
- `mapped_results_tg_best_model.csv` - Full results (2,937,780 predictions)
- `mapping_summary.json` - Statistics
- `mapped_results_sample.csv` - Random sample (1,000 rows)

### Visualizations
- `distribution_tg_best_model.{png,svg,pdf}` - Tg distribution
- `density_plots_best_model/` - 5 density plots + data

## Mapping Coverage

**Total Combinations:** 2,937,780

| Feature | Min | Max | Step | Count |
|---------|-----|-----|------|-------|
| Lignin (wt%) | 0 | 70 | 1 | 70 |
| Co-polyol type (PTHF) | 250, 650, 1000 | - | discrete | 3 |
| r | 0.6 | 1.4 | 0.05 | 17 |
| Copolyol (wt%) | 0 | 66 | 2 | 34 |
| Isocyanate (wt%) | 0 | 20 | 0.5 | 41 |

## Performance Estimates

| Step | Time | Output Size |
|------|------|-------------|
| Retrain model | 5-15 min | ~50 MB |
| Generate mapping | 10-30 min | ~300 MB |
| Visualize distribution | <1 min | ~5 MB |
| Create density plots | 2-5 min | ~10 MB |
| **Total** | **20-50 min** | **~365 MB** |

## Next Steps

1. **Run the workflow** using one of the methods above
2. **Review `mapping_summary.json`** for Tg statistics
3. **Examine density plots** to understand feature-Tg relationships
4. **Use `mapped_results_tg_best_model.csv`** for:
   - Finding optimal formulations
   - Identifying Tg ranges
   - Guiding experimental design

## Advantages of This Approach

✅ **Reproducible:** All steps documented and automated  
✅ **Validated:** Uses best model from systematic wrapper experiments  
✅ **Comprehensive:** Covers entire feasible feature space  
✅ **Well-documented:** README, quickstart, and inline comments  
✅ **Flexible:** Easy to adjust feature ranges and parameters  
✅ **Efficient:** Batch processing with progress tracking  

## Files Organization

```
7.Mapping/
├── Scripts (new)
│   ├── retrain_best_model.py
│   ├── mapping_best_model.py
│   ├── visualize_distribution_best_model.py
│   ├── visualize_density_plots_best_model.py
│   └── run_complete_workflow.py
│
├── Documentation (new)
│   ├── README_BEST_MODEL_MAPPING.md
│   ├── QUICKSTART.md
│   └── SUMMARY.md
│
├── Scripts (original)
│   ├── Mapping.py
│   ├── Distribution of Predicted Tg Values_mapped_results.py
│   ├── Density plot for mapping data.py
│   └── Merging and labelling plots.py
│
└── Outputs (generated after running)
    ├── Model files (*.joblib, *.json)
    ├── Mapping results (*.csv)
    └── Visualizations (*.png, *.svg, *.pdf)
```

## Support

- **Detailed guide:** `README_BEST_MODEL_MAPPING.md`
- **Quick start:** `QUICKSTART.md`
- **Model info:** `../4.Wrapper/Stratified_fixed_split_16_val_16_test/top_10_models_validation_with_test.csv`
