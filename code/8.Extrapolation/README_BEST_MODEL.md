# Extrapolation Analysis - Best Model

## Overview

This directory contains scripts to perform extrapolation analysis for the best model (5 features). The analysis finds optimal input parameters to achieve target Tg values and visualizes the model's extrapolation capabilities.

## Best Model Configuration

- **Features (5):** Lignin (wt%), Co-polyol type (PTHF), r, Copolyol (wt%), Isocyanate (wt%)
- **Constraint:** Lignin (wt%) + Copolyol (wt%) = 100%
- **Base estimators:** 10
- **Performance:** Validation R² = 0.465, Test R² = 0.565

## Files

### New Scripts (Best Model)

1. **`adaptive_grid_search_best_model.py`**
   - Finds optimal input parameters for target Tg values
   - Uses iterative grid refinement
   - Enforces composition constraint (Lignin + Copolyol = 100%)
   - Output: `closest_inputs_best_model.csv`

2. **`extrapolation_plot_best_model_simple.py`**
   - Creates visualization of target vs predicted Tg
   - Highlights extrapolated data points
   - Includes inset zoom regions
   - Output: PNG, SVG, PDF plots

### Legacy Scripts (7 features)

- `Adaptive_grid_search_with_linginWT_restrictions.py` - Original 7-feature version
- `Extrapolation of the closes_inputs_plot_v2_2.py` - Original visualization

## Workflow

### Step 1: Run Adaptive Grid Search

```bash
cd C:\Users\sacaru\digilignin\DigiLignin\8.Extrapolation
python adaptive_grid_search_best_model.py
```

**What it does:**
- Loads trained models from `../7.Mapping/`
- Defines 50 target Tg values from -17°C to 100°C
- For each target, finds optimal input parameters using adaptive grid search
- Saves results to `closest_inputs_best_model.csv`

**Expected runtime:** 2-5 minutes

**Output:**
```
Successfully found parameters for 50/50 targets
Mean error: ~4.4°C
All compositions sum to 100%: True
```

### Step 2: Create Extrapolation Plot

```bash
python extrapolation_plot_best_model_simple.py
```

**What it does:**
- Loads results from Step 1
- Creates scatter plot of target vs predicted Tg
- Highlights extrapolated points (outside training range -8°C to 96°C)
- Adds inset zoom regions for low and high Tg areas

**Expected runtime:** < 1 minute

**Output files:**
- `extrapolation_plot_best_model.png`
- `extrapolation_plot_best_model.svg`
- `extrapolation_plot_best_model.pdf`

## Results Interpretation

### Adaptive Grid Search Results

The CSV file contains:
- **Lignin (wt%):** Optimal lignin content
- **Co-polyol type (PTHF):** Optimal PTHF molecular weight (250, 650, or 1000)
- **r:** Optimal NCO/OH ratio
- **Copolyol (wt%):** Optimal copolyol content (= 100 - Lignin)
- **Isocyanate (wt%):** Optimal isocyanate content
- **Target_Tg:** Desired Tg value
- **Predicted_Tg:** Model's prediction
- **Total_wt%:** Sum of Lignin + Copolyol (should be 100%)
- **Error:** Absolute difference between target and predicted

### Extrapolation Plot

The plot shows:
- **Black dashed line:** Perfect prediction (y = x)
- **Colored points:** All data points (color = predicted Tg)
- **Red points:** Extrapolated data (target Tg outside training range)
- **Insets:** Zoomed views of low and high Tg regions

**Extrapolation regions:**
- **Low Tg:** Target < -8°C (below training minimum)
- **High Tg:** Target > 96°C (above training maximum)

## Customization

### Adjust Target Tg Range

Edit `adaptive_grid_search_best_model.py`:

```python
# Line ~175
target_tgs = list(np.linspace(-17, 100, 50))  # Change range or number of points
```

### Modify Grid Search Parameters

Edit `adaptive_grid_search_best_model.py`:

```python
# Line ~61 - Initial grid points
lignin_points = np.linspace(0, 100, 5)  # Change number of points

grid_points = [
    [250, 650, 1000],           # PTHF types
    np.linspace(0.6, 1.4, 5),   # r range
    np.linspace(0, 20, 5),      # Isocyanate range
]

# Line ~26 - Number of refinement iterations
n_iterations=3  # Increase for more precision
```

### Adjust Extrapolation Plot Regions

Edit `extrapolation_plot_best_model_simple.py`:

```python
# Line ~17 - Training range
training_range=(-8, 96)  # Adjust based on your training data

# Line ~62-65 - Inset zoom regions
ax_inset1.set_xlim(-20, 0)   # Low Tg region
ax_inset2.set_xlim(95, 105)  # High Tg region
```

## Troubleshooting

### Issue: "Could not find model files"

**Solution:** Ensure models are trained in `../7.Mapping/`:
```bash
cd ../7.Mapping
python retrain_best_model.py
```

### Issue: Grid search takes too long

**Solutions:**
1. Reduce number of target Tg values
2. Decrease number of iterations
3. Reduce grid point density

### Issue: Large prediction errors

**Possible causes:**
1. Target Tg far outside training range
2. Insufficient grid refinement iterations
3. Model limitations for extreme compositions

**Solutions:**
- Increase `n_iterations` in adaptive_grid_search
- Check if target Tg is reasonable for the material system
- Review model performance in training range

## Key Findings

From the best model extrapolation analysis:

1. **Mean prediction error:** ~4.4°C across all targets
2. **Composition constraint:** All solutions satisfy Lignin + Copolyol = 100%
3. **Extrapolation capability:** Model can predict beyond training range, but with increased uncertainty
4. **Optimal parameters:** Vary systematically with target Tg

## Next Steps

1. **Analyze results:** Review `closest_inputs_best_model.csv` for patterns
2. **Validate predictions:** Compare with experimental data if available
3. **Explore relationships:** Use parallel coordinates plot (see `../9.Parallel coordinates plot/`)
4. **Optimize formulations:** Use results to guide experimental design

## Related Files

- **Model training:** `../7.Mapping/retrain_best_model.py`
- **Parallel coordinates:** `../9.Parallel coordinates plot/parallel_coordinates_best_model.py`
- **Model performance:** `../4.Wrapper/Stratified_fixed_split_16_val_16_test/top_10_models_validation_with_test.csv`
