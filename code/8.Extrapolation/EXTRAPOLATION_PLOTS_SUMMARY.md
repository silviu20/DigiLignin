# Extrapolation Plots Summary

## Available Plot Versions

### 1. Simple Version (No Labels)
**File:** `extrapolation_plot_best_model_simple.py`

**Features:**
- Clean scatter plot of target vs predicted Tg
- Two inset zoom regions (low and high Tg)
- Red highlighting for extrapolated points
- No text labels on points

**Use when:** You want a clean, uncluttered visualization

---

### 2. Labeled Version (With Arrows) ⭐ RECOMMENDED
**File:** `extrapolation_plot_best_model_labeled.py`

**Features:**
- Scatter plot with labeled extrapolated points
- Arrows pointing from labels to data points
- Insets positioned at deviation regions
- Shows exact (Target, Predicted) coordinates
- Larger figure size (14×11) for better readability
- Bold labels with white background boxes

**Use when:** You need to show exact values and deviations

**Key Improvements:**
- ✅ Insets automatically positioned at deviation points
- ✅ Labels show (Target Tg, Predicted Tg) for all extrapolated points
- ✅ Arrows connect labels to points (if adjustText available)
- ✅ White background boxes for label readability
- ✅ Deviation summary printed after plotting

---

## Extrapolation Results

### Deviation Summary

**Largest deviations (extrapolated points):**

| Target Tg (°C) | Predicted Tg (°C) | Deviation (°C) |
|----------------|-------------------|----------------|
| -17.0 | 5.6 | 22.6 |
| 100.0 | 79.4 | 20.6 |
| -14.6 | 5.6 | 20.2 |
| 97.6 | 79.4 | 18.2 |
| -12.2 | 5.6 | 17.8 |

### Key Observations

1. **Low Temperature Extrapolation:**
   - Target range: -17°C to -10°C
   - All predict to ~5.6°C
   - Model underestimates low Tg capability
   - Deviation: 13-23°C

2. **High Temperature Extrapolation:**
   - Target range: 97°C to 100°C
   - All predict to ~79.4°C
   - Model underestimates high Tg capability
   - Deviation: 18-21°C

3. **Interpolation Region:**
   - Target range: -8°C to 96°C (training range)
   - Much better predictions
   - Lower deviations

---

## Plot Features Explained

### Main Plot
- **X-axis:** Target Tg values (-30°C to 130°C)
- **Y-axis:** Predicted Tg values (-30°C to 130°C)
- **Black dashed line:** Perfect prediction (y = x)
- **Color scale:** Predicted Tg (inferno colormap)
- **Red points:** Extrapolated data (outside training range)

### Inset 1: Low Tg Region
- **Location:** Lower right
- **Focus:** -20°C to 0°C range
- **Shows:** Where low Tg predictions deviate from ideal
- **Labels:** All extrapolated points with coordinates

### Inset 2: High Tg Region
- **Location:** Upper left
- **Focus:** 95°C to 105°C range
- **Shows:** Where high Tg predictions deviate from ideal
- **Labels:** All extrapolated points with coordinates

---

## How to Use

### Generate Labeled Plot

```bash
cd C:\Users\sacaru\digilignin\DigiLignin\8.Extrapolation
python extrapolation_plot_best_model_labeled.py
```

### Output Files
- `extrapolation_plot_best_model_labeled.png` (600 DPI)
- `extrapolation_plot_best_model_labeled.svg` (vector)
- `extrapolation_plot_best_model_labeled.pdf` (publication-ready)

---

## Customization Options

### Adjust Inset Positions

The insets automatically position based on data, but you can manually adjust:

```python
# Line 95-100 - Low Tg inset
x_min_inset = low_extrap['Target_Tg'].min() - 2
x_max_inset = max(low_extrap['Target_Tg'].max() + 2, training_range[0] + 2)

# Line 145-150 - High Tg inset  
x_min_inset = min(high_extrap['Target_Tg'].min() - 2, training_range[1] - 2)
x_max_inset = high_extrap['Target_Tg'].max() + 2
```

### Change Label Font Size

```python
# Line 118 - Inset 1 labels
label = f"({row['Target_Tg']:.1f}, {row['Predicted_Tg']:.1f})"
texts1.append(ax_inset1.text(row['Target_Tg'], row['Predicted_Tg'], 
                          label, fontsize=16, fontweight='bold'))  # Change fontsize

# Line 168 - Inset 2 labels
texts2.append(ax_inset2.text(row['Target_Tg'], row['Predicted_Tg'], 
                           label, fontsize=16, fontweight='bold'))  # Change fontsize
```

### Modify Arrow Style

If adjustText is installed:

```python
# Line 124-130 - Arrow properties
arrowprops=dict(
    arrowstyle='->',      # Options: '->', '-|>', '-[', 'fancy'
    color='darkred',      # Arrow color
    lw=1.5,              # Line width
    alpha=0.8            # Transparency
)
```

---

## Installation Note

For best results with automatic label positioning, install adjustText:

```bash
pip install adjustText
```

**Without adjustText:** Labels will have white background boxes but no arrows
**With adjustText:** Labels will have arrows pointing to points with automatic overlap avoidance

---

## Comparison with Original Code

### Similarities
✅ Two inset zoom regions  
✅ Labels on extrapolated points  
✅ Arrows pointing to data points  
✅ Multiple output formats (PNG, SVG, PDF)  

### Improvements
✅ Automatic inset positioning based on actual deviations  
✅ Larger, more readable figure size  
✅ Bold labels with better contrast  
✅ Deviation summary in console output  
✅ Adapted for 5-feature model  
✅ Works with or without adjustText library  

---

## Troubleshooting

### Labels overlap
- Install adjustText: `pip install adjustText`
- Or manually adjust inset bounds to give more space

### Insets too small
- Change width/height percentages in `inset_axes()` calls
- Currently: 45% width, 35% height (low Tg), 32% height (high Tg)

### Can't see all labels
- Increase inset bounds (x_min_inset, x_max_inset, etc.)
- Reduce label font size
- Increase figure size in line 54

---

## Next Steps

1. ✅ Review generated plots
2. Use for publication or presentation
3. Compare with experimental validation data
4. Adjust model or feature ranges if needed
5. Generate parallel coordinates plot for feature analysis
