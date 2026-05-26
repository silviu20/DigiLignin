# Parallel Coordinates Plot - Best Model

## Overview

This directory contains scripts to create interactive parallel coordinates plots for the best model (5 features). These plots visualize relationships between input features and predicted Tg values.

## Best Model Configuration

- **Features (5):** Lignin (wt%), Co-polyol type (PTHF), r, Copolyol (wt%), Isocyanate (wt%)
- **Base estimators:** 10
- **Performance:** Validation R² = 0.465, Test R² = 0.565

## Files

### New Scripts (Best Model)

1. **`parallel_coordinates_best_model.py`**
   - Creates interactive parallel coordinates plot
   - Uses Plotly for interactivity
   - Shows relationships between all 5 features and Tg
   - Output: HTML file that opens in browser

### Legacy Scripts (7 features)

- `Parallel Coordinates Plot.py` - Original 7-feature version

## Workflow

### Prerequisites

Ensure you have run the extrapolation analysis:
```bash
cd ../8.Extrapolation
python adaptive_grid_search_best_model.py
```

### Run Parallel Coordinates Plot

```bash
cd C:\Users\sacaru\digilignin\DigiLignin\9.Parallel coordinates plot
python parallel_coordinates_best_model.py
```

**What it does:**
- Loads results from `../8.Extrapolation/closest_inputs_best_model.csv`
- Creates interactive parallel coordinates plot
- Saves HTML file
- Opens plot in default web browser

**Expected runtime:** < 1 minute

**Output:**
- `parallel_coordinates_best_model.html` - Interactive plot
- Browser window with visualization

## Using the Interactive Plot

### Features

1. **Color coding:** Lines colored by predicted Tg value
2. **Hover information:** Hover over lines to see exact values
3. **Filtering:** Click and drag on any axis to filter data
4. **Zoom:** Use mouse wheel or pinch to zoom
5. **Pan:** Click and drag to pan

### Interpreting the Plot

**Axes (left to right):**
1. **Lignin (wt%):** 0-100%
2. **Co-polyol type (PTHF):** 250, 650, or 1000 g/mol
3. **r (Ratio):** 0.6-1.4 (NCO/OH ratio)
4. **Copolyol (wt%):** 0-100%
5. **Isocyanate (wt%):** 0-20%
6. **Predicted Tg:** -20°C to 100°C

**Line colors:**
- **Purple/Blue:** Low Tg values
- **Green/Yellow:** Medium Tg values
- **Orange/Red:** High Tg values

### Analysis Tips

1. **Identify trends:** Look for parallel lines indicating correlated features
2. **Find patterns:** Crossing lines suggest complex interactions
3. **Filter by Tg:** Drag on Tg axis to see only specific temperature ranges
4. **Compare formulations:** Select specific Tg range to see corresponding input combinations

## Customization

### Modify Axis Ranges

Edit `parallel_coordinates_best_model.py`:

```python
# Lines 33-75 - Adjust tick values for each axis
dict(
    range=[df['Lignin (wt%)'].min(), df['Lignin (wt%)'].max()],
    label='<b>Lignin (wt%)</b>',
    values=df['Lignin (wt%)'],
    tickvals=list(range(0, 101, 10))  # Change tick spacing
),
```

### Change Color Scale

Edit `parallel_coordinates_best_model.py`:

```python
# Line 22 - Change colorscale
colorscale='Viridis',  # Options: 'Viridis', 'Plasma', 'Inferno', 'Magma', 'Cividis'
```

### Adjust Font Sizes

Edit `parallel_coordinates_best_model.py`:

```python
# Lines 100-106 - Update layout
parallel_coords.update_layout(
    plot_bgcolor='white',
    font=dict(size=22, color='black'),  # Change font size
    hoverlabel=dict(font_size=20)       # Change hover font size
)
```

## Key Insights from Parallel Coordinates

### Typical Patterns

1. **High Tg formulations:**
   - High lignin content
   - Low PTHF molecular weight (250)
   - Higher r values
   - Lower copolyol content

2. **Low Tg formulations:**
   - Low lignin content
   - High PTHF molecular weight (1000)
   - Lower r values
   - Higher copolyol content

3. **Feature interactions:**
   - Lignin and Copolyol are complementary (sum to 100%)
   - PTHF type has discrete effect on Tg
   - r and Isocyanate show complex interactions

## Troubleshooting

### Issue: "File not found"

**Solution:** Run extrapolation analysis first:
```bash
cd ../8.Extrapolation
python adaptive_grid_search_best_model.py
```

### Issue: Plot doesn't open in browser

**Solutions:**
1. Check that HTML file was created
2. Manually open `parallel_coordinates_best_model.html` in browser
3. Try different renderer:
   ```python
   pio.renderers.default = "browser"  # or "notebook", "png", etc.
   ```

### Issue: Plot is too slow/laggy

**Solutions:**
1. Reduce number of data points in extrapolation analysis
2. Simplify plot by removing some axes
3. Use static image export instead of interactive HTML

## Export Options

### Save as Static Image

Add to script:

```python
# After creating the plot
parallel_coords.write_image("parallel_coordinates.png", width=1920, height=1080)
parallel_coords.write_image("parallel_coordinates.pdf")
```

**Note:** Requires `kaleido` package:
```bash
pip install kaleido
```

### Save as Interactive HTML

Already included in script:
```python
parallel_coords.write_html('parallel_coordinates_best_model.html')
```

## Applications

1. **Formulation design:** Identify input combinations for target Tg
2. **Sensitivity analysis:** See which features most affect Tg
3. **Constraint exploration:** Understand feasible parameter space
4. **Communication:** Share interactive plots with collaborators
5. **Publication:** Export high-quality static images

## Related Files

- **Extrapolation analysis:** `../8.Extrapolation/adaptive_grid_search_best_model.py`
- **Model training:** `../7.Mapping/retrain_best_model.py`
- **Model performance:** `../4.Wrapper/Stratified_fixed_split_16_val_16_test/top_10_models_validation_with_test.csv`

## References

- Plotly documentation: https://plotly.com/python/parallel-coordinates-plot/
- Parallel coordinates theory: https://en.wikipedia.org/wiki/Parallel_coordinates
