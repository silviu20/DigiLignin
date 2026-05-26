# Split Statistics Table (Table SY) - Documentation

## Overview
This directory contains the split statistics table and visualizations for the stratified fixed-split ensemble method, suitable for inclusion in the manuscript.

## Generated Files

### 1. **Table Files**
- **`split_statistics_table.csv`** - CSV format for Excel/data analysis
- **`split_statistics_table.tex`** - LaTeX format ready for manuscript inclusion

### 2. **Visualization Files**
- **`split_statistics_visualization.png/pdf/svg`** - Three-panel visualization showing:
  - Mean ± Std Dev for each split
  - Min-Max range for each split
  - Sample size distribution
  
- **`split_distribution_comparison.png/pdf/svg`** - Box plot style comparison of distributions across splits

### 3. **Source Data**
- **`split_statistics.json`** - Raw statistics from the stratified splitting process

### 4. **Scripts**
- **`create_split_statistics_table.py`** - Generates the table files
- **`visualize_split_statistics.py`** - Creates the visualizations

## Table SY: Split Statistics Summary

| Data Split | N Samples | Mean (°C) | Std Dev (°C) | Min (°C) | Max (°C) | Range (°C) |
|------------|-----------|-----------|--------------|----------|----------|------------|
| Training   | 104       | 45.04     | 27.00        | -16.61   | 100.20   | 116.81     |
| Validation | 16        | 40.43     | 28.74        | -17.74   | 76.90    | 94.64      |
| Test       | 16        | 41.26     | 28.24        | -16.74   | 77.37    | 94.11      |

## Key Insights

### Distribution Quality
- **Total samples**: 136
- **Training set**: 104 samples (76.5%)
- **Validation set**: 16 samples (11.8%)
- **Test set**: 16 samples (11.8%)

### Distribution Similarity
- **Overall mean Tg**: 44.05°C
- **Training mean deviation**: 0.99°C (2.2% from overall mean)
- **Validation mean deviation**: 3.62°C (8.2% from overall mean)
- **Test mean deviation**: 2.79°C (6.3% from overall mean)

### Why This Matters
✅ **Representative sampling**: All splits cover similar target ranges  
✅ **Stratified approach**: Systematic sampling ensures even distribution  
✅ **Small set reliability**: Despite only 16 samples each, validation and test sets are representative  
✅ **Reproducibility**: Fixed splits enable consistent comparison across experiments  

## Usage in Manuscript

### For LaTeX Documents
Simply include the file `split_statistics_table.tex` in your manuscript:

```latex
\input{split_statistics_table.tex}
```

Or reference it in text:
```latex
The stratified splitting process resulted in representative distributions 
across all data partitions (Table~\ref{tab:split_statistics}).
```

### For Word Documents
1. Open `split_statistics_table.csv` in Excel
2. Format as needed
3. Copy and paste into Word document

### For Figures
Use the visualization files:
- **Figure for manuscript**: `split_statistics_visualization.pdf` (high quality, vector format)
- **Figure for presentations**: `split_statistics_visualization.png` (300 DPI)
- **Supplementary figure**: `split_distribution_comparison.pdf` (box plot comparison)

## Regenerating the Files

If you need to regenerate the table or visualizations:

```bash
# Generate table files
python create_split_statistics_table.py

# Generate visualizations
python visualize_split_statistics.py
```

## Citation Text Suggestion

For the manuscript methods section:

> "The stratified splitting process ensured representative target variable 
> distribution across all data partitions. As shown in Table SY, the training 
> (n=104), validation (n=16), and test (n=16) sets exhibited similar statistical 
> characteristics, with mean Tg values of 45.04±27.00°C, 40.43±28.74°C, and 
> 41.26±28.24°C, respectively. The mean deviations from the overall dataset mean 
> (44.05°C) were minimal (2.2%, 8.2%, and 6.3% for training, validation, and test 
> sets, respectively), confirming the effectiveness of the stratified sampling 
> approach in maintaining representative distributions despite the small validation 
> and test set sizes."

## Notes

- All statistics are calculated on the original (unscaled) target variable (Tg in °C)
- The stratified splitting uses systematic sampling with a fixed random seed (42)
- The same splits are used consistently across all 819 experiments (63 feature combinations × 13 n_estimators values)

