import pandas as pd
import numpy as np

# Read the results data
df = pd.read_csv('Fixed_Stacking_Ensemble/fixed_stacking_results_all_combinations.csv')

# Add number of features column
df['Number of Features'] = df['Feature Combination'].apply(lambda x: len(eval(x)))

# Get top 5 performing combinations (by MAE)
top_5 = df.nsmallest(5, 'MAE Validation').copy()

# Format the table for better display
def format_features(feature_combo):
    """Format feature combination for better readability"""
    features = eval(feature_combo)
    return ', '.join(features)

# Create formatted table
table_data = []
for idx, row in top_5.iterrows():
    table_data.append({
        'Rank': len(table_data) + 1,
        'Features Used': format_features(row['Feature Combination']),
        'Number of Features': int(row['Number of Features']),
        'MAE Validation (°C)': f"{row['MAE Validation']:.3f}",
        'MSE Validation': f"{row['MSE Validation']:.3f}",
        'R² Validation': f"{row['R-squared Validation']:.3f}",
        'MAE Train (°C)': f"{row['Train MAE']:.3f}",
        'MSE Train': f"{row['Train MSE']:.3f}",
        'R² Train': f"{row['Train R-squared']:.3f}",
        'Number of Estimators': int(row['Number of Estimators']),
        'Generalization Gap (°C)': f"{row['MAE Validation'] - row['Train MAE']:.3f}"
    })

# Create DataFrame for display
table_df = pd.DataFrame(table_data)

# Display the table
print("\n" + "="*120)
print("TOP 5 PERFORMING FEATURE COMBINATIONS - FIXED STACKING ENSEMBLE")
print("="*120)
print()

# Print formatted table
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 50)

print(table_df.to_string(index=False))

print("\n" + "="*120)
print("DETAILED METRICS")
print("="*120)

# Print detailed information for each combination
for idx, row in top_5.iterrows():
    rank = list(top_5.index).index(idx) + 1
    print(f"\nRank {rank}:")
    print(f"  Features: {format_features(row['Feature Combination'])}")
    print(f"  Number of Features: {int(row['Number of Features'])}")
    print(f"  Number of Estimators: {int(row['Number of Estimators'])}")
    print(f"  Validation Metrics:")
    print(f"    MAE: {row['MAE Validation']:.3f}°C")
    print(f"    MSE: {row['MSE Validation']:.3f}")
    print(f"    R²: {row['R-squared Validation']:.3f}")
    print(f"    MAE CI: [{row['Validation MAE CI Lower']:.3f}, {row['Validation MAE CI Upper']:.3f}]")
    print(f"  Training Metrics:")
    print(f"    MAE: {row['Train MAE']:.3f}°C")
    print(f"    MSE: {row['Train MSE']:.3f}")
    print(f"    R²: {row['Train R-squared']:.3f}")
    print(f"    Train MAE CI: [{row['Train MAE CI Lower']:.3f}, {row['Train MAE CI Upper']:.3f}]")
    print(f"  Generalization Gap: {row['MAE Validation'] - row['Train MAE']:.3f}°C")

# Save table to CSV
table_df.to_csv('top_5_combinations_table.csv', index=False)
print(f"\nTable saved to: top_5_combinations_table.csv")

# Create LaTeX table for academic papers
latex_table = """
\\begin{table}[h]
\\centering
\\caption{Top 5 Performing Feature Combinations - Fixed Stacking Ensemble}
\\label{tab:top_combinations}
\\begin{tabular}{clcccccc}
\\hline
Rank & Features Used & \\# Features & MAE Val (°C) & MSE Val & R² Val & \\# Estimators \\\\
\\hline
"""

for idx, row in top_5.iterrows():
    rank = list(top_5.index).index(idx) + 1
    features = format_features(row['Feature Combination'])
    # Truncate long feature names for LaTeX
    if len(features) > 40:
        features = features[:37] + "..."
    
    latex_table += f"{rank} & {features} & {int(row['Number of Features'])} & {row['MAE Validation']:.3f} & {row['MSE Validation']:.1f} & {row['R-squared Validation']:.3f} & {int(row['Number of Estimators'])} \\\\\n"

latex_table += """\\hline
\\end{tabular}
\\end{table}
"""

# Save LaTeX table
with open('top_5_combinations_latex.txt', 'w') as f:
    f.write(latex_table)

print(f"LaTeX table saved to: top_5_combinations_latex.txt")
print("="*120)
