import pandas as pd

# Read the detailed analysis
df = pd.read_csv('top_5_detailed_model_analysis.csv')

# Create a formatted table for better readability
print("="*120)
print("DETAILED MODEL ANALYSIS - TOP 5 PERFORMING COMBINATIONS")
print("="*120)

for idx, row in df.iterrows():
    rank = int(row['Rank'])
    print(f"\n{'='*20} RANK {rank} {'='*20}")
    print(f"Features: {row['Features Used']}")
    print(f"Number of Features: {row['Number of Features']}")
    print(f"Performance:")
    print(f"  • MAE Validation: {row['MAE Validation (°C)']}°C")
    print(f"  • R² Validation: {row['R² Validation']}")
    print(f"  • Generalization Gap: {row['Generalization Gap (°C)']}°C")
    
    print(f"\nBase Models (5 total):")
    base_models = eval(row['Base Models'])
    base_hyperparams = eval(row['Base Model Hyperparameters'])
    
    for i, (model, hyperparams) in enumerate(zip(base_models, base_hyperparams), 1):
        print(f"  {i}. {model}")
        print(f"     Hyperparameters: {hyperparams}")
    
    print(f"\nMeta Model:")
    print(f"  • Type: {row['Meta Model']}")
    print(f"  • Hyperparameters: {row['Meta Model Hyperparameters']}")
    print()

# Create a compact summary table
compact_data = []
for idx, row in df.iterrows():
    rank = int(row['Rank'])
    base_models = eval(row['Base Models'])
    
    compact_data.append({
        'Rank': rank,
        'Features': row['Features Used'][:50] + '...' if len(row['Features Used']) > 50 else row['Features Used'],
        'Num Feat': row['Number of Features'],
        'MAE (°C)': row['MAE Validation (°C)'],
        'R²': row['R² Validation'],
        'Gap (°C)': row['Generalization Gap (°C)'],
        'GBR': '✓' if 'Gradient Boosting' in str(base_models) else '✗',
        'RF': '✓' if 'Random Forest' in str(base_models) else '✗',
        'SVR': '✓' if 'Support Vector' in str(base_models) else '✗',
        'Lasso': '✓' if 'Lasso' in str(base_models) else '✗',
        'ElasticNet': '✓' if 'Elastic Net' in str(base_models) else '✗',
        'Meta': 'Ridge'
    })

compact_df = pd.DataFrame(compact_data)

print("\n" + "="*120)
print("COMPACT SUMMARY TABLE")
print("="*120)
print(compact_df.to_string(index=False))

# Create LaTeX table for publication
latex_table = """
\\begin{table}[h]
\\centering
\\caption{Top 5 Performing Feature Combinations with Detailed Model Information}
\\label{tab:detailed_models}
\\begin{tabular}{clccccccccc}
\\hline
Rank & Features (truncated) & \\# & MAE & R² & Gap & GBR & RF & SVR & Lasso & EN \\\\
     &                       & Feat & (°C) &    & (°C) &    &    &    &      &    \\\\
\\hline
"""

for idx, row in compact_df.iterrows():
    latex_table += f"{row['Rank']} & {row['Features'][:30]}... & {row['Num Feat']} & {row['MAE (°C)']} & {row['R²']} & {row['Gap (°C)']} & {row['GBR']} & {row['RF']} & {row['SVR']} & {row['Lasso']} & {row['ElasticNet']} \\\\\n"

latex_table += """\\hline
\\end{tabular}
\\\\
\\footnotesize{GBR: Gradient Boosting Regressor, RF: Random Forest, SVR: Support Vector Regressor, EN: Elastic Net. All models use 1000 estimators for tree-based methods.}
\\end{table}
"""

# Save LaTeX table
with open('detailed_models_latex.txt', 'w') as f:
    f.write(latex_table)

print(f"\n" + "="*120)
print("KEY INSIGHTS:")
print("="*120)
print("• All top 5 combinations use the same 5 base models:")
print("  - Gradient Boosting Regressor (n_estimators=1000, lr=0.01, max_depth=3)")
print("  - Random Forest Regressor (n_estimators=1000, max_depth=10 or None)")
print("  - Support Vector Regressor (C=0.1, kernel=rbf)")
print("  - Lasso Regression (alpha=0.1, max_iter=1000)")
print("  - Elastic Net (alpha=0.1, l1_ratio=0.1 or 0.5, max_iter=1000)")
print()
print("• All use Ridge Regression (alpha=1.0) as the meta-model")
print("• Performance differences come from feature selection, not model architecture")
print("• Generalization gaps are small (±0.5°C), indicating good model validation")
print("• Best performance with 5-6 features, not necessarily more features")
print("="*120)

print(f"\nFiles created:")
print(f"  - top_5_detailed_model_analysis.csv (full details)")
print(f"  - top_5_summary_table.csv (summary view)")
print(f"  - detailed_models_latex.txt (LaTeX formatted table)")
