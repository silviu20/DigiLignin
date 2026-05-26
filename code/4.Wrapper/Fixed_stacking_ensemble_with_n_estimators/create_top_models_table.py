# -*- coding: utf-8 -*-
"""
Create Top 5 Best Models Table from Comprehensive N_Estimators Analysis
"""

import pandas as pd
import numpy as np

def create_top_models_table():
    """Create a formatted table of the top 5 best performing models."""
    
    # Load the results
    print("Loading comprehensive results...")
    df = pd.read_csv('all_combinations_n_estimators_results.csv')
    
    # Parse feature combinations from string to list for better display
    df['Features_Display'] = df['Feature Combination'].str.replace("'", "").str.replace("[", "").str.replace("]", "").str.replace(", ", " + ")
    
    # Sort by MAE Validation (lower is better) and get top 5
    top_5_mae = df.nsmallest(5, 'MAE Validation').copy()
    
    # Sort by R² Validation (higher is better) and get top 5
    top_5_r2 = df.nlargest(5, 'R2 Validation').copy()
    
    # Create the main table based on MAE (primary metric)
    top_5_table = top_5_mae[['Features_Display', 'Number of Features', 'n_estimators', 
                             'MAE Validation', 'R2 Validation', 'MSE Validation']].copy()
    
    # Rename columns for better display
    top_5_table.columns = ['Feature Combination', 'Num Features', 'N_Estimators', 
                          'MAE (°C)', 'R²', 'MSE']
    
    # Add confidence intervals
    top_5_table['MAE CI'] = top_5_mae.apply(lambda row: 
        f"[{row['Validation MAE CI Lower']:.2f}, {row['Validation MAE CI Upper']:.2f}]", axis=1)
    top_5_table['R² CI'] = top_5_mae.apply(lambda row: 
        f"[{row['Validation R2 CI Lower']:.3f}, {row['Validation R2 CI Upper']:.3f}]", axis=1)
    
    # Reorder columns
    top_5_table = top_5_table[['Feature Combination', 'Num Features', 'N_Estimators', 
                              'MAE (°C)', 'MAE CI', 'R²', 'R² CI', 'MSE']]
    
    # Add rank
    top_5_table.insert(0, 'Rank', range(1, 6))
    
    # Format the table
    pd.set_option('display.max_colwidth', 80)
    pd.set_option('display.precision', 3)
    
    print("\n" + "="*120)
    print("TOP 5 BEST PERFORMING MODELS (Ranked by MAE)")
    print("="*120)
    print(top_5_table.to_string(index=False))
    print("="*120)
    
    # Save to CSV
    top_5_table.to_csv('top_5_best_models.csv', index=False)
    print(f"\nTable saved to: top_5_best_models.csv")
    
    # Create additional analysis
    print("\n" + "="*80)
    print("ADDITIONAL ANALYSIS")
    print("="*80)
    
    # Best performance by n_estimators
    print("\nBest Performance by N_Estimators:")
    best_by_n_estimators = df.loc[df.groupby('n_estimators')['MAE Validation'].idxmin()]
    for _, row in best_by_n_estimators.iterrows():
        print(f"  {row['n_estimators']:4d}: MAE = {row['MAE Validation']:.3f}°C, "
              f"Features = {row['Number of Features']}, "
              f"R² = {row['R2 Validation']:.3f}")
    
    # Performance by feature count
    print("\nBest Performance by Number of Features:")
    best_by_features = df.loc[df.groupby('Number of Features')['MAE Validation'].idxmin()]
    for _, row in best_by_features.iterrows():
        print(f"  {row['Number of Features']} features: MAE = {row['MAE Validation']:.3f}°C, "
              f"N_Estimators = {row['n_estimators']}, "
              f"R² = {row['R2 Validation']:.3f}")
    
    # Overall statistics
    print(f"\nOverall Statistics:")
    print(f"  Total experiments: {len(df)}")
    print(f"  Feature combinations: {df['Feature Combination'].nunique()}")
    print(f"  N_Estimators tested: {df['n_estimators'].nunique()}")
    print(f"  Best overall MAE: {df['MAE Validation'].min():.3f}°C")
    print(f"  Best overall R²: {df['R2 Validation'].max():.3f}")
    print(f"  Average MAE: {df['MAE Validation'].mean():.3f}°C")
    print(f"  Average R²: {df['R2 Validation'].mean():.3f}")
    
    # Create comparison with R² ranking
    print("\n" + "="*80)
    print("COMPARISON: TOP 5 BY MAE vs TOP 5 BY R²")
    print("="*80)
    
    print("\nTop 5 by MAE (already shown above):")
    for i, (_, row) in enumerate(top_5_mae.iterrows(), 1):
        print(f"  {i}. MAE: {row['MAE Validation']:.3f}°C, R²: {row['R2 Validation']:.3f}, "
              f"Estimators: {row['n_estimators']}, Features: {row['Number of Features']}")
    
    print("\nTop 5 by R²:")
    for i, (_, row) in enumerate(top_5_r2.iterrows(), 1):
        print(f"  {i}. R²: {row['R2 Validation']:.3f}, MAE: {row['MAE Validation']:.3f}°C, "
              f"Estimators: {row['n_estimators']}, Features: {row['Number of Features']}")
    
    # Check overlap
    mae_features = set(top_5_mae['Feature Combination'])
    r2_features = set(top_5_r2['Feature Combination'])
    overlap = mae_features.intersection(r2_features)
    
    print(f"\nOverlap between MAE and R² top 5: {len(overlap)} models")
    if overlap:
        print("Models that appear in both rankings:")
        for i, feature in enumerate(overlap, 1):
            mae_row = top_5_mae[top_5_mae['Feature Combination'] == feature].iloc[0]
            r2_row = top_5_r2[top_5_r2['Feature Combination'] == feature].iloc[0]
            print(f"  {i}. Features: {mae_row['Number of Features']}, "
                  f"Estimators: {mae_row['n_estimators']}")
    
    return top_5_table

if __name__ == "__main__":
    top_5_table = create_top_models_table()
