# -*- coding: utf-8 -*-
"""
Create Top 5 Best Models Table from Comprehensive N_Estimators Analysis
Excluding models that contain Sratio(%) feature
"""

import pandas as pd
import numpy as np

def create_top_models_no_sratio():
    """Create a formatted table of the top 5 best performing models without Sratio feature."""
    
    # Load the results
    print("Loading comprehensive results...")
    df = pd.read_csv('all_combinations_n_estimators_results.csv')
    
    # Filter out models that contain Sratio(%) in their feature combination
    print("Filtering out models with Sratio(%) feature...")
    df_no_sratio = df[~df['Feature Combination'].str.contains('Sratio\\(\\%\\)', case=False, na=False, regex=True)].copy()
    
    print(f"Original total experiments: {len(df)}")
    print(f"Experiments without Sratio(%): {len(df_no_sratio)}")
    print(f"Filtered out: {len(df) - len(df_no_sratio)} experiments")
    
    # Parse feature combinations from string to list for better display
    df_no_sratio['Features_Display'] = df_no_sratio['Feature Combination'].str.replace("'", "").str.replace("[", "").str.replace("]", "").str.replace(", ", " + ")
    
    # Sort by MAE Validation (lower is better) and get top 5
    top_5_mae = df_no_sratio.nsmallest(5, 'MAE Validation').copy()
    
    # Sort by R² Validation (higher is better) and get top 5
    top_5_r2 = df_no_sratio.nlargest(5, 'R2 Validation').copy()
    
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
    print("TOP 5 BEST PERFORMING MODELS WITHOUT SRATIO(%) (Ranked by MAE)")
    print("="*120)
    print(top_5_table.to_string(index=False))
    print("="*120)
    
    # Save to CSV
    top_5_table.to_csv('top_5_best_models_no_sratio.csv', index=False)
    print(f"\nTable saved to: top_5_best_models_no_sratio.csv")
    
    # Create additional analysis
    print("\n" + "="*80)
    print("ADDITIONAL ANALYSIS - WITHOUT SRATIO(%)")
    print("="*80)
    
    # Best performance by n_estimators (without Sratio)
    print("\nBest Performance by N_Estimators (without Sratio):")
    best_by_n_estimators = df_no_sratio.loc[df_no_sratio.groupby('n_estimators')['MAE Validation'].idxmin()]
    for _, row in best_by_n_estimators.iterrows():
        print(f"  {row['n_estimators']:4d}: MAE = {row['MAE Validation']:.3f}°C, "
              f"Features = {row['Number of Features']}, "
              f"R² = {row['R2 Validation']:.3f}")
    
    # Performance by feature count (without Sratio)
    print("\nBest Performance by Number of Features (without Sratio):")
    best_by_features = df_no_sratio.loc[df_no_sratio.groupby('Number of Features')['MAE Validation'].idxmin()]
    for _, row in best_by_features.iterrows():
        print(f"  {row['Number of Features']} features: MAE = {row['MAE Validation']:.3f}°C, "
              f"N_Estimators = {row['n_estimators']}, "
              f"R² = {row['R2 Validation']:.3f}")
    
    # Overall statistics (without Sratio)
    print(f"\nOverall Statistics (without Sratio):")
    print(f"  Total experiments: {len(df_no_sratio)}")
    print(f"  Feature combinations: {df_no_sratio['Feature Combination'].nunique()}")
    print(f"  N_Estimators tested: {df_no_sratio['n_estimators'].nunique()}")
    print(f"  Best MAE: {df_no_sratio['MAE Validation'].min():.3f}°C")
    print(f"  Best R²: {df_no_sratio['R2 Validation'].max():.3f}")
    print(f"  Average MAE: {df_no_sratio['MAE Validation'].mean():.3f}°C")
    print(f"  Average R²: {df_no_sratio['R2 Validation'].mean():.3f}")
    
    # Comparison with original results
    print("\n" + "="*80)
    print("COMPARISON: WITH SRATIO(%) vs WITHOUT SRATIO(%)")
    print("="*80)
    
    # Original best (with Sratio)
    original_best = df.nsmallest(1, 'MAE Validation').iloc[0]
    no_sratio_best = df_no_sratio.nsmallest(1, 'MAE Validation').iloc[0]
    
    print(f"\nBest Performance Comparison:")
    print(f"  WITH Sratio(%):    MAE = {original_best['MAE Validation']:.3f}°C, "
          f"R² = {original_best['R2 Validation']:.3f}, "
          f"Features = {original_best['Number of Features']}, "
          f"Estimators = {original_best['n_estimators']}")
    print(f"  WITHOUT Sratio(%): MAE = {no_sratio_best['MAE Validation']:.3f}°C, "
          f"R² = {no_sratio_best['R2 Validation']:.3f}, "
          f"Features = {no_sratio_best['Number of Features']}, "
          f"Estimators = {no_sratio_best['n_estimators']}")
    
    # Performance difference
    mae_diff = no_sratio_best['MAE Validation'] - original_best['MAE Validation']
    r2_diff = original_best['R2 Validation'] - no_sratio_best['R2 Validation']
    
    print(f"\nPerformance Impact of Removing Sratio(%):")
    print(f"  MAE increase: {mae_diff:.3f}°C ({(mae_diff/original_best['MAE Validation'])*100:.1f}%)")
    print(f"  R² decrease: {r2_diff:.3f} ({(r2_diff/original_best['R2 Validation'])*100:.1f}%)")
    
    # Create comparison with R² ranking (without Sratio)
    print("\n" + "="*80)
    print("COMPARISON: TOP 5 BY MAE vs TOP 5 BY R² (WITHOUT SRATIO)")
    print("="*80)
    
    print("\nTop 5 by MAE (without Sratio):")
    for i, (_, row) in enumerate(top_5_mae.iterrows(), 1):
        print(f"  {i}. MAE: {row['MAE Validation']:.3f}°C, R²: {row['R2 Validation']:.3f}, "
              f"Estimators: {row['n_estimators']}, Features: {row['Number of Features']}")
    
    print("\nTop 5 by R² (without Sratio):")
    for i, (_, row) in enumerate(top_5_r2.iterrows(), 1):
        print(f"  {i}. R²: {row['R2 Validation']:.3f}, MAE: {row['MAE Validation']:.3f}°C, "
              f"Estimators: {row['n_estimators']}, Features: {row['Number of Features']}")
    
    # Check overlap
    mae_features = set(top_5_mae['Feature Combination'])
    r2_features = set(top_5_r2['Feature Combination'])
    overlap = mae_features.intersection(r2_features)
    
    print(f"\nOverlap between MAE and R² top 5 (without Sratio): {len(overlap)} models")
    if overlap:
        print("Models that appear in both rankings:")
        for i, feature in enumerate(overlap, 1):
            mae_row = top_5_mae[top_5_mae['Feature Combination'] == feature].iloc[0]
            r2_row = top_5_r2[top_5_r2['Feature Combination'] == feature].iloc[0]
            print(f"  {i}. Features: {mae_row['Number of Features']}, "
                  f"Estimators: {mae_row['n_estimators']}")
    
    # Feature frequency analysis
    print("\n" + "="*80)
    print("FEATURE FREQUENCY ANALYSIS (WITHOUT SRATIO)")
    print("="*80)
    
    # Count feature occurrences in top 10 models
    top_10_no_sratio = df_no_sratio.nsmallest(10, 'MAE Validation')
    feature_counts = {}
    
    for _, row in top_10_no_sratio.iterrows():
        # Parse the feature combination
        features = eval(row['Feature Combination'])
        for feature in features:
            feature_counts[feature] = feature_counts.get(feature, 0) + 1
    
    print("\nFeature frequency in top 10 models (without Sratio):")
    for feature, count in sorted(feature_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / 10) * 100
        print(f"  {feature}: {count}/10 ({percentage:.0f}%)")
    
    return top_5_table

if __name__ == "__main__":
    top_5_table = create_top_models_no_sratio()
