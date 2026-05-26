# -*- coding: utf-8 -*-
"""
Create Comprehensive Tabular Summary of Validation Results
Combining results from both with and without Sratio(%) analyses
"""

import pandas as pd
import numpy as np

def create_comprehensive_summary():
    """Create a comprehensive summary table combining both analyses."""
    
    # Load both result tables
    print("Loading result tables...")
    df_with_sratio = pd.read_csv('top_5_best_models.csv')
    df_no_sratio = pd.read_csv('top_5_best_models_no_sratio.csv')
    
    # Clean up feature names for better display
    def clean_feature_names(feature_str):
        """Clean feature names for display."""
        return feature_str.replace('Lignin (wt%)', 'Lignin (wt%)') \
                         .replace('Co-polyol type (PTHF)', 'Co-polyol type (PTHF)') \
                         .replace('r', 'Ratio') \
                         .replace('Copolyol (wt%)', 'Co-polyol (wt%)') \
                         .replace('Isocyanate (wt%)', 'Isocyanate (wt%)') \
                         .replace('Isocyanate (mmol NCO)', 'Isocyanate (mmol NCO)') \
                         .replace('Isocyonate type', 'Isocyanate type') \
                         .replace('tin(II) octoate', 'Tin(II) octoate') \
                         .replace('Sratio(%)', 'Swelling ratio (%)')
    
    # Process both dataframes
    df_with_sratio['Features_Clean'] = df_with_sratio['Feature Combination'].apply(clean_feature_names)
    df_no_sratio['Features_Clean'] = df_no_sratio['Feature Combination'].apply(clean_feature_names)
    
    # Create comprehensive summary table
    comprehensive_data = []
    
    # Add top 5 with Sratio
    for _, row in df_with_sratio.iterrows():
        comprehensive_data.append({
            'Model': 'Stacking Ensemble',
            'R2': f"{row['R²']:.3f}",
            'MSE': f"{row['MSE']:.2f}",
            'MAE': f"{row['MAE (°C)']:.3f}",
            'Number of base estimators in tree-base models': int(row['N_Estimators']),
            'Features': row['Features_Clean'],
            'Dataset': 'With Sratio(%)',
            'Rank': int(row['Rank'])
        })
    
    # Add top 5 without Sratio
    for _, row in df_no_sratio.iterrows():
        comprehensive_data.append({
            'Model': 'Stacking Ensemble',
            'R2': f"{row['R²']:.3f}",
            'MSE': f"{row['MSE']:.2f}",
            'MAE': f"{row['MAE (°C)']:.3f}",
            'Number of base estimators in tree-base models': int(row['N_Estimators']),
            'Features': row['Features_Clean'],
            'Dataset': 'Without Sratio(%)',
            'Rank': int(row['Rank'])
        })
    
    # Create DataFrame
    df_comprehensive = pd.DataFrame(comprehensive_data)
    
    # Sort by MAE (ascending) to show best performance first
    df_comprehensive['MAE_float'] = df_comprehensive['MAE'].astype(float)
    df_comprehensive = df_comprehensive.sort_values('MAE_float').drop('MAE_float', axis=1)
    
    # Add overall rank
    df_comprehensive['Overall_Rank'] = range(1, len(df_comprehensive) + 1)
    
    # Reorder columns for final display
    final_columns = ['Model', 'R2', 'MSE', 'MAE', 'Number of base estimators in tree-base models', 
                    'Features', 'Dataset', 'Overall_Rank']
    df_comprehensive = df_comprehensive[final_columns]
    
    # Display the table
    pd.set_option('display.max_colwidth', 120)
    pd.set_option('display.precision', 3)
    
    print("\n" + "="*150)
    print("COMPREHENSIVE VALIDATION RESULTS SUMMARY")
    print("="*150)
    print("Top 10 Stacking Ensemble Models (All Feature Combinations)")
    print("="*150)
    
    # Display without the Dataset and Overall_Rank columns for cleaner output
    display_df = df_comprehensive.drop(['Dataset', 'Overall_Rank'], axis=1)
    print(display_df.to_string(index=False))
    
    print("="*150)
    
    # Save comprehensive table
    df_comprehensive.to_csv('comprehensive_validation_summary.csv', index=False)
    print(f"\nComprehensive table saved to: comprehensive_validation_summary.csv")
    
    # Create separate tables for each dataset
    print("\n" + "="*100)
    print("TOP 5 MODELS WITH SRATIO(%)")
    print("="*100)
    
    with_sratio_table = df_comprehensive[df_comprehensive['Dataset'] == 'With Sratio(%)'].copy()
    with_sratio_table = with_sratio_table.drop(['Dataset', 'Overall_Rank'], axis=1)
    with_sratio_table['Rank'] = range(1, 6)
    with_sratio_table = with_sratio_table[['Rank'] + [col for col in with_sratio_table.columns if col != 'Rank']]
    print(with_sratio_table.to_string(index=False))
    
    print("\n" + "="*100)
    print("TOP 5 MODELS WITHOUT SRATIO(%)")
    print("="*100)
    
    no_sratio_table = df_comprehensive[df_comprehensive['Dataset'] == 'Without Sratio(%)'].copy()
    no_sratio_table = no_sratio_table.drop(['Dataset', 'Overall_Rank'], axis=1)
    no_sratio_table['Rank'] = range(1, 6)
    no_sratio_table = no_sratio_table[['Rank'] + [col for col in no_sratio_table.columns if col != 'Rank']]
    print(no_sratio_table.to_string(index=False))
    
    # Statistical analysis
    print("\n" + "="*100)
    print("STATISTICAL ANALYSIS")
    print("="*100)
    
    with_sratio_stats = df_comprehensive[df_comprehensive['Dataset'] == 'With Sratio(%)']
    no_sratio_stats = df_comprehensive[df_comprehensive['Dataset'] == 'Without Sratio(%)']
    
    print(f"\nWith Sratio(%) - Top 5 Models:")
    print(f"  Average MAE: {with_sratio_stats['MAE'].astype(float).mean():.3f}°C")
    print(f"  Average R²: {with_sratio_stats['R2'].astype(float).mean():.3f}")
    print(f"  Average MSE: {with_sratio_stats['MSE'].astype(float).mean():.2f}")
    print(f"  Average Estimators: {with_sratio_stats['Number of base estimators in tree-base models'].mean():.0f}")
    print(f"  Best MAE: {with_sratio_stats['MAE'].astype(float).min():.3f}°C")
    print(f"  Best R²: {with_sratio_stats['R2'].astype(float).max():.3f}")
    
    print(f"\nWithout Sratio(%) - Top 5 Models:")
    print(f"  Average MAE: {no_sratio_stats['MAE'].astype(float).mean():.3f}°C")
    print(f"  Average R²: {no_sratio_stats['R2'].astype(float).mean():.3f}")
    print(f"  Average MSE: {no_sratio_stats['MSE'].astype(float).mean():.2f}")
    print(f"  Average Estimators: {no_sratio_stats['Number of base estimators in tree-base models'].mean():.0f}")
    print(f"  Best MAE: {no_sratio_stats['MAE'].astype(float).min():.3f}°C")
    print(f"  Best R²: {no_sratio_stats['R2'].astype(float).max():.3f}")
    
    # Performance comparison
    best_with = with_sratio_stats.loc[with_sratio_stats['MAE'].astype(float).idxmin()]
    best_without = no_sratio_stats.loc[no_sratio_stats['MAE'].astype(float).idxmin()]
    
    print(f"\nHead-to-Head Comparison (Best Models):")
    print(f"  With Sratio(%):    MAE = {best_with['MAE']}°C, R² = {best_with['R2']}, Estimators = {best_with['Number of base estimators in tree-base models']}")
    print(f"  Without Sratio(%): MAE = {best_without['MAE']}°C, R² = {best_without['R2']}, Estimators = {best_without['Number of base estimators in tree-base models']}")
    
    # Feature analysis
    print("\n" + "="*100)
    print("FEATURE ANALYSIS")
    print("="*100)
    
    # Count feature frequency in top models
    all_features = []
    for features in df_comprehensive['Features']:
        feature_list = [f.strip() for f in features.split(',')]
        all_features.extend(feature_list)
    
    feature_counts = {}
    for feature in all_features:
        feature_counts[feature] = feature_counts.get(feature, 0) + 1
    
    print(f"\nFeature frequency in top 10 models:")
    for feature, count in sorted(feature_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / 10) * 100
        print(f"  {feature}: {count}/10 ({percentage:.0f}%)")
    
    # Save separate tables
    with_sratio_table.to_csv('top_5_with_sratio_summary.csv', index=False)
    no_sratio_table.to_csv('top_5_without_sratio_summary.csv', index=False)
    
    print(f"\nAdditional files saved:")
    print(f"  - top_5_with_sratio_summary.csv")
    print(f"  - top_5_without_sratio_summary.csv")
    
    return df_comprehensive

if __name__ == "__main__":
    comprehensive_table = create_comprehensive_summary()
