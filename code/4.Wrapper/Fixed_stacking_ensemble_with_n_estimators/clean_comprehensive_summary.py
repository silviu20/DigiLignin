# -*- coding: utf-8 -*-
"""
Create Clean Comprehensive Tabular Summary of Validation Results
Properly formatted table matching the requested format
"""

import pandas as pd

def create_clean_summary():
    """Create a clean, properly formatted comprehensive summary table."""
    
    # Load both result tables
    print("Loading result tables...")
    df_with_sratio = pd.read_csv('top_5_best_models.csv')
    df_no_sratio = pd.read_csv('top_5_best_models_no_sratio.csv')
    
    # Create the comprehensive summary data
    summary_data = []
    
    # Add top 5 with Sratio
    for _, row in df_with_sratio.iterrows():
        # Clean feature names
        features = row['Feature Combination'].replace("'", "").replace("[", "").replace("]", "").replace(", ", ", ")
        features = features.replace('Lignin (wt%)', 'Lignin (wt%)')
        features = features.replace('Co-polyol type (PTHF)', 'Co-polyol type (PTHF)')
        features = features.replace('r', 'Ratio')
        features = features.replace('Copolyol (wt%)', 'Co-polyol (wt%)')
        features = features.replace('Isocyanate (wt%)', 'Isocyanate (wt%)')
        features = features.replace('Isocyanate (mmol NCO)', 'Isocyanate (mmol NCO)')
        features = features.replace('Isocyonate type', 'Isocyanate type')
        features = features.replace('tin(II) octoate', 'Tin(II) octoate')
        features = features.replace('Sratio(%)', 'Swelling ratio (%)')
        
        summary_data.append({
            'Model': 'Stacking Ensemble',
            'R2': f"{row['R²']:.3f}",
            'MSE': f"{row['MSE']:.2f}",
            'MAE': f"{row['MAE (°C)']:.3f}",
            'Number of base estimators in tree-base models': int(row['N_Estimators']),
            'Features': features
        })
    
    # Add top 5 without Sratio
    for _, row in df_no_sratio.iterrows():
        # Clean feature names
        features = row['Feature Combination'].replace("'", "").replace("[", "").replace("]", "").replace(", ", ", ")
        features = features.replace('Lignin (wt%)', 'Lignin (wt%)')
        features = features.replace('Co-polyol type (PTHF)', 'Co-polyol type (PTHF)')
        features = features.replace('r', 'Ratio')
        features = features.replace('Copolyol (wt%)', 'Co-polyol (wt%)')
        features = features.replace('Isocyanate (wt%)', 'Isocyanate (wt%)')
        features = features.replace('Isocyanate (mmol NCO)', 'Isocyanate (mmol NCO)')
        features = features.replace('Isocyonate type', 'Isocyanate type')
        features = features.replace('tin(II) octoate', 'Tin(II) octoate')
        # Note: Sratio is not present in this dataset
        
        summary_data.append({
            'Model': 'Stacking Ensemble',
            'R2': f"{row['R²']:.3f}",
            'MSE': f"{row['MSE']:.2f}",
            'MAE': f"{row['MAE (°C)']:.3f}",
            'Number of base estimators in tree-base models': int(row['N_Estimators']),
            'Features': features
        })
    
    # Create DataFrame
    df_summary = pd.DataFrame(summary_data)
    
    # Sort by MAE (ascending) to show best performance first
    df_summary['MAE_float'] = df_summary['MAE'].astype(float)
    df_summary = df_summary.sort_values('MAE_float').drop('MAE_float', axis=1)
    
    # Display the table
    pd.set_option('display.max_colwidth', 100)
    pd.set_option('display.precision', 3)
    
    print("\n" + "="*150)
    print("COMPREHENSIVE VALIDATION RESULTS SUMMARY")
    print("="*150)
    print("Top 10 Stacking Ensemble Models (All Feature Combinations)")
    print("="*150)
    
    print(df_summary.to_string(index=False))
    
    print("="*150)
    
    # Save the clean table
    df_summary.to_csv('clean_comprehensive_validation_summary.csv', index=False)
    print(f"\nClean comprehensive table saved to: clean_comprehensive_validation_summary.csv")
    
    # Create separate formatted tables
    print("\n" + "="*120)
    print("TOP 5 MODELS WITH SRATIO(%)")
    print("="*120)
    
    with_sratio_data = []
    for i, (_, row) in enumerate(df_with_sratio.iterrows(), 1):
        features = row['Feature Combination'].replace("'", "").replace("[", "").replace("]", "").replace(", ", ", ")
        features = features.replace('Lignin (wt%)', 'Lignin (wt%)')
        features = features.replace('Co-polyol type (PTHF)', 'Co-polyol type (PTHF)')
        features = features.replace('r', 'Ratio')
        features = features.replace('Copolyol (wt%)', 'Co-polyol (wt%)')
        features = features.replace('Isocyanate (wt%)', 'Isocyanate (wt%)')
        features = features.replace('Isocyanate (mmol NCO)', 'Isocyanate (mmol NCO)')
        features = features.replace('Isocyonate type', 'Isocyanate type')
        features = features.replace('tin(II) octoate', 'Tin(II) octoate')
        features = features.replace('Sratio(%)', 'Swelling ratio (%)')
        
        with_sratio_data.append({
            'Model': 'Stacking Ensemble',
            'R2': f"{row['R²']:.3f}",
            'MSE': f"{row['MSE']:.2f}",
            'MAE': f"{row['MAE (°C)']:.3f}",
            'Number of base estimators in tree-base models': int(row['N_Estimators']),
            'Features': features
        })
    
    df_with_sratio_clean = pd.DataFrame(with_sratio_data)
    print(df_with_sratio_clean.to_string(index=False))
    
    print("\n" + "="*120)
    print("TOP 5 MODELS WITHOUT SRATIO(%)")
    print("="*120)
    
    no_sratio_data = []
    for i, (_, row) in enumerate(df_no_sratio.iterrows(), 1):
        features = row['Feature Combination'].replace("'", "").replace("[", "").replace("]", "").replace(", ", ", ")
        features = features.replace('Lignin (wt%)', 'Lignin (wt%)')
        features = features.replace('Co-polyol type (PTHF)', 'Co-polyol type (PTHF)')
        features = features.replace('r', 'Ratio')
        features = features.replace('Copolyol (wt%)', 'Co-polyol (wt%)')
        features = features.replace('Isocyanate (wt%)', 'Isocyanate (wt%)')
        features = features.replace('Isocyanate (mmol NCO)', 'Isocyanate (mmol NCO)')
        features = features.replace('Isocyonate type', 'Isocyanate type')
        features = features.replace('tin(II) octoate', 'Tin(II) octoate')
        
        no_sratio_data.append({
            'Model': 'Stacking Ensemble',
            'R2': f"{row['R²']:.3f}",
            'MSE': f"{row['MSE']:.2f}",
            'MAE': f"{row['MAE (°C)']:.3f}",
            'Number of base estimators in tree-base models': int(row['N_Estimators']),
            'Features': features
        })
    
    df_no_sratio_clean = pd.DataFrame(no_sratio_data)
    print(df_no_sratio_clean.to_string(index=False))
    
    # Save separate tables
    df_with_sratio_clean.to_csv('formatted_top_5_with_sratio.csv', index=False)
    df_no_sratio_clean.to_csv('formatted_top_5_without_sratio.csv', index=False)
    
    print(f"\nAdditional files saved:")
    print(f"  - formatted_top_5_with_sratio.csv")
    print(f"  - formatted_top_5_without_sratio.csv")
    
    # Summary statistics
    print("\n" + "="*120)
    print("SUMMARY STATISTICS")
    print("="*120)
    
    print(f"\nWith Sratio(%) - Top 5 Models:")
    print(f"  Best MAE: {df_with_sratio['MAE (°C)'].min():.3f}°C")
    print(f"  Best R²: {df_with_sratio['R²'].max():.3f}")
    print(f"  Average Estimators: {df_with_sratio['N_Estimators'].mean():.0f}")
    
    print(f"\nWithout Sratio(%) - Top 5 Models:")
    print(f"  Best MAE: {df_no_sratio['MAE (°C)'].min():.3f}°C")
    print(f"  Best R²: {df_no_sratio['R²'].max():.3f}")
    print(f"  Average Estimators: {df_no_sratio['N_Estimators'].mean():.0f}")
    
    return df_summary

if __name__ == "__main__":
    clean_summary = create_clean_summary()
