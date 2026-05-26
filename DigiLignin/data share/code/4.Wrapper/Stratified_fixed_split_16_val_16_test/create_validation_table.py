# -*- coding: utf-8 -*-
"""
Create table for top 10 models based on validation MAE with corresponding test data
"""

import pandas as pd
import ast

def parse_features(feature_str):
    """Parse feature string to list."""
    try:
        return ast.literal_eval(feature_str)
    except:
        return feature_str

def main():
    """Load results and create top 10 validation table with test data."""
    print("Loading results...")
    df = pd.read_csv('fixed_split_results.csv')
    
    # Sort by Validation MAE (lowest first) and get top 10
    top_10_val = df.nsmallest(10, 'Validation MAE').copy()
    
    print("\n" + "="*140)
    print("TOP 10 STACKED ENSEMBLE MODELS (Lowest Validation MAE) WITH TEST DATA")
    print("="*140)
    
    # Create formatted table with both validation and test data
    print(f"{'Rank':<4} {'Val R²':<8} {'Val MSE':<10} {'Val MAE':<9} {'Test R²':<8} {'Test MSE':<10} {'Test MAE':<9} {'n_est':<6} {'Features':<50}")
    print("-" * 140)
    
    for idx, (_, row) in enumerate(top_10_val.iterrows(), 1):
        features = parse_features(row['Feature Combination'])
        features_str = ', '.join(features)
        
        # Truncate features if too long
        if len(features_str) > 47:
            features_str = features_str[:44] + '...'
        
        print(f"{idx:<4} {row['Validation R2']:<8.3f} {row['Validation MSE']:<10.1f} {row['Validation MAE']:<9.3f} {row['Test R2']:<8.3f} {row['Test MSE']:<10.1f} {row['Test MAE']:<9.3f} {row['n_estimators']:<6} {features_str:<50}")
    
    print("-" * 140)
    
    # Additional details
    print(f"\nValidation Key Insights:")
    print(f"• Best Validation MAE: {top_10_val['Validation MAE'].min():.3f}°C (Rank 1)")
    print(f"• Best Validation R²: {top_10_val['Validation R2'].max():.3f}")
    print(f"• n_estimators range: {top_10_val['n_estimators'].min()} - {top_10_val['n_estimators'].max()}")
    print(f"• Most common feature count: {top_10_val['Number of Features'].mode().iloc[0]} features")
    
    print(f"\nTest Performance for Same Models:")
    print(f"• Best Test MAE: {top_10_val['Test MAE'].min():.3f}°C")
    print(f"• Best Test R²: {top_10_val['Test R2'].max():.3f}")
    print(f"• Average Test MAE: {top_10_val['Test MAE'].mean():.3f}°C")
    print(f"• Average Test R²: {top_10_val['Test R2'].mean():.3f}")
    
    # Generalization analysis
    top_10_val['Generalization_Gap'] = top_10_val['Test MAE'] - top_10_val['Validation MAE']
    print(f"\nGeneralization Analysis:")
    print(f"• Average Gap (Test - Val): {top_10_val['Generalization_Gap'].mean():+.3f}°C")
    print(f"• Gap Range: {top_10_val['Generalization_Gap'].min():+.3f} to {top_10_val['Generalization_Gap'].max():+.3f}°C")
    print(f"• Models with positive gap (worse on test): {(top_10_val['Generalization_Gap'] > 0).sum()}/10")
    
    # Save validation table with test data (full precision)
    clean_table = []
    for idx, (_, row) in enumerate(top_10_val.iterrows(), 1):
        features = parse_features(row['Feature Combination'])
        clean_table.append({
            'Rank': idx,
            'Validation_R2': row['Validation R2'],
            'Validation_MSE': row['Validation MSE'],
            'Validation_MAE': row['Validation MAE'],
            'Test_R2': row['Test R2'],
            'Test_MSE': row['Test MSE'],
            'Test_MAE': row['Test MAE'],
            'Generalization_Gap': row['Test MAE'] - row['Validation MAE'],
            'n_estimators': int(row['n_estimators']),
            'Features': ', '.join(features),
            'Number_of_Features': int(row['Number of Features'])
        })
    
    clean_df = pd.DataFrame(clean_table)
    clean_df.to_csv('top_10_models_validation_with_test.csv', index=False)
    print(f"\nEnhanced validation table saved to: top_10_models_validation_with_test.csv")
    
    return clean_df

if __name__ == "__main__":
    table = main()
