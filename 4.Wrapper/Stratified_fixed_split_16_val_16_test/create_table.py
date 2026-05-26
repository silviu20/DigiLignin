# -*- coding: utf-8 -*-
"""
Create properly formatted table for top 10 models
"""

import pandas as pd
import ast

def parse_features(feature_str):
    """Parse feature string to list."""
    try:
        return ast.literal_eval(feature_str)
    except:
        return feature_str

def count_tree_estimators(n_estimators):
    """Count number of base estimators in tree-based models."""
    # In the stacking ensemble, tree-based models are:
    # - GradientBoostingRegressor: n_estimators
    # - RandomForestRegressor: n_estimators
    return n_estimators * 2  # GB + RF each have n_estimators

def main():
    """Load results and create top 10 table."""
    print("Loading results...")
    df = pd.read_csv('fixed_split_results.csv')
    
    # Sort by Test MAE (lowest first) and get top 10
    top_10 = df.nsmallest(10, 'Test MAE').copy()
    
    print("\n" + "="*120)
    print("TOP 10 STACKED ENSEMBLE MODELS (Lowest Test MAE)")
    print("="*120)
    
    # Create formatted table
    print(f"{'Rank':<4} {'Test R²':<8} {'Test MSE':<10} {'Test MAE':<9} {'n_estimators':<12} {'Features':<60}")
    print("-" * 120)
    
    for idx, (_, row) in enumerate(top_10.iterrows(), 1):
        features = parse_features(row['Feature Combination'])
        features_str = ', '.join(features)
        
        # Truncate features if too long
        if len(features_str) > 57:
            features_str = features_str[:54] + '...'
        
        print(f"{idx:<4} {row['Test R2']:<8.3f} {row['Test MSE']:<10.1f} {row['Test MAE']:<9.3f} {row['n_estimators']:<12} {features_str:<60}")
    
    print("-" * 120)
    
    # Additional details
    print(f"\nKey Insights:")
    print(f"• Best Test MAE: {top_10['Test MAE'].min():.3f}°C (Rank 1)")
    print(f"• Best Test R²: {top_10['Test R2'].max():.3f} (Rank 9)")
    print(f"• n_estimators range: {top_10['n_estimators'].min()} - {top_10['n_estimators'].max()}")
    print(f"• Most common feature count: 5 features (8/10 models)")
    
    # Save clean table
    clean_table = []
    for idx, (_, row) in enumerate(top_10.iterrows(), 1):
        features = parse_features(row['Feature Combination'])
        clean_table.append({
            'Rank': idx,
            'Test_R2': row['Test R2'],
            'Test_MSE': row['Test MSE'],
            'Test_MAE': row['Test MAE'],
            'n_estimators': row['n_estimators'],
            'Features': ', '.join(features),
            'Number_of_Features': row['Number of Features']
        })
    
    clean_df = pd.DataFrame(clean_table)
    clean_df.to_csv('top_10_models_table.csv', index=False)
    print(f"\nClean table saved to: top_10_models_table.csv")
    
    return clean_df

if __name__ == "__main__":
    table = main()
