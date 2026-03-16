# -*- coding: utf-8 -*-
"""
Analyze results to find top 10 models based on lowest MAE
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
    # - SVR, Lasso, ElasticNet: 0 (not tree-based)
    return n_estimators * 2  # GB + RF each have n_estimators

def main():
    """Load results and create top 10 table."""
    print("Loading results...")
    df = pd.read_csv('fixed_split_results.csv')
    
    print(f"Total results: {len(df)}")
    
    # Sort by Test MAE (lowest first) and get top 10
    top_10 = df.nsmallest(10, 'Test MAE').copy()
    
    print("\n" + "="*80)
    print("TOP 10 STACKED ENSEMBLE MODELS (Lowest Test MAE)")
    print("="*80)
    
    # Create table
    table_data = []
    
    for idx, (_, row) in enumerate(top_10.iterrows(), 1):
        features = parse_features(row['Feature Combination'])
        tree_estimators = count_tree_estimators(row['n_estimators'])
        
        table_data.append({
            'Rank': idx,
            'Test R²': f"{row['Test R2']:.3f}",
            'Test MSE': f"{row['Test MSE']:.1f}",
            'Test MAE': f"{row['Test MAE']:.3f}",
            'Tree Base Estimators': tree_estimators,
            'Total n_estimators': row['n_estimators'],
            'Features': ', '.join(features),
            'Num Features': row['Number of Features']
        })
    
    # Convert to DataFrame for nice display
    table_df = pd.DataFrame(table_data)
    
    # Display table
    print("\nTop 10 Models Table:")
    print("-" * 120)
    print(f"{'Rank':<4} {'Test R²':<8} {'Test MSE':<10} {'Test MAE':<9} {'Tree Est':<10} {'Total n':<8} {'Num Feat':<8} {'Features':<50}")
    print("-" * 120)
    
    for _, row in table_df.iterrows():
        features_truncated = row['Features'][:47] + '...' if len(row['Features']) > 50 else row['Features']
        print(f"{int(row['Rank']):<4} {row['Test R²']:<8} {row['Test MSE']:<10} {row['Test MAE']:<9} {row['Tree Base Estimators']:<10} {row['Total n_estimators']:<8} {row['Num Features']:<8} {features_truncated:<50}")
    
    print("-" * 120)
    
    # Detailed analysis
    print("\nDetailed Analysis:")
    print("="*50)
    
    print(f"Best Test MAE: {top_10['Test MAE'].min():.3f}°C")
    print(f"Worst Test MAE (in top 10): {top_10['Test MAE'].max():.3f}°C")
    print(f"Average Test MAE (top 10): {top_10['Test MAE'].mean():.3f}°C")
    
    print(f"\nBest Test R²: {top_10['Test R2'].max():.3f}")
    print(f"Average Test R² (top 10): {top_10['Test R2'].mean():.3f}")
    
    print(f"\nMost common n_estimators in top 10:")
    n_est_counts = top_10['n_estimators'].value_counts()
    for n_est, count in n_est_counts.items():
        print(f"  {n_est}: {count} models")
    
    print(f"\nMost common feature count in top 10:")
    feat_counts = top_10['Number of Features'].value_counts()
    for num_feat, count in feat_counts.items():
        print(f"  {num_feat} features: {count} models")
    
    # Save detailed table
    table_df.to_csv('top_10_models_detailed.csv', index=False)
    print(f"\nDetailed table saved to: top_10_models_detailed.csv")
    
    return table_df

if __name__ == "__main__":
    table = main()
