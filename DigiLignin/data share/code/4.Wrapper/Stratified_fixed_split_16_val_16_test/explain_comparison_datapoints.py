# -*- coding: utf-8 -*-
"""
Show exactly what validation datapoints are used in compare_with_oof.py
"""

import pandas as pd

def main():
    """Demonstrate the validation datapoints in comparison code."""
    print("VALIDATION DATAPOINTS IN compare_with_oof.py")
    print("="*60)
    
    # Load results
    df_fixed = pd.read_csv('fixed_split_results.csv')
    
    print("\n1. PLOT 1: Validation MAE Comparison by N_Estimators")
    print("   Code: fixed_val_mae = df_fixed.groupby('n_estimators')['Validation MAE'].mean()")
    print("   This IS calculating AVERAGES by n_estimators!")
    
    print("\n   What this means:")
    for n_est in [10, 100, 500]:
        models_with_n_est = df_fixed[df_fixed['n_estimators'] == n_est]
        avg_mae = models_with_n_est['Validation MAE'].mean()
        count = len(models_with_n_est)
        print(f"   - n_estimators={n_est}: Average of {count} models = {avg_mae:.3f}°C")
    
    print("\n2. PLOT 2: Validation R² Comparison by N_Estimators") 
    print("   Code: fixed_val_r2 = df_fixed.groupby('n_estimators')['Validation R2'].mean()")
    print("   This IS calculating AVERAGES by n_estimators!")
    
    print("\n3. PLOT 3: Best Validation Performance Comparison")
    print("   Code: fixed_best_val = df_fixed.groupby('n_estimators')['Validation MAE'].min()")
    print("   This shows the BEST (minimum) MAE for each n_estimators")
    
    print("\n4. PLOT 4: Validation Performance Distribution Comparison")
    print("   Code: fixed_val_mae_dist = df_fixed['Validation MAE'].values")
    print("   This shows ALL individual validation MAE values (no averaging)")
    
    print("\n5. SUMMARY:")
    print("   - Plots 1 & 2: Use AVERAGES by n_estimators")
    print("   - Plot 3: Uses BEST values by n_estimators") 
    print("   - Plot 4: Uses ALL individual values (no averaging)")
    print("   - The table output shows individual model values (no averaging)")
    
    print("\n6. KEY DISTINCTION:")
    print("   - TABLES: Individual model performances")
    print("   - PLOTS 1&2: Averaged performances for trend visualization")
    print("   - PLOTS 3&4: Best performances and distributions")

if __name__ == "__main__":
    main()
