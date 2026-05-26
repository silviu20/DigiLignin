# -*- coding: utf-8 -*-
"""
Demonstrate that validation metrics are actual values, not averages
"""

import pandas as pd

def main():
    """Show the difference between actual values and averages."""
    print("VALIDATION METRICS EXPLANATION")
    print("="*60)
    
    # Load results
    df = pd.read_csv('fixed_split_results.csv')
    
    print("\n1. WHAT THE TABLE SHOWS:")
    print("   Each row in top_10_models_validation.csv shows:")
    print("   - Validation MAE: ACTUAL performance on 16 validation samples")
    print("   - Validation R²: ACTUAL performance on 16 validation samples") 
    print("   - Validation MSE: ACTUAL performance on 16 validation samples")
    print("   - These are NOT averages across multiple runs")
    
    print("\n2. EXAMPLE FROM RAW DATA:")
    # Find the best validation model
    best_val = df.nsmallest(1, 'Validation MAE').iloc[0]
    print(f"   Best validation model (n_estimators={best_val['n_estimators']}):")
    print(f"   - Validation MAE: {best_val['Validation MAE']:.3f}°C")
    print(f"   - Validation R²: {best_val['Validation R2']:.3f}")
    print(f"   - Validation MSE: {best_val['Validation MSE']:.1f}")
    print(f"   - This is performance on {int(best_val['Val Size'])} validation samples")
    
    print("\n3. WHAT WOULD BE AVERAGES:")
    print("   If we wanted averages, we would calculate:")
    avg_val_mae = df['Validation MAE'].mean()
    avg_val_r2 = df['Validation R2'].mean()
    avg_val_mse = df['Validation MSE'].mean()
    print(f"   - Average Validation MAE across ALL models: {avg_val_mae:.3f}°C")
    print(f"   - Average Validation R² across ALL models: {avg_val_r2:.3f}")
    print(f"   - Average Validation MSE across ALL models: {avg_val_mse:.1f}")
    
    print("\n4. VALIDATION SETUP:")
    print("   - Fixed split: 16 validation samples")
    print("   - Each model is evaluated ONCE on these 16 samples")
    print("   - No cross-validation, no multiple runs")
    print("   - Each metric is a single calculation, not an average")
    
    print("\n5. KEY POINT:")
    print("   The validation metrics in the table are the ACTUAL performance")
    print("   of each specific model configuration on the validation set.")
    print("   They are not averaged across multiple runs or folds.")

if __name__ == "__main__":
    main()
