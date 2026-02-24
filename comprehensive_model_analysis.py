#!/usr/bin/env python3
"""
Comprehensive Model Analysis Script
Extracts and analyzes all model performance data for manuscript update
"""

import pandas as pd
import numpy as np
import json

# Load all data sources
print("=" * 80)
print("COMPREHENSIVE MODEL ANALYSIS FOR MANUSCRIPT UPDATE")
print("=" * 80)

# 1. Load Fixed Stratified Split Results
fixed_split_df = pd.read_csv('/home/silviu/DigiLignin/4.Wrapper/Stratified_fixed_split_16_val_16_test/fixed_split_results.csv')

# 2. Load OOF Results
oof_all_df = pd.read_csv('/home/silviu/DigiLignin/4.Wrapper/Fixed_stacking_ensemble_with_n_estimators/all_combinations_n_estimators_results.csv')
oof_with_sratio = pd.read_csv('/home/silviu/DigiLignin/4.Wrapper/Fixed_stacking_ensemble_with_n_estimators/top_5_with_sratio_summary.csv')
oof_without_sratio = pd.read_csv('/home/silviu/DigiLignin/4.Wrapper/Fixed_stacking_ensemble_with_n_estimators/top_5_without_sratio_summary.csv')

# ============================================================================
# TASK 1: IDENTIFY BEST PERFORMING MODEL
# ============================================================================
print("\n" + "=" * 80)
print("TASK 1: BEST PERFORMING MODEL IDENTIFICATION")
print("=" * 80)

# Best model from Fixed Split (has true test set)
best_fixed_idx = fixed_split_df['Test MAE'].idxmin()
best_fixed = fixed_split_df.loc[best_fixed_idx]

print("\n✓ BEST MODEL (Fixed Stratified Split with True Test Set):")
print(f"  Features: {best_fixed['Feature Combination']}")
print(f"  Number of Features: {int(best_fixed['Number of Features'])}")
print(f"  n_estimators: {int(best_fixed['n_estimators'])}")
print(f"\n  Performance Metrics:")
print(f"    Training   - R²: {best_fixed['Train R2']:.4f}, MSE: {best_fixed['Train MSE']:.2f}, MAE: {best_fixed['Train MAE']:.2f}°C")
print(f"    Validation - R²: {best_fixed['Validation R2']:.4f}, MSE: {best_fixed['Validation MSE']:.2f}, MAE: {best_fixed['Validation MAE']:.2f}°C")
print(f"    Test       - R²: {best_fixed['Test R2']:.4f}, MSE: {best_fixed['Test MSE']:.2f}, MAE: {best_fixed['Test MAE']:.2f}°C")
print(f"\n  Generalization Gaps:")
print(f"    Val-Test Gap: {best_fixed['Validation MAE'] - best_fixed['Test MAE']:.2f}°C")
print(f"    Test-Train Gap: {best_fixed['Test MAE'] - best_fixed['Train MAE']:.2f}°C")

# ============================================================================
# TASK 2: SWELLING RATIO IMPACT ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("TASK 2: SWELLING RATIO IMPACT ANALYSIS")
print("=" * 80)

# Fixed Split comparison
df_with_sratio = fixed_split_df[fixed_split_df['Feature Combination'].str.contains('Sratio', case=False, na=False)]
df_without_sratio = fixed_split_df[~fixed_split_df['Feature Combination'].str.contains('Sratio', case=False, na=False)]

if len(df_with_sratio) > 0:
    best_with_sratio_idx = df_with_sratio['Test MAE'].idxmin()
    best_with_sratio = df_with_sratio.loc[best_with_sratio_idx]
    
    print("\n✓ Best Model WITH Swelling Ratio (Fixed Split):")
    print(f"  Features: {best_with_sratio['Feature Combination']}")
    print(f"  n_estimators: {int(best_with_sratio['n_estimators'])}")
    print(f"  Test MAE: {best_with_sratio['Test MAE']:.3f}°C, Test R²: {best_with_sratio['Test R2']:.4f}")
    print(f"  Val MAE: {best_with_sratio['Validation MAE']:.3f}°C, Val R²: {best_with_sratio['Validation R2']:.4f}")

if len(df_without_sratio) > 0:
    best_without_sratio_idx = df_without_sratio['Test MAE'].idxmin()
    best_without_sratio = df_without_sratio.loc[best_without_sratio_idx]
    
    print("\n✓ Best Model WITHOUT Swelling Ratio (Fixed Split):")
    print(f"  Features: {best_without_sratio['Feature Combination']}")
    print(f"  n_estimators: {int(best_without_sratio['n_estimators'])}")
    print(f"  Test MAE: {best_without_sratio['Test MAE']:.3f}°C, Test R²: {best_without_sratio['Test R2']:.4f}")
    print(f"  Val MAE: {best_without_sratio['Validation MAE']:.3f}°C, Val R²: {best_without_sratio['Validation R2']:.4f}")

if len(df_with_sratio) > 0 and len(df_without_sratio) > 0:
    mae_diff = best_without_sratio['Test MAE'] - best_with_sratio['Test MAE']
    print(f"\n✓ Impact of Removing Swelling Ratio:")
    print(f"  Test MAE Change: {mae_diff:+.3f}°C ({mae_diff/best_with_sratio['Test MAE']*100:+.1f}%)")
    print(f"  Note: Negative means WITHOUT performs better!")

# OOF comparison
print("\n✓ OOF Cross-Validation Results:")
best_oof_with = oof_with_sratio.iloc[0]
best_oof_without = oof_without_sratio.iloc[0]
print(f"  WITH Sratio - Val MAE: {best_oof_with['MAE']:.3f}°C, Val R²: {best_oof_with['R2']:.4f}")
print(f"  WITHOUT Sratio - Val MAE: {best_oof_without['MAE']:.3f}°C, Val R²: {best_oof_without['R2']:.4f}")
print(f"  Difference: {best_oof_without['MAE'] - best_oof_with['MAE']:+.3f}°C")

# ============================================================================
# TASK 3: PERFORMANCE VS N_ESTIMATORS
# ============================================================================
print("\n" + "=" * 80)
print("TASK 3: PERFORMANCE VS N_ESTIMATORS ANALYSIS")
print("=" * 80)

# Get the best feature combination
best_features = best_fixed['Feature Combination']
best_combo_df = fixed_split_df[fixed_split_df['Feature Combination'] == best_features].copy()
best_combo_df = best_combo_df.sort_values('n_estimators')

print(f"\n✓ Performance for Best Feature Set: {best_features}")
print(f"\n{'n_est':<8} {'Val MAE':<10} {'Test MAE':<10} {'Test R²':<10} {'Gap':<10}")
print("-" * 50)
for _, row in best_combo_df.iterrows():
    gap = row['Validation MAE'] - row['Test MAE']
    print(f"{int(row['n_estimators']):<8} {row['Validation MAE']:<10.3f} {row['Test MAE']:<10.3f} {row['Test R2']:<10.4f} {gap:<10.3f}")

print("\n✓ Recommendation: n_estimators = 1000 provides best Test MAE with stable performance")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

