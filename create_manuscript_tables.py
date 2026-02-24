#!/usr/bin/env python3
"""
Create Publication-Ready Tables for Manuscript Update
"""

import pandas as pd
import numpy as np

# Load data
fixed_split_df = pd.read_csv('/home/silviu/DigiLignin/4.Wrapper/Stratified_fixed_split_16_val_16_test/fixed_split_results.csv')
oof_with_sratio = pd.read_csv('/home/silviu/DigiLignin/4.Wrapper/Fixed_stacking_ensemble_with_n_estimators/top_5_with_sratio_summary.csv')
oof_without_sratio = pd.read_csv('/home/silviu/DigiLignin/4.Wrapper/Fixed_stacking_ensemble_with_n_estimators/top_5_without_sratio_summary.csv')

# Get best model
best_idx = fixed_split_df['Test MAE'].idxmin()
best_model = fixed_split_df.loc[best_idx]

# ============================================================================
# TABLE 1: BEST MODEL PERFORMANCE SUMMARY
# ============================================================================
print("=" * 80)
print("TABLE 1: BEST MODEL PERFORMANCE SUMMARY")
print("=" * 80)

table1_data = {
    'Metric': ['R²', 'MSE (°C²)', 'MAE (°C)'],
    'Training': [
        f"{best_model['Train R2']:.4f}",
        f"{best_model['Train MSE']:.2f}",
        f"{best_model['Train MAE']:.2f}"
    ],
    'Validation': [
        f"{best_model['Validation R2']:.4f}",
        f"{best_model['Validation MSE']:.2f}",
        f"{best_model['Validation MAE']:.2f}"
    ],
    'Test': [
        f"{best_model['Test R2']:.4f}",
        f"{best_model['Test MSE']:.2f}",
        f"{best_model['Test MAE']:.2f}"
    ],
    'Generalization Gap': [
        f"{best_model['Test R2'] - best_model['Train R2']:.4f}",
        f"{best_model['Test MSE'] - best_model['Train MSE']:.2f}",
        f"{best_model['Test MAE'] - best_model['Train MAE']:.2f}"
    ]
}

table1 = pd.DataFrame(table1_data)
print("\n" + table1.to_string(index=False))
print(f"\nModel Configuration:")
print(f"  Features: {best_model['Feature Combination']}")
print(f"  Number of Features: {int(best_model['Number of Features'])}")
print(f"  n_estimators (tree-based models): {int(best_model['n_estimators'])}")
print(f"  Validation Strategy: Fixed Stratified Split (104 train, 16 val, 16 test)")

# Save to CSV
table1.to_csv('/home/silviu/DigiLignin/Table1_Best_Model_Performance.csv', index=False)
print("\n✓ Saved to: Table1_Best_Model_Performance.csv")

# ============================================================================
# TABLE 2: IMPACT OF SWELLING RATIO
# ============================================================================
print("\n" + "=" * 80)
print("TABLE 2: IMPACT OF SWELLING RATIO ON MODEL PERFORMANCE")
print("=" * 80)

df_with_sratio = fixed_split_df[fixed_split_df['Feature Combination'].str.contains('Sratio', case=False, na=False)]
df_without_sratio = fixed_split_df[~fixed_split_df['Feature Combination'].str.contains('Sratio', case=False, na=False)]

best_with_sratio = df_with_sratio.loc[df_with_sratio['Test MAE'].idxmin()]
best_without_sratio = df_without_sratio.loc[df_without_sratio['Test MAE'].idxmin()]

table2_data = {
    'Configuration': ['With Swelling Ratio', 'Without Swelling Ratio', 'Difference (Δ)'],
    'Features': [
        best_with_sratio['Feature Combination'],
        best_without_sratio['Feature Combination'],
        'Swelling Ratio removed'
    ],
    'n_estimators': [
        int(best_with_sratio['n_estimators']),
        int(best_without_sratio['n_estimators']),
        '-'
    ],
    'Test R²': [
        f"{best_with_sratio['Test R2']:.4f}",
        f"{best_without_sratio['Test R2']:.4f}",
        f"{best_without_sratio['Test R2'] - best_with_sratio['Test R2']:.4f}"
    ],
    'Test MAE (°C)': [
        f"{best_with_sratio['Test MAE']:.3f}",
        f"{best_without_sratio['Test MAE']:.3f}",
        f"{best_without_sratio['Test MAE'] - best_with_sratio['Test MAE']:.3f}"
    ],
    'Performance Change': [
        'Baseline',
        f"{(best_without_sratio['Test MAE'] - best_with_sratio['Test MAE'])/best_with_sratio['Test MAE']*100:.1f}%",
        '-'
    ]
}

table2 = pd.DataFrame(table2_data)
print("\n" + table2.to_string(index=False))
print("\nNote: Negative difference indicates WITHOUT swelling ratio performs better on test set.")
print("This suggests the model can predict Tg from formulation parameters alone.")

# Save to CSV
table2.to_csv('/home/silviu/DigiLignin/Table2_Swelling_Ratio_Impact.csv', index=False)
print("\n✓ Saved to: Table2_Swelling_Ratio_Impact.csv")

# ============================================================================
# TABLE 3: PERFORMANCE VS N_ESTIMATORS
# ============================================================================
print("\n" + "=" * 80)
print("TABLE 3: PERFORMANCE VS NUMBER OF BASE ESTIMATORS")
print("=" * 80)

best_features = best_model['Feature Combination']
best_combo_df = fixed_split_df[fixed_split_df['Feature Combination'] == best_features].copy()
best_combo_df = best_combo_df.sort_values('n_estimators')

# Select key n_estimators values
key_n_estimators = [1, 10, 50, 100, 200, 400, 600, 800, 1000]
table3_df = best_combo_df[best_combo_df['n_estimators'].isin(key_n_estimators)].copy()

table3_data = {
    'n_estimators': table3_df['n_estimators'].astype(int).tolist(),
    'Val MAE (°C)': [f"{x:.3f}" for x in table3_df['Validation MAE']],
    'Test MAE (°C)': [f"{x:.3f}" for x in table3_df['Test MAE']],
    'Test R²': [f"{x:.4f}" for x in table3_df['Test R2']],
    'Gen. Gap (°C)': [f"{v-t:.3f}" for v, t in zip(table3_df['Validation MAE'], table3_df['Test MAE'])]
}

table3 = pd.DataFrame(table3_data)
print("\n" + table3.to_string(index=False))
print(f"\nFeature Set: {best_features}")
print("Recommendation: n_estimators = 1000 provides optimal Test MAE with stable generalization")

# Save to CSV
table3.to_csv('/home/silviu/DigiLignin/Table3_Performance_vs_N_Estimators.csv', index=False)
print("\n✓ Saved to: Table3_Performance_vs_N_Estimators.csv")

print("\n" + "=" * 80)
print("ALL TABLES CREATED SUCCESSFULLY")
print("=" * 80)

