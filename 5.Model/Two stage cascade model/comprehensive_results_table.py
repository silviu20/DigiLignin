import pandas as pd
import numpy as np

# Data from the results files
data = {
    'Model': [
        'Original Stacked Ensemble',
        'Fixed Stacked Ensemble', 
        'Baseline: Formulation Only',
        'Stage 1: Swelling Prediction',
        'Stage 2: Tg Prediction (Cascade)'
    ],
    'MAE Validation (°C)': [
        11.31,  # Original Stacking
        16.38,  # Fixed Stacking
        17.07,  # Baseline
        24.83,  # Stage 1 (swelling - different units)
        16.67   # Stage 2 (Cascade)
    ],
    'MAE Train (°C)': [
        11.33,  # Original Stacking
        16.00,  # Fixed Stacking
        16.93,  # Baseline
        23.10,  # Stage 1 (swelling - different units)
        16.56   # Stage 2 (Cascade)
    ],
    'R² Validation': [
        0.687,  # Original Stacking
        0.295,  # Fixed Stacking
        0.286,  # Baseline
        0.669,  # Stage 1
        0.296   # Stage 2
    ],
    'R² Train': [
        0.683,  # Original Stacking
        0.392,  # Fixed Stacking
        0.341,  # Baseline
        0.742,  # Stage 1
        0.373   # Stage 2
    ],
    'Generalizability (Val MAE - Train MAE)': [
        -0.02,  # Original Stacking (slight negative = good)
        0.38,   # Fixed Stacking
        0.13,   # Baseline
        1.73,   # Stage 1
        0.11    # Stage 2
    ],
    'Key Features': [
        'All formulation + swelling',
        'Reduced formulation + swelling',
        'Formulation only',
        'Formulation only',
        'Formulation + predicted swelling'
    ],
    'Data Leakage Issue': [
        'Yes (original)',
        'No (fixed)',
        'No',
        'No',
        'No'
    ],
    'Practical Use': [
        'Limited (requires synthesis)',
        'Limited (requires synthesis)',
        'High (fully predictive)',
        'High (fully predictive)',
        'High (fully predictive)'
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# Format the table
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 50)

# Add ranking based on MAE (lower is better)
df['MAE Rank'] = df['MAE Validation (°C)'].rank(method='min', ascending=True).astype(int)

# Add improvement percentages
baseline_mae = df.loc[df['Model'] == 'Baseline: Formulation Only', 'MAE Validation (°C)'].iloc[0]
df['Improvement vs Baseline (%)'] = ((baseline_mae - df['MAE Validation (°C)']) / baseline_mae * 100).round(2)

# Reorder columns for better presentation
columns_order = [
    'Model', 'MAE Rank', 'MAE Validation (°C)', 'MAE Train (°C)', 
    'Improvement vs Baseline (%)', 'R² Validation', 'R² Train',
    'Generalizability (Val MAE - Train MAE)', 'Key Features', 
    'Data Leakage Issue', 'Practical Use'
]

df = df[columns_order]

# Display the table
print("="*120)
print("COMPREHENSIVE MODEL COMPARISON TABLE")
print("="*120)
print()

# Print with formatting
for idx, row in df.iterrows():
    print(f"{'Model:':<20} {row['Model']}")
    print(f"{'MAE Rank:':<20} {row['MAE Rank']}")
    print(f"{'MAE Validation:':<20} {row['MAE Validation (°C)']:.2f}°C")
    print(f"{'MAE Train:':<20} {row['MAE Train (°C)']:.2f}°C")
    print(f"{'Improvement vs Baseline:':<20} {row['Improvement vs Baseline (%)']:+.2f}%")
    print(f"{'R² Validation:':<20} {row['R² Validation']:.3f}")
    print(f"{'R² Train:':<20} {row['R² Train']:.3f}")
    print(f"{'Generalizability:':<20} {row['Generalizability (Val MAE - Train MAE)']:.2f}")
    print(f"{'Key Features:':<20} {row['Key Features']}")
    print(f"{'Data Leakage Issue:':<20} {row['Data Leakage Issue']}")
    print(f"{'Practical Use:':<20} {row['Practical Use']}")
    print("-"*80)

print()
print("KEY INSIGHTS:")
print("-"*80)
print("1. Original Stacked Ensemble shows best MAE (11.31°C) but has data leakage")
print("2. Fixed Stacked Ensemble addresses leakage but performance drops to 16.38°C")
print("3. Baseline (formulation only) achieves 17.07°C with full practicality")
print("4. Cascade Model improves baseline to 16.67°C while remaining fully predictive")
print("5. Stage 1 (swelling prediction) has different units (%) and target")
print()
print("RECOMMENDATIONS:")
print("-"*80)
print("• For research: Use Fixed Stacked Ensemble (most accurate without leakage)")
print("• For practical application: Use Cascade Model (best balance of accuracy and practicality)")
print("• For quick predictions: Use Baseline (simplest, reasonable accuracy)")

# Save to CSV
df.to_csv('Comprehensive_Model_Comparison.csv', index=False)
print(f"\nTable saved to: Comprehensive_Model_Comparison.csv")
