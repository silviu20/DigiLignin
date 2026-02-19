import matplotlib.pyplot as plt
import numpy as np

# Actual results from running the original method
original_r2 = 0.998
original_mae = 0.835

# Results from nested cross-validation (corrected method)
nested_r2_mean = 0.280
nested_r2_ci_lower = 0.020
nested_r2_ci_upper = 0.539
nested_mae_mean = 777.379
nested_mae_ci_lower = 649.543
nested_mae_ci_upper = 905.215

# Create comprehensive comparison visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Original vs Corrected Stacking Implementation\n(Actual DigiLignin Dataset Results)', 
             fontsize=16, fontweight='bold')

# 1. R² Comparison
ax1 = axes[0, 0]
methods = ['Original\n(Biased)', 'Corrected\n(Unbiased)']
r2_values = [original_r2, nested_r2_mean]
colors = ['red', 'green']
errors = [0, (nested_r2_mean - nested_r2_ci_lower, nested_r2_ci_upper - nested_r2_mean)]

bars = ax1.bar(methods, r2_values, color=colors, alpha=0.7)
ax1.set_ylabel('R² Score', fontsize=12)
ax1.set_title('R² Score Comparison', fontsize=14, fontweight='bold')
ax1.set_ylim(0, 1)

# Add value labels and confidence intervals
for i, (bar, value) in enumerate(zip(bars, r2_values)):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Add confidence interval for corrected method
    if i == 1:
        ax1.errorbar(bar.get_x() + bar.get_width()/2., height, 
                    yerr=[[nested_r2_mean - nested_r2_ci_lower], [nested_r2_ci_upper - nested_r2_mean]], 
                    fmt='none', ecolor='black', capsize=5, capthick=2)

ax1.annotate('⚠️ SEVERE DATA LEAKAGE\nMassively Inflated Performance', 
            xy=(0, original_r2), xytext=(0, original_r2 - 0.3),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            ha='center', fontsize=10, color='red')

# 2. MAE Comparison
ax2 = axes[0, 1]
mae_values = [original_mae, nested_mae_mean]

bars = ax2.bar(methods, mae_values, color=colors, alpha=0.7)
ax2.set_ylabel('MAE (°C)', fontsize=12)
ax2.set_title('MAE Comparison', fontsize=14, fontweight='bold')

# Add value labels and confidence intervals
for i, (bar, value) in enumerate(zip(bars, mae_values)):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + max(mae_values)*0.01,
            f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # Add confidence interval for corrected method
    if i == 1:
        ax2.errorbar(bar.get_x() + bar.get_width()/2., height, 
                    yerr=[[nested_mae_mean - nested_mae_ci_lower], [nested_mae_ci_upper - nested_mae_mean]], 
                    fmt='none', ecolor='black', capsize=5, capthick=2)

ax2.annotate('⚠️ SEVERE UNDERESTIMATION\nError 1000x too low!', 
            xy=(0, original_mae), xytext=(0, original_mae + max(mae_values)*0.3),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            ha='center', fontsize=10, color='red')

# 3. Performance Difference Summary
ax3 = axes[1, 0]
ax3.axis('off')

# Calculate differences
r2_inflation = original_r2 - nested_r2_mean
mae_underestimation_factor = nested_mae_mean / original_mae

summary_text = f"""ACTUAL PERFORMANCE COMPARISON

Original Method (with Data Leakage):
• R²: {original_r2:.3f}
• MAE: {original_mae:.3f}°C
• ❌ SEVERE data leakage in meta-features
• ❌ Meta-model trained on predictions from same data
• ❌ In-sample predictions reported as validation

Corrected Method (Unbiased):
• R²: {nested_r2_mean:.3f} [{nested_r2_ci_lower:.3f}, {nested_r2_ci_upper:.3f}]
• MAE: {nested_mae_mean:.1f}°C [{nested_mae_ci_lower:.1f}, {nested_mae_ci_upper:.1f}]
• ✅ Proper OOF predictions for meta-features
• ✅ Nested cross-validation
• ✅ True generalization performance

IMPACT OF DATA LEAKAGE:
• R² Inflation: +{r2_inflation:.3f} ({r2_inflation/nested_r2_mean*100:.0f}% too high!)
• MAE Underestimation: {mae_underestimation_factor:.0f}x too low!
• Completely misleading performance claims

CRITICAL FINDINGS:
• Original method reports near-perfect R² (0.998)
• True performance is much lower (R² ≈ 0.28)
• Original MAE is 1000x underestimated
• This is a CLASSIC data leakage example"""

ax3.text(0.05, 0.95, summary_text, transform=ax3.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# 4. Methodology Comparison
ax4 = axes[1, 1]
ax4.axis('off')

methodology_text = f"""METHODOLOGY COMPARISON

ORIGINAL METHOD (FLAWED):
1. Train base models on 100% of data
2. Generate meta-features on SAME data
3. Train meta-model on leaked predictions
4. Report in-sample performance as validation
5. ❌ DATA LEAKAGE at every step

CORRECTED METHOD (PROPER):
1. Nested cross-validation
   • Outer CV: Performance estimation
   • Inner CV: Hyperparameter tuning
2. Out-of-fold (OOF) predictions only
3. Meta-model trained on OOF predictions
4. Held-out test set for final evaluation
5. ✅ No data leakage anywhere

VALIDATION APPROACH:
• Original: Biased in-sample validation
• Corrected: Unbiased nested CV + test set

RECOMMENDATION:
✅ Use corrected implementation
✅ Report unbiased metrics only
✅ Update all manuscript figures
✅ Document methodological fix"""

ax4.text(0.05, 0.95, methodology_text, transform=ax4.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

plt.tight_layout()

# Save the comparison plot
for ext in ['tiff', 'pdf', 'eps', 'svg', 'jpg']:
    plt.savefig(f'digilignin_actual_comparison.{ext}', dpi=600, bbox_inches='tight')

plt.show()

print("=== ACTUAL COMPARISON RESULTS ===")
print(f"Original Method R²: {original_r2:.3f} (SEVERELY INFLATED)")
print(f"Corrected Method R²: {nested_r2_mean:.3f} (TRUE PERFORMANCE)")
print(f"R² Inflation: +{r2_inflation:.3f} ({r2_inflation/nested_r2_mean*100:.0f}%)")
print(f"")
print(f"Original Method MAE: {original_mae:.3f}°C (SEVERELY UNDERESTIMATED)")
print(f"Corrected Method MAE: {nested_mae_mean:.1f}°C (TRUE PERFORMANCE)")
print(f"MAE Underestimation Factor: {mae_underestimation_factor:.0f}x")
print(f"")
print("CONCLUSION: The original method has MASSIVE data leakage that makes")
print("the model appear nearly perfect when it actually has modest performance.")
