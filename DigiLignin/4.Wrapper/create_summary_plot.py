import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib as mpl

# Set style parameters for a more scientific look
plt.style.use('default')
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 14
mpl.rcParams['axes.linewidth'] = 1.5

# Read the results data
df = pd.read_csv('Fixed_Stacking_Ensemble/fixed_stacking_results_all_combinations.csv')

# Add number of features column
df['Number of Features'] = df['Feature Combination'].apply(lambda x: len(eval(x)))

# Create summary plot
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Fixed Stacking Ensemble - Feature Combination Analysis', fontsize=16, fontweight='bold')

# Plot 1: MAE vs Number of Features
ax1.scatter(df['Number of Features'], df['MAE Validation'], alpha=0.6, s=50, color='#0b53c1')
ax1.set_xlabel('Number of Features')
ax1.set_ylabel('MAE Validation (°C)')
ax1.set_title('MAE vs Number of Features')
ax1.grid(True, alpha=0.3)

# Add trend line
z = np.polyfit(df['Number of Features'], df['MAE Validation'], 1)
p = np.poly1d(z)
ax1.plot(df['Number of Features'], p(df['Number of Features']), "r--", alpha=0.8)

# Plot 2: R² vs Number of Features
ax2.scatter(df['Number of Features'], df['R-squared Validation'], alpha=0.6, s=50, color='#33a02c')
ax2.set_xlabel('Number of Features')
ax2.set_ylabel('R² Validation')
ax2.set_title('R² vs Number of Features')
ax2.grid(True, alpha=0.3)

# Add trend line
z2 = np.polyfit(df['Number of Features'], df['R-squared Validation'], 1)
p2 = np.poly1d(z2)
ax2.plot(df['Number of Features'], p2(df['Number of Features']), "r--", alpha=0.8)

# Plot 3: Top 10 Best Performing Combinations
top_10 = df.nsmallest(10, 'MAE Validation')
top_10_sorted = top_10.sort_values('MAE Validation')
y_pos = np.arange(len(top_10_sorted))
bars = ax3.barh(y_pos, top_10_sorted['MAE Validation'], color='#e31a1c', alpha=0.8)
ax3.set_yticks(y_pos)
ax3.set_yticklabels([f"{feat} ({n} feats)" for feat, n in zip(top_10_sorted['Feature Combination'], top_10_sorted['Number of Features'])], fontsize=8)
ax3.set_xlabel('MAE Validation (°C)')
ax3.set_title('Top 10 Best Performing Combinations')
ax3.grid(True, alpha=0.3, axis='x')

# Add value labels
for i, (bar, mae) in enumerate(zip(bars, top_10_sorted['MAE Validation'])):
    width = bar.get_width()
    ax3.text(width + 0.1, bar.get_y() + bar.get_height()/2., 
             f'{mae:.1f}', ha='left', va='center', fontsize=9, fontweight='bold')

# Plot 4: Train vs Validation MAE
ax4.scatter(df['Train MAE'], df['MAE Validation'], alpha=0.6, s=50, color='#ff7f00')
ax4.plot([df['Train MAE'].min(), df['Train MAE'].max()], 
         [df['Train MAE'].min(), df['Train MAE'].max()], 'k--', alpha=0.8, label='Perfect Generalization')
ax4.set_xlabel('Train MAE (°C)')
ax4.set_ylabel('Validation MAE (°C)')
ax4.set_title('Train vs Validation MAE')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Calculate generalization gap
df['Generalization Gap'] = df['MAE Validation'] - df['Train MAE']
avg_gap = df['Generalization Gap'].mean()
ax4.text(0.05, 0.95, f'Avg Gap: {avg_gap:.2f}°C', 
         transform=ax4.transAxes, fontsize=12, 
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('stacking_ensemble_summary.png', dpi=300, bbox_inches='tight')
plt.show()

# Print summary statistics
print("\n" + "="*60)
print("STACKING ENSEMBLE RESULTS SUMMARY")
print("="*60)
print(f"Total feature combinations tested: {len(df)}")
print(f"Best MAE: {df['MAE Validation'].min():.2f}°C")
print(f"Worst MAE: {df['MAE Validation'].max():.2f}°C")
print(f"Average MAE: {df['MAE Validation'].mean():.2f}°C")
print(f"Best R²: {df['R-squared Validation'].max():.3f}")
print(f"Average R²: {df['R-squared Validation'].mean():.3f}")
print(f"Average generalization gap: {avg_gap:.2f}°C")

print(f"\nBest performing combination:")
best = df.loc[df['MAE Validation'].idxmin()]
print(f"Features: {best['Feature Combination']}")
print(f"Number of features: {best['Number of Features']}")
print(f"MAE: {best['MAE Validation']:.2f}°C")
print(f"R²: {best['R-squared Validation']:.3f}")
print(f"Train MAE: {best['Train MAE']:.2f}°C")
print("="*60)
