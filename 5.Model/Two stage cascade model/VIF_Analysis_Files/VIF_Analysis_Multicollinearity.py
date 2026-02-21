# -*- coding: utf-8 -*-
"""
VIF Analysis for Multicollinearity Detection

This script addresses Reviewer #2's concern about multicollinearity in the feature set.

Problem: Several features are mathematically related or redundant:
  - Lignin (wt%) + Co-polyol (wt%) ≈ constant (complementary features)
  - Isocyanate (wt%) and Isocyanate (mmol NCO) are highly correlated
  - Ratio [NCO]/[OH] is derived from other features

Solution: Calculate Variance Inflation Factor (VIF) for all features
  - VIF > 10: High multicollinearity (should remove)
  - VIF 5-10: Moderate multicollinearity (consider removing)
  - VIF < 5: Low multicollinearity (keep)

@author: Fixed implementation addressing reviewer concerns
"""

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import RobustScaler
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_vif(df, features):
    """
    Calculate Variance Inflation Factor for each feature.

    VIF measures how much the variance of a regression coefficient is inflated
    due to multicollinearity with other features.

    Args:
        df: DataFrame with all data
        features: List of feature names to analyze

    Returns:
        vif_df: DataFrame with VIF values for each feature
    """
    print("\n" + "="*80)
    print("VARIANCE INFLATION FACTOR (VIF) ANALYSIS")
    print("="*80)
    print("\nCalculating VIF for features:")
    for feat in features:
        print(f"  - {feat}")

    # Extract features
    X = df[features].copy()

    # Scale features (VIF can be sensitive to scale)
    scaler = RobustScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=features,
        index=X.index
    )

    # Calculate VIF for each feature
    vif_data = []
    for i, feature in enumerate(features):
        vif_value = variance_inflation_factor(X_scaled.values, i)
        vif_data.append({
            'Feature': feature,
            'VIF': vif_value
        })
        print(f"\n  {feature}:")
        print(f"    VIF = {vif_value:.2f}", end="")

        if vif_value > 10:
            print(" [WARNING] HIGH multicollinearity - REMOVE")
        elif vif_value > 5:
            print(" [WARNING] MODERATE multicollinearity - CONSIDER REMOVING")
        else:
            print(" [OK] Low multicollinearity - KEEP")

    vif_df = pd.DataFrame(vif_data).sort_values('VIF', ascending=False)

    return vif_df


def plot_vif_results(vif_df, save_path='VIF_Analysis.png'):
    """
    Create visualization of VIF results.

    Args:
        vif_df: DataFrame with VIF values
        save_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create color map based on VIF thresholds
    colors = []
    for vif in vif_df['VIF']:
        if vif > 10:
            colors.append('#e31a1c')  # Red - high multicollinearity
        elif vif > 5:
            colors.append('#ff7f00')  # Orange - moderate
        else:
            colors.append('#33a02c')  # Green - low

    # Create horizontal bar plot
    bars = ax.barh(vif_df['Feature'], vif_df['VIF'], color=colors, alpha=0.7)

    # Add threshold lines
    ax.axvline(x=5, color='orange', linestyle='--', linewidth=2, label='VIF = 5 (Moderate)')
    ax.axvline(x=10, color='red', linestyle='--', linewidth=2, label='VIF = 10 (High)')

    # Labels and title
    ax.set_xlabel('Variance Inflation Factor (VIF)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Features', fontsize=12, fontweight='bold')
    ax.set_title('Multicollinearity Analysis - VIF Values', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='x', alpha=0.3)

    # Add VIF values as text
    for i, (feature, vif) in enumerate(zip(vif_df['Feature'], vif_df['VIF'])):
        ax.text(vif + 0.5, i, f'{vif:.2f}', va='center', fontsize=10)

    plt.tight_layout()

    # Save in multiple formats
    for ext in ['png', 'pdf', 'svg']:
        plt.savefig(f'VIF_Analysis.{ext}', dpi=300, bbox_inches='tight')

    print(f"\n[OK] VIF plot saved")
    plt.close()


def recommend_feature_reduction(vif_df, threshold=10):
    """
    Recommend which features to remove based on VIF analysis.

    Args:
        vif_df: DataFrame with VIF values
        threshold: VIF threshold for removal (default: 10)

    Returns:
        recommendations: Dictionary with keep/remove lists
    """
    print("\n" + "="*80)
    print("FEATURE REDUCTION RECOMMENDATIONS")
    print("="*80)

    high_vif = vif_df[vif_df['VIF'] > threshold]
    moderate_vif = vif_df[(vif_df['VIF'] > 5) & (vif_df['VIF'] <= threshold)]
    low_vif = vif_df[vif_df['VIF'] <= 5]

    print(f"\n1. HIGH multicollinearity (VIF > {threshold}): MUST REMOVE")
    if len(high_vif) > 0:
        for _, row in high_vif.iterrows():
            print(f"   [X] {row['Feature']} (VIF = {row['VIF']:.2f})")
    else:
        print("   [OK] None")

    print(f"\n2. MODERATE multicollinearity (5 < VIF ≤ {threshold}): CONSIDER REMOVING")
    if len(moderate_vif) > 0:
        for _, row in moderate_vif.iterrows():
            print(f"   [!] {row['Feature']} (VIF = {row['VIF']:.2f})")
    else:
        print("   [OK] None")

    print(f"\n3. LOW multicollinearity (VIF ≤ 5): KEEP")
    for _, row in low_vif.iterrows():
        print(f"   [OK] {row['Feature']} (VIF = {row['VIF']:.2f})")

    recommendations = {
        'remove_high': high_vif['Feature'].tolist(),
        'consider_removing': moderate_vif['Feature'].tolist(),
        'keep': low_vif['Feature'].tolist()
    }

    return recommendations


def propose_reduced_feature_set(vif_df, df):
    """
    Propose a reduced feature set by iteratively removing high-VIF features.

    Strategy:
    1. Remove features with VIF > 10 one at a time (highest first)
    2. Recalculate VIF after each removal
    3. Stop when all VIF < 10

    Args:
        vif_df: Initial VIF DataFrame
        df: Full dataset

    Returns:
        final_features: List of features to keep
        vif_history: History of VIF calculations
    """
    print("\n" + "="*80)
    print("ITERATIVE FEATURE REDUCTION")
    print("="*80)

    current_features = vif_df['Feature'].tolist()
    vif_history = []
    iteration = 0

    while True:
        iteration += 1
        print(f"\n--- Iteration {iteration} ---")
        print(f"Current features ({len(current_features)}): {current_features}")

        # Calculate VIF for current feature set
        current_vif = calculate_vif(df, current_features)
        vif_history.append(current_vif.copy())

        # Check if any VIF > 10
        max_vif_row = current_vif.iloc[0]  # Already sorted by VIF descending

        if max_vif_row['VIF'] <= 10:
            print(f"\n[OK] All VIF values <= 10. Stopping.")
            break

        # Remove feature with highest VIF
        feature_to_remove = max_vif_row['Feature']
        print(f"\n[X] Removing '{feature_to_remove}' (VIF = {max_vif_row['VIF']:.2f})")

        # Ask for user confirmation or provide automatic logic
        # For automatic removal, we need domain knowledge
        current_features.remove(feature_to_remove)

        if len(current_features) < 3:
            print("\n[WARNING] Only 2 features remaining. Stopping to avoid over-reduction.")
            break

    print("\n" + "="*80)
    print("FINAL REDUCED FEATURE SET")
    print("="*80)
    print(f"\nReduced from {len(vif_df)} to {len(current_features)} features:")
    for feat in current_features:
        vif_val = current_vif[current_vif['Feature'] == feat]['VIF'].values[0]
        print(f"  [OK] {feat} (VIF = {vif_val:.2f})")

    return current_features, vif_history


def compare_feature_sets(df, original_features, reduced_features):
    """
    Compare correlation matrices between original and reduced feature sets.

    Args:
        df: DataFrame with all data
        original_features: Original feature list
        reduced_features: Reduced feature list
    """
    print("\n" + "="*80)
    print("CORRELATION COMPARISON")
    print("="*80)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Original features correlation
    corr_original = df[original_features].corr()
    sns.heatmap(corr_original, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, vmin=-1, vmax=1, ax=ax1, cbar_kws={'label': 'Correlation'})
    ax1.set_title(f'Original Features ({len(original_features)})', fontsize=14, fontweight='bold')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')

    # Reduced features correlation
    corr_reduced = df[reduced_features].corr()
    sns.heatmap(corr_reduced, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, vmin=-1, vmax=1, ax=ax2, cbar_kws={'label': 'Correlation'})
    ax2.set_title(f'Reduced Features ({len(reduced_features)})', fontsize=14, fontweight='bold')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()

    for ext in ['png', 'pdf']:
        plt.savefig(f'Correlation_Comparison.{ext}', dpi=300, bbox_inches='tight')

    print("\n[OK] Correlation comparison plot saved")
    plt.close()

    # Print max absolute correlations
    print("\nMaximum absolute correlations:")
    print(f"  Original features: {corr_original.abs().where(~np.eye(len(corr_original), dtype=bool)).max().max():.3f}")
    print(f"  Reduced features: {corr_reduced.abs().where(~np.eye(len(corr_reduced), dtype=bool)).max().max():.3f}")


# Main script
if __name__ == "__main__":
    """
    Main execution script for VIF analysis and multicollinearity detection.

    This addresses Reviewer #2's concern about redundant features.
    """

    print("="*80)
    print("MULTICOLLINEARITY ANALYSIS - VIF CALCULATION")
    print("Addressing Reviewer #2's Concern about Redundant Features")
    print("="*80)

    try:
        # Define all formulation features (excluding swelling ratio for now)
        all_features = [
            'Lignin (wt%)',
            'Co-polyol (wt%)',
            'Co-polyol type (PTHF)',
            'Isocyanate (wt%)',
            'Isocyanate (mmol NCO)',
            'Isocyanate type',
            'Ratio',
            'Tin(II) octoate'
        ]

        print(f"\nAnalyzing {len(all_features)} formulation features")
        print("(Swelling ratio excluded as it will be handled by cascade model)")

        # Calculate VIF
        vif_df = calculate_vif(df, all_features)

        # Plot results
        plot_vif_results(vif_df)

        # Get recommendations
        recommendations = recommend_feature_reduction(vif_df, threshold=10)

        # Propose reduced feature set
        reduced_features, vif_history = propose_reduced_feature_set(vif_df, df)

        # Compare correlation matrices
        compare_feature_sets(df, all_features, reduced_features)

        # Save results
        vif_df.to_csv('VIF_Analysis_Results.csv', index=False)

        # Save reduced feature set
        with open('Reduced_Feature_Set.txt', 'w') as f:
            f.write("REDUCED FEATURE SET (VIF < 10)\n")
            f.write("="*50 + "\n\n")
            for feat in reduced_features:
                f.write(f"{feat}\n")

        print("\n" + "="*80)
        print("[COMPLETE] VIF ANALYSIS")
        print("="*80)
        print("\nFiles saved:")
        print("  - VIF_Analysis_Results.csv")
        print("  - VIF_Analysis.png/pdf/svg")
        print("  - Correlation_Comparison.png/pdf")
        print("  - Reduced_Feature_Set.txt")

        print("\nNext steps:")
        print("  1. Update all model scripts to use reduced feature set")
        print("  2. Re-run models with reduced features")
        print("  3. Compare performance: original vs reduced features")
        print("  4. Update manuscript to report VIF analysis")

    except NameError:
        print("\n[WARNING] ERROR: DataFrame 'df' not found.")
        print("\nPlease load your data first:")
        print("  from '1.Loading and Preprocessing.Loading and preprocessing' import main")
        print("  df = main()")

