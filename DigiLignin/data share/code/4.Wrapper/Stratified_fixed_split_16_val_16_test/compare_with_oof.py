# -*- coding: utf-8 -*-
"""
Comparison Script: Fixed Split vs OOF Method
Analyzes and visualizes differences between the two approaches
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_results():
    """Load results from both methods."""
    print("Loading results from both methods...")
    
    # Load fixed split results
    fixed_split_path = 'fixed_split_results.csv'
    if not os.path.exists(fixed_split_path):
        print(f"Error: {fixed_split_path} not found. Run run_fixed_split_experiments.py first.")
        return None, None
    
    df_fixed = pd.read_csv(fixed_split_path)
    print(f"Fixed split results loaded: {len(df_fixed)} rows")
    
    # Load OOF results
    oof_path = '../Fixed_stacking_ensemble_with_n_estimators/all_combinations_n_estimators_results.csv'
    if not os.path.exists(oof_path):
        print(f"Warning: {oof_path} not found. OOF comparison will be skipped.")
        return df_fixed, None
    
    df_oof = pd.read_csv(oof_path)
    print(f"OOF results loaded: {len(df_oof)} rows")
    
    return df_fixed, df_oof

def compare_best_models(df_fixed, df_oof):
    """Compare best models from both methods using validation data."""
    print("\n" + "="*80)
    print("BEST MODEL COMPARISON (Validation Focus)")
    print("="*80)
    
    # Fixed split best (by validation MAE)
    best_fixed = df_fixed.nsmallest(1, 'Validation MAE').iloc[0]
    print("\nBest Fixed Split Model (Validation MAE):")
    print(f"  Combination ID: {best_fixed['Combination ID']}")
    print(f"  N_Estimators: {best_fixed['n_estimators']}")
    print(f"  Number of Features: {best_fixed['Number of Features']}")
    print(f"  Validation MAE: {best_fixed['Validation MAE']:.3f}°C")
    print(f"  Validation R²: {best_fixed['Validation R2']:.3f}")
    print(f"  Test MAE: {best_fixed['Test MAE']:.3f}°C (for reference)")
    print(f"  Test R²: {best_fixed['Test R2']:.3f} (for reference)")
    
    if df_oof is not None:
        # OOF best (by validation MAE)
        best_oof = df_oof.nsmallest(1, 'MAE Validation').iloc[0]
        print("\nBest OOF Model (Validation MAE):")
        print(f"  N_Estimators: {best_oof['n_estimators']}")
        print(f"  Number of Features: {best_oof['Number of Features']}")
        print(f"  Validation MAE: {best_oof['MAE Validation']:.3f}°C")
        print(f"  Validation R²: {best_oof['R2 Validation']:.3f}")
        print(f"  Validation MAE CI: [{best_oof['Validation MAE CI Lower']:.3f}, {best_oof['Validation MAE CI Upper']:.3f}]")
        
        print("\nValidation Comparison:")
        print(f"  Fixed Split Val MAE - OOF Val MAE: {best_fixed['Validation MAE'] - best_oof['MAE Validation']:.3f}°C")
        print(f"  Fixed Split Val R² - OOF Val R²: {best_fixed['Validation R2'] - best_oof['R2 Validation']:.3f}")

def compare_by_n_estimators(df_fixed, df_oof):
    """Compare performance trends by n_estimators using validation data."""
    print("\n" + "="*80)
    print("VALIDATION PERFORMANCE BY N_ESTIMATORS")
    print("="*80)
    
    estimator_values = sorted(df_fixed['n_estimators'].unique())
    
    print("\nAverage Validation Performance:")
    print(f"{'N_Est':>6} | {'Fixed Val MAE':>14} | {'Fixed Val R²':>13} | {'OOF Val MAE':>13} | {'OOF Val R²':>12} | {'MAE Diff':>9} | {'R² Diff':>8}")
    print("-" * 95)
    
    for n_est in estimator_values:
        fixed_val_mae = df_fixed[df_fixed['n_estimators'] == n_est]['Validation MAE'].mean()
        fixed_val_r2 = df_fixed[df_fixed['n_estimators'] == n_est]['Validation R2'].mean()
        
        if df_oof is not None:
            oof_val_mae = df_oof[df_oof['n_estimators'] == n_est]['MAE Validation'].mean()
            oof_val_r2 = df_oof[df_oof['n_estimators'] == n_est]['R2 Validation'].mean()
            mae_diff = fixed_val_mae - oof_val_mae
            r2_diff = fixed_val_r2 - oof_val_r2
            print(f"{n_est:6d} | {fixed_val_mae:14.3f} | {fixed_val_r2:13.3f} | {oof_val_mae:13.3f} | {oof_val_r2:12.3f} | {mae_diff:+9.3f} | {r2_diff:+8.3f}")
        else:
            print(f"{n_est:6d} | {fixed_val_mae:14.3f} | {fixed_val_r2:13.3f} | {'N/A':>13} | {'N/A':>12} | {'N/A':>9} | {'N/A':>8}")

def create_comparison_plots(df_fixed, df_oof):
    """Create comprehensive comparison plots."""
    print("\nCreating comparison plots...")
    
    if df_oof is None:
        print("OOF results not available. Creating fixed split plots only.")
        create_fixed_split_only_plots(df_fixed)
        return
    
    fig = plt.figure(figsize=(20, 12))
    fig.patch.set_facecolor('white')
    
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.25, 
                          left=0.06, right=0.95, top=0.92, bottom=0.08)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])
    
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        ax.set_facecolor('white')
    
    # Plot 1: Validation MAE Comparison by N_Estimators
    fixed_val_mae = df_fixed.groupby('n_estimators')['Validation MAE'].mean()
    fixed_test_mae = df_fixed.groupby('n_estimators')['Test MAE'].mean()
    oof_val_mae = df_oof.groupby('n_estimators')['MAE Validation'].mean()
    
    ax1.plot(fixed_val_mae.index, fixed_val_mae, 'o-', color='#4C72B0', 
             markersize=8, label='Fixed Split Val', linewidth=2)
    ax1.plot(fixed_test_mae.index, fixed_test_mae, 's--', color='#55A868', 
             markersize=6, label='Fixed Split Test (ref)', linewidth=1, alpha=0.7)
    ax1.plot(oof_val_mae.index, oof_val_mae, '^-', color='#C44E52', 
             markersize=8, label='OOF Val', linewidth=2)
    ax1.set_xlabel('Number of Estimators', fontsize=12)
    ax1.set_ylabel('MAE (°C)', fontsize=12)
    ax1.set_title('A: Validation MAE Comparison', fontsize=14, fontweight='bold', pad=15)
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.legend(fontsize=10)
    
    # Plot 2: Validation R² Comparison by N_Estimators
    fixed_val_r2 = df_fixed.groupby('n_estimators')['Validation R2'].mean()
    fixed_test_r2 = df_fixed.groupby('n_estimators')['Test R2'].mean()
    oof_val_r2 = df_oof.groupby('n_estimators')['R2 Validation'].mean()
    
    ax2.plot(fixed_val_r2.index, fixed_val_r2, 'o-', color='#4C72B0', 
             markersize=8, label='Fixed Split Val', linewidth=2)
    ax2.plot(fixed_test_r2.index, fixed_test_r2, 's--', color='#55A868', 
             markersize=6, label='Fixed Split Test (ref)', linewidth=1, alpha=0.7)
    ax2.plot(oof_val_r2.index, oof_val_r2, '^-', color='#C44E52', 
             markersize=8, label='OOF Val', linewidth=2)
    ax2.set_xlabel('Number of Estimators', fontsize=12)
    ax2.set_ylabel('R²', fontsize=12)
    ax2.set_title('B: Validation R² Comparison', fontsize=14, fontweight='bold', pad=15)
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.legend(fontsize=10)
    
    # Plot 3: Best Validation Performance Comparison
    fixed_best_val = df_fixed.groupby('n_estimators')['Validation MAE'].min()
    oof_best_val = df_oof.groupby('n_estimators')['MAE Validation'].min()
    
    ax3.plot(fixed_best_val.index, fixed_best_val, 'o-', color='#4C72B0', 
             markersize=8, label='Fixed Split Best Val', linewidth=2)
    ax3.plot(oof_best_val.index, oof_best_val, '^-', color='#C44E52', 
             markersize=8, label='OOF Best Val', linewidth=2)
    ax3.set_xlabel('Number of Estimators', fontsize=12)
    ax3.set_ylabel('MAE (°C)', fontsize=12)
    ax3.set_title('C: Best Validation Performance', fontsize=14, fontweight='bold', pad=15)
    ax3.grid(True, linestyle='--', alpha=0.3)
    ax3.legend(fontsize=10)
    
    # Plot 4: Validation Performance Distribution Comparison
    fixed_val_mae_dist = df_fixed['Validation MAE'].values
    oof_val_mae_dist = df_oof['MAE Validation'].values
    
    ax4.hist(fixed_val_mae_dist, bins=30, alpha=0.6, color='#4C72B0', 
             label='Fixed Split Val', edgecolor='black')
    ax4.hist(oof_val_mae_dist, bins=30, alpha=0.6, color='#C44E52', 
             label='OOF Val', edgecolor='black')
    ax4.set_xlabel('Validation MAE (°C)', fontsize=12)
    ax4.set_ylabel('Frequency', fontsize=12)
    ax4.set_title('D: Validation MAE Distribution', fontsize=14, fontweight='bold', pad=15)
    ax4.legend(fontsize=10)
    ax4.grid(True, linestyle='--', alpha=0.3, axis='y')
    
    # Plot 5: Scatter Plot - Fixed Val vs OOF Val
    # Match by n_estimators and number of features
    merged = pd.merge(
        df_fixed.groupby(['n_estimators', 'Number of Features'])['Validation MAE'].mean().reset_index(),
        df_oof.groupby(['n_estimators', 'Number of Features'])['MAE Validation'].mean().reset_index(),
        on=['n_estimators', 'Number of Features'],
        suffixes=('_fixed', '_oof')
    )
    
    ax5.scatter(merged['MAE Validation'], merged['Validation MAE'], 
                alpha=0.5, s=50, c='#4C72B0', edgecolors='black')
    
    # Add diagonal line
    min_val = min(merged['MAE Validation'].min(), merged['Validation MAE'].min())
    max_val = max(merged['MAE Validation'].max(), merged['Validation MAE'].max())
    ax5.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='y=x')
    
    ax5.set_xlabel('OOF Validation MAE (°C)', fontsize=12)
    ax5.set_ylabel('Fixed Split Validation MAE (°C)', fontsize=12)
    ax5.set_title('E: Fixed Val vs OOF Val', fontsize=14, fontweight='bold', pad=15)
    ax5.grid(True, linestyle='--', alpha=0.3)
    ax5.legend(fontsize=10)
    
    # Calculate correlation
    corr = merged['MAE Validation'].corr(merged['Validation MAE'])
    ax5.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
             transform=ax5.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 6: Validation Difference Analysis
    merged['Difference'] = merged['Validation MAE'] - merged['MAE Validation']
    
    ax6.scatter(merged['n_estimators'], merged['Difference'], 
                alpha=0.5, s=50, c=merged['Number of Features'], 
                cmap='viridis', edgecolors='black')
    ax6.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax6.set_xlabel('Number of Estimators', fontsize=12)
    ax6.set_ylabel('Difference (Fixed Val - OOF Val) (°C)', fontsize=12)
    ax6.set_title('F: Validation Performance Difference', fontsize=14, fontweight='bold', pad=15)
    ax6.grid(True, linestyle='--', alpha=0.3)
    
    cbar = plt.colorbar(ax6.collections[0], ax=ax6)
    cbar.set_label('Number of Features', fontsize=10)
    
    fig.suptitle('Fixed Split vs OOF Validation Method Comparison', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    for ext in ['png', 'tiff', 'pdf', 'svg']:
        plt.savefig(f'method_comparison.{ext}', 
                   dpi=300, bbox_inches='tight', facecolor='white')
    
    plt.show()

def create_fixed_split_only_plots(df_fixed):
    """Create plots for fixed split results only."""
    fig = plt.figure(figsize=(15, 10))
    fig.patch.set_facecolor('white')
    
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25, 
                          left=0.08, right=0.95, top=0.92, bottom=0.08)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Plot 1: Val vs Test MAE
    val_mae = df_fixed.groupby('n_estimators')['Validation MAE'].mean()
    test_mae = df_fixed.groupby('n_estimators')['Test MAE'].mean()
    
    ax1.plot(val_mae.index, val_mae, 'o-', color='#4C72B0', 
             markersize=8, label='Validation', linewidth=2)
    ax1.plot(test_mae.index, test_mae, 's-', color='#55A868', 
             markersize=8, label='Test', linewidth=2)
    ax1.set_xlabel('Number of Estimators', fontsize=12)
    ax1.set_ylabel('MAE (°C)', fontsize=12)
    ax1.set_title('A: Validation vs Test MAE', fontsize=14, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.legend(fontsize=10)
    
    # Plot 2: Val vs Test R²
    val_r2 = df_fixed.groupby('n_estimators')['Validation R2'].mean()
    test_r2 = df_fixed.groupby('n_estimators')['Test R2'].mean()
    
    ax2.plot(val_r2.index, val_r2, 'o-', color='#4C72B0', 
             markersize=8, label='Validation', linewidth=2)
    ax2.plot(test_r2.index, test_r2, 's-', color='#55A868', 
             markersize=8, label='Test', linewidth=2)
    ax2.set_xlabel('Number of Estimators', fontsize=12)
    ax2.set_ylabel('R²', fontsize=12)
    ax2.set_title('B: Validation vs Test R²', fontsize=14, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.legend(fontsize=10)
    
    # Plot 3: Generalization Gap
    gap = df_fixed.groupby('n_estimators').apply(
        lambda x: (x['Test MAE'] - x['Validation MAE']).mean()
    )
    
    ax3.plot(gap.index, gap, 'o-', color='#C44E52', markersize=8, linewidth=2)
    ax3.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax3.set_xlabel('Number of Estimators', fontsize=12)
    ax3.set_ylabel('Generalization Gap (Test - Val MAE) (°C)', fontsize=12)
    ax3.set_title('C: Generalization Gap', fontsize=14, fontweight='bold')
    ax3.grid(True, linestyle='--', alpha=0.3)
    
    # Plot 4: Performance by Feature Count
    feature_stats = df_fixed.groupby('Number of Features').agg({
        'Validation MAE': 'mean',
        'Test MAE': 'mean'
    })
    
    x_pos = np.arange(len(feature_stats))
    width = 0.35
    
    ax4.bar(x_pos - width/2, feature_stats['Validation MAE'], width, 
            label='Validation', alpha=0.7, color='#4C72B0', edgecolor='black')
    ax4.bar(x_pos + width/2, feature_stats['Test MAE'], width, 
            label='Test', alpha=0.7, color='#55A868', edgecolor='black')
    ax4.set_xlabel('Number of Features', fontsize=12)
    ax4.set_ylabel('Average MAE (°C)', fontsize=12)
    ax4.set_title('D: Performance by Feature Count', fontsize=14, fontweight='bold')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(feature_stats.index)
    ax4.legend(fontsize=10)
    ax4.grid(True, linestyle='--', alpha=0.3, axis='y')
    
    fig.suptitle('Fixed Split Analysis', fontsize=16, fontweight='bold', y=0.98)
    
    for ext in ['png', 'tiff', 'pdf', 'svg']:
        plt.savefig(f'fixed_split_summary.{ext}', 
                   dpi=300, bbox_inches='tight', facecolor='white')
    
    plt.show()

def analyze_generalization(df_fixed):
    """Analyze generalization gap between validation and test sets."""
    print("\n" + "="*80)
    print("GENERALIZATION ANALYSIS")
    print("="*80)
    
    df_fixed['Generalization_Gap'] = df_fixed['Test MAE'] - df_fixed['Validation MAE']
    
    print(f"\nOverall Generalization Statistics:")
    print(f"  Mean Gap: {df_fixed['Generalization_Gap'].mean():.3f}°C")
    print(f"  Std Gap: {df_fixed['Generalization_Gap'].std():.3f}°C")
    print(f"  Min Gap: {df_fixed['Generalization_Gap'].min():.3f}°C")
    print(f"  Max Gap: {df_fixed['Generalization_Gap'].max():.3f}°C")
    print(f"  Median Gap: {df_fixed['Generalization_Gap'].median():.3f}°C")
    
    print(f"\nGeneralization Gap by N_Estimators:")
    gap_by_n = df_fixed.groupby('n_estimators')['Generalization_Gap'].agg(['mean', 'std'])
    for n_est, stats in gap_by_n.iterrows():
        print(f"  {n_est:4d}: {stats['mean']:+.3f} ± {stats['std']:.3f}°C")
    
    print(f"\nGeneralization Gap by Feature Count:")
    gap_by_feat = df_fixed.groupby('Number of Features')['Generalization_Gap'].agg(['mean', 'std'])
    for n_feat, stats in gap_by_feat.iterrows():
        print(f"  {n_feat} features: {stats['mean']:+.3f} ± {stats['std']:.3f}°C")

def main():
    """Main execution function."""
    print("="*80)
    print("FIXED SPLIT VS OOF METHOD COMPARISON")
    print("="*80)
    
    # Load results
    df_fixed, df_oof = load_results()
    
    if df_fixed is None:
        print("Error: Fixed split results not found. Exiting.")
        return
    
    # Compare best models
    compare_best_models(df_fixed, df_oof)
    
    # Compare by n_estimators
    compare_by_n_estimators(df_fixed, df_oof)
    
    # Analyze generalization
    analyze_generalization(df_fixed)
    
    # Create comparison plots
    create_comparison_plots(df_fixed, df_oof)
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)
    print("Generated Files:")
    if df_oof is not None:
        print("  - method_comparison.png/tiff/pdf/svg")
    else:
        print("  - fixed_split_summary.png/tiff/pdf/svg")
    print("="*80)

if __name__ == "__main__":
    main()
