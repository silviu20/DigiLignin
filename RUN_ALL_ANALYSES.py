#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Master Script to Run All Analyses for Manuscript Revision

This script executes all critical and high-priority analyses:
1. Load and preprocess data
2. VIF analysis for multicollinearity
3. Fixed stacking ensemble (no data leakage)
4. Two-stage cascade model (swelling ratio issue)

@author: Silviu
@date: 2026-02-15
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add paths for imports
sys.path.append(str(Path(__file__).parent / "1.Loading and Preprocessing"))
sys.path.append(str(Path(__file__).parent / "5.Model"))

print("="*80)
print("MASTER ANALYSIS SCRIPT - MANUSCRIPT REVISION")
print("="*80)
print("\nThis script will run all analyses needed for the manuscript revision.")
print("Estimated time: 10-30 minutes depending on your hardware.\n")

# ============================================================================
# STEP 1: LOAD AND PREPROCESS DATA
# ============================================================================

print("\n" + "="*80)
print("STEP 1: LOADING AND PREPROCESSING DATA")
print("="*80)

try:
    # Try to load from Excel file
    data_path = Path(__file__).parent / "dataset.csv.xlsx"
    
    if data_path.exists():
        print(f"\nLoading data from: {data_path}")
        df_raw = pd.read_excel(data_path)
        print(f"✓ Data loaded: {len(df_raw)} rows, {len(df_raw.columns)} columns")
    else:
        print(f"\n⚠ File not found: {data_path}")
        print("Please specify the correct data file path.")
        sys.exit(1)
    
    # Preprocess data
    print("\nPreprocessing data...")
    print(f"  Original columns: {list(df_raw.columns)}")

    # Rename columns to match expected names
    column_mapping = {
        'Copolyol (wt%)': 'Co-polyol (wt%)',
        'Isocyonate type': 'Isocyanate type',
        'r': 'Ratio',
        'tin(II) octoate': 'Tin(II) octoate',
        'Tg(deg C)': 'Tg (°C)',
        'Sratio(%)': 'Swelling ratio (%)'
    }

    df = df_raw.rename(columns=column_mapping)
    print(f"  ✓ Columns renamed to standard format")

    # Drop rows with missing Tg
    df = df.dropna(subset=['Tg (°C)'])
    print(f"  After removing NaN Tg: {len(df)} rows")

    # Map categorical values
    if 'Isocyanate type' in df.columns:
        isocyanate_mapping = {'N3600': 1, 'HDI': 0, 0: np.nan}
        df['Isocyanate type'] = df['Isocyanate type'].map(isocyanate_mapping)
        df['Isocyanate type'] = df['Isocyanate type'].fillna(0)
        print("  ✓ Isocyanate type encoded")

    # Verify required columns exist
    required_columns = [
        'Lignin (wt%)', 'Co-polyol (wt%)', 'Co-polyol type (PTHF)',
        'Isocyanate (wt%)', 'Isocyanate (mmol NCO)', 'Isocyanate type',
        'Ratio', 'Tin(II) octoate', 'Swelling ratio (%)', 'Tg (°C)'
    ]

    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        print(f"\n⚠ Missing columns: {missing_cols}")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)
    
    print(f"\n✓ Data preprocessing complete")
    print(f"  Final dataset: {len(df)} samples")
    print(f"  Features: {len(required_columns) - 1}")
    
except Exception as e:
    print(f"\n❌ Error loading data: {e}")
    print("\nPlease ensure:")
    print("  1. dataset.csv.xlsx exists in the project root")
    print("  2. File contains all required columns")
    sys.exit(1)

# ============================================================================
# STEP 2: VIF ANALYSIS FOR MULTICOLLINEARITY
# ============================================================================

print("\n" + "="*80)
print("STEP 2: VIF ANALYSIS FOR MULTICOLLINEARITY")
print("="*80)

try:
    from VIF_Analysis_Multicollinearity import (
        calculate_vif, plot_vif_results, recommend_feature_reduction,
        propose_reduced_feature_set, compare_feature_sets
    )
    
    # Define formulation features (excluding swelling ratio)
    formulation_features = [
        'Lignin (wt%)', 'Co-polyol (wt%)', 'Co-polyol type (PTHF)',
        'Isocyanate (wt%)', 'Isocyanate (mmol NCO)', 'Isocyanate type',
        'Ratio', 'Tin(II) octoate'
    ]
    
    print(f"\nAnalyzing {len(formulation_features)} formulation features...")
    
    # Calculate VIF
    vif_df = calculate_vif(df, formulation_features)
    
    # Plot results
    plot_vif_results(vif_df)
    
    # Get recommendations
    recommendations = recommend_feature_reduction(vif_df, threshold=10)
    
    # Propose reduced feature set
    reduced_features, vif_history = propose_reduced_feature_set(vif_df, df)
    
    # Compare correlation matrices
    compare_feature_sets(df, formulation_features, reduced_features)
    
    # Save results
    vif_df.to_csv('VIF_Analysis_Results.csv', index=False)
    
    with open('Reduced_Feature_Set.txt', 'w') as f:
        f.write("REDUCED FEATURE SET (VIF < 10)\n")
        f.write("="*50 + "\n\n")
        for feat in reduced_features:
            f.write(f"{feat}\n")
    
    print("\n✓ VIF Analysis complete")
    print(f"  Results saved to: VIF_Analysis_Results.csv")
    print(f"  Reduced features: {len(reduced_features)}/{len(formulation_features)}")
    
except Exception as e:
    print(f"\n⚠ VIF Analysis failed: {e}")
    print("Continuing with remaining analyses...")
    reduced_features = formulation_features  # Use all features if VIF fails

# ============================================================================
# STEP 3: FIXED STACKING ENSEMBLE (NO DATA LEAKAGE)
# ============================================================================

print("\n" + "="*80)
print("STEP 3: FIXED STACKING ENSEMBLE (NO DATA LEAKAGE)")
print("="*80)

try:
    from Stacked_Ensembles_Fixed import run_multiple_times_fixed, plot_results_fixed
    
    # Define features and target
    x_features = reduced_features + ['Swelling ratio (%)']  # Include swelling for now
    x = df[x_features]
    y = df[['Tg (°C)']]
    
    print(f"\nRunning fixed stacking ensemble...")
    print(f"  Features: {x_features}")
    print(f"  Samples: {len(df)}")
    
    # Run fixed stacking
    results_df, best_models = run_multiple_times_fixed(
        x, y,
        num_runs=1,
        n_estimators_list=[1000]
    )
    
    # Display results
    print("\n" + "="*80)
    print("FIXED STACKING RESULTS")
    print("="*80)
    print(results_df[['Model', 'MAE Validation', 'Train MAE',
                      'Generalizability (Val MAE - Train MAE)']].to_string(index=False))
    
    # Save results
    results_df.to_csv('Fixed_Stacking_Results.csv', index=False)
    
    print("\n✓ Fixed Stacking complete")
    print(f"  Results saved to: Fixed_Stacking_Results.csv")
    
except Exception as e:
    print(f"\n❌ Fixed Stacking failed: {e}")
    import traceback
    traceback.print_exc()
    print("Continuing with remaining analyses...")

# ============================================================================
# STEP 4: TWO-STAGE CASCADE MODEL (SWELLING RATIO ISSUE)
# ============================================================================

print("\n" + "="*80)
print("STEP 4: TWO-STAGE CASCADE MODEL (SWELLING RATIO ISSUE)")
print("="*80)

try:
    from Two_Stage_Cascade_Model import run_complete_cascade_analysis

    print(f"\nRunning two-stage cascade analysis...")
    print("  This compares 3 models:")
    print("    1. Baseline: Formulation only → Tg")
    print("    2. Stage 1: Formulation → Swelling")
    print("    3. Stage 2: Formulation + Predicted Swelling → Tg")

    # Run cascade analysis
    cascade_results_df, cascade_models = run_complete_cascade_analysis(df, n_estimators=1000)

    print("\n✓ Cascade Model complete")
    print(f"  Results saved to: cascade_model_results.csv")
    print(f"  Models saved to: stage1_swelling_models.joblib, stage2_tg_models.joblib")

except Exception as e:
    print(f"\n❌ Cascade Model failed: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("✓ ALL ANALYSES COMPLETE")
print("="*80)

print("\n📊 GENERATED FILES:")
print("\n1. VIF Analysis:")
print("  - VIF_Analysis_Results.csv")
print("  - Reduced_Feature_Set.txt")
print("  - VIF_Analysis.png/pdf/svg")
print("  - Correlation_Comparison.png/pdf")

print("\n2. Fixed Stacking Ensemble:")
print("  - Fixed_Stacking_Results.csv")
print("  - Fixed_Stacking_Actual_vs_Predicted.png/pdf/svg")
print("  - stacking_results_fixed_run_*.csv")

print("\n3. Cascade Model:")
print("  - cascade_model_results.csv")
print("  - stage1_swelling_models.joblib")
print("  - stage2_tg_models.joblib")

print("\n" + "="*80)
print("📝 NEXT STEPS FOR MANUSCRIPT:")
print("="*80)
print("\n1. Review VIF_Analysis_Results.csv")
print("   → Identify which features to remove (VIF > 10)")
print("   → Document in manuscript methodology")

print("\n2. Review Fixed_Stacking_Results.csv")
print("   → Report honest MAE (expected 10-15°C)")
print("   → Update all figures with new predictions")
print("   → Compare to original results (6.66°C was inflated)")

print("\n3. Review cascade_model_results.csv")
print("   → Compare Baseline vs Cascade performance")
print("   → Discuss trade-offs in manuscript")
print("   → Emphasize predictive design capability")

print("\n4. Integrate manuscript sections:")
print("   → Add mechanistic interpretation (DRAFT_Mechanistic_Interpretation_Section.md)")
print("   → Enhance introduction (DRAFT_Introduction_Enhancement.md)")
print("   → Update methodology with OOF predictions and cascade model")
print("   → Update results with new metrics")

print("\n5. Create response to reviewers")
print("   → Document how each concern was addressed")
print("   → Reference specific results from analyses")

print("\n" + "="*80)
print("🎉 READY FOR MANUSCRIPT REVISION!")
print("="*80)

