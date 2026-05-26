# -*- coding: utf-8 -*-
"""
Complete workflow script to run all mapping steps sequentially
Executes: retrain -> map -> visualize distribution -> visualize density plots
"""

import subprocess
import sys
import os
from datetime import datetime

def run_script(script_name, description):
    """
    Run a Python script and report status.
    
    Args:
        script_name: Name of the script to run
        description: Description of what the script does
    
    Returns:
        bool: True if successful, False otherwise
    """
    print("\n" + "="*80)
    print(f"STEP: {description}")
    print("="*80)
    print(f"Running: {script_name}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-"*80)
    
    start_time = datetime.now()
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            capture_output=False,
            text=True
        )
        
        elapsed = (datetime.now() - start_time).total_seconds()
        print("-"*80)
        print(f"✓ COMPLETED in {elapsed/60:.1f} minutes")
        print("="*80)
        return True
        
    except subprocess.CalledProcessError as e:
        elapsed = (datetime.now() - start_time).total_seconds()
        print("-"*80)
        print(f"✗ FAILED after {elapsed/60:.1f} minutes")
        print(f"Error: {e}")
        print("="*80)
        return False
    except KeyboardInterrupt:
        print("\n\n" + "="*80)
        print("⚠ INTERRUPTED BY USER")
        print("="*80)
        return False

def check_file_exists(filename):
    """Check if a file exists and report its size."""
    if os.path.exists(filename):
        size = os.path.getsize(filename)
        if size > 1e6:
            size_str = f"{size/1e6:.1f} MB"
        elif size > 1e3:
            size_str = f"{size/1e3:.1f} KB"
        else:
            size_str = f"{size} bytes"
        print(f"  ✓ {filename} ({size_str})")
        return True
    else:
        print(f"  ✗ {filename} (not found)")
        return False

def main():
    """Main workflow execution."""
    print("="*80)
    print("COMPLETE MAPPING WORKFLOW - BEST MODEL")
    print("="*80)
    print("This script will run all steps sequentially:")
    print("  1. Retrain the best model")
    print("  2. Generate mapping across feature space")
    print("  3. Visualize Tg distribution")
    print("  4. Create density plots")
    print("\nEstimated total time: 20-50 minutes (depends on hardware)")
    print("="*80)
    
    response = input("\nDo you want to continue? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("Workflow cancelled by user.")
        return
    
    workflow_start = datetime.now()
    
    # Step 1: Retrain model
    success = run_script(
        'retrain_best_model.py',
        'Retrain Best Model (10 estimators, 5 features)'
    )
    
    if not success:
        print("\n❌ Workflow stopped: Model retraining failed")
        return
    
    # Verify model files
    print("\nVerifying model files...")
    required_files = [
        'best_model_base_models.joblib',
        'best_model_meta_model.joblib',
        'best_model_x_scaler.joblib',
        'best_model_y_scaler.joblib',
        'best_model_metadata.json'
    ]
    
    all_present = all(check_file_exists(f) for f in required_files)
    if not all_present:
        print("\n❌ Workflow stopped: Some model files are missing")
        return
    
    # Step 2: Generate mapping
    success = run_script(
        'mapping_best_model.py',
        'Generate Mapping (~2.9M predictions)'
    )
    
    if not success:
        print("\n❌ Workflow stopped: Mapping generation failed")
        return
    
    # Verify mapping file
    print("\nVerifying mapping file...")
    if not check_file_exists('mapped_results_tg_best_model.csv'):
        print("\n❌ Workflow stopped: Mapping file not found")
        return
    
    # Step 3: Visualize distribution
    success = run_script(
        'visualize_distribution_best_model.py',
        'Visualize Tg Distribution'
    )
    
    if not success:
        print("\n⚠ Warning: Distribution visualization failed (continuing anyway)")
    
    # Step 4: Create density plots
    success = run_script(
        'visualize_density_plots_best_model.py',
        'Create Density Plots (5 plots)'
    )
    
    if not success:
        print("\n⚠ Warning: Density plot creation failed (continuing anyway)")
    
    # Final summary
    total_elapsed = (datetime.now() - workflow_start).total_seconds()
    
    print("\n" + "="*80)
    print("WORKFLOW COMPLETED!")
    print("="*80)
    print(f"Total time: {total_elapsed/60:.1f} minutes")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nGenerated files:")
    
    # List all output files
    output_files = [
        'best_model_base_models.joblib',
        'best_model_meta_model.joblib',
        'best_model_x_scaler.joblib',
        'best_model_y_scaler.joblib',
        'best_model_metadata.json',
        'mapped_results_tg_best_model.csv',
        'mapping_summary.json',
        'mapped_results_sample.csv',
        'distribution_tg_best_model.png',
        'distribution_tg_best_model.svg',
        'distribution_tg_best_model.pdf'
    ]
    
    for filename in output_files:
        check_file_exists(filename)
    
    if os.path.exists('density_plots_best_model'):
        print(f"  ✓ density_plots_best_model/ (directory with 5 plots)")
    
    print("\nNext steps:")
    print("  1. Review mapping_summary.json for statistics")
    print("  2. Examine distribution plots to see Tg range")
    print("  3. Analyze density plots to understand feature-Tg relationships")
    print("  4. Use mapped_results_tg_best_model.csv for further analysis")
    print("="*80)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n" + "="*80)
        print("⚠ WORKFLOW INTERRUPTED BY USER")
        print("="*80)
        sys.exit(1)
