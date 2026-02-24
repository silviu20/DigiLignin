# -*- coding: utf-8 -*-
"""
Setup Verification Script
Checks all prerequisites before running the experiment
"""

import sys
import os
from pathlib import Path

def check_python_version():
    """Check Python version."""
    print("Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 7:
        print(f"  ✓ Python {version.major}.{version.minor}.{version.micro} (OK)")
        return True
    else:
        print(f"  ✗ Python {version.major}.{version.minor}.{version.micro} (Need 3.7+)")
        return False

def check_packages():
    """Check required packages."""
    print("\nChecking required packages...")
    
    required_packages = {
        'numpy': 'numpy',
        'pandas': 'pandas',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'sklearn': 'scikit-learn',
        'scipy': 'scipy',
        'joblib': 'joblib'
    }
    
    all_ok = True
    for module_name, package_name in required_packages.items():
        try:
            __import__(module_name)
            print(f"  ✓ {package_name}")
        except ImportError:
            print(f"  ✗ {package_name} (Not installed)")
            all_ok = False
    
    return all_ok

def check_dataset():
    """Check dataset availability."""
    print("\nChecking dataset...")
    
    dataset_path = Path('../Fixed_Stacking_Ensemble/dataset.xlsx')
    
    if dataset_path.exists():
        print(f"  ✓ Dataset found at: {dataset_path.absolute()}")
        
        # Try to load it
        try:
            import pandas as pd
            df = pd.read_excel(dataset_path)
            print(f"  ✓ Dataset loaded successfully")
            print(f"    Shape: {df.shape}")
            
            # Check for required columns
            required_cols = ['Tg(deg C)', 'Lignin (wt%)', 'Co-polyol type (PTHF)', 'r']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                print(f"  ✗ Missing required columns: {missing_cols}")
                return False
            else:
                print(f"  ✓ All required columns present")
                return True
                
        except Exception as e:
            print(f"  ✗ Error loading dataset: {e}")
            return False
    else:
        print(f"  ✗ Dataset not found at: {dataset_path.absolute()}")
        print(f"    Expected location: C:\\Users\\sacaru\\digilignin\\DigiLignin\\4.Wrapper\\Fixed_Stacking_Ensemble\\dataset.xlsx")
        return False

def check_preprocessing_module():
    """Check preprocessing module availability."""
    print("\nChecking preprocessing module...")
    
    preprocessing_path = Path('../../1.Loading and Preprocessing/Loading and preprocessing.py')
    
    if preprocessing_path.exists():
        print(f"  ✓ Preprocessing module found at: {preprocessing_path.absolute()}")
        
        # Try to import it
        try:
            sys.path.append(str(preprocessing_path.parent))
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "loading_preprocessing", 
                preprocessing_path
            )
            loading_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(loading_module)
            
            # Check for required functions
            if hasattr(loading_module, 'read_csv_with_encoding') and \
               hasattr(loading_module, 'map_categorical_values'):
                print(f"  ✓ Required functions found")
                return True
            else:
                print(f"  ✗ Required functions not found in module")
                return False
                
        except Exception as e:
            print(f"  ✗ Error importing preprocessing module: {e}")
            return False
    else:
        print(f"  ✗ Preprocessing module not found at: {preprocessing_path.absolute()}")
        return False

def check_directory_structure():
    """Check directory structure."""
    print("\nChecking directory structure...")
    
    models_dir = Path('models')
    
    if models_dir.exists():
        print(f"  ✓ models/ directory exists")
    else:
        print(f"  ! models/ directory not found, creating...")
        try:
            models_dir.mkdir(parents=True, exist_ok=True)
            print(f"  ✓ models/ directory created")
        except Exception as e:
            print(f"  ✗ Error creating models/ directory: {e}")
            return False
    
    return True

def check_disk_space():
    """Check available disk space."""
    print("\nChecking disk space...")
    
    try:
        import shutil
        total, used, free = shutil.disk_usage(os.getcwd())
        
        free_gb = free / (1024**3)
        print(f"  Available space: {free_gb:.2f} GB")
        
        if free_gb < 2:
            print(f"  ⚠ Warning: Low disk space (need ~2GB for all models)")
            return False
        else:
            print(f"  ✓ Sufficient disk space")
            return True
            
    except Exception as e:
        print(f"  ! Could not check disk space: {e}")
        return True  # Don't fail on this

def estimate_runtime():
    """Estimate runtime."""
    print("\nRuntime Estimation...")
    
    n_combinations = 511  # 2^9 - 1 combinations
    n_estimators_values = 13  # Number of n_estimators to test
    total_runs = n_combinations * n_estimators_values
    
    # Estimate 2-3 minutes per run on average
    min_minutes = total_runs * 2
    max_minutes = total_runs * 5
    
    print(f"  Total combinations: {n_combinations}")
    print(f"  N_estimators values: {n_estimators_values}")
    print(f"  Total runs: {total_runs}")
    print(f"  Estimated time: {min_minutes/60:.1f} - {max_minutes/60:.1f} hours")
    print(f"  Note: Actual time varies by hardware and n_estimators value")

def main():
    """Main verification function."""
    print("="*80)
    print("SETUP VERIFICATION FOR FIXED SPLIT EXPERIMENT")
    print("="*80)
    
    checks = []
    
    # Run all checks
    checks.append(("Python Version", check_python_version()))
    checks.append(("Required Packages", check_packages()))
    checks.append(("Dataset", check_dataset()))
    checks.append(("Preprocessing Module", check_preprocessing_module()))
    checks.append(("Directory Structure", check_directory_structure()))
    checks.append(("Disk Space", check_disk_space()))
    
    # Runtime estimation
    estimate_runtime()
    
    # Summary
    print("\n" + "="*80)
    print("VERIFICATION SUMMARY")
    print("="*80)
    
    all_passed = True
    for check_name, result in checks:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {check_name:.<30} {status}")
        if not result:
            all_passed = False
    
    print("="*80)
    
    if all_passed:
        print("\n✓ All checks passed! You're ready to run the experiment.")
        print("\nNext steps:")
        print("  1. Run: python run_fixed_split_experiments.py")
        print("  2. Wait for completion (15-40 hours)")
        print("  3. Run: python compare_with_oof.py")
        print("\nFor quick testing, edit run_fixed_split_experiments.py to use fewer combinations.")
    else:
        print("\n✗ Some checks failed. Please fix the issues above before running.")
        print("\nCommon fixes:")
        print("  - Install missing packages: pip install <package_name>")
        print("  - Verify dataset location")
        print("  - Check file paths")
    
    print("="*80)
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
