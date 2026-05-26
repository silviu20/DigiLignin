# -*- coding: utf-8 -*-
"""
Test script to verify checkpoint and resume functionality
Tests with a small subset of combinations
"""

import numpy as np
import pandas as pd
import sys
import os

# Modify the main script to test with fewer combinations
def create_test_script():
    """Create a test version with limited combinations."""
    
    # Read the main script
    with open('run_fixed_split_experiments.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Modify to use only first 2 combinations and 3 n_estimators values
    test_content = content.replace(
        "estimator_values = [1, 10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]",
        "estimator_values = [1, 10, 50]  # TEST: Limited values"
    )
    
    # Add a limiter for combinations
    test_content = test_content.replace(
        "for combo_idx, feature_combination in enumerate(all_combinations):",
        "for combo_idx, feature_combination in enumerate(all_combinations[:2]):  # TEST: Only first 2 combinations"
    )
    
    # Save test version
    with open('test_run.py', 'w', encoding='utf-8') as f:
        f.write(test_content)
    
    print("✓ Created test_run.py with limited combinations")
    print("  - Testing only first 2 feature combinations")
    print("  - Testing only 3 n_estimators values: [1, 10, 50]")
    print("  - Total experiments: 2 × 3 = 6")
    print("\nTo test checkpoint functionality:")
    print("  1. Run: python test_run.py")
    print("  2. Stop it after first combination (Ctrl+C)")
    print("  3. Run again: python test_run.py")
    print("  4. It should resume from where it stopped")

if __name__ == "__main__":
    create_test_script()
