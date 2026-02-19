#!/usr/bin/env python3
"""
Check if all required Python packages are installed.
"""

import sys

print("Checking required Python packages...\n")

required_packages = {
    'numpy': 'numpy',
    'pandas': 'pandas',
    'matplotlib': 'matplotlib',
    'seaborn': 'seaborn',
    'sklearn': 'scikit-learn',
    'scipy': 'scipy',
    'statsmodels': 'statsmodels',
    'joblib': 'joblib',
    'openpyxl': 'openpyxl'  # For reading Excel files
}

missing_packages = []
installed_packages = []

for module_name, package_name in required_packages.items():
    try:
        __import__(module_name)
        installed_packages.append(f"✓ {package_name}")
    except ImportError:
        missing_packages.append(package_name)
        print(f"❌ {package_name} - NOT INSTALLED")

print("\nInstalled packages:")
for pkg in installed_packages:
    print(f"  {pkg}")

if missing_packages:
    print("\n" + "="*60)
    print("⚠ MISSING PACKAGES")
    print("="*60)
    print("\nThe following packages need to be installed:")
    for pkg in missing_packages:
        print(f"  - {pkg}")
    
    print("\nTo install all missing packages, run:")
    print(f"  pip install {' '.join(missing_packages)}")
    print("\nOr install all at once:")
    print("  pip install numpy pandas matplotlib seaborn scikit-learn scipy statsmodels joblib openpyxl")
    sys.exit(1)
else:
    print("\n" + "="*60)
    print("✓ ALL REQUIRED PACKAGES INSTALLED")
    print("="*60)
    print("\nYou're ready to run the analyses!")
    print("Execute: python RUN_ALL_ANALYSES.py")

