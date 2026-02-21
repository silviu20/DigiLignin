#!/bin/bash
# Run analyses step by step with progress tracking

echo "================================================================================"
echo "STEP-BY-STEP ANALYSIS EXECUTION"
echo "================================================================================"
echo ""

# Step 1: VIF Analysis
echo "STEP 1/3: Running VIF Analysis..."
python3 -c ""
import sys
sys.path.append('5.Model')
import pandas as pd
import numpy as np

# Load data
df = pd.read_excel('dataset.csv.xlsx')
df = df.rename(columns={
    'Copolyol (wt%)': 'Co-polyol (wt%)',
    'Isocyonate type': 'Isocyanate type',
    'r': 'Ratio',
    'tin(II) octoate': 'Tin(II) octoate',
    'Tg(deg C)': 'Tg (°C)',
    'Sratio(%)': 'Swelling ratio (%)'
})
df = df.dropna(subset=['Tg (°C)'])

# Encode isocyanate type
isocyanate_mapping = {'N3600': 1, 'HDI': 0, 0: np.nan}
df['Isocyanate type'] = df['Isocyanate type'].map(isocyanate_mapping).fillna(0)

print(f'Data loaded: {len(df)} samples')

# Run VIF analysis
from VIF_Analysis_Multicollinearity import calculate_vif, plot_vif_results, recommend_feature_reduction, propose_reduced_feature_set

formulation_features = [
    'Lignin (wt%)', 'Co-polyol (wt%)', 'Co-polyol type (PTHF)',
    'Isocyanate (wt%)', 'Isocyanate (mmol NCO)', 'Isocyanate type',
    'Ratio', 'Tin(II) octoate'
]

vif_df = calculate_vif(df, formulation_features)
plot_vif_results(vif_df)
recommendations = recommend_feature_reduction(vif_df, threshold=10)
reduced_features, vif_history = propose_reduced_feature_set(vif_df, df)

# Save results
vif_df.to_csv('VIF_Analysis_Results.csv', index=False)
with open('Reduced_Feature_Set.txt', 'w') as f:
    f.write('REDUCED FEATURE SET (VIF < 10)\n')
    f.write('='*50 + '\n\n')
    for feat in reduced_features:
        f.write(f'{feat}\n')

print('✓ VIF Analysis complete')
print(f'  Results: VIF_Analysis_Results.csv')
print(f'  Plots: VIF_Analysis.png/pdf')
print(f'  Reduced features: {len(reduced_features)}/{len(formulation_features)}')
" 2>&1 | tee step1_vif.log

if [ $? -eq 0 ]; then
    echo "✓ Step 1 complete"
else
    echo "✗ Step 1 failed - check step1_vif.log"
    exit 1
fi

echo ""
echo "================================================================================"
echo "STEP 2/3: Running Fixed Stacking Ensemble..."
python3 -c "
import sys
sys.path.append('5.Model')
import pandas as pd
import numpy as np

# Load data
df = pd.read_excel('dataset.csv.xlsx')
df = df.rename(columns={
    'Copolyol (wt%)': 'Co-polyol (wt%)',
    'Isocyonate type': 'Isocyanate type',
    'r': 'Ratio',
    'tin(II) octoate': 'Tin(II) octoate',
    'Tg(deg C)': 'Tg (°C)',
    'Sratio(%)': 'Swelling ratio (%)'
})
df = df.dropna(subset=['Tg (°C)'])
isocyanate_mapping = {'N3600': 1, 'HDI': 0, 0: np.nan}
df['Isocyanate type'] = df['Isocyanate type'].map(isocyanate_mapping).fillna(0)

# Load reduced features
with open('Reduced_Feature_Set.txt', 'r') as f:
    lines = f.readlines()
    reduced_features = [line.strip() for line in lines[3:] if line.strip()]

# Run fixed stacking
from Stacked_Ensembles_Fixed import run_multiple_times_fixed

x_features = reduced_features + ['Swelling ratio (%)']
x = df[x_features]
y = df[['Tg (°C)']]

print(f'Running fixed stacking with {len(x_features)} features...')
results_df, best_models = run_multiple_times_fixed(x, y, num_runs=1, n_estimators_list=[1000])

results_df.to_csv('Fixed_Stacking_Results.csv', index=False)
print('✓ Fixed Stacking complete')
print(f'  Results: Fixed_Stacking_Results.csv')
" 2>&1 | tee step2_stacking.log

if [ $? -eq 0 ]; then
    echo "✓ Step 2 complete"
else
    echo "✗ Step 2 failed - check step2_stacking.log"
    exit 1
fi

echo ""
echo "================================================================================"
echo "STEP 3/3: Running Cascade Model..."
python3 -c "
import sys
sys.path.append('5.Model')
import pandas as pd
import numpy as np

# Load data
df = pd.read_excel('dataset.csv.xlsx')
df = df.rename(columns={
    'Copolyol (wt%)': 'Co-polyol (wt%)',
    'Isocyonate type': 'Isocyanate type',
    'r': 'Ratio',
    'tin(II) octoate': 'Tin(II) octoate',
    'Tg(deg C)': 'Tg (°C)',
    'Sratio(%)': 'Swelling ratio (%)'
})
df = df.dropna(subset=['Tg (°C)'])
isocyanate_mapping = {'N3600': 1, 'HDI': 0, 0: np.nan}
df['Isocyanate type'] = df['Isocyanate type'].map(isocyanate_mapping).fillna(0)

# Run cascade analysis
from Two_Stage_Cascade_Model import run_complete_cascade_analysis

print('Running two-stage cascade analysis...')
cascade_results_df, cascade_models = run_complete_cascade_analysis(df, n_estimators=1000)

print('✓ Cascade Model complete')
print(f'  Results: cascade_model_results.csv')
" 2>&1 | tee step3_cascade.log

if [ $? -eq 0 ]; then
    echo "✓ Step 3 complete"
else
    echo "✗ Step 3 failed - check step3_cascade.log"
    exit 1
fi

echo ""
echo "================================================================================"
echo "✓ ALL ANALYSES COMPLETE!"
echo "================================================================================"
echo ""
echo "Generated files:"
ls -lh VIF_Analysis* Fixed_Stacking* cascade_model* Reduced_Feature_Set.txt 2>/dev/null
echo ""
echo "Check the log files for details:"
echo "  - step1_vif.log"
echo "  - step2_stacking.log"
echo "  - step3_cascade.log"

