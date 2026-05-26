import pandas as pd
import numpy as np
import joblib
import json
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Lasso, ElasticNet, Ridge
import os

# Read the results data
df = pd.read_csv('Fixed_Stacking_Ensemble/fixed_stacking_results_all_combinations.csv')

# Add number of features column
df['Number of Features'] = df['Feature Combination'].apply(lambda x: len(eval(x)))

# Get top 5 performing combinations (by MAE)
top_5 = df.nsmallest(5, 'MAE Validation').copy()

def get_model_details(model):
    """Extract detailed information about a trained model"""
    details = {}
    
    if isinstance(model, GradientBoostingRegressor):
        details['Model Type'] = 'Gradient Boosting Regressor'
        details['N Estimators'] = model.n_estimators
        details['Learning Rate'] = model.learning_rate
        details['Max Depth'] = model.max_depth
        details['Random State'] = model.random_state
        
    elif isinstance(model, RandomForestRegressor):
        details['Model Type'] = 'Random Forest Regressor'
        details['N Estimators'] = model.n_estimators
        details['Max Depth'] = model.max_depth
        details['Min Samples Split'] = model.min_samples_split
        details['Random State'] = model.random_state
        
    elif isinstance(model, SVR):
        details['Model Type'] = 'Support Vector Regressor'
        details['C'] = model.C
        details['Kernel'] = model.kernel
        details['Gamma'] = model.gamma
        
    elif isinstance(model, Lasso):
        details['Model Type'] = 'Lasso Regression'
        details['Alpha'] = model.alpha
        details['Max Iter'] = model.max_iter
        details['Random State'] = model.random_state
        
    elif isinstance(model, ElasticNet):
        details['Model Type'] = 'Elastic Net'
        details['Alpha'] = model.alpha
        details['L1 Ratio'] = model.l1_ratio
        details['Max Iter'] = model.max_iter
        details['Random State'] = model.random_state
        
    elif isinstance(model, Ridge):
        details['Model Type'] = 'Ridge Regression (Meta-model)'
        details['Alpha'] = model.alpha
        details['Random State'] = model.random_state
        
    else:
        details['Model Type'] = str(type(model).__name__)
    
    return details

def format_hyperparams(details):
    """Format hyperparameters for display"""
    if details['Model Type'] == 'Gradient Boosting Regressor':
        return f"n_estimators={details['N Estimators']}, lr={details['Learning Rate']}, max_depth={details['Max Depth']}"
    elif details['Model Type'] == 'Random Forest Regressor':
        return f"n_estimators={details['N Estimators']}, max_depth={details['Max Depth']}, min_samples_split={details['Min Samples Split']}"
    elif details['Model Type'] == 'Support Vector Regressor':
        return f"C={details['C']}, kernel={details['Kernel']}, gamma={details['Gamma']}"
    elif details['Model Type'] == 'Lasso Regression':
        return f"alpha={details['Alpha']}, max_iter={details['Max Iter']}"
    elif details['Model Type'] == 'Elastic Net':
        return f"alpha={details['Alpha']}, l1_ratio={details['L1 Ratio']}, max_iter={details['Max Iter']}"
    elif details['Model Type'] == 'Ridge Regression (Meta-model)':
        return f"alpha={details['Alpha']}"
    else:
        return "N/A"

# Create detailed analysis
detailed_results = []

print("Analyzing saved models for top 5 combinations...")
print("="*80)

for idx, row in top_5.iterrows():
    rank = list(top_5.index).index(idx) + 1
    feature_combo = row['Feature Combination']
    feature_str = '_'.join(eval(feature_combo)).replace(' ', '_').replace('(', '').replace(')', '').replace('%', 'pct')
    
    print(f"\nRank {rank}: {feature_combo}")
    print("-" * 60)
    
    # Try to load the saved models
    try:
        # Load base models
        base_models = joblib.load(f'Fixed_Stacking_Ensemble/base_models_fixed_run_1_{feature_str}.joblib')
        meta_model = joblib.load(f'Fixed_Stacking_Ensemble/meta_model_fixed_run_1_{feature_str}.joblib')
        
        # Extract base model details
        base_model_details = []
        for i, model in enumerate(base_models):
            details = get_model_details(model)
            hyperparams = format_hyperparams(details)
            base_model_details.append({
                'Base Model': details['Model Type'],
                'Hyperparameters': hyperparams
            })
        
        # Extract meta model details
        meta_details = get_model_details(meta_model)
        meta_hyperparams = format_hyperparams(meta_details)
        
        # Add to results
        detailed_results.append({
            'Rank': rank,
            'Features Used': ', '.join(eval(feature_combo)),
            'Number of Features': int(row['Number of Features']),
            'MAE Validation (°C)': f"{row['MAE Validation']:.3f}",
            'R² Validation': f"{row['R-squared Validation']:.3f}",
            'Base Models': str([bm['Base Model'] for bm in base_model_details]),
            'Base Model Hyperparameters': str([bm['Hyperparameters'] for bm in base_model_details]),
            'Meta Model': meta_details['Model Type'],
            'Meta Model Hyperparameters': meta_hyperparams,
            'Generalization Gap (°C)': f"{row['MAE Validation'] - row['Train MAE']:.3f}"
        })
        
        print(f"✓ Successfully loaded models for rank {rank}")
        
        # Print detailed information
        print(f"Base Models ({len(base_models)}):")
        for i, bm in enumerate(base_model_details):
            print(f"  {i+1}. {bm['Base Model']}")
            print(f"     Hyperparameters: {bm['Hyperparameters']}")
        
        print(f"Meta Model: {meta_details['Model Type']}")
        print(f"  Hyperparameters: {meta_hyperparams}")
        
    except FileNotFoundError as e:
        print(f"✗ Could not load models for rank {rank}: {e}")
        # Add placeholder information
        detailed_results.append({
            'Rank': rank,
            'Features Used': ', '.join(eval(feature_combo)),
            'Number of Features': int(row['Number of Features']),
            'MAE Validation (°C)': f"{row['MAE Validation']:.3f}",
            'R² Validation': f"{row['R-squared Validation']:.3f}",
            'Base Models': 'Models not found',
            'Base Model Hyperparameters': 'N/A',
            'Meta Model': 'N/A',
            'Meta Model Hyperparameters': 'N/A',
            'Generalization Gap (°C)': f"{row['MAE Validation'] - row['Train MAE']:.3f}"
        })

# Create DataFrame for display
detailed_df = pd.DataFrame(detailed_results)

# Save detailed table
detailed_df.to_csv('top_5_detailed_model_analysis.csv', index=False)

print("\n" + "="*80)
print("DETAILED MODEL ANALYSIS - TOP 5 PERFORMING COMBINATIONS")
print("="*80)

# Display formatted table
for _, row in detailed_df.iterrows():
    print(f"\nRank {row['Rank']}:")
    print(f"  Features: {row['Features Used']}")
    print(f"  Number of Features: {row['Number of Features']}")
    print(f"  MAE Validation: {row['MAE Validation (°C)']}°C")
    print(f"  R² Validation: {row['R² Validation']}")
    print(f"  Generalization Gap: {row['Generalization Gap (°C)']}°C")
    print(f"  Base Models: {row['Base Models']}")
    print(f"  Meta Model: {row['Meta Model']}")
    print(f"  Meta Hyperparameters: {row['Meta Model Hyperparameters']}")

# Create a more readable summary table
summary_data = []
for _, row in detailed_df.iterrows():
    summary_data.append({
        'Rank': row['Rank'],
        'Features': row['Features Used'][:60] + '...' if len(row['Features Used']) > 60 else row['Features Used'],
        'Num Features': row['Number of Features'],
        'MAE (°C)': row['MAE Validation (°C)'],
        'R²': row['R² Validation'],
        'Base Models': len(eval(row['Base Models'])) if row['Base Models'] != 'Models not found' else 'N/A',
        'Meta Model': row['Meta Model'],
        'Gap (°C)': row['Generalization Gap (°C)']
    })

summary_df = pd.DataFrame(summary_data)
print(f"\n" + "="*80)
print("SUMMARY TABLE")
print("="*80)
print(summary_df.to_string(index=False))

# Save summary table
summary_df.to_csv('top_5_summary_table.csv', index=False)

print(f"\nFiles saved:")
print(f"  - top_5_detailed_model_analysis.csv (full details)")
print(f"  - top_5_summary_table.csv (summary view)")
print("="*80)
