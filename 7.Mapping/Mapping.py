# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 09:36:23 2024

@author: P70090917
"""

import numpy as np
import pandas as pd
import itertools
import joblib

def map_target_batch(base_models, meta_model, X_scaler, y_scaler, batch_size=10000):
    feature_values = [
        np.arange(0, 70, 1),  # 'Lignin (wt%)'
        [250, 650, 1000],  # 'Co-polyol type (PTHF)'
        np.arange(0.6, 1.4 + 0.2, 0.05),  # 'Ratio'
        np.linspace(0, 66, 5),  # 'Co-polyol (wt%)',
        np.arange(0, 20, 0.4),  # 'Isocyanate (mmol NCO)'
        np.arange(0, 2, 0.5),  # 'Tin(II) octoate'
        np.linspace(0, 472, 2)  # 'Swelling ratio (%)'
    ]
    
    # Define input columns
    input_columns = ['Lignin (wt%)', 'Co-polyol type (PTHF)', 'Ratio', 'Co-polyol (wt%)', 
                    'Isocyanate (mmol NCO)', 'Tin(II) octoate', 'Swelling ratio (%)']
    
    # All columns including the target
    all_columns = input_columns + ['Tg (°C)']
    
    results = pd.DataFrame(columns=all_columns)
    
    total_combinations = np.prod([len(fv) for fv in feature_values])
    print(f"Total combinations to process: {total_combinations}")
    
    for i, combo in enumerate(itertools.product(*feature_values)):
        if i % batch_size == 0:
            print(f"Processing batch {i // batch_size + 1}")
            batch = []
        
        input_data = pd.DataFrame([combo], columns=input_columns)
        input_data_scaled = X_scaler.transform(input_data)
        
        base_predictions = np.column_stack([model.predict(input_data_scaled) for model in base_models])
        prediction_scaled = meta_model.predict(base_predictions)
        prediction = y_scaler.inverse_transform(prediction_scaled.reshape(-1, 1))[0][0]
        
        batch.append((*combo, prediction))
        
        if (i + 1) % batch_size == 0 or i == total_combinations - 1:
            batch_df = pd.DataFrame(batch, columns=all_columns)
            results = pd.concat([results, batch_df], ignore_index=True)
            batch = []
    
    return results

# Rest of your code remains the same
run_number = 1
base_models = joblib.load(f'base_models_{run_number}_best_Lignin_wtpct_Co-polyol_type_PTHF_Ratio_Co-polyol_wtpct_Isocyanate_mmol_NCO_TinII_octoate_Swelling_ratio_pct.joblib')
meta_model = joblib.load(f'meta_model_{run_number}_best_Lignin_wtpct_Co-polyol_type_PTHF_Ratio_Co-polyol_wtpct_Isocyanate_mmol_NCO_TinII_octoate_Swelling_ratio_pct.joblib')
X_scaler = joblib.load(f'X_scaler_{run_number}_best_Lignin_wtpct_Co-polyol_type_PTHF_Ratio_Co-polyol_wtpct_Isocyanate_mmol_NCO_TinII_octoate_Swelling_ratio_pct.joblib')
y_scaler = joblib.load(f'y_scaler_{run_number}_best_Lignin_wtpct_Co-polyol_type_PTHF_Ratio_Co-polyol_wtpct_Isocyanate_mmol_NCO_TinII_octoate_Swelling_ratio_pct.joblib')

# Map the target using all feature variations
mapped_results = map_target_batch(base_models, meta_model, X_scaler, y_scaler)

# Save the mapped results
mapped_results.to_csv('mapped_results_tg.csv', index=False)
print("Mapped results saved to 'mapped_results_tg.csv'")