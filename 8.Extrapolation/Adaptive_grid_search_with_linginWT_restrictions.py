import numpy as np
import pandas as pd
import itertools
import joblib
from sklearn.neighbors import NearestNeighbors

def predict_tg(combo, base_models, meta_model, X_scaler, y_scaler):
    """
    Predict Tg for a given combination of input parameters
    """
    input_data = pd.DataFrame([combo], columns=['Lignin (wt%)', 'Co-polyol type (PTHF)', 'Ratio', 'Co-polyol (wt%)', 
                    'Isocyanate (mmol NCO)', 'Tin(II) octoate', 'Swelling ratio (%)'])
    input_data_scaled = X_scaler.transform(input_data)
    base_predictions = np.column_stack([model.predict(input_data_scaled) for model in base_models])
    prediction_scaled = meta_model.predict(base_predictions)
    prediction = y_scaler.inverse_transform(prediction_scaled.reshape(-1, 1))[0][0]
    return prediction

def is_valid_composition(lignin_wt, copolyol_wt, tolerance=1e-10):
    """
    Check if the composition sums exactly to 100%
    Added tolerance to account for floating point arithmetic
    """
    return abs((lignin_wt + copolyol_wt) - 100.0) <= tolerance

def adaptive_grid_search(target_tg, base_models, meta_model, X_scaler, y_scaler, n_iterations=2):
    """
    Perform adaptive grid search to find parameters that yield the target Tg,
    with constraints that Lignin + Co-polyol must equal 100%
    """
    input_params = ['Lignin (wt%)', 'Co-polyol type (PTHF)', 'Ratio', 'Co-polyol (wt%)', 
                    'Isocyanate (mmol NCO)', 'Tin(II) octoate', 'Swelling ratio (%)']
    
    # For Lignin, we'll create a range of points from 0 to 100
    lignin_points = np.linspace(0, 100, 4)
    
    # Other grid points
    grid_points = [
        [250, 650, 1000],  # 'Co-polyol type (PTHF)'
        np.linspace(0.6, 1.4, 5),  # 'Ratio'
        np.linspace(0, 20, 5),  # 'Isocyanate (mmol NCO)'
        np.linspace(0, 2, 4),  # 'Tin(II) octoate'
        np.linspace(0, 472, 5)  # 'Swelling ratio (%)'
    ]

    best_params = None
    best_tg_diff = float('inf')

    for iteration in range(n_iterations):
        valid_combos = []
        
        # For each Lignin percentage, calculate the corresponding Co-polyol percentage
        for lignin_wt in lignin_points:
            copolyol_wt = 100 - lignin_wt  # Co-polyol is always 100% - Lignin%
            
            # Skip if either percentage is negative
            if lignin_wt < 0 or copolyol_wt < 0:
                continue
                
            # Generate combinations for other parameters
            for other_params in itertools.product(*grid_points):
                # Insert Co-polyol wt% at the correct position (after PTHF type)
                combo = (lignin_wt, other_params[0], other_params[1], copolyol_wt) + other_params[2:]
                valid_combos.append(combo)
        
        # If no valid combinations found, adjust the grid
        if not valid_combos:
            print(f"Warning: No valid combinations found in iteration {iteration + 1}")
            continue
        
        # Evaluate all valid combinations
        for combo in valid_combos:
            predicted_tg = predict_tg(combo, base_models, meta_model, X_scaler, y_scaler)
            tg_diff = abs(predicted_tg - target_tg)
            
            if tg_diff < best_tg_diff:
                best_tg_diff = tg_diff
                best_params = combo
        
        # Refine the grid around the best parameters for next iteration
        if best_params is not None:
            lignin_wt = best_params[0]
            lignin_points = np.linspace(
                max(0, lignin_wt - 25),
                min(100, lignin_wt + 25),
                4
            )
            
            # Update other grid points
            new_grid_points = []
            for i, (param, gp) in enumerate(zip(best_params[1:], grid_points)):
                if isinstance(gp, np.ndarray):
                    span = (gp[1] - gp[0]) / 2
                    min_val = max(gp[0], param - span)
                    max_val = min(gp[-1], param + span)
                    new_grid_points.append(np.linspace(min_val, max_val, len(gp)))
                else:
                    new_grid_points.append(gp)
            grid_points = new_grid_points

    if best_params is None:
        raise ValueError("No valid parameter combination found")

    return dict(zip(input_params, best_params)), predict_tg(best_params, base_models, meta_model, X_scaler, y_scaler)

def find_closest_inputs_adaptive_grid(target_tgs, base_models, meta_model, X_scaler, y_scaler):
    """
    Find closest inputs for multiple target Tg values using adaptive grid search
    """
    closest_inputs = []
    for target_tg in target_tgs:
        print(f"Processing target Tg: {target_tg}")
        try:
            best_params, predicted_tg = adaptive_grid_search(target_tg, base_models, meta_model, X_scaler, y_scaler)
            best_params['Target_Tg'] = target_tg
            best_params['Predicted_Tg'] = predicted_tg
            best_params['Total_wt%'] = best_params['Lignin (wt%)'] + best_params['Co-polyol (wt%)']
            closest_inputs.append(best_params)
        except ValueError as e:
            print(f"Warning: Could not find valid parameters for Tg = {target_tg}")
            continue
    
    return pd.DataFrame(closest_inputs)



# Load the saved models and scalers
run_number = 1  # Change this to the appropriate run number
base_models = joblib.load(f'base_models_1_best_Lignin_wtpct_Co-polyol_type_PTHF_Ratio_Co-polyol_wtpct_Isocyanate_mmol_NCO_TinII_octoate_Swelling_ratio_pct.joblib')
meta_model = joblib.load(f'meta_model_{run_number}_best_Lignin_wtpct_Co-polyol_type_PTHF_Ratio_Co-polyol_wtpct_Isocyanate_mmol_NCO_TinII_octoate_Swelling_ratio_pct.joblib')
X_scaler = joblib.load(f'X_scaler_{run_number}_best_Lignin_wtpct_Co-polyol_type_PTHF_Ratio_Co-polyol_wtpct_Isocyanate_mmol_NCO_TinII_octoate_Swelling_ratio_pct.joblib')
y_scaler = joblib.load(f'y_scaler_{run_number}_best_Lignin_wtpct_Co-polyol_type_PTHF_Ratio_Co-polyol_wtpct_Isocyanate_mmol_NCO_TinII_octoate_Swelling_ratio_pct.joblib')

# Define target Tg values - reduced range for testing
# target_tgs = [-17, 0, 20, 40, 60, 80, 100]  # Just testing a few key points
# target_tgs = list(range(-17, 100))  # From -17 to 100
target_tgs = list(np.linspace(-17, 100, 50))

# Find closest inputs for all target Tgs
closest_inputs = find_closest_inputs_adaptive_grid(target_tgs, base_models, meta_model, X_scaler, y_scaler)

# Save results
closest_inputs.to_csv('closest_inputs_test_results.csv', index=False)
print("Test results saved to 'closest_inputs_test_results.csv'")