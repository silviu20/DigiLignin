# -*- coding: utf-8 -*-
"""
Adaptive Grid Search for Best Model (5 features)
Finds optimal input parameters to achieve target Tg values
Features: Lignin (wt%), Co-polyol type (PTHF), r, Copolyol (wt%), Isocyanate (wt%)

Constraint: Lignin (wt%) + Copolyol (wt%) = 100%
"""

import numpy as np
import pandas as pd
import itertools
import joblib
import sys
import os

def predict_tg(combo, base_models, meta_model, X_scaler, y_scaler):
    """
    Predict Tg for a given combination of input parameters.
    
    Args:
        combo: Tuple of (Lignin wt%, Co-polyol type, r, Copolyol wt%, Isocyanate wt%)
        base_models: List of trained base models
        meta_model: Trained meta-model
        X_scaler: Feature scaler
        y_scaler: Target scaler
    
    Returns:
        float: Predicted Tg value
    """
    input_data = pd.DataFrame([combo], columns=['Lignin (wt%)', 'Co-polyol type (PTHF)', 'r', 
                                                  'Copolyol (wt%)', 'Isocyanate (wt%)'])
    input_data_scaled = X_scaler.transform(input_data)
    base_predictions = np.column_stack([model.predict(input_data_scaled) for model in base_models])
    prediction_scaled = meta_model.predict(base_predictions)
    prediction = y_scaler.inverse_transform(prediction_scaled.reshape(-1, 1))[0][0]
    return prediction

def is_valid_composition(lignin_wt, copolyol_wt, tolerance=1e-10):
    """
    Check if the composition sums exactly to 100%.
    
    Args:
        lignin_wt: Lignin weight percentage
        copolyol_wt: Copolyol weight percentage
        tolerance: Tolerance for floating point comparison
    
    Returns:
        bool: True if composition is valid
    """
    return abs((lignin_wt + copolyol_wt) - 100.0) <= tolerance

def adaptive_grid_search(target_tg, base_models, meta_model, X_scaler, y_scaler, n_iterations=3):
    """
    Perform adaptive grid search to find parameters that yield the target Tg.
    Constraint: Lignin + Copolyol must equal 100%
    
    Args:
        target_tg: Target glass transition temperature
        base_models: List of trained base models
        meta_model: Trained meta-model
        X_scaler: Feature scaler
        y_scaler: Target scaler
        n_iterations: Number of refinement iterations
    
    Returns:
        tuple: (best_params dict, predicted_tg)
    """
    input_params = ['Lignin (wt%)', 'Co-polyol type (PTHF)', 'r', 'Copolyol (wt%)', 'Isocyanate (wt%)']
    
    # Initial grid points
    lignin_points = np.linspace(0, 100, 5)  # Lignin from 0 to 100%
    
    # Other grid points (excluding Copolyol which is calculated)
    grid_points = [
        [250, 650, 1000],           # 'Co-polyol type (PTHF)' - discrete values
        np.linspace(0.6, 1.4, 5),   # 'r' - ratio
        np.linspace(0, 20, 5),      # 'Isocyanate (wt%)'
    ]

    best_params = None
    best_tg_diff = float('inf')

    for iteration in range(n_iterations):
        valid_combos = []
        
        # For each Lignin percentage, calculate the corresponding Copolyol percentage
        for lignin_wt in lignin_points:
            copolyol_wt = 100 - lignin_wt  # Copolyol is always 100% - Lignin%
            
            # Skip if either percentage is negative
            if lignin_wt < 0 or copolyol_wt < 0:
                continue
                
            # Generate combinations for other parameters
            for other_params in itertools.product(*grid_points):
                # Create combo: (Lignin, PTHF type, r, Copolyol, Isocyanate)
                combo = (lignin_wt, other_params[0], other_params[1], copolyol_wt, other_params[2])
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
        if best_params is not None and iteration < n_iterations - 1:
            lignin_wt = best_params[0]
            
            # Refine lignin range
            lignin_span = 25 / (iteration + 1)  # Reduce span each iteration
            lignin_points = np.linspace(
                max(0, lignin_wt - lignin_span),
                min(100, lignin_wt + lignin_span),
                5
            )
            
            # Update other grid points (skip PTHF type as it's discrete)
            new_grid_points = []
            
            # PTHF type stays the same (discrete values)
            new_grid_points.append(grid_points[0])
            
            # Refine r (ratio)
            r_val = best_params[2]
            r_span = (1.4 - 0.6) / (2 * (iteration + 2))
            new_grid_points.append(np.linspace(
                max(0.6, r_val - r_span),
                min(1.4, r_val + r_span),
                5
            ))
            
            # Refine Isocyanate
            iso_val = best_params[4]
            iso_span = 20 / (2 * (iteration + 2))
            new_grid_points.append(np.linspace(
                max(0, iso_val - iso_span),
                min(20, iso_val + iso_span),
                5
            ))
            
            grid_points = new_grid_points

    if best_params is None:
        raise ValueError("No valid parameter combination found")

    return dict(zip(input_params, best_params)), predict_tg(best_params, base_models, meta_model, X_scaler, y_scaler)

def find_closest_inputs_adaptive_grid(target_tgs, base_models, meta_model, X_scaler, y_scaler):
    """
    Find closest inputs for multiple target Tg values using adaptive grid search.
    
    Args:
        target_tgs: List of target Tg values
        base_models: List of trained base models
        meta_model: Trained meta-model
        X_scaler: Feature scaler
        y_scaler: Target scaler
    
    Returns:
        pd.DataFrame: Results with optimal parameters for each target Tg
    """
    closest_inputs = []
    total = len(target_tgs)
    
    for idx, target_tg in enumerate(target_tgs, 1):
        print(f"Processing target Tg: {target_tg:.1f}°C ({idx}/{total})")
        try:
            best_params, predicted_tg = adaptive_grid_search(target_tg, base_models, meta_model, X_scaler, y_scaler)
            best_params['Target_Tg'] = target_tg
            best_params['Predicted_Tg'] = predicted_tg
            best_params['Total_wt%'] = best_params['Lignin (wt%)'] + best_params['Copolyol (wt%)']
            best_params['Error'] = abs(predicted_tg - target_tg)
            closest_inputs.append(best_params)
        except ValueError as e:
            print(f"  Warning: Could not find valid parameters for Tg = {target_tg}")
            continue
    
    return pd.DataFrame(closest_inputs)

def main():
    """Main execution function."""
    print("="*80)
    print("ADAPTIVE GRID SEARCH - BEST MODEL")
    print("="*80)
    print("Finding optimal input parameters for target Tg values")
    print("Model: 10 base estimators, 5 features")
    print("Constraint: Lignin (wt%) + Copolyol (wt%) = 100%")
    print("="*80)
    
    # Load the saved models and scalers
    model_dir = '../7.Mapping'
    print("\nLoading models and scalers...")
    
    try:
        base_models = joblib.load(os.path.join(model_dir, 'best_model_base_models.joblib'))
        print("  ✓ Loaded base_models")
        
        meta_model = joblib.load(os.path.join(model_dir, 'best_model_meta_model.joblib'))
        print("  ✓ Loaded meta_model")
        
        X_scaler = joblib.load(os.path.join(model_dir, 'best_model_x_scaler.joblib'))
        print("  ✓ Loaded x_scaler")
        
        y_scaler = joblib.load(os.path.join(model_dir, 'best_model_y_scaler.joblib'))
        print("  ✓ Loaded y_scaler")
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: Could not find model files!")
        print(f"   {e}")
        print("\nPlease ensure models are trained in ../7.Mapping/")
        return
    
    # Define target Tg values
    print("\nDefining target Tg range...")
    target_tgs = list(np.linspace(-17, 100, 50))  # 50 points from -17°C to 100°C
    print(f"  Target range: {target_tgs[0]:.1f}°C to {target_tgs[-1]:.1f}°C")
    print(f"  Number of targets: {len(target_tgs)}")
    
    # Find closest inputs for all target Tgs
    print("\n" + "="*80)
    print("STARTING ADAPTIVE GRID SEARCH")
    print("="*80)
    
    closest_inputs = find_closest_inputs_adaptive_grid(target_tgs, base_models, meta_model, X_scaler, y_scaler)
    
    # Display summary statistics
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"Successfully found parameters for {len(closest_inputs)}/{len(target_tgs)} targets")
    print(f"\nPrediction error statistics:")
    print(f"  Mean error: {closest_inputs['Error'].mean():.2f}°C")
    print(f"  Max error: {closest_inputs['Error'].max():.2f}°C")
    print(f"  Min error: {closest_inputs['Error'].min():.2f}°C")
    
    print(f"\nComposition validation:")
    all_valid = all(closest_inputs['Total_wt%'].apply(lambda x: abs(x - 100) < 0.01))
    print(f"  All compositions sum to 100%: {all_valid}")
    
    # Save results
    output_filename = 'closest_inputs_best_model.csv'
    closest_inputs.to_csv(output_filename, index=False)
    print(f"\n✓ Results saved to '{output_filename}'")
    
    # Display sample results
    print("\nSample results (first 5):")
    print(closest_inputs.head().to_string())
    
    print("\n" + "="*80)
    print("ADAPTIVE GRID SEARCH COMPLETED!")
    print("="*80)

if __name__ == "__main__":
    main()
