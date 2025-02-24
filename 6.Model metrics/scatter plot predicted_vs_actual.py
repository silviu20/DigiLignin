# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 10:40:49 2024

@author: P70090917
"""

'''This code needs to be in the same folder as the basemodel-joblib for it to work. 
The feature combination needs to be set (in this case it was extracted from the wrapper section results csv file)'''

import numpy as np
import matplotlib.pyplot as plt
import joblib

def plot_model_results(X, y, feature_combination, run_number='1_best'):
    """Plot the regression and residual plots using saved models."""
    import numpy as np
    import matplotlib.pyplot as plt
    import joblib
    
    # Create feature string for loading models
    feature_str = '_'.join(feature_combination).replace(' ', '_').replace('(', '').replace(')', '').replace('%', 'pct')
    
    # Load the saved models and scalers
    try:
        base_models = joblib.load(f'base_models_{run_number}_{feature_str}.joblib')
        meta_model = joblib.load(f'meta_model_{run_number}_{feature_str}.joblib')
        X_scaler = joblib.load(f'X_scaler_{run_number}_{feature_str}.joblib')
        y_scaler = joblib.load(f'y_scaler_{run_number}_{feature_str}.joblib')
        print("Models and scalers loaded successfully.")
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    # Prepare the data
    X_subset = X[feature_combination]
    X_scaled = X_scaler.transform(X_subset)
    y_scaled = y_scaler.transform(y)

    # Generate predictions from base models
    meta_features = np.zeros((X_scaled.shape[0], len(base_models)))
    for i, model in enumerate(base_models):
        meta_features[:, i] = model.predict(X_scaled)

    # Generate final predictions using meta model
    y_pred_scaled = meta_model.predict(meta_features)

    # Unscale the predictions
    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))
    y_true = y_scaler.inverse_transform(y_scaled)

    # Calculate correlation coefficient
    correlation_coef = np.corrcoef(y_true.ravel(), y_pred.ravel())[0, 1]
    print(f"\nCorrelation Analysis:")
    print(f"Pearson correlation coefficient between actual and predicted values: {correlation_coef:.4f}")
    print(f"R² (square of correlation coefficient): {correlation_coef**2:.4f}")

    # Create the plots with white background
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.patch.set_facecolor('white')
    ax1.set_facecolor('white')
    ax2.set_facecolor('white')

    # Define common tick intervals
    x_interval = 10  # Interval for x-axis
    y_interval = 10  # Interval for y-axis
    residual_y_interval = 5  # Interval for y-axis on the residual plot

    # Plot the regression plot
    # Add grid first so it's behind the points
    ax1.grid(True, linestyle='--', alpha=0.7, color='#E0E0E0')
    
    ax1.scatter(y_true, y_true, color='blue', alpha=0.6, label='Actual Values')
    ax1.scatter(y_true, y_pred, color='red', alpha=0.6, label='Predicted Values')
    ax1.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'k--', lw=2, label='Ideal Fit')
    ax1.set_xlabel('Actual Values', fontsize=14)
    ax1.set_ylabel('Predicted Values', fontsize=14)
    ax1.tick_params(axis='x', labelsize=14)
    ax1.tick_params(axis='y', labelsize=14)
    ax1.xaxis.set_major_locator(plt.MultipleLocator(x_interval))
    ax1.yaxis.set_major_locator(plt.MultipleLocator(y_interval))
    
    # Add correlation coefficient to the plot
    ax1.text(0.05, 0.95, f'r = {correlation_coef:.4f}', 
             transform=ax1.transAxes, fontsize=12, 
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    
    # Legend with white background for plot A
    legend1 = ax1.legend(prop={'size': 10}, 
                        facecolor='white', 
                        framealpha=1,
                        edgecolor='#666666')
    
    ax1.set_xlim(min(y_true), max(y_true))
    ax1.set_ylim(min(y_true), max(y_true))
    ax1.text(0.85, 0.95, 'A', transform=ax1.transAxes, fontsize=16, 
             color='black', fontweight='bold', ha='center', va='center')

    # Plot the residual plot
    # Add grid first so it's behind the points
    ax2.grid(True, linestyle='--', alpha=0.7, color='#E0E0E0')
    
    residuals = y_true - y_pred
    ax2.scatter(y_pred, residuals, alpha=0.6, color='green', label='Residuals')
    ax2.axhline(y=0, color='k', linestyle='--', label='Zero Residual Line')
    ax2.set_xlabel('Predicted Values', fontsize=14)
    ax2.set_ylabel('Residuals', fontsize=14)
    ax2.tick_params(axis='x', labelsize=14)
    ax2.tick_params(axis='y', labelsize=14)
    
    # Set major tick intervals
    ax2.xaxis.set_major_locator(plt.MultipleLocator(x_interval))
    
    # Calculate appropriate y-axis limits and ticks for residual plot
    max_abs_residual = max(abs(max(residuals)), abs(min(residuals)))
    y_max = np.ceil(max_abs_residual / residual_y_interval) * residual_y_interval
    y_min = -y_max
    
    ax2.set_ylim(y_min, y_max)
    ax2.yaxis.set_major_locator(plt.MultipleLocator(residual_y_interval))
    
    # Add horizontal grid lines
    ax2.yaxis.grid(True, linestyle='--', alpha=0.7, color='#E0E0E0')
    
    # Legend with white background for plot B
    legend2 = ax2.legend(prop={'size': 10}, 
                        facecolor='white', 
                        framealpha=1,
                        edgecolor='#666666')
    
    ax2.set_xlim(min(y_pred), max(y_pred))
    ax2.text(0.85, 0.95, 'B', transform=ax2.transAxes, fontsize=16, 
             color='black', fontweight='bold', ha='center', va='center')

    # Make spines (plot borders) slightly grey
    for ax in [ax1, ax2]:
        for spine in ax.spines.values():
            spine.set_edgecolor('#666666')

    # Adjust the spacing between subplots
    plt.subplots_adjust(wspace=0.2)

    # Save the figure in multiple formats
    for ext in ['tiff', 'pdf', 'eps', 'svg', 'jpg']:
        plt.savefig(f'Actual_vs_Predicted_and_Residuals_Plots.{ext}', 
                   dpi=600, bbox_inches='tight', facecolor='white')

    # Show the figure
    plt.show()
    return correlation_coef

# Example usage:
# Assuming you have your data in X and y, and know your feature combination:
feature_combination = ['Lignin (wt%)', 'Co-polyol type (PTHF)', 'Ratio', 'Co-polyol (wt%)', 'Isocyanate (mmol NCO)', 'Tin(II) octoate', 'Swelling ratio (%)']
correlation = plot_model_results(X, y, feature_combination)