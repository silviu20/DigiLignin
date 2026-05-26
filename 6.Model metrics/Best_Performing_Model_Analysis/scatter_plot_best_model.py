# -*- coding: utf-8 -*-
"""
Analysis for the Best Performing Stacking Ensemble Model
Rank 1: ['Lignin (wt%)', 'Co-polyol type (PTHF)', 'r', 'Copolyol (wt%)', 'Sratio(%)']
MAE: 15.498°C, R²: 0.353, Generalization Gap: 0.114°C
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import sys
import os

# Add path to import preprocessing functions
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '1.Loading and Preprocessing'))
import importlib.util

# Import preprocessing module
spec = importlib.util.spec_from_file_location("loading_preprocessing", 
    os.path.join(os.path.dirname(__file__), '..', '..', '1.Loading and Preprocessing', 'Loading and preprocessing.py'))
loading_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loading_module)

read_csv_with_encoding = loading_module.read_csv_with_encoding
map_categorical_values = loading_module.map_categorical_values

def load_and_prepare_data():
    """Load and prepare the dataset using the best performing feature combination."""
    print("Loading and preparing data...")
    
    # Load dataset
    df = pd.read_excel(os.path.join(os.path.dirname(__file__), '..', '..', '4.Wrapper', 'Fixed_Stacking_Ensemble', 'dataset.xlsx'))
    
    # Remove rows with NaN values in target variable
    df_clean = df.dropna(subset=['Tg(deg C)'])
    print(f"Dataset shape after cleaning: {df_clean.shape}")
    
    # Map categorical values (same as in the original training)
    isocyanate_mapping = {'N3600': 1, 'HDI': 0, 0: np.nan}
    if 'Isocyonate type' in df_clean.columns:
        df_clean = map_categorical_values(df_clean, 'Isocyonate type', isocyanate_mapping)
        df_clean = df_clean.fillna(0)
    
    # Define the best performing feature combination
    best_features = ['Lignin (wt%)', 'Co-polyol type (PTHF)', 'r', 'Copolyol (wt%)', 'Sratio(%)']
    
    X = df_clean[best_features]
    y = df_clean[['Tg(deg C)']]
    
    print(f"Using features: {best_features}")
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")
    
    return X, y, best_features

def plot_best_model_results(X, y, feature_combination):
    """Plot comprehensive analysis for the best performing model."""
    import numpy as np
    import matplotlib.pyplot as plt
    import joblib
    
    # Create feature string for loading models
    feature_str = '_'.join(feature_combination).replace(' ', '_').replace('(', '').replace(')', '').replace('%', 'pct')
    
    # Load the saved models and scalers
    model_path = os.path.join(os.path.dirname(__file__), '..', '..', '4.Wrapper', 'Fixed_Stacking_Ensemble')
    
    try:
        base_models = joblib.load(os.path.join(model_path, f'base_models_fixed_run_1_{feature_str}.joblib'))
        meta_model = joblib.load(os.path.join(model_path, f'meta_model_fixed_run_1_{feature_str}.joblib'))
        X_scaler = joblib.load(os.path.join(model_path, f'x_scaler_fixed_run_1_{feature_str}.joblib'))
        y_scaler = joblib.load(os.path.join(model_path, f'y_scaler_fixed_run_1_{feature_str}.joblib'))
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

    # Calculate comprehensive metrics
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    correlation_coef = np.corrcoef(y_true.ravel(), y_pred.ravel())[0, 1]
    
    print(f"\nModel Performance Metrics:")
    print(f"MAE: {mae:.3f}°C")
    print(f"MSE: {mse:.3f}")
    print(f"R²: {r2:.3f}")
    print(f"Pearson correlation: {correlation_coef:.4f}")
    print(f"RMSE: {np.sqrt(mse):.3f}°C")

    # Create comprehensive plots with better spacing
    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor('white')
    
    # Create subplot layout with better spacing
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3, 
                          left=0.08, right=0.95, top=0.92, bottom=0.08)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Set background colors for all subplots
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_facecolor('white')

    # Plot 1: Actual vs Predicted (Regression plot)
    ax1.grid(True, linestyle='--', alpha=0.7, color='#E0E0E0')
    ax1.scatter(y_true, y_true, color='blue', alpha=0.6, s=30, label='Actual Values')
    ax1.scatter(y_true, y_pred, color='red', alpha=0.6, s=30, label='Predicted Values')
    ax1.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'k--', lw=2, label='Ideal Fit')
    ax1.set_xlabel('Actual Tg (°C)', fontsize=12)
    ax1.set_ylabel('Predicted Tg (°C)', fontsize=12)
    ax1.set_title('A: Actual vs Predicted', fontsize=14, fontweight='bold', pad=15)
    ax1.tick_params(axis='both', labelsize=10)
    
    # Add correlation coefficient
    ax1.text(0.05, 0.95, f'R² = {r2:.3f}\nr = {correlation_coef:.3f}', 
             transform=ax1.transAxes, fontsize=10, 
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    ax1.legend(fontsize=9, loc='lower right')
    
    # Plot 2: Residuals
    ax2.grid(True, linestyle='--', alpha=0.7, color='#E0E0E0')
    residuals = y_true - y_pred
    ax2.scatter(y_pred, residuals, alpha=0.6, color='green', s=30)
    ax2.axhline(y=0, color='k', linestyle='--', lw=2)
    ax2.set_xlabel('Predicted Tg (°C)', fontsize=12)
    ax2.set_ylabel('Residuals (°C)', fontsize=12)
    ax2.set_title('B: Residual Plot', fontsize=14, fontweight='bold', pad=15)
    ax2.tick_params(axis='both', labelsize=10)
    
    # Add residual statistics
    ax2.text(0.05, 0.95, f'Mean: {np.mean(residuals):.2f}\nStd: {np.std(residuals):.2f}', 
             transform=ax2.transAxes, fontsize=10, 
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    # Plot 3: Residual Histogram
    ax3.grid(True, linestyle='--', alpha=0.7, color='#E0E0E0', axis='y')
    ax3.hist(residuals, bins=20, alpha=0.7, color='orange', edgecolor='black')
    ax3.axvline(x=0, color='k', linestyle='--', lw=2)
    ax3.set_xlabel('Residuals (°C)', fontsize=12)
    ax3.set_ylabel('Frequency', fontsize=12)
    ax3.set_title('C: Residual Distribution', fontsize=14, fontweight='bold', pad=15)
    ax3.tick_params(axis='both', labelsize=10)

    # Plot 4: Prediction Error Analysis
    ax4.grid(True, linestyle='--', alpha=0.7, color='#E0E0E0')
    prediction_errors = np.abs(residuals)
    ax4.scatter(range(len(prediction_errors)), prediction_errors, alpha=0.6, color='purple', s=30)
    ax4.axhline(y=mae, color='r', linestyle='--', lw=2, label=f'MAE = {mae:.2f}°C')
    ax4.set_xlabel('Sample Index', fontsize=12)
    ax4.set_ylabel('Absolute Error (°C)', fontsize=12)
    ax4.set_title('D: Prediction Error Analysis', fontsize=14, fontweight='bold', pad=15)
    ax4.tick_params(axis='both', labelsize=10)
    ax4.legend(fontsize=9)

    # Make spines slightly grey
    for ax in [ax1, ax2, ax3, ax4]:
        for spine in ax.spines.values():
            spine.set_edgecolor('#666666')

    # Adjust layout - remove tight_layout since we're using gridspec
    # plt.tight_layout()  # Commented out as we're using gridspec spacing
    
    # Add main title with better positioning
    fig.suptitle('Best Performing Stacking Ensemble Model Analysis\n' + 
                 f'Features: {", ".join(feature_combination)}\n' +
                 f'MAE: {mae:.3f}°C, R²: {r2:.3f}', 
                 fontsize=16, fontweight='bold', y=0.98)

    # Save the figure in multiple formats
    for ext in ['png', 'tiff', 'pdf', 'svg']:
        plt.savefig(f'Best_Model_Comprehensive_Analysis.{ext}', 
                   dpi=300, bbox_inches='tight', facecolor='white')

    # Show the figure
    plt.show()
    
    return {
        'mae': mae,
        'mse': mse,
        'r2': r2,
        'correlation': correlation_coef,
        'rmse': np.sqrt(mse),
        'residuals_mean': np.mean(residuals),
        'residuals_std': np.std(residuals)
    }

def main():
    """Main execution function."""
    print("="*80)
    print("BEST PERFORMING STACKING ENSEMBLE MODEL ANALYSIS")
    print("="*80)
    print("Rank 1 Model Configuration:")
    print("Features: ['Lignin (wt%)', 'Co-polyol type (PTHF)', 'r', 'Copolyol (wt%)', 'Sratio(%)']")
    print("Expected Performance: MAE = 15.498°C, R² = 0.353")
    print("="*80)
    
    # Load and prepare data
    X, y, best_features = load_and_prepare_data()
    
    # Run analysis
    metrics = plot_best_model_results(X, y, best_features)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("Generated Files:")
    print("  - Best_Model_Comprehensive_Analysis.png")
    print("  - Best_Model_Comprehensive_Analysis.tiff")
    print("  - Best_Model_Comprehensive_Analysis.pdf")
    print("  - Best_Model_Comprehensive_Analysis.svg")
    print("="*80)

if __name__ == "__main__":
    main()
