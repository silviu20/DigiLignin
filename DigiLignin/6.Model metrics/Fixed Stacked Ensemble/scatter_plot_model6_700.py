# -*- coding: utf-8 -*-
"""
Regression Plot for Model #6 at 700 Base Estimators
Model #6 Features: ['Lignin (wt%)', 'Co-polyol type (PTHF)', 'r', 'Copolyol (wt%)', 'Isocyonate type']
Best Performance: MAE = 15.714°C, R² = 0.337 at 700 estimators
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import RepeatedKFold, GridSearchCV, cross_val_predict
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import scipy.stats as stats

# Add the parent directory to path to import preprocessing module
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '1.Loading and Preprocessing'))

# Import the module with the correct filename
import importlib.util
spec = importlib.util.spec_from_file_location("loading_preprocessing", 
    os.path.join(os.path.dirname(__file__), '..', '1.Loading and Preprocessing', 'Loading and preprocessing.py'))
loading_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loading_module)

# Get the functions we need
read_csv_with_encoding = loading_module.read_csv_with_encoding
map_categorical_values = loading_module.map_categorical_values

# Set global random seed
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def create_base_models(n_estimators):
    """Create base model configurations with hyperparameter grids."""
    return [
        (GradientBoostingRegressor(random_state=RANDOM_SEED), {
            'n_estimators': [n_estimators],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }),
        (RandomForestRegressor(random_state=RANDOM_SEED), {
            'n_estimators': [n_estimators],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }),
        (SVR(), {
            'C': [0.1, 1, 10],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto']
        }),
        (Lasso(random_state=RANDOM_SEED), {
            'alpha': [0.1, 1, 10],
            'max_iter': [1000, 5000]
        }),
        (ElasticNet(random_state=RANDOM_SEED), {
            'alpha': [0.1, 1, 10],
            'l1_ratio': [0.1, 0.5, 0.9],
            'max_iter': [1000, 5000]
        })
    ]

def generate_oof_predictions(x_train, y_train, model, param_grid, cv_inner=5):
    """Generate out-of-fold predictions for a single base model."""
    # Tune hyperparameters using GridSearchCV
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv_inner,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    grid_search.fit(x_train, y_train.ravel())

    best_model = grid_search.best_estimator_

    # Generate OOF predictions using cross_val_predict
    oof_predictions = cross_val_predict(
        best_model,
        x_train,
        y_train.ravel(),
        cv=cv_inner,
        n_jobs=-1
    )

    # Retrain on full training set for final model
    best_model.fit(x_train, y_train.ravel())

    return oof_predictions, best_model

def train_model6_700(X, y, feature_combination):
    """Train Model #6 with 700 estimators and return trained models."""
    base_model_configs = create_base_models(700)
    
    print("Training Model #6 with 700 estimators...")
    
    # Use the full dataset for final model training
    # Scale data
    x_scaler = RobustScaler()
    y_scaler = RobustScaler()
    x_scaled = x_scaler.fit_transform(X)
    y_scaled = y_scaler.fit_transform(y)

    # Generate OOF predictions for each base model
    oof_meta_features = np.zeros((x_scaled.shape[0], len(base_model_configs)))
    trained_base_models = []

    for i, (model, param_grid) in enumerate(base_model_configs):
        print(f"  Training base model {i+1}/5...")
        # Generate OOF predictions
        oof_preds, best_model = generate_oof_predictions(
            x_scaled,
            y_scaled,
            model,
            param_grid,
            cv_inner=5
        )

        oof_meta_features[:, i] = oof_preds
        trained_base_models.append(best_model)

    # Train meta-model on OOF predictions
    meta_model = Ridge(random_state=RANDOM_SEED)
    meta_model.fit(oof_meta_features, y_scaled.ravel())

    print("Model training completed!")
    return trained_base_models, meta_model, x_scaler, y_scaler

def plot_model6_regression(X, y, feature_combination):
    """Create regression plot for Model #6 at 700 estimators."""
    
    print("="*80)
    print("MODEL #6 REGRESSION PLOT (700 ESTIMATORS)")
    print("="*80)
    print(f"Features: {feature_combination}")
    print("="*80)
    
    # Train the model
    base_models, meta_model, x_scaler, y_scaler = train_model6_700(X, y, feature_combination)
    
    # Prepare the data for prediction
    X_subset = X[feature_combination]
    X_scaled = x_scaler.transform(X_subset)
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

    # Calculate performance metrics
    correlation_coef = np.corrcoef(y_true.ravel(), y_pred.ravel())[0, 1]
    r2_score_val = r2_score(y_true, y_pred)
    mae_val = mean_absolute_error(y_true, y_pred)
    mse_val = mean_squared_error(y_true, y_pred)
    
    print(f"\nPerformance Metrics:")
    print(f"Pearson correlation coefficient: {correlation_coef:.4f}")
    print(f"R²: {r2_score_val:.4f}")
    print(f"MAE: {mae_val:.3f}°C")
    print(f"MSE: {mse_val:.3f}")

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
    ax1.grid(True, linestyle='--', alpha=0.7, color='#E0E0E0')
    
    ax1.scatter(y_true, y_true, color='blue', alpha=0.6, label='Actual Values', s=30)
    ax1.scatter(y_true, y_pred, color='red', alpha=0.6, label='Predicted Values', s=30)
    ax1.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'k--', lw=2, label='Ideal Fit')
    ax1.set_xlabel('Actual Values (°C)', fontsize=14)
    ax1.set_ylabel('Predicted Values (°C)', fontsize=14)
    ax1.tick_params(axis='x', labelsize=14)
    ax1.tick_params(axis='y', labelsize=14)
    ax1.xaxis.set_major_locator(plt.MultipleLocator(x_interval))
    ax1.yaxis.set_major_locator(plt.MultipleLocator(y_interval))
    
    # Add correlation coefficient to the plot
    ax1.text(0.05, 0.95, f'r = {correlation_coef:.4f}', 
             transform=ax1.transAxes, fontsize=12, 
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    
    # Add performance metrics
    metrics_text = f'R² = {r2_score_val:.3f}\nMAE = {mae_val:.2f}°C'
    ax1.text(0.05, 0.80, metrics_text, 
             transform=ax1.transAxes, fontsize=10, 
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    
    # Legend
    legend1 = ax1.legend(prop={'size': 10}, 
                        facecolor='white', 
                        framealpha=1,
                        edgecolor='#666666',
                        loc='lower right')
    
    ax1.set_xlim(min(y_true), max(y_true))
    ax1.set_ylim(min(y_true), max(y_true))
    ax1.text(0.85, 0.95, 'A', transform=ax1.transAxes, fontsize=16, 
             color='black', fontweight='bold', ha='center', va='center')

    # Plot the residual plot
    ax2.grid(True, linestyle='--', alpha=0.7, color='#E0E0E0')
    
    residuals = y_true - y_pred
    ax2.scatter(y_pred, residuals, alpha=0.6, color='green', label='Residuals', s=30)
    ax2.axhline(y=0, color='k', linestyle='--', label='Zero Residual Line')
    ax2.set_xlabel('Predicted Values (°C)', fontsize=14)
    ax2.set_ylabel('Residuals (°C)', fontsize=14)
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
    
    # Legend
    legend2 = ax2.legend(prop={'size': 10}, 
                        facecolor='white', 
                        framealpha=1,
                        edgecolor='#666666',
                        loc='upper right')
    
    ax2.set_xlim(min(y_pred), max(y_pred))
    ax2.text(0.85, 0.95, 'B', transform=ax2.transAxes, fontsize=16, 
             color='black', fontweight='bold', ha='center', va='center')

    # Make spines (plot borders) slightly grey
    for ax in [ax1, ax2]:
        for spine in ax.spines.values():
            spine.set_edgecolor('#666666')

    # Adjust the spacing between subplots
    plt.subplots_adjust(wspace=0.2)

    # Add main title
    fig.suptitle('Model #6 Regression Analysis (700 Base Estimators)\nFeatures: Lignin, Co-polyol type, Ratio, Co-polyol, Isocyanate type', 
                 fontsize=14, fontweight='bold', y=0.98)

    # Save the figure in multiple formats
    for ext in ['tiff', 'pdf', 'eps', 'svg', 'jpg', 'png']:
        plt.savefig(f'Model6_700_Regression_Plot.{ext}', 
                   dpi=600, bbox_inches='tight', facecolor='white')

    # Show the figure
    plt.show()
    
    return correlation_coef, r2_score_val, mae_val, mse_val

def load_and_prepare_data():
    """Load and prepare the dataset using Model #6 feature combination."""
    print("Loading and preparing data...")
    
    # Load dataset
    df = pd.read_excel('../4.Wrapper/Fixed_Stacking_Ensemble/dataset.xlsx')
    
    # Remove rows with NaN values in target variable
    df_clean = df.dropna(subset=['Tg(deg C)'])
    print(f"Dataset shape after cleaning: {df_clean.shape}")
    
    # Map categorical values
    isocyanate_mapping = {'N3600': 1, 'HDI': 0, 0: np.nan}
    if 'Isocyonate type' in df_clean.columns:
        df_clean = map_categorical_values(df_clean, 'Isocyonate type', isocyanate_mapping)
        df_clean = df_clean.fillna(0)
    
    # Model #6 feature combination (best without Swelling ratio)
    model6_features = ['Lignin (wt%)', 'Co-polyol type (PTHF)', 'r', 'Copolyol (wt%)', 'Isocyonate type']
    
    X = df_clean[model6_features]
    y = df_clean[['Tg(deg C)']]
    
    print(f"Using Model #6 features: {model6_features}")
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")
    
    return X, y, model6_features

def main():
    """Main execution function."""
    
    # Load and prepare data
    X, y, model6_features = load_and_prepare_data()
    
    # Create regression plot for Model #6 at 700 estimators
    correlation, r2, mae, mse = plot_model6_regression(X, y, model6_features)
    
    print("\n" + "="*80)
    print("MODEL #6 (700 ESTIMATORS) - FINAL RESULTS")
    print("="*80)
    print(f"Correlation Coefficient: {correlation:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"MAE: {mae:.3f}°C")
    print(f"MSE: {mse:.3f}")
    print("="*80)
    print("Generated Files:")
    print("  - Model6_700_Regression_Plot.tiff/pdf/eps/svg/jpg/png")
    print("="*80)

if __name__ == "__main__":
    main()
