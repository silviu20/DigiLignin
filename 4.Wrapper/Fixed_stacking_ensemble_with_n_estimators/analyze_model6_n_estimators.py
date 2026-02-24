# -*- coding: utf-8 -*-
"""
Analyze Model #6 (Best without Swelling ratio) Performance Across N_Estimators
Model #6 Features: ['Lignin (wt%)', 'Co-polyol type (PTHF)', 'r', 'Copolyol (wt%)', 'Isocyonate type']
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '1.Loading and Preprocessing'))

# Import the module with the correct filename
import importlib.util
spec = importlib.util.spec_from_file_location("loading_preprocessing", 
    os.path.join(os.path.dirname(__file__), '..', '..', '1.Loading and Preprocessing', 'Loading and preprocessing.py'))
loading_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loading_module)

# Get the functions we need
read_csv_with_encoding = loading_module.read_csv_with_encoding
map_categorical_values = loading_module.map_categorical_values

# Set global random seed
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def calculate_confidence_intervals(metric_values, confidence=0.95):
    """Calculate confidence intervals for given metric values."""
    n = len(metric_values)
    mean = np.mean(metric_values)
    se = stats.sem(metric_values)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return mean - h, mean + h

def calculate_metrics(y_true, y_pred, y_scaler):
    """Calculate R2, MSE, and MAE metrics."""
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    y_true_unscaled = y_scaler.inverse_transform(y_true)
    y_pred_unscaled = y_scaler.inverse_transform(y_pred)

    r2 = r2_score(y_true_unscaled, y_pred_unscaled)
    mse = mean_squared_error(y_true_unscaled, y_pred_unscaled)
    mae = mean_absolute_error(y_true_unscaled, y_pred_unscaled)

    return r2, mse, mae

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

def run_stacking_with_estimators(X, y, feature_combination, n_estimators, outer_cv_splits):
    """Run stacking ensemble with specified number of estimators."""
    base_model_configs = create_base_models(n_estimators)

    # Storage for CV results
    cv_scores = {
        'r2': [], 'mse': [], 'mae': [],
        'train_r2': [], 'train_mse': [], 'train_mae': []
    }

    print(f"  Processing n_estimators = {n_estimators}...")

    for fold_idx, (train_index, val_index) in enumerate(outer_cv_splits):
        # Split data
        x_train_outer, x_val_outer = X.iloc[train_index], X.iloc[val_index]
        y_train_outer, y_val_outer = y.iloc[train_index], y.iloc[val_index]

        # Scale data
        x_scaler = RobustScaler()
        y_scaler = RobustScaler()
        x_train_scaled = x_scaler.fit_transform(x_train_outer)
        x_val_scaled = x_scaler.transform(x_val_outer)
        y_train_scaled = y_scaler.fit_transform(y_train_outer)
        y_val_scaled = y_scaler.transform(y_val_outer)

        # Generate OOF predictions for each base model
        oof_meta_features = np.zeros((x_train_scaled.shape[0], len(base_model_configs)))
        val_meta_features = np.zeros((x_val_scaled.shape[0], len(base_model_configs)))

        for i, (model, param_grid) in enumerate(base_model_configs):
            # Generate OOF predictions
            oof_preds, best_model = generate_oof_predictions(
                x_train_scaled,
                y_train_scaled,
                model,
                param_grid,
                cv_inner=5
            )

            oof_meta_features[:, i] = oof_preds
            val_meta_features[:, i] = best_model.predict(x_val_scaled)

        # Train meta-model on OOF predictions
        meta_model = Ridge(random_state=RANDOM_SEED)
        meta_model.fit(oof_meta_features, y_train_scaled.ravel())

        # Evaluate meta-model
        train_meta_pred = meta_model.predict(oof_meta_features)
        val_meta_pred = meta_model.predict(val_meta_features)

        train_r2, train_mse, train_mae = calculate_metrics(y_train_scaled, train_meta_pred.reshape(-1, 1), y_scaler)
        val_r2, val_mse, val_mae = calculate_metrics(y_val_scaled, val_meta_pred.reshape(-1, 1), y_scaler)

        cv_scores['train_r2'].append(train_r2)
        cv_scores['train_mse'].append(train_mse)
        cv_scores['train_mae'].append(train_mae)
        cv_scores['r2'].append(val_r2)
        cv_scores['mse'].append(val_mse)
        cv_scores['mae'].append(val_mae)

    # Calculate validation metrics and CIs
    r2_mean, r2_ci_lower, r2_ci_upper = np.mean(cv_scores['r2']), *calculate_confidence_intervals(cv_scores['r2'])
    mse_mean, mse_ci_lower, mse_ci_upper = np.mean(cv_scores['mse']), *calculate_confidence_intervals(cv_scores['mse'])
    mae_mean, mae_ci_lower, mae_ci_upper = np.mean(cv_scores['mae']), *calculate_confidence_intervals(cv_scores['mae'])

    # Calculate training metrics and CIs
    train_r2_mean, train_r2_ci_lower, train_r2_ci_upper = np.mean(cv_scores['train_r2']), *calculate_confidence_intervals(cv_scores['train_r2'])
    train_mse_mean, train_mse_ci_lower, train_mse_ci_upper = np.mean(cv_scores['train_mse']), *calculate_confidence_intervals(cv_scores['train_mse'])
    train_mae_mean, train_mae_ci_lower, train_mae_ci_upper = np.mean(cv_scores['train_mae']), *calculate_confidence_intervals(cv_scores['train_mae'])

    return {
        'n_estimators': n_estimators,
        'R2 Validation': r2_mean,
        'MSE Validation': mse_mean,
        'MAE Validation': mae_mean,
        'Validation R2 CI Lower': r2_ci_lower,
        'Validation R2 CI Upper': r2_ci_upper,
        'Validation MSE CI Lower': mse_ci_lower,
        'Validation MSE CI Upper': mse_ci_upper,
        'Validation MAE CI Lower': mae_ci_lower,
        'Validation MAE CI Upper': mae_ci_upper,
        'Train R2': train_r2_mean,
        'Train MSE': train_mse_mean,
        'Train MAE': train_mae_mean,
        'Train R2 CI Lower': train_r2_ci_lower,
        'Train R2 CI Upper': train_r2_ci_upper,
        'Train MSE CI Lower': train_mse_ci_lower,
        'Train MSE CI Upper': train_mse_ci_upper,
        'Train MAE CI Lower': train_mae_ci_lower,
        'Train MAE CI Upper': train_mae_ci_upper
    }

def load_and_prepare_data():
    """Load and prepare the dataset using Model #6 feature combination."""
    print("Loading and preparing data...")
    
    # Load dataset
    df = pd.read_excel('../Fixed_Stacking_Ensemble/dataset.xlsx')
    
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
    """Main execution function for Model #6 n_estimators analysis."""
    print("="*80)
    print("MODEL #6 (BEST WITHOUT SWELLING RATIO) N_ESTIMATORS ANALYSIS")
    print("="*80)
    print("Features: ['Lignin (wt%)', 'Co-polyol type (PTHF)', 'r', 'Copolyol (wt%)', 'Isocyonate type']")
    print("Testing n_estimators values: 1, 10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000")
    print("="*80)
    
    # Load and prepare data
    X, y, model6_features = load_and_prepare_data()
    
    # Define estimator values to test
    estimator_values = [1, 10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    
    # Create CV splits
    cv = RepeatedKFold(n_splits=5, n_repeats=2, random_state=RANDOM_SEED)
    cv_splits = list(cv.split(X))
    
    # Run analysis for each estimator value
    results = []
    for n_estimators in estimator_values:
        result = run_stacking_with_estimators(X, y, model6_features, n_estimators, cv_splits)
        results.append(result)
    
    # Convert to DataFrame
    df_results = pd.DataFrame(results)
    
    # Save results
    df_results.to_csv('model6_n_estimators_results.csv', index=False)
    print(f"Results saved to: model6_n_estimators_results.csv")
    
    # Create data for plotting (adapt for existing plotting script)
    plot_data = {
        'n_estimators': df_results['n_estimators'].values,
        'R2 Validation': df_results['R2 Validation'].values,
        'R2 Validation CI Lower': df_results['Validation R2 CI Lower'].values,
        'R2 Validation CI Upper': df_results['Validation R2 CI Upper'].values,
        'Train R2': df_results['Train R2'].values,
        'Train R2 CI Lower': df_results['Train R2 CI Lower'].values,
        'Train R2 CI Upper': df_results['Train R2 CI Upper'].values,
        'MSE Validation': df_results['MSE Validation'].values,
        'MSE Validation CI Lower': df_results['Validation MSE CI Lower'].values,
        'MSE Validation CI Upper': df_results['Validation MSE CI Upper'].values,
        'Train MSE': df_results['Train MSE'].values,
        'Train MSE CI Lower': df_results['Train MSE CI Lower'].values,
        'Train MSE CI Upper': df_results['Train MSE CI Upper'].values,
        'MAE Validation': df_results['MAE Validation'].values,
        'MAE Validation CI Lower': df_results['Validation MAE CI Lower'].values,
        'MAE Validation CI Upper': df_results['Validation MAE CI Upper'].values,
        'Train MAE': df_results['Train MAE'].values,
        'Train MAE CI Lower': df_results['Train MAE CI Lower'].values,
        'Train MAE CI Upper': df_results['Train MAE CI Upper'].values
    }
    
    # Save plot data
    plot_df = pd.DataFrame(plot_data)
    plot_df.to_csv('model6_plot_data.csv', index=False)
    print(f"Plot data saved to: model6_plot_data.csv")
    
    # Print summary
    print("\n" + "="*80)
    print("MODEL #6 N_ESTIMATORS ANALYSIS SUMMARY")
    print("="*80)
    
    # Find best performance
    best_mae_idx = df_results['MAE Validation'].idxmin()
    best_r2_idx = df_results['R2 Validation'].idxmax()
    
    print(f"\nBest MAE Performance:")
    best_mae_row = df_results.loc[best_mae_idx]
    print(f"  Estimators: {best_mae_row['n_estimators']}")
    print(f"  MAE: {best_mae_row['MAE Validation']:.3f}°C")
    print(f"  R²: {best_mae_row['R2 Validation']:.3f}")
    
    print(f"\nBest R² Performance:")
    best_r2_row = df_results.loc[best_r2_idx]
    print(f"  Estimators: {best_r2_row['n_estimators']}")
    print(f"  R²: {best_r2_row['R2 Validation']:.3f}")
    print(f"  MAE: {best_r2_row['MAE Validation']:.3f}°C")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("Generated Files:")
    print("  - model6_n_estimators_results.csv")
    print("  - model6_plot_data.csv")
    print("="*80)
    
    return plot_data

if __name__ == "__main__":
    plot_data = main()
