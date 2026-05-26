# -*- coding: utf-8 -*-
"""
Fixed Stacked Ensemble Implementation for Wrapper - Addresses Data Leakage Issue

This implementation uses proper out-of-fold (OOF) predictions to prevent data leakage.
The key changes:
1. Base models generate OOF predictions using nested cross-validation
2. Meta-model is trained only on OOF predictions, never on full dataset
3. Proper nested CV for final evaluation
4. Feature combination testing integrated

@author: Fixed implementation addressing reviewer concerns
"""

import numpy as np
import random
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import joblib
import itertools
import re
import sys
import os
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import RepeatedKFold, GridSearchCV, cross_val_predict
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

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

def set_global_random_seed(seed):
    """Set random seed for numpy and random modules."""
    np.random.seed(seed)
    random.seed(seed)

def scale_columns_with_robust_scaler(data, scaler=None):
    """Scale data columns using RobustScaler."""
    if scaler is None:
        scaler = RobustScaler()
        scaled_data = scaler.fit_transform(data)
    else:
        scaled_data = scaler.transform(data)
    return scaled_data, scaler

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

def save_models(base_models, meta_model, x_scaler, y_scaler, feature_combination, run_number):
    """Save models and scalers to files."""
    feature_str = '_'.join(feature_combination).replace(' ', '_').replace('(', '').replace(')', '').replace('%', 'pct')
    joblib.dump(base_models, f'base_models_fixed_run_{run_number}_{feature_str}.joblib')
    joblib.dump(meta_model, f'meta_model_fixed_run_{run_number}_{feature_str}.joblib')
    joblib.dump(x_scaler, f'x_scaler_fixed_run_{run_number}_{feature_str}.joblib')
    joblib.dump(y_scaler, f'y_scaler_fixed_run_{run_number}_{feature_str}.joblib')
    print(f"Fixed models and scalers from run {run_number} with features {feature_str} saved successfully.")

def generate_oof_predictions(x_train, y_train, model, param_grid, cv_inner=5):
    """
    Generate out-of-fold predictions for a single base model.

    This is CRITICAL to prevent data leakage. The model never sees the validation
    data during training, and predictions are made only on held-out folds.

    Args:
        x_train: Training features
        y_train: Training target
        model: Base model to train
        param_grid: Hyperparameter grid for tuning
        cv_inner: Number of inner CV folds for OOF generation

    Returns:
        oof_predictions: Out-of-fold predictions for the training set
        best_model: Best trained model
    """
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
    # This ensures each sample's prediction is made when it's in the validation fold
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

def run_stacking_with_proper_oof(x, y, feature_combination, n_estimators, outer_cv_splits):
    """
    Run stacking ensemble with proper OOF predictions to prevent data leakage.

    This implements the CORRECT stacking procedure:
    1. For each outer CV fold:
       a. Split into train_outer and val_outer
       b. For train_outer, generate OOF predictions from base models using inner CV
       c. Train meta-model on OOF predictions
       d. Predict on val_outer using base models → meta-model
    2. Report metrics from outer CV folds only

    Args:
        x: Features DataFrame
        y: Target DataFrame
        feature_combination: List of feature names being used
        n_estimators: Number of estimators for tree-based models
        outer_cv_splits: Outer CV splits for evaluation

    Returns:
        results: Dictionary with performance metrics
        best_base_models: List of trained base models
        meta_model: Trained meta-model
    """
    base_model_configs = create_base_models(n_estimators)

    # Storage for CV results
    cv_scores = {
        'r2': [], 'mse': [], 'mae': [],
        'train_r2': [], 'train_mse': [], 'train_mae': []
    }

    base_model_cv_scores = {i: {
        'r2': [], 'mse': [], 'mae': [],
        'train_r2': [], 'train_mse': [], 'train_mae': []
    } for i in range(len(base_model_configs))}

    print(f"\nRunning stacking with {len(feature_combination)} features using proper OOF predictions...")

    for fold_idx, (train_index, val_index) in enumerate(outer_cv_splits):
        print(f"  Processing outer fold {fold_idx + 1}/{len(outer_cv_splits)}...")

        # Split data
        x_train_outer, x_val_outer = x.iloc[train_index], x.iloc[val_index]
        y_train_outer, y_val_outer = y.iloc[train_index], y.iloc[val_index]

        # Scale data
        x_train_scaled, x_scaler = scale_columns_with_robust_scaler(x_train_outer)
        x_val_scaled = x_scaler.transform(x_val_outer)
        y_train_scaled, y_scaler = scale_columns_with_robust_scaler(y_train_outer)
        y_val_scaled = y_scaler.transform(y_val_outer)

        # Generate OOF predictions for each base model
        oof_meta_features = np.zeros((x_train_scaled.shape[0], len(base_model_configs)))
        val_meta_features = np.zeros((x_val_scaled.shape[0], len(base_model_configs)))
        trained_base_models = []

        for i, (model, param_grid) in enumerate(base_model_configs):
            model_name = model.__class__.__name__

            # Generate OOF predictions (CRITICAL: prevents data leakage)
            oof_preds, best_model = generate_oof_predictions(
                x_train_scaled,
                y_train_scaled,
                model,
                param_grid,
                cv_inner=5
            )

            oof_meta_features[:, i] = oof_preds
            val_meta_features[:, i] = best_model.predict(x_val_scaled)
            trained_base_models.append(best_model)

            # Calculate base model metrics
            train_pred = best_model.predict(x_train_scaled)
            val_pred = best_model.predict(x_val_scaled)

            train_r2, train_mse, train_mae = calculate_metrics(y_train_scaled, train_pred, y_scaler)
            val_r2, val_mse, val_mae = calculate_metrics(y_val_scaled, val_pred, y_scaler)

            base_model_cv_scores[i]['train_r2'].append(train_r2)
            base_model_cv_scores[i]['train_mse'].append(train_mse)
            base_model_cv_scores[i]['train_mae'].append(train_mae)
            base_model_cv_scores[i]['r2'].append(val_r2)
            base_model_cv_scores[i]['mse'].append(val_mse)
            base_model_cv_scores[i]['mae'].append(val_mae)

        # Train meta-model on OOF predictions (NOT on full dataset!)
        meta_model = Ridge(random_state=RANDOM_SEED)
        meta_model.fit(oof_meta_features, y_train_scaled.ravel())

        # Evaluate meta-model
        train_meta_pred = meta_model.predict(oof_meta_features)
        val_meta_pred = meta_model.predict(val_meta_features)

        train_r2, train_mse, train_mae = calculate_metrics(y_train_scaled, train_meta_pred, y_scaler)
        val_r2, val_mse, val_mae = calculate_metrics(y_val_scaled, val_meta_pred, y_scaler)

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

    stacking_result = {
        'Feature Combination': feature_combination,
        'Model': 'Fixed Stacking Ensemble',
        'R-squared Validation': r2_mean,
        'MSE Validation': mse_mean,
        'MAE Validation': mae_mean,
        'Validation R-squared CI Lower': r2_ci_lower,
        'Validation R-squared CI Upper': r2_ci_upper,
        'Validation MSE CI Lower': mse_ci_lower,
        'Validation MSE CI Upper': mse_ci_upper,
        'Validation MAE CI Lower': mae_ci_lower,
        'Validation MAE CI Upper': mae_ci_upper,
        'Train R-squared': train_r2_mean,
        'Train MSE': train_mse_mean,
        'Train MAE': train_mae_mean,
        'Train R-squared CI Lower': train_r2_ci_lower,
        'Train R-squared CI Upper': train_r2_ci_upper,
        'Train MSE CI Lower': train_mse_ci_lower,
        'Train MSE CI Upper': train_mse_ci_upper,
        'Train MAE CI Lower': train_mae_ci_lower,
        'Train MAE CI Upper': train_mae_ci_upper,
        'Number of Estimators': n_estimators
    }

    return stacking_result, trained_base_models, meta_model, x_scaler, y_scaler

def run_multiple_times(X, y, mandatory_features, optional_features, num_runs=1):
    """
    Run fixed stacking ensemble multiple times with different feature combinations.
    
    Args:
        X: Features DataFrame
        y: Target DataFrame
        mandatory_features: List of features that must be included
        optional_features: List of features that can be optionally included
        num_runs: Number of times to run the entire process
    
    Returns:
        all_results: List of all results across runs and feature combinations
        best_models: Best models from the final run
    """
    all_results = []
    best_models = []
    best_scalers = []
    
    for run in range(1, num_runs + 1):
        print(f"\n{'='*80}")
        print(f"FIXED STACKING ENSEMBLE - RUN {run}")
        print(f"{'='*80}")
        
        # Set the random seed for reproducibility
        set_global_random_seed(RANDOM_SEED)
        
        # Get consistent CV splits using the global random seed
        cv_splits = RepeatedKFold(n_splits=5, n_repeats=2, random_state=RANDOM_SEED)
        outer_cv_splits = list(cv_splits.split(X))
        
        best_mae = float('inf')
        best_meta_model = None
        best_base_models_overall = None
        best_feature_combination = None
        
        # Test different feature combinations
        for r in range(1, len(optional_features) + 1):
            for optional_combo in itertools.combinations(optional_features, r):
                features_to_use = mandatory_features + list(optional_combo)
                
                print(f"\nTesting feature combination: {features_to_use}")
                
                X_subset = X[features_to_use]
                
                # Run fixed stacking with proper OOF
                stacking_result, base_models, meta_model, x_scaler, y_scaler = run_stacking_with_proper_oof(
                    X_subset, y, features_to_use, 1000, outer_cv_splits
                )
                
                stacking_result['Run Number'] = run
                
                all_results.append(stacking_result)
                
                # Save models and scalers for this combination
                save_models(base_models, meta_model, x_scaler, y_scaler, features_to_use, run)
                
                # Track the best overall model
                if stacking_result['MAE Validation'] < best_mae:
                    best_mae = stacking_result['MAE Validation']
                    best_meta_model = meta_model
                    best_base_models_overall = base_models
                    best_feature_combination = features_to_use
                    best_x_scaler = x_scaler
                    best_y_scaler = y_scaler
        
        print(f"\nBest feature combination in run {run}: {best_feature_combination}")
        print(f"Best MAE: {best_mae:.2f}")
        
        best_models.append((best_base_models_overall, best_meta_model))
        best_scalers.append((best_x_scaler, best_y_scaler))
        
        # Save results for this run
        df_results = pd.DataFrame(all_results)
        mode = 'w' if run == 1 else 'a'
        header = True if run == 1 else False
        
        df_results.to_csv(f"fixed_stacking_results_all_combinations.csv", 
                         mode=mode, 
                         header=header, 
                         index=False)
        
        print(f"Results from run {run} added to fixed_stacking_results_all_combinations.csv")
    
    return all_results, best_models, best_scalers

if __name__ == "__main__":
    # Load dataset using the preprocessing module
    # Since we have an Excel file, we'll use pandas directly but follow the preprocessing pattern
    print("Loading dataset...")
    df = pd.read_excel('dataset.xlsx')
    
    # Remove rows with NaN values in target variable (following preprocessing pattern)
    print(f"Original dataset shape: {df.shape}")
    df_clean = df.dropna(subset=['Tg(deg C)'])
    print(f"Dataset shape after removing NaN target values: {df_clean.shape}")
    print(f"Removed {df.shape[0] - df_clean.shape[0]} rows with NaN target values")
    
    # Map categorical values (similar to preprocessing module)
    # Note: The column name in our dataset is 'Isocyonate type' (with typo)
    isocyanate_mapping = {'N3600': 1, 'HDI': 0, 0: np.nan}
    if 'Isocyonate type' in df_clean.columns:
        df_clean = map_categorical_values(df_clean, 'Isocyonate type', isocyanate_mapping)
        df_clean = df_clean.fillna(0)
    
    X = df_clean[['Sample name', 'Lignin (wt%)', 'Copolyol (wt%)',
           'Co-polyol type (PTHF)', 'Isocyanate (wt%)', 'Isocyanate (mmol NCO)',
           'Isocyonate type', 'r', 'tin(II) octoate', 
           'Sratio(%)']]
    y = df_clean[['Tg(deg C)']]
    
    # List of mandatory and optional features
    mandatory_features = ['Lignin (wt%)', 'Co-polyol type (PTHF)', 'r']
    optional_features = ['Copolyol (wt%)', 'Isocyanate (wt%)', 'Isocyanate (mmol NCO)', 'Isocyonate type', 'tin(II) octoate', 'Sratio(%)']
    
    # Run the process
    all_results, best_models, best_scalers = run_multiple_times(X, y, mandatory_features, optional_features, num_runs=1)
    
    # Use the best models and scalers from the last run
    best_base_models, meta_model = best_models[-1]
    X_scaler, y_scaler = best_scalers[-1]
    
    print(f"\nFinal Results Summary:")
    print(f"Total feature combinations tested: {len(all_results)}")
    print(f"Best models saved successfully.")
