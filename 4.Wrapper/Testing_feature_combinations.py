# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 10:58:16 2024

@author: P70090917
"""

import numpy as np
import random
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import joblib
import itertools
import re
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split, RepeatedKFold, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Set global random seed
RANDOM_SEED = 42

def set_global_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

# Model Persistence
def save_models(base_models, meta_model, X_scaler, y_scaler, feature_combination, run_number):
    feature_str = '_'.join(feature_combination).replace(' ', '_').replace('(', '').replace(')', '').replace('%', 'pct')
    joblib.dump(base_models, f'base_models_run_{run_number}_{feature_str}.joblib')
    joblib.dump(meta_model, f'meta_model_run_{run_number}_{feature_str}.joblib')
    joblib.dump(X_scaler, f'X_scaler_run_{run_number}_{feature_str}.joblib')
    joblib.dump(y_scaler, f'y_scaler_run_{run_number}_{feature_str}.joblib')
    print(f"Models and scalers from run {run_number} with features {feature_str} saved successfully.")

def load_models(run_number, feature_combination):
    feature_str = '_'.join(feature_combination).replace(' ', '_').replace('(', '').replace(')', '').replace('%', 'pct')
    base_models = joblib.load(f'base_models_run_{run_number}_{feature_str}.joblib')
    meta_model = joblib.load(f'meta_model_run_{run_number}_{feature_str}.joblib')
    X_scaler = joblib.load(f'X_scaler_run_{run_number}_{feature_str}.joblib')
    y_scaler = joblib.load(f'y_scaler_run_{run_number}_{feature_str}.joblib')
    print(f"Models and scalers from run {run_number} with features {feature_str} loaded successfully.")
    return base_models, meta_model, X_scaler, y_scaler

def get_consistent_cv_splits(X, n_splits=5, n_repeats=2, random_state=None):
    rskf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    return list(rskf.split(X))

def scale_columns_with_robust_scaler(data, scaler=None):
    if scaler is None:
        scaler = RobustScaler()
        scaled_data = scaler.fit_transform(data)
    else:
        scaled_data = scaler.transform(data)
    return scaled_data, scaler

def calculate_confidence_intervals(metric_values, confidence=0.95):
    n = len(metric_values)
    mean = np.mean(metric_values)
    se = stats.sem(metric_values)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return mean - h, mean + h

def calculate_metrics(y_true, y_pred, y_scaler):
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

def run_base_models_with_tuning(X, y, n_estimators, cv_splits, feature_combination):
    base_models = [
        (GradientBoostingRegressor(), {
            'n_estimators': [n_estimators],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }),
        (RandomForestRegressor(), {
            'n_estimators': [n_estimators],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }),
        (SVR(), {
            'C': [0.1, 1, 10],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto']
        }),
        (Lasso(), {
            'alpha': [0.1, 1, 10],
            'max_iter': [1000, 5000]
        }),
        (ElasticNet(), {
            'alpha': [0.1, 1, 10],
            'l1_ratio': [0.1, 0.5, 0.9],
            'max_iter': [1000, 5000]
        })
    ]

    base_model_results = []
    best_base_models = []

    for model, param_grid in base_models:
        model_name = model.__class__.__name__
        print(f"Tuning {model_name}...")

        cv_scores = {'r2': [], 'mse': [], 'mae': [], 'train_r2': [], 'train_mse': [], 'train_mae': []}
        all_predictions = []
        all_actuals = []
        all_train_predictions = []
        all_train_actuals = []

        for train_index, val_index in cv_splits:
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]

            X_train_scaled, X_scaler = scale_columns_with_robust_scaler(X_train)
            X_val_scaled = X_scaler.transform(X_val)
            y_train_scaled, y_scaler = scale_columns_with_robust_scaler(y_train)
            y_val_scaled = y_scaler.transform(y_val)

            grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
            grid_search.fit(X_train_scaled, y_train_scaled.ravel())

            best_model = grid_search.best_estimator_
            val_pred = best_model.predict(X_val_scaled)
            train_pred = best_model.predict(X_train_scaled)

            # Calculate metrics
            r2, mse, mae = calculate_metrics(y_val_scaled, val_pred.reshape(-1, 1), y_scaler)
            train_r2, train_mse, train_mae = calculate_metrics(y_train_scaled, train_pred.reshape(-1, 1), y_scaler)

            # Store validation predictions and actuals
            all_predictions.extend(y_scaler.inverse_transform(val_pred.reshape(-1, 1)).ravel())
            all_actuals.extend(y_scaler.inverse_transform(y_val_scaled).ravel())
            
            # Store training predictions and actuals
            all_train_predictions.extend(y_scaler.inverse_transform(train_pred.reshape(-1, 1)).ravel())
            all_train_actuals.extend(y_scaler.inverse_transform(y_train_scaled).ravel())

            # Store metrics
            cv_scores['r2'].append(r2)
            cv_scores['mse'].append(mse)
            cv_scores['mae'].append(mae)
            cv_scores['train_r2'].append(train_r2)
            cv_scores['train_mse'].append(train_mse)
            cv_scores['train_mae'].append(train_mae)

        # Convert predictions to numpy arrays
        all_predictions = np.array(all_predictions)
        all_actuals = np.array(all_actuals)
        all_train_predictions = np.array(all_train_predictions)
        all_train_actuals = np.array(all_train_actuals)

        # Calculate residuals
        validation_residuals = all_actuals - all_predictions
        training_residuals = all_train_actuals - all_train_predictions

        # Save predictions for this model
        validation_df = pd.DataFrame({
            'Dataset': 'Validation',
            'Actual': all_actuals,
            'Predicted': all_predictions,
            'Residuals': validation_residuals
        })
        
        training_df = pd.DataFrame({
            'Dataset': 'Training',
            'Actual': all_train_actuals,
            'Predicted': all_train_predictions,
            'Residuals': training_residuals
        })
        
        predictions_df = pd.concat([training_df, validation_df], ignore_index=True)
        
        feature_str = '_'.join(feature_combination).replace(' ', '_').replace('(', '').replace(')', '').replace('%', 'pct')
        # predictions_df.to_csv(f'predictions_{model_name}_{n_estimators}_{feature_str}.csv', index=False)

        # Calculate validation metrics and CIs
        r2_mean, r2_ci_lower, r2_ci_upper = np.mean(cv_scores['r2']), *calculate_confidence_intervals(cv_scores['r2'])
        mse_mean, mse_ci_lower, mse_ci_upper = np.mean(cv_scores['mse']), *calculate_confidence_intervals(cv_scores['mse'])
        mae_mean, mae_ci_lower, mae_ci_upper = np.mean(cv_scores['mae']), *calculate_confidence_intervals(cv_scores['mae'])

        # Calculate training metrics and CIs
        train_r2_mean, train_r2_ci_lower, train_r2_ci_upper = np.mean(cv_scores['train_r2']), *calculate_confidence_intervals(cv_scores['train_r2'])
        train_mse_mean, train_mse_ci_lower, train_mse_ci_upper = np.mean(cv_scores['train_mse']), *calculate_confidence_intervals(cv_scores['train_mse'])
        train_mae_mean, train_mae_ci_lower, train_mae_ci_upper = np.mean(cv_scores['train_mae']), *calculate_confidence_intervals(cv_scores['train_mae'])

        base_model_results.append({
            'Feature Combination': feature_combination,
            'Model': f"{model_name} (n={n_estimators})",
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
            'Train MAE CI Upper': train_mae_ci_upper
        })

        # Fit final model on full dataset
        X_scaled, X_scaler = scale_columns_with_robust_scaler(X)
        y_scaled, y_scaler = scale_columns_with_robust_scaler(y)
        best_model.fit(X_scaled, y_scaled.ravel())
        best_base_models.append(best_model)

    return base_model_results, best_base_models

def run_meta_model_with_tuning(X, y, best_base_models, cv_splits, feature_combination):
    meta_model = Ridge()

    cv_scores = {'r2': [], 'mse': [], 'mae': [], 'train_r2': [], 'train_mse': [], 'train_mae': []}
    all_predictions = []
    all_actuals = []
    all_train_predictions = []
    all_train_actuals = []
    
    for train_index, test_index in cv_splits:
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        X_train_scaled, X_scaler = scale_columns_with_robust_scaler(X_train)
        X_test_scaled = X_scaler.transform(X_test)
        y_train_scaled, y_scaler = scale_columns_with_robust_scaler(y_train)
        y_test_scaled = y_scaler.transform(y_test)

        train_meta_features = np.zeros((X_train_scaled.shape[0], len(best_base_models)))
        test_meta_features = np.zeros((X_test_scaled.shape[0], len(best_base_models)))

        for i, base_model in enumerate(best_base_models):
            train_meta_features[:, i] = base_model.predict(X_train_scaled)
            test_meta_features[:, i] = base_model.predict(X_test_scaled)

        meta_model.fit(train_meta_features, y_train_scaled.ravel())

        train_predictions = meta_model.predict(train_meta_features)
        train_r2, train_mse, train_mae = calculate_metrics(y_train_scaled, train_predictions.reshape(-1, 1), y_scaler)

        test_predictions = meta_model.predict(test_meta_features)
        r2, mse, mae = calculate_metrics(y_test_scaled, test_predictions.reshape(-1, 1), y_scaler)

        # Store validation predictions and actuals
        all_predictions.extend(y_scaler.inverse_transform(test_predictions.reshape(-1, 1)).ravel())
        all_actuals.extend(y_scaler.inverse_transform(y_test_scaled).ravel())
        
        # Store training predictions and actuals
        all_train_predictions.extend(y_scaler.inverse_transform(train_predictions.reshape(-1, 1)).ravel())
        all_train_actuals.extend(y_scaler.inverse_transform(y_train_scaled).ravel())

        cv_scores['r2'].append(r2)
        cv_scores['mse'].append(mse)
        cv_scores['mae'].append(mae)
        cv_scores['train_r2'].append(train_r2)
        cv_scores['train_mse'].append(train_mse)
        cv_scores['train_mae'].append(train_mae)

    # Convert lists to numpy arrays
    all_predictions = np.array(all_predictions)
    all_actuals = np.array(all_actuals)
    all_train_predictions = np.array(all_train_predictions)
    all_train_actuals = np.array(all_train_actuals)
    
    # Calculate residuals for both training and validation
    validation_residuals = all_actuals - all_predictions
    training_residuals = all_train_actuals - all_train_predictions

    # Create separate DataFrames for training and validation
    validation_df = pd.DataFrame({
        'Dataset': 'Validation',
        'Actual': all_actuals,
        'Predicted': all_predictions,
        'Residuals': validation_residuals
    })
    
    training_df = pd.DataFrame({
        'Dataset': 'Training',
        'Actual': all_train_actuals,
        'Predicted': all_train_predictions,
        'Residuals': training_residuals
    })
    
    # Combine the DataFrames
    predictions_df = pd.concat([training_df, validation_df], ignore_index=True)
    
    # Save predictions
    feature_str = '_'.join(feature_combination).replace(' ', '_').replace('(', '').replace(')', '').replace('%', 'pct')
    # predictions_df.to_csv(f'predictions_{feature_str}.csv', index=False)

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
        'Model': 'Stacking Ensemble',
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
        'Train MAE CI Upper': train_mae_ci_upper
    }

    return stacking_result, meta_model

def run_multiple_times(X, y, mandatory_features, optional_features, num_runs=1):
    best_models = []
    best_scalers = []
    
    # Generate all possible combinations of optional features
    feature_combinations = []
    for r in range(len(optional_features) + 1):
        feature_combinations.extend(itertools.combinations(optional_features, r))
    
    for run in range(1, num_runs + 1):
        print(f"\nStarting Run {run}")
        set_global_random_seed(RANDOM_SEED)

        all_results = []
        best_mae = float('inf')
        best_meta_model = None
        best_feature_combination = None
        best_base_models_overall = None

        estimator_counts = [1, 10, 50, 100,200,300,400,500,600,700,800,900,1000]

        for n_estimators in estimator_counts:
            for optional_subset in feature_combinations:
                features_to_use = mandatory_features + list(optional_subset)
                print(f"\nUsing features: {features_to_use}")
                
                X_subset = X[features_to_use]
                y_subset = y
                cv_splits = get_consistent_cv_splits(X_subset, random_state=RANDOM_SEED)

                print(f"\nRunning base models with {n_estimators} estimators...")
                base_model_results, best_base_models = run_base_models_with_tuning(X_subset, y_subset, n_estimators, cv_splits, features_to_use)
                
                # Add n_estimators information to each result
                for result in base_model_results:
                    result['Number of Estimators'] = n_estimators
                    result['Run Number'] = run
                
                all_results.extend(base_model_results)

                print(f"\nRunning meta model with {n_estimators} estimators...")
                stacking_result, meta_model = run_meta_model_with_tuning(X_subset, y_subset, best_base_models, cv_splits, features_to_use)
                
                # Add n_estimators information to stacking result
                stacking_result['Number of Estimators'] = n_estimators
                stacking_result['Run Number'] = run
                
                all_results.append(stacking_result)

                # Save models and scalers for this combination
                X_scaled, X_scaler = scale_columns_with_robust_scaler(X[features_to_use])
                y_scaled, y_scaler = scale_columns_with_robust_scaler(y)

                # Track the best overall model
                if stacking_result['MAE Validation'] < best_mae:
                    best_mae = stacking_result['MAE Validation']
                    best_meta_model = meta_model
                    best_base_models_overall = best_base_models
                    best_feature_combination = features_to_use

        # Save the best overall models for this run
        X_scaled, X_scaler = scale_columns_with_robust_scaler(X[best_feature_combination])
        y_scaled, y_scaler = scale_columns_with_robust_scaler(y)
        
        # Save models and scalers for the best combination
        save_models(best_base_models_overall, best_meta_model, X_scaler, y_scaler, best_feature_combination, f"{run}_best")
        
        best_models.append((best_base_models_overall, best_meta_model, best_feature_combination))
        best_scalers.append((X_scaler, y_scaler))

        # Create DataFrame with all results and save
        df_results = pd.DataFrame(all_results)
        
        # Reorder columns for better readability
        column_order = [
            'Run Number',
            'Number of Estimators',
            'Feature Combination',
            'Model',
            'R-squared Validation',
            'MSE Validation',
            'MAE Validation',
            'Validation R-squared CI Lower',
            'Validation R-squared CI Upper',
            'Validation MSE CI Lower',
            'Validation MSE CI Upper',
            'Validation MAE CI Lower',
            'Validation MAE CI Upper',
            'Train R-squared',
            'Train MSE',
            'Train MAE',
            'Train R-squared CI Lower',
            'Train R-squared CI Upper',
            'Train MSE CI Lower',
            'Train MSE CI Upper',
            'Train MAE CI Lower',
            'Train MAE CI Upper'
        ]
        
        df_results = df_results[column_order]
        
        # Save results - if first run create new file, otherwise append
        mode = 'w' if run == 1 else 'a'
        header = True if run == 1 else False
        
        df_results.to_csv(f"stacking_results_all_combinations.csv", 
                         mode=mode, 
                         header=header, 
                         index=False)
        
        print(f"Results from run {run} added to stacking_results_all_combinations.csv")

    return best_models, best_scalers

def save_models(base_models, meta_model, X_scaler, y_scaler, feature_combination, identifier):
    """
    Save models with more detailed identifiers
    """
    feature_str = '_'.join(feature_combination).replace(' ', '_').replace('(', '').replace(')', '').replace('%', 'pct')
    
    joblib.dump(base_models, f'base_models_{identifier}_{feature_str}.joblib')
    joblib.dump(meta_model, f'meta_model_{identifier}_{feature_str}.joblib')
    joblib.dump(X_scaler, f'X_scaler_{identifier}_{feature_str}.joblib')
    joblib.dump(y_scaler, f'y_scaler_{identifier}_{feature_str}.joblib')
    print(f"Models and scalers for {identifier} with features {feature_str} saved successfully.")

if __name__ == "__main__":
    X = df[['Sample name', 'Lignin (wt%)', 'Co-polyol (wt%)',
           'Co-polyol type (PTHF)', 'Isocyanate (wt%)', 'Isocyanate (mmol NCO)',
           'Isocyanate type', 'Ratio', 'Tin(II) octoate', 
           'Swelling ratio (%)']]
    y = df[['Tg (°C)']]
    
    # List of mandatory and optional features
    mandatory_features = ['Lignin (wt%)', 'Co-polyol type (PTHF)','Ratio']
    optional_features = ['Co-polyol (wt%)', 'Isocyanate (wt%)', 'Isocyanate (mmol NCO)','Isocyanate type', 'Tin(II) octoate', 'Swelling ratio (%)']
    
    # Run the process
    best_models, best_scalers = run_multiple_times(X, y, mandatory_features, optional_features, num_runs=1)
    
    # Use the best models and scalers from the last run
    best_base_models, meta_model, best_feature_combination = best_models[-1]
    X_scaler, y_scaler = best_scalers[-1]
    
    # # Plot the results
    # plot_results(X, y, best_base_models, meta_model, X_scaler, y_scaler, best_feature_combination)
