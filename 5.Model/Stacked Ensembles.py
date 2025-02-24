# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 12:47:56 2024

@author: P70090917
"""

import numpy as np
import random
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split, RepeatedKFold, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Set global random seed
RANDOM_SEED = 42

def set_global_random_seed(seed):
    """Set random seed for numpy and random modules."""
    np.random.seed(seed)
    random.seed(seed)

def save_models(base_models, meta_model, x_scaler, y_scaler, run_number):
    """Save models and scalers to files."""
    joblib.dump(base_models, f'base_models_run_{run_number}.joblib')
    joblib.dump(meta_model, f'meta_model_run_{run_number}.joblib')
    joblib.dump(x_scaler, f'x_scaler_run_{run_number}.joblib')
    joblib.dump(y_scaler, f'y_scaler_run_{run_number}.joblib')
    print(f"Models and scalers from run {run_number} saved successfully.")

def load_models(run_number):
    """Load models and scalers from files."""
    base_models = joblib.load(f'base_models_run_{run_number}.joblib')
    meta_model = joblib.load(f'meta_model_run_{run_number}.joblib')
    x_scaler = joblib.load(f'x_scaler_run_{run_number}.joblib')
    y_scaler = joblib.load(f'y_scaler_run_{run_number}.joblib')
    print(f"Models and scalers from run {run_number} loaded successfully.")
    return base_models, meta_model, x_scaler, y_scaler

def get_consistent_cv_splits(x, n_splits=5, n_repeats=2, random_state=None):
    """Get consistent cross-validation splits."""
    rskf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    return list(rskf.split(x))

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

def run_base_models_with_tuning(x, y, n_estimators, cv_splits):
    """Run and tune base models."""
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

        for train_index, val_index in cv_splits:
            x_train, x_val = x.iloc[train_index], x.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]

            x_train_scaled, x_scaler = scale_columns_with_robust_scaler(x_train)
            x_val_scaled = x_scaler.transform(x_val)
            y_train_scaled, y_scaler = scale_columns_with_robust_scaler(y_train)
            y_val_scaled = y_scaler.transform(y_val)

            grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
            grid_search.fit(x_train_scaled, y_train_scaled.ravel())

            best_model = grid_search.best_estimator_
            val_pred = best_model.predict(x_val_scaled)
            train_pred = best_model.predict(x_train_scaled)

            r2, mse, mae = calculate_metrics(y_val_scaled, val_pred, y_scaler)
            train_r2, train_mse, train_mae = calculate_metrics(y_train_scaled, train_pred, y_scaler)

            cv_scores['r2'].append(r2)
            cv_scores['mse'].append(mse)
            cv_scores['mae'].append(mae)
            cv_scores['train_r2'].append(train_r2)
            cv_scores['train_mse'].append(train_mse)
            cv_scores['train_mae'].append(train_mae)

        r2_mean, r2_ci_lower, r2_ci_upper = np.mean(cv_scores['r2']), *calculate_confidence_intervals(cv_scores['r2'])
        mse_mean, mse_ci_lower, mse_ci_upper = np.mean(cv_scores['mse']), *calculate_confidence_intervals(cv_scores['mse'])
        mae_mean, mae_ci_lower, mae_ci_upper = np.mean(cv_scores['mae']), *calculate_confidence_intervals(cv_scores['mae'])

        train_r2_mean, train_r2_ci_lower, train_r2_ci_upper = np.mean(cv_scores['train_r2']), *calculate_confidence_intervals(cv_scores['train_r2'])
        train_mse_mean, train_mse_ci_lower, train_mse_ci_upper = np.mean(cv_scores['train_mse']), *calculate_confidence_intervals(cv_scores['train_mse'])
        train_mae_mean, train_mae_ci_lower, train_mae_ci_upper = np.mean(cv_scores['train_mae']), *calculate_confidence_intervals(cv_scores['train_mae'])

        base_model_results.append({
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

        x_scaled, x_scaler = scale_columns_with_robust_scaler(x)
        y_scaled, y_scaler = scale_columns_with_robust_scaler(y)
        best_model.fit(x_scaled, y_scaled.ravel())
        best_base_models.append(best_model)

    return base_model_results, best_base_models

def run_meta_model_with_tuning(x, y, best_base_models, cv_splits):
    """Run and tune meta model."""
    meta_model = Ridge()

    cv_scores = {'r2': [], 'mse': [], 'mae': [], 'train_r2': [], 'train_mse': [], 'train_mae': []}
    for train_index, test_index in cv_splits:
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        x_train_scaled, x_scaler = scale_columns_with_robust_scaler(x_train)
        x_test_scaled = x_scaler.transform(x_test)
        y_train_scaled, y_scaler = scale_columns_with_robust_scaler(y_train)
        y_test_scaled = y_scaler.transform(y_test)

        train_meta_features = np.zeros((x_train_scaled.shape[0], len(best_base_models)))
        test_meta_features = np.zeros((x_test_scaled.shape[0], len(best_base_models)))

        for i, base_model in enumerate(best_base_models):
            train_meta_features[:, i] = base_model.predict(x_train_scaled)
            test_meta_features[:, i] = base_model.predict(x_test_scaled)

        meta_model.fit(train_meta_features, y_train_scaled.ravel())

        train_predictions = meta_model.predict(train_meta_features)
        train_r2, train_mse, train_mae = calculate_metrics(y_train_scaled, train_predictions, y_scaler)

        test_predictions = meta_model.predict(test_meta_features)
        r2, mse, mae = calculate_metrics(y_test_scaled, test_predictions, y_scaler)

        cv_scores['r2'].append(r2)
        cv_scores['mse'].append(mse)
        cv_scores['mae'].append(mae)
        cv_scores['train_r2'].append(train_r2)
        cv_scores['train_mse'].append(train_mse)
        cv_scores['train_mae'].append(train_mae)

    r2_mean, r2_ci_lower, r2_ci_upper = np.mean(cv_scores['r2']), *calculate_confidence_intervals(cv_scores['r2'])
    mse_mean, mse_ci_lower, mse_ci_upper = np.mean(cv_scores['mse']), *calculate_confidence_intervals(cv_scores['mse'])
    mae_mean, mae_ci_lower, mae_ci_upper = np.mean(cv_scores['mae']), *calculate_confidence_intervals(cv_scores['mae'])

    train_r2_mean, train_r2_ci_lower, train_r2_ci_upper = np.mean(cv_scores['train_r2']), *calculate_confidence_intervals(cv_scores['train_r2'])
    train_mse_mean, train_mse_ci_lower, train_mse_ci_upper = np.mean(cv_scores['train_mse']), *calculate_confidence_intervals(cv_scores['train_mse'])
    train_mae_mean, train_mae_ci_lower, train_mae_ci_upper = np.mean(cv_scores['train_mae']), *calculate_confidence_intervals(cv_scores['train_mae'])

    x_scaled, x_scaler = scale_columns_with_robust_scaler(x)
    y_scaled, y_scaler = scale_columns_with_robust_scaler(y)

    meta_features = np.zeros((x_scaled.shape[0], len(best_base_models)))
    for i, base_model in enumerate(best_base_models):
        meta_features[:, i] = base_model.predict(x_scaled)

    meta_model.fit(meta_features, y_scaled.ravel())

    stacking_result = {
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

def run_multiple_times(x, y, num_runs=3):
    """Run the entire process multiple times."""
    best_models = []
    best_scalers = []
    
    for run in range(1, num_runs + 1):
        print(f"\nStarting Run {run}")

        # Set the random seed for reproducibility
        set_global_random_seed(RANDOM_SEED)

        # Get consistent CV splits using the global random seed
        cv_splits = get_consistent_cv_splits(x, random_state=RANDOM_SEED)

        all_results = []
        best_meta_model = None
        best_mae = float('inf')

        estimator_counts = [1000]

        for n_estimators in estimator_counts:
            print(f"\nRunning base models with {n_estimators} estimators...")
            base_model_results, best_base_models = run_base_models_with_tuning(x, y, n_estimators, cv_splits)

            # Add feature information to base model results
            for result in base_model_results:
                result['N Estimators'] = n_estimators

            all_results.extend(base_model_results)

            print(f"\nRunning meta model with {n_estimators} estimators...")
            stacking_result, meta_model = run_meta_model_with_tuning(x, y, best_base_models, cv_splits)

            # Add feature information to the stacking result
            stacking_result['N Estimators'] = n_estimators

            all_results.append(stacking_result)

            if stacking_result['MAE Validation'] < best_mae:
                best_mae = stacking_result['MAE Validation']
                best_meta_model = meta_model

        # Create DataFrame from all results
        df_results = pd.DataFrame(all_results)

        # Save results to CSV
        file_name = f"stacking_results_tuned_run_{run}.csv"
        df_results.to_csv(file_name, index=False)
        print(f"Results saved to {file_name}")

        # Save models
        x_scaled, x_scaler = scale_columns_with_robust_scaler(x)
        y_scaled, y_scaler = scale_columns_with_robust_scaler(y)
        save_models(best_base_models, best_meta_model, x_scaler, y_scaler, run)

        best_models.append((best_base_models, best_meta_model))
        best_scalers.append((x_scaler, y_scaler))

    return best_models, best_scalers

def plot_results(x, y, best_base_models, meta_model, x_scaler, y_scaler):
    """Plot the regression and residual plots."""
    # Prepare the data
    x_scaled = x_scaler.transform(x)
    y_scaled = y_scaler.transform(y)

    # Generate predictions from the base models and the meta model
    meta_features = np.zeros((x_scaled.shape[0], len(best_base_models)))
    for i, model in enumerate(best_base_models):
        meta_features[:, i] = model.predict(x_scaled)

    # Final prediction using the meta model
    y_pred_scaled = meta_model.predict(meta_features)

    # Unscale the predictions
    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))
    y_true = y_scaler.inverse_transform(y_scaled)

    # Create the plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Define common tick intervals
    x_interval = 10  # Interval for x-axis
    y_interval = 10  # Interval for y-axis
    residual_y_interval = 10  # Interval for y-axis on the residual plot

    # Plot the regression plot
    ax1.scatter(y_true, y_true, color='blue', alpha=0.6, label='Actual Values')
    ax1.scatter(y_true, y_pred, color='red', alpha=0.6, label='Predicted Values')
    ax1.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'k--', lw=2, label='Ideal Fit')
    ax1.set_xlabel('Actual Values', fontsize=14)
    ax1.set_ylabel('Predicted Values', fontsize=14)
    ax1.grid(linestyle='--', alpha=0.7)
    ax1.tick_params(axis='x', labelsize=14)
    ax1.tick_params(axis='y', labelsize=14)
    ax1.xaxis.set_major_locator(plt.MultipleLocator(x_interval))
    ax1.yaxis.set_major_locator(plt.MultipleLocator(y_interval))
    ax1.legend(prop={'size': 10})
    ax1.set_xlim(min(y_true), max(y_true))
    ax1.set_ylim(min(y_true), max(y_true))
    ax1.text(0.85, 0.95, 'A', transform=ax1.transAxes, fontsize=16, color='black', fontweight='bold', ha='center', va='center')

    # Plot the residual plot
    residuals = y_true - y_pred
    ax2.scatter(y_pred, residuals, alpha=0.6, color='green', label='Residuals')
    ax2.axhline(y=0, color='k', linestyle='--', label='Zero Residual Line')
    ax2.set_xlabel('Predicted Values', fontsize=14)
    ax2.set_ylabel('Residuals', fontsize=14)
    ax2.grid(linestyle='--', alpha=0.7)
    ax2.tick_params(axis='x', labelsize=14)
    ax2.tick_params(axis='y', labelsize=14)
    ax2.xaxis.set_major_locator(plt.MultipleLocator(x_interval))
    ax2.yaxis.set_major_locator(plt.MultipleLocator(residual_y_interval))
    ax2.legend(prop={'size': 10})
    ax2.set_xlim(min(y_pred), max(y_pred))
    ax2.set_ylim(min(residuals), max(residuals))
    ax2.text(0.85, 0.95, 'B', transform=ax2.transAxes, fontsize=16, color='black', fontweight='bold', ha='center', va='center')

    # Adjust the spacing between subplots
    plt.subplots_adjust(wspace=0.2)

    # Save the figure in multiple formats
    for ext in ['tiff', 'pdf', 'eps', 'svg', 'jpg']:
        plt.savefig(f'Actual_vs_Predicted_and_Residuals_Plots.{ext}', dpi=600, bbox_inches='tight')

    # Show the figure
    plt.show()

# Main script
if __name__ == "__main__":
    # Load your data into df here
    # For example:
    # df = pd.read_csv('your_data.csv')

    x = df[['Lignin (wt%)', 'Ratio', 'Co-polyol type (PTHF)', 'Isocyanate (mmol NCO)', 
            'Isocyanate type', 'Tin(II) octoate', 'Swelling ratio (%)']]
    y = df[['Tg (°C)']]

    # Run the process
    best_models, best_scalers = run_multiple_times(x, y, num_runs=1)

    # Use the best models and scalers from the last run
    best_base_models, meta_model = best_models[-1]
    x_scaler, y_scaler = best_scalers[-1]

    # Plot the results
    plot_results(x, y, best_base_models, meta_model, x_scaler, y_scaler)