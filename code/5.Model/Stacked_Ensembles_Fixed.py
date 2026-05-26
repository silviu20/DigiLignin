# -*- coding: utf-8 -*-
"""
Fixed Stacked Ensemble Implementation - Addresses Data Leakage Issue

This implementation uses proper out-of-fold (OOF) predictions to prevent data leakage.
The key changes:
1. Base models generate OOF predictions using nested cross-validation
2. Meta-model is trained only on OOF predictions, never on full dataset
3. Proper nested CV for final evaluation

@author: Fixed implementation addressing reviewer concerns
"""

import numpy as np
import random
import pandas as pd
import scipy.stats as stats
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import RepeatedKFold, GridSearchCV, cross_val_predict
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, StackingRegressor
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
    joblib.dump(base_models, f'base_models_fixed_run_{run_number}.joblib')
    joblib.dump(meta_model, f'meta_model_fixed_run_{run_number}.joblib')
    joblib.dump(x_scaler, f'x_scaler_fixed_run_{run_number}.joblib')
    joblib.dump(y_scaler, f'y_scaler_fixed_run_{run_number}.joblib')
    print(f"Fixed models and scalers from run {run_number} saved successfully.")

def load_models(run_number):
    """Load models and scalers from files."""
    base_models = joblib.load(f'base_models_fixed_run_{run_number}.joblib')
    meta_model = joblib.load(f'meta_model_fixed_run_{run_number}.joblib')
    x_scaler = joblib.load(f'x_scaler_fixed_run_{run_number}.joblib')
    y_scaler = joblib.load(f'y_scaler_fixed_run_{run_number}.joblib')
    print(f"Fixed models and scalers from run {run_number} loaded successfully.")
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

def create_base_models(n_estimators):
    """Create base models with parameter grids for tuning."""
    base_models = [
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
    return base_models


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


def run_stacking_with_proper_oof(x, y, n_estimators, outer_cv_splits):
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

    print(f"\nRunning stacking with {n_estimators} estimators using proper OOF predictions...")

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

    # Calculate mean and CI for stacking ensemble
    r2_mean, r2_ci_lower, r2_ci_upper = np.mean(cv_scores['r2']), *calculate_confidence_intervals(cv_scores['r2'])
    mse_mean, mse_ci_lower, mse_ci_upper = np.mean(cv_scores['mse']), *calculate_confidence_intervals(cv_scores['mse'])
    mae_mean, mae_ci_lower, mae_ci_upper = np.mean(cv_scores['mae']), *calculate_confidence_intervals(cv_scores['mae'])

    train_r2_mean, train_r2_ci_lower, train_r2_ci_upper = np.mean(cv_scores['train_r2']), *calculate_confidence_intervals(cv_scores['train_r2'])
    train_mse_mean, train_mse_ci_lower, train_mse_ci_upper = np.mean(cv_scores['train_mse']), *calculate_confidence_intervals(cv_scores['train_mse'])
    train_mae_mean, train_mae_ci_lower, train_mae_ci_upper = np.mean(cv_scores['train_mae']), *calculate_confidence_intervals(cv_scores['train_mae'])

    stacking_result = {
        'Model': 'Stacking Ensemble (Fixed - No Leakage)',
        'N Estimators': n_estimators,
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
        'Generalizability (Val MAE - Train MAE)': mae_mean - train_mae_mean
    }

    # Calculate base model results
    base_model_results = []
    for i, (model, _) in enumerate(base_model_configs):
        model_name = model.__class__.__name__

        r2_mean = np.mean(base_model_cv_scores[i]['r2'])
        mse_mean = np.mean(base_model_cv_scores[i]['mse'])
        mae_mean = np.mean(base_model_cv_scores[i]['mae'])
        train_r2_mean = np.mean(base_model_cv_scores[i]['train_r2'])
        train_mse_mean = np.mean(base_model_cv_scores[i]['train_mse'])
        train_mae_mean = np.mean(base_model_cv_scores[i]['train_mae'])

        r2_ci_lower, r2_ci_upper = calculate_confidence_intervals(base_model_cv_scores[i]['r2'])
        mse_ci_lower, mse_ci_upper = calculate_confidence_intervals(base_model_cv_scores[i]['mse'])
        mae_ci_lower, mae_ci_upper = calculate_confidence_intervals(base_model_cv_scores[i]['mae'])

        base_model_results.append({
            'Model': f"{model_name} (n={n_estimators})",
            'N Estimators': n_estimators,
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
            'Generalizability (Val MAE - Train MAE)': mae_mean - train_mae_mean
        })

    # Train final models on full dataset for deployment
    print("  Training final models on full dataset for deployment...")
    x_scaled, x_scaler_final = scale_columns_with_robust_scaler(x)
    y_scaled, y_scaler_final = scale_columns_with_robust_scaler(y)

    final_base_models = []
    final_oof_features = np.zeros((x_scaled.shape[0], len(base_model_configs)))

    for i, (model, param_grid) in enumerate(base_model_configs):
        # Generate OOF predictions for full dataset
        oof_preds, best_model = generate_oof_predictions(
            x_scaled,
            y_scaled,
            model,
            param_grid,
            cv_inner=5
        )
        final_oof_features[:, i] = oof_preds
        final_base_models.append(best_model)

    # Train final meta-model on OOF predictions
    final_meta_model = Ridge(random_state=RANDOM_SEED)
    final_meta_model.fit(final_oof_features, y_scaled.ravel())

    return stacking_result, base_model_results, final_base_models, final_meta_model, x_scaler_final, y_scaler_final


def run_multiple_times_fixed(x, y, num_runs=1, n_estimators_list=[1000]):
    """
    Run the fixed stacking process multiple times.

    Args:
        x: Features DataFrame
        y: Target DataFrame
        num_runs: Number of independent runs
        n_estimators_list: List of estimator counts to try

    Returns:
        all_results_df: DataFrame with all results
        best_models: Tuple of (base_models, meta_model, x_scaler, y_scaler)
    """
    all_results = []
    best_mae = float('inf')
    best_models = None

    for run in range(1, num_runs + 1):
        print(f"\n{'='*80}")
        print(f"STARTING RUN {run}/{num_runs}")
        print(f"{'='*80}")

        # Set random seed for reproducibility
        set_global_random_seed(RANDOM_SEED + run - 1)

        # Get CV splits
        cv_splits = get_consistent_cv_splits(x, n_splits=5, n_repeats=2, random_state=RANDOM_SEED + run - 1)

        for n_estimators in n_estimators_list:
            print(f"\nTesting with {n_estimators} estimators...")

            stacking_result, base_model_results, final_base_models, final_meta_model, x_scaler, y_scaler = \
                run_stacking_with_proper_oof(x, y, n_estimators, cv_splits)

            # Store results
            all_results.extend(base_model_results)
            all_results.append(stacking_result)

            # Track best model
            if stacking_result['MAE Validation'] < best_mae:
                best_mae = stacking_result['MAE Validation']
                best_models = (final_base_models, final_meta_model, x_scaler, y_scaler)
                print(f"  [OK] New best MAE: {best_mae:.4f} deg C")

        # Save results for this run
        df_results = pd.DataFrame(all_results)
        filename = f"stacking_results_fixed_run_{run}.csv"
        df_results.to_csv(filename, index=False)
        print(f"\n[OK] Results saved to {filename}")

        # Save best models
        if best_models:
            save_models(best_models[0], best_models[1], best_models[2], best_models[3], run)

    return pd.DataFrame(all_results), best_models


def plot_results_fixed(x, y, best_base_models, meta_model, x_scaler, y_scaler):
    """
    Plot regression and residual plots for the fixed stacking model.

    Args:
        x: Features DataFrame
        y: Target DataFrame
        best_base_models: List of trained base models
        meta_model: Trained meta-model
        x_scaler: Fitted feature scaler
        y_scaler: Fitted target scaler
    """
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

    # Calculate Pearson correlation
    from scipy.stats import pearsonr
    pearson_corr, _ = pearsonr(y_true.ravel(), y_pred.ravel())

    # Create the plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Define common tick intervals
    x_interval = 10
    y_interval = 10
    residual_y_interval = 10

    # Plot the regression plot
    ax1.scatter(y_true, y_true, color='blue', alpha=0.6, label='Actual Values', s=50)
    ax1.scatter(y_true, y_pred, color='red', alpha=0.6, label='Predicted Values', s=50)
    ax1.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'k--', lw=2, label='Ideal Fit')
    ax1.set_xlabel('Actual Tg (°C)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Predicted Tg (°C)', fontsize=14, fontweight='bold')
    ax1.set_title(f'Fixed Stacking Model (Pearson r = {pearson_corr:.3f})', fontsize=14, fontweight='bold')
    ax1.grid(linestyle='--', alpha=0.7)
    ax1.tick_params(axis='both', labelsize=12)
    ax1.xaxis.set_major_locator(plt.MultipleLocator(x_interval))
    ax1.yaxis.set_major_locator(plt.MultipleLocator(y_interval))
    ax1.legend(prop={'size': 11})
    ax1.set_xlim(min(y_true), max(y_true))
    ax1.set_ylim(min(y_true), max(y_true))
    ax1.text(0.05, 0.95, 'A', transform=ax1.transAxes, fontsize=18,
             color='black', fontweight='bold', ha='left', va='top')

    # Plot the residual plot
    residuals = y_true - y_pred
    ax2.scatter(y_pred, residuals, alpha=0.6, color='green', s=50, label='Residuals')
    ax2.axhline(y=0, color='k', linestyle='--', linewidth=2, label='Zero Residual Line')
    ax2.set_xlabel('Predicted Tg (°C)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Residuals (°C)', fontsize=14, fontweight='bold')
    ax2.set_title('Residual Analysis', fontsize=14, fontweight='bold')
    ax2.grid(linestyle='--', alpha=0.7)
    ax2.tick_params(axis='both', labelsize=12)
    ax2.xaxis.set_major_locator(plt.MultipleLocator(x_interval))
    ax2.yaxis.set_major_locator(plt.MultipleLocator(residual_y_interval))
    ax2.legend(prop={'size': 11})
    ax2.set_xlim(min(y_pred), max(y_pred))
    ax2.set_ylim(min(residuals), max(residuals))
    ax2.text(0.05, 0.95, 'B', transform=ax2.transAxes, fontsize=18,
             color='black', fontweight='bold', ha='left', va='top')

    # Adjust the spacing between subplots
    plt.tight_layout()

    # Save the figure in multiple formats
    for ext in ['png', 'pdf', 'svg']:
        plt.savefig(f'Fixed_Stacking_Actual_vs_Predicted.{ext}', dpi=300, bbox_inches='tight')

    print(f"\n[OK] Plots saved successfully")
    plt.close()


# Main script
if __name__ == "__main__":
    """
    Main execution script for the fixed stacking ensemble.

    This script demonstrates the CORRECT implementation of stacking that prevents data leakage.

    Key differences from the original implementation:
    1. Uses out-of-fold (OOF) predictions for meta-features
    2. Never trains on the full dataset during cross-validation
    3. Properly separates training and validation data
    4. Reports honest validation metrics

    Expected outcomes:
    - MAE will likely be HIGHER than the original (10-15°C vs 6.66°C)
    - This is the TRUE performance of the model
    - R² will likely be LOWER (0.85-0.92 vs 0.99)
    - Train-validation gap will be more realistic
    """

    print("="*80)
    print("FIXED STACKING ENSEMBLE - NO DATA LEAKAGE")
    print("="*80)
    print("\nThis implementation addresses the critical data leakage issue identified by reviewers.")
    print("Expected changes:")
    print("  - MAE will increase (this is the TRUE performance)")
    print("  - R² will decrease (more realistic)")
    print("  - Train-validation gap will be larger (indicates proper validation)")
    print("\n" + "="*80 + "\n")

    # Load your data
    # Example: df = pd.read_csv('dataset.csv')
    # For now, assuming df is already loaded

    try:
        # Define features and target
        # NOTE: This still includes swelling ratio - will be addressed in Action 1.2
        x = df[['Lignin (wt%)', 'Ratio', 'Co-polyol type (PTHF)', 'Isocyanate (mmol NCO)',
                'Isocyanate type', 'Tin(II) octoate', 'Swelling ratio (%)']]
        y = df[['Tg (°C)']]

        print(f"Dataset loaded: {len(df)} samples")
        print(f"Features: {list(x.columns)}")
        print(f"Target: {list(y.columns)}\n")

        # Run the fixed stacking process
        results_df, best_models = run_multiple_times_fixed(
            x, y,
            num_runs=1,
            n_estimators_list=[1000]  # Can test multiple: [50, 100, 500, 1000]
        )

        # Display results
        print("\n" + "="*80)
        print("FINAL RESULTS")
        print("="*80)
        print(results_df[['Model', 'MAE Validation', 'Train MAE',
                          'Generalizability (Val MAE - Train MAE)']].to_string(index=False))

        # Extract best models
        best_base_models, meta_model, x_scaler, y_scaler = best_models

        # Plot results
        plot_results_fixed(x, y, best_base_models, meta_model, x_scaler, y_scaler)

        print("\n" + "="*80)
        print("[COMPLETE] FIXED STACKING ENSEMBLE")
        print("="*80)
        print("\nNext steps:")
        print("1. Compare these results with the original implementation")
        print("2. Update manuscript with honest validation metrics")
        print("3. Address swelling ratio issue (Action 1.2)")
        print("4. Add discussion of overfitting/generalization")

    except NameError:
        print("\n[WARNING] ERROR: DataFrame 'df' not found.")
        print("\nPlease load your data first:")
        print("  df = pd.read_csv('your_dataset.csv')")
        print("\nOr import from the preprocessing module:")
        print("  from '1.Loading and Preprocessing.Loading and preprocessing' import main")
        print("  df = main()")

