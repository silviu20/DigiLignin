# -*- coding: utf-8 -*-
"""
Stacked Ensemble Performance Progression Analysis
Shows performance metrics at various base estimators (1 to 1000) for the best feature combination
Best Features: ['Lignin (wt%)', 'Co-polyol type (PTHF)', 'r', 'Copolyol (wt%)', 'Sratio(%)']
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import joblib
import sys
import os
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import RepeatedKFold, cross_val_predict
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import scipy.stats as stats

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

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def create_base_models(n_estimators):
    """Create base model configurations with hyperparameter grids (same as wrapper)."""
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

def run_stacking_with_estimators(X, y, n_estimators, cv_splits):
    """Run stacking ensemble with specified number of estimators (same as wrapper methodology)."""
    base_model_configs = create_base_models(n_estimators)
    
    # Storage for CV results
    cv_scores = {
        'r2': [], 'mse': [], 'mae': [],
        'train_r2': [], 'train_mse': [], 'train_mae': []
    }

    print(f"  Processing n_estimators = {n_estimators}...")
    
    for fold_idx, (train_index, val_index) in enumerate(cv_splits):
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
            # Generate OOF predictions using GridSearchCV (same as wrapper)
            from sklearn.model_selection import GridSearchCV, cross_val_predict
            
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=5,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            grid_search.fit(x_train_scaled, y_train_outer.values.ravel())
            
            best_model = grid_search.best_estimator_
            
            # Generate OOF predictions using cross_val_predict
            oof_preds = cross_val_predict(
                best_model,
                x_train_scaled,
                y_train_outer.values.ravel(),
                cv=5,
                n_jobs=-1
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

    # Calculate metrics and confidence intervals
    r2_mean, r2_ci_lower, r2_ci_upper = np.mean(cv_scores['r2']), *calculate_confidence_intervals(cv_scores['r2'])
    mse_mean, mse_ci_lower, mse_ci_upper = np.mean(cv_scores['mse']), *calculate_confidence_intervals(cv_scores['mse'])
    mae_mean, mae_ci_lower, mae_ci_upper = np.mean(cv_scores['mae']), *calculate_confidence_intervals(cv_scores['mae'])

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

def plot_metric(ax, x, y_val, yerr_val, y_test, yerr_test, title, ylabel, color_val, color_test):
    """Plot metric with error bars and annotations."""
    ax.errorbar(x, y_val, yerr=yerr_val, fmt='o', capsize=5, color=color_val, ecolor=color_val, 
                alpha=0.7, markersize=10, label='Validation')
    ax.fill_between(x, y_val - yerr_val[0], y_val + yerr_val[1], alpha=0.2, color=color_val)

    ax.errorbar(x, y_test, yerr=yerr_test, fmt='s', capsize=3, color=color_test, ecolor=color_test, 
                alpha=0.5, markersize=5, linestyle='--', label='Train')
    ax.fill_between(x, y_test - yerr_test[0], y_test + yerr_test[1], alpha=0.1, color=color_test)

    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Number of Estimators', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Annotate key points
    for i, (x_val, y_val_point, y_test_point) in enumerate(zip(x, y_val, y_test)):
        if i == 0 or i == len(x) - 1 or x_val in [10, 50, 100]:  # Annotate key points
            y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
            offset = y_range * 0.03
            
            ax.text(x_val, y_val_point + offset, f'{y_val_point:.3f}', 
                    fontsize=8, ha='center', va='bottom', color=color_val, fontweight='bold')
            ax.text(x_val, y_test_point - offset, f'{y_test_point:.3f}', 
                    fontsize=8, ha='center', va='top', color=color_test, fontweight='bold')

def load_and_prepare_data():
    """Load and prepare the dataset using the best performing feature combination."""
    print("Loading and preparing data...")
    
    # Load dataset
    df = pd.read_excel(os.path.join(os.path.dirname(__file__), '..', '..', '4.Wrapper', 'Fixed_Stacking_Ensemble', 'dataset.xlsx'))
    
    # Remove rows with NaN values in target variable
    df_clean = df.dropna(subset=['Tg(deg C)'])
    print(f"Dataset shape after cleaning: {df_clean.shape}")
    
    # Map categorical values
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

def main():
    """Main execution function."""
    print("="*80)
    print("STACKED ENSEMBLE PERFORMANCE PROGRESSION ANALYSIS")
    print("="*80)
    print("Best Features: ['Lignin (wt%)', 'Co-polyol type (PTHF)', 'r', 'Copolyol (wt%)', 'Sratio(%)']")
    print("Analyzing performance across different numbers of base estimators...")
    print("="*80)
    
    # Load and prepare data
    X, y, best_features = load_and_prepare_data()
    
    # Define estimator values to test
    estimator_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 75, 100, 150, 200, 300, 400, 500, 750, 1000]
    
    # Create CV splits
    cv = RepeatedKFold(n_splits=5, n_repeats=2, random_state=RANDOM_SEED)
    cv_splits = list(cv.split(X))
    
    # Run analysis for each estimator value
    results = []
    for n_estimators in estimator_values:
        result = run_stacking_with_estimators(X, y, n_estimators, cv_splits)
        results.append(result)
    
    # Convert to DataFrame
    df_results = pd.DataFrame(results)
    
    # Save results
    df_results.to_csv('stacking_ensemble_progression_results.csv', index=False)
    print(f"Results saved to: stacking_ensemble_progression_results.csv")
    
    # Create progression plots
    print("\nCreating performance progression plots...")
    
    # Create figure with better spacing
    fig = plt.figure(figsize=(18, 6))
    fig.patch.set_facecolor('white')
    
    # Create subplot layout with better spacing
    gs = fig.add_gridspec(1, 3, hspace=0.3, wspace=0.25, 
                          left=0.06, right=0.95, top=0.85, bottom=0.15)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    
    # Set background colors
    for ax in [ax1, ax2, ax3]:
        ax.set_facecolor('white')
    
    # Plot metrics
    plot_metric(ax1, df_results['n_estimators'], df_results['R2 Validation'],
                yerr_val=[df_results['R2 Validation'] - df_results['Validation R2 CI Lower'],
                          df_results['Validation R2 CI Upper'] - df_results['R2 Validation']],
                y_test=df_results['Train R2'],
                yerr_test=[df_results['Train R2'] - df_results['Train R2 CI Lower'],
                           df_results['Train R2 CI Upper'] - df_results['Train R2']],
                title='A: R-squared Progression', ylabel='R-squared', 
                color_val='#4C72B0', color_test='#D55E00')

    plot_metric(ax2, df_results['n_estimators'], df_results['MSE Validation'],
                yerr_val=[df_results['MSE Validation'] - df_results['Validation MSE CI Lower'],
                          df_results['Validation MSE CI Upper'] - df_results['MSE Validation']],
                y_test=df_results['Train MSE'],
                yerr_test=[df_results['Train MSE'] - df_results['Train MSE CI Lower'],
                           df_results['Train MSE CI Upper'] - df_results['Train MSE']],
                title='B: MSE Progression', ylabel='MSE', 
                color_val='#55A868', color_test='#CC79A7')

    plot_metric(ax3, df_results['n_estimators'], df_results['MAE Validation'],
                yerr_val=[df_results['MAE Validation'] - df_results['Validation MAE CI Lower'],
                          df_results['Validation MAE CI Upper'] - df_results['MAE Validation']],
                y_test=df_results['Train MAE'],
                yerr_test=[df_results['Train MAE'] - df_results['Train MAE CI Lower'],
                           df_results['Train MAE CI Upper'] - df_results['Train MAE']],
                title='C: MAE Progression', ylabel='MAE (°C)', 
                color_val='#C44E52', color_test='#0072B2')

    # Add main title
    fig.suptitle('Stacked Ensemble Performance Progression\nBest Feature Combination', 
                 fontsize=16, fontweight='bold', y=0.95)

    # Save the figure in multiple formats
    for ext in ['png', 'tiff', 'pdf', 'svg']:
        plt.savefig(f'Stacking_Ensemble_Progression.{ext}', 
                   dpi=300, bbox_inches='tight', facecolor='white')

    plt.show()
    
    # Print summary
    print("\n" + "="*80)
    print("PROGRESSION ANALYSIS SUMMARY")
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
    
    print(f"\nPerformance at 1000 estimators:")
    final_row = df_results[df_results['n_estimators'] == 1000].iloc[0]
    print(f"  MAE: {final_row['MAE Validation']:.3f}°C")
    print(f"  R²: {final_row['R2 Validation']:.3f}")
    print(f"  MSE: {final_row['MSE Validation']:.3f}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("Generated Files:")
    print("  - stacking_ensemble_progression_results.csv")
    print("  - Stacking_Ensemble_Progression.png/tiff/pdf/svg")
    print("="*80)

if __name__ == "__main__":
    main()
