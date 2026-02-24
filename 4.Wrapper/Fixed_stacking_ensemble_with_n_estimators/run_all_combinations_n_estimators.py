# -*- coding: utf-8 -*-
"""
Stacked Ensemble Analysis with Different n_estimators Values Across ALL Feature Combinations
Tests performance progression: 1, 10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000
Across all feature combinations (mandatory + optional features)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import sys
import os
import itertools
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

    print(f"    Processing n_estimators = {n_estimators}...")

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
        'Feature Combination': feature_combination,
        'Number of Features': len(feature_combination),
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
    """Load and prepare the dataset."""
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
    
    # Define features (same as original wrapper)
    X = df_clean[['Sample name', 'Lignin (wt%)', 'Copolyol (wt%)',
           'Co-polyol type (PTHF)', 'Isocyanate (wt%)', 'Isocyanate (mmol NCO)',
           'Isocyonate type', 'r', 'tin(II) octoate', 
           'Sratio(%)']]
    y = df_clean[['Tg(deg C)']]
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")
    
    return X, y

def create_progression_plots(df_results):
    """Create progression plots for all feature combinations."""
    print("\nCreating comprehensive progression plots...")
    
    # Create figure with better spacing
    fig = plt.figure(figsize=(20, 12))
    fig.patch.set_facecolor('white')
    
    # Create subplot layout with better spacing
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.25, 
                          left=0.06, right=0.95, top=0.92, bottom=0.08)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])
    
    # Set background colors
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        ax.set_facecolor('white')
    
    # Plot 1: MAE vs N_Estimators (averaged across all combinations)
    avg_mae = df_results.groupby('n_estimators')['MAE Validation'].mean()
    avg_mae_ci_lower = df_results.groupby('n_estimators')['Validation MAE CI Lower'].mean()
    avg_mae_ci_upper = df_results.groupby('n_estimators')['Validation MAE CI Upper'].mean()
    
    ax1.errorbar(avg_mae.index, avg_mae, 
                yerr=[avg_mae - avg_mae_ci_lower, avg_mae_ci_upper - avg_mae],
                fmt='o-', capsize=5, color='#C44E52', ecolor='#C44E52', 
                alpha=0.7, markersize=8, label='Validation MAE')
    ax1.fill_between(avg_mae.index, avg_mae_ci_lower, avg_mae_ci_upper, alpha=0.2, color='#C44E52')
    ax1.set_xlabel('Number of Estimators', fontsize=12)
    ax1.set_ylabel('MAE (°C)', fontsize=12)
    ax1.set_title('A: Average MAE vs N_Estimators', fontsize=14, fontweight='bold', pad=15)
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.legend(fontsize=10)
    
    # Plot 2: R² vs N_Estimators (averaged across all combinations)
    avg_r2 = df_results.groupby('n_estimators')['R2 Validation'].mean()
    avg_r2_ci_lower = df_results.groupby('n_estimators')['Validation R2 CI Lower'].mean()
    avg_r2_ci_upper = df_results.groupby('n_estimators')['Validation R2 CI Upper'].mean()
    
    ax2.errorbar(avg_r2.index, avg_r2, 
                yerr=[avg_r2 - avg_r2_ci_lower, avg_r2_ci_upper - avg_r2],
                fmt='o-', capsize=5, color='#4C72B0', ecolor='#4C72B0', 
                alpha=0.7, markersize=8, label='Validation R²')
    ax2.fill_between(avg_r2.index, avg_r2_ci_lower, avg_r2_ci_upper, alpha=0.2, color='#4C72B0')
    ax2.set_xlabel('Number of Estimators', fontsize=12)
    ax2.set_ylabel('R²', fontsize=12)
    ax2.set_title('B: Average R² vs N_Estimators', fontsize=14, fontweight='bold', pad=15)
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.legend(fontsize=10)
    
    # Plot 3: Best Performance vs N_Estimators
    best_mae = df_results.groupby('n_estimators')['MAE Validation'].min()
    best_r2 = df_results.groupby('n_estimators')['R2 Validation'].max()
    
    ax3_twin = ax3.twinx()
    ax3.plot(best_mae.index, best_mae, 'o-', color='#C44E52', markersize=8, label='Best MAE')
    ax3_twin.plot(best_r2.index, best_r2, 's-', color='#4C72B0', markersize=8, label='Best R²')
    ax3.set_xlabel('Number of Estimators', fontsize=12)
    ax3.set_ylabel('Best MAE (°C)', fontsize=12, color='#C44E52')
    ax3_twin.set_ylabel('Best R²', fontsize=12, color='#4C72B0')
    ax3.set_title('C: Best Performance vs N_Estimators', fontsize=14, fontweight='bold', pad=15)
    ax3.grid(True, linestyle='--', alpha=0.3)
    ax3.tick_params(axis='y', labelcolor='#C44E52')
    ax3_twin.tick_params(axis='y', labelcolor='#4C72B0')
    
    # Plot 4: Performance Distribution by Number of Features
    feature_performance = df_results.groupby('Number of Features')['MAE Validation'].agg(['mean', 'std', 'min', 'max'])
    
    x_pos = np.arange(len(feature_performance))
    ax4.bar(x_pos, feature_performance['mean'], yerr=feature_performance['std'], 
            alpha=0.7, color='skyblue', edgecolor='black', capsize=5)
    ax4.set_xlabel('Number of Features', fontsize=12)
    ax4.set_ylabel('Average MAE (°C)', fontsize=12)
    ax4.set_title('D: Performance by Feature Count', fontsize=14, fontweight='bold', pad=15)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(feature_performance.index)
    ax4.grid(True, linestyle='--', alpha=0.3, axis='y')
    
    # Plot 5: Heatmap of Performance
    pivot_data = df_results.pivot_table(values='MAE Validation', 
                                      index='Number of Features', 
                                      columns='n_estimators', 
                                      aggfunc='mean')
    
    im = ax5.imshow(pivot_data.values, cmap='RdYlBu_r', aspect='auto')
    ax5.set_xticks(range(len(pivot_data.columns)))
    ax5.set_xticklabels(pivot_data.columns, rotation=45)
    ax5.set_yticks(range(len(pivot_data.index)))
    ax5.set_yticklabels(pivot_data.index)
    ax5.set_xlabel('Number of Estimators', fontsize=12)
    ax5.set_ylabel('Number of Features', fontsize=12)
    ax5.set_title('E: MAE Heatmap (Features vs Estimators)', fontsize=14, fontweight='bold', pad=15)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax5)
    cbar.set_label('MAE (°C)', fontsize=10)
    
    # Plot 6: Top 10 Best Combinations
    top_10 = df_results.nsmallest(10, 'MAE Validation')
    feature_labels = [str(fc)[:30] + '...' if len(str(fc)) > 30 else str(fc) for fc in top_10['Feature Combination']]
    
    bars = ax6.barh(range(len(feature_labels)), top_10['MAE Validation'], 
                   alpha=0.7, color='lightcoral', edgecolor='black')
    ax6.set_yticks(range(len(feature_labels)))
    ax6.set_yticklabels(feature_labels, fontsize=8)
    ax6.set_xlabel('MAE (°C)', fontsize=12)
    ax6.set_title('F: Top 10 Best Combinations', fontsize=14, fontweight='bold', pad=15)
    ax6.grid(True, linestyle='--', alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, mae) in enumerate(zip(bars, top_10['MAE Validation'])):
        width = bar.get_width()
        ax6.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                f'{mae:.2f}', ha='left', va='center', fontsize=8)
    
    # Add main title
    fig.suptitle('Comprehensive N_Estimators Analysis - All Feature Combinations', 
                 fontsize=16, fontweight='bold', y=0.98)

    # Save the figure in multiple formats
    for ext in ['png', 'tiff', 'pdf', 'svg']:
        plt.savefig(f'all_combinations_n_estimators_analysis.{ext}', 
                   dpi=300, bbox_inches='tight', facecolor='white')

    plt.show()

def main():
    """Main execution function."""
    print("="*80)
    print("COMPREHENSIVE N_ESTIMATORS ANALYSIS - ALL FEATURE COMBINATIONS")
    print("="*80)
    print("Testing n_estimators values: 1, 10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000")
    print("Across all feature combinations (mandatory + optional)")
    print("="*80)
    
    # Load and prepare data
    X, y = load_and_prepare_data()
    
    # Define mandatory and optional features (same as original wrapper)
    mandatory_features = ['Lignin (wt%)', 'Co-polyol type (PTHF)', 'r']
    optional_features = ['Copolyol (wt%)', 'Isocyanate (wt%)', 'Isocyanate (mmol NCO)', 'Isocyonate type', 'tin(II) octoate', 'Sratio(%)']
    
    # Generate all feature combinations
    all_combinations = []
    for r in range(1, len(optional_features) + 1):
        for optional_combo in itertools.combinations(optional_features, r):
            features_to_use = mandatory_features + list(optional_combo)
            all_combinations.append(features_to_use)
    
    print(f"Total feature combinations to test: {len(all_combinations)}")
    
    # Define estimator values to test
    estimator_values = [1, 10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    
    # Create CV splits
    cv = RepeatedKFold(n_splits=5, n_repeats=2, random_state=RANDOM_SEED)
    cv_splits = list(cv.split(X))
    
    # Run analysis for each combination and estimator value
    all_results = []
    
    for combo_idx, feature_combination in enumerate(all_combinations):
        print(f"\n{'='*60}")
        print(f"Feature Combination {combo_idx + 1}/{len(all_combinations)}")
        print(f"Features: {feature_combination}")
        print(f"Number of Features: {len(feature_combination)}")
        print('='*60)
        
        X_subset = X[feature_combination]
        
        # Test all estimator values for this combination
        for n_estimators in estimator_values:
            result = run_stacking_with_estimators(X_subset, y, feature_combination, n_estimators, cv_splits)
            all_results.append(result)
        
        print(f"Completed all n_estimators for combination {combo_idx + 1}")
    
    # Convert to DataFrame
    df_results = pd.DataFrame(all_results)
    
    # Save results
    df_results.to_csv('all_combinations_n_estimators_results.csv', index=False)
    print(f"\nResults saved to: all_combinations_n_estimators_results.csv")
    print(f"Total results: {len(df_results)} rows")
    
    # Create comprehensive plots
    create_progression_plots(df_results)
    
    # Print summary statistics
    print("\n" + "="*80)
    print("COMPREHENSIVE ANALYSIS SUMMARY")
    print("="*80)
    
    # Overall best performance
    best_overall = df_results.nsmallest(1, 'MAE Validation').iloc[0]
    print(f"\nOverall Best Performance:")
    print(f"  Features: {best_overall['Feature Combination']}")
    print(f"  N_Estimators: {best_overall['n_estimators']}")
    print(f"  MAE: {best_overall['MAE Validation']:.3f}°C")
    print(f"  R²: {best_overall['R2 Validation']:.3f}")
    print(f"  Number of Features: {best_overall['Number of Features']}")
    
    # Best performance by n_estimators
    print(f"\nBest Performance by N_Estimators:")
    for n_est in estimator_values:
        best_for_n = df_results[df_results['n_estimators'] == n_est].nsmallest(1, 'MAE Validation').iloc[0]
        print(f"  {n_est:4d}: MAE = {best_for_n['MAE Validation']:.3f}°C, Features = {best_for_n['Number of Features']}")
    
    # Performance by feature count
    print(f"\nAverage Performance by Feature Count:")
    feature_stats = df_results.groupby('Number of Features')['MAE Validation'].agg(['mean', 'std', 'min', 'count'])
    for num_feat, stats in feature_stats.iterrows():
        print(f"  {num_feat} features: MAE = {stats['mean']:.3f}±{stats['std']:.3f}°C (min: {stats['min']:.3f}, combos: {stats['count']})")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("Generated Files:")
    print("  - all_combinations_n_estimators_results.csv")
    print("  - all_combinations_n_estimators_analysis.png/tiff/pdf/svg")
    print("="*80)

if __name__ == "__main__":
    main()
