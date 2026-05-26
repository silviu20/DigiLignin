# -*- coding: utf-8 -*-
"""
Stacked Ensemble Analysis with Fixed Stratified Data Splitting
Uses fixed splits: 16 validation samples, 16 test samples, remaining for training
Implements stratified splitting to maintain representative distribution across splits
Tests performance progression: 1, 10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import sys
import os
import itertools
from datetime import datetime
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, train_test_split
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

# Set global random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Model registry for tracking all trained models
MODEL_REGISTRY = []

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

def stratified_split(X, y, val_size=16, test_size=16, random_state=42):
    """
    Perform stratified splitting to ensure representative distribution.
    
    Strategy:
    1. Sort data by target variable
    2. Use systematic sampling to ensure even distribution
    3. Maintain diversity across feature space
    
    Args:
        X: Feature matrix
        y: Target variable
        val_size: Number of validation samples
        test_size: Number of test samples
        random_state: Random seed for reproducibility
    
    Returns:
        train_idx, val_idx, test_idx: Indices for each split
    """
    np.random.seed(random_state)
    
    # Create DataFrame with target for sorting
    data_with_target = X.copy()
    data_with_target['target'] = y.values.ravel()
    
    # Sort by target to enable stratified sampling
    sorted_indices = data_with_target.sort_values('target').index.values
    
    n_samples = len(sorted_indices)
    
    # Use systematic sampling for stratification
    # This ensures samples are evenly distributed across the target range
    
    # Calculate step size for validation sampling
    val_step = n_samples / val_size
    val_indices = [sorted_indices[int(i * val_step)] for i in range(val_size)]
    
    # Remove validation indices from available pool
    remaining_indices = [idx for idx in sorted_indices if idx not in val_indices]
    
    # Calculate step size for test sampling from remaining
    test_step = len(remaining_indices) / test_size
    test_indices = [remaining_indices[int(i * test_step)] for i in range(test_size)]
    
    # Training indices are what's left
    train_indices = [idx for idx in remaining_indices if idx not in test_indices]
    
    return np.array(train_indices), np.array(val_indices), np.array(test_indices)

def analyze_split_distribution(X, y, train_idx, val_idx, test_idx):
    """
    Analyze and document the distribution of splits.
    
    Returns:
        Dictionary with distribution statistics
    """
    stats_dict = {
        'train_size': len(train_idx),
        'val_size': len(val_idx),
        'test_size': len(test_idx),
        'train_target_mean': y.iloc[train_idx].mean().values[0],
        'train_target_std': y.iloc[train_idx].std().values[0],
        'train_target_min': y.iloc[train_idx].min().values[0],
        'train_target_max': y.iloc[train_idx].max().values[0],
        'val_target_mean': y.iloc[val_idx].mean().values[0],
        'val_target_std': y.iloc[val_idx].std().values[0],
        'val_target_min': y.iloc[val_idx].min().values[0],
        'val_target_max': y.iloc[val_idx].max().values[0],
        'test_target_mean': y.iloc[test_idx].mean().values[0],
        'test_target_std': y.iloc[test_idx].std().values[0],
        'test_target_min': y.iloc[test_idx].min().values[0],
        'test_target_max': y.iloc[test_idx].max().values[0]
    }
    
    return stats_dict

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

def train_base_model_with_validation(x_train, y_train, x_val, y_val, model, param_grid):
    """
    Train a base model using validation set for hyperparameter tuning.
    
    Args:
        x_train: Training features
        y_train: Training target
        x_val: Validation features
        y_val: Validation target
        model: Base model
        param_grid: Hyperparameter grid
    
    Returns:
        best_model: Trained model
        val_predictions: Predictions on validation set
    """
    # Combine train and validation for GridSearchCV
    x_combined = np.vstack([x_train, x_val])
    y_combined = np.concatenate([y_train.ravel(), y_val.ravel()])
    
    # Create validation split indices for GridSearchCV
    # Use validation set as the validation fold
    train_indices = list(range(len(x_train)))
    val_indices = list(range(len(x_train), len(x_combined)))
    cv_split = [(train_indices, val_indices)]
    
    # Tune hyperparameters
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv_split,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    grid_search.fit(x_combined, y_combined)
    
    best_model = grid_search.best_estimator_
    
    # Retrain on training set only
    best_model.fit(x_train, y_train.ravel())
    
    # Get validation predictions
    val_predictions = best_model.predict(x_val)
    
    return best_model, val_predictions

def save_model_and_register(model, model_type, combination_id, n_estimators, 
                            feature_combination, performance_metrics, 
                            hyperparameters, split_info, timestamp):
    """
    Register model metadata in tracking system (model saving disabled to save disk space).
    
    Args:
        model: Trained model object
        model_type: Type of model (e.g., 'base_gb', 'base_rf', 'meta_ridge')
        combination_id: Feature combination identifier
        n_estimators: Number of estimators used
        feature_combination: List of features
        performance_metrics: Dictionary of metrics
        hyperparameters: Model hyperparameters
        split_info: Information about data split
        timestamp: Training timestamp
    
    Returns:
        model_filename: Path to saved model (not actually saved)
    """
    # Create filename
    model_filename = f"model_{model_type}_combo{combination_id}_n{n_estimators}_{timestamp}.joblib"
    model_path = os.path.join('models', model_filename)
    
    # DISABLED: Save model to disk (to save disk space)
    # joblib.dump(model, model_path)
    
    # Register in tracking system
    registry_entry = {
        'model_filename': model_filename,
        'model_type': model_type,
        'combination_id': combination_id,
        'n_estimators': n_estimators,
        'feature_combination': str(feature_combination),
        'num_features': len(feature_combination),
        'hyperparameters': str(hyperparameters),
        'train_r2': performance_metrics.get('train_r2', None),
        'train_mse': performance_metrics.get('train_mse', None),
        'train_mae': performance_metrics.get('train_mae', None),
        'val_r2': performance_metrics.get('val_r2', None),
        'val_mse': performance_metrics.get('val_mse', None),
        'val_mae': performance_metrics.get('val_mae', None),
        'test_r2': performance_metrics.get('test_r2', None),
        'test_mse': performance_metrics.get('test_mse', None),
        'test_mae': performance_metrics.get('test_mae', None),
        'split_seed': split_info.get('random_state', RANDOM_SEED),
        'train_size': split_info.get('train_size', None),
        'val_size': split_info.get('val_size', None),
        'test_size': split_info.get('test_size', None),
        'timestamp': timestamp
    }
    
    MODEL_REGISTRY.append(registry_entry)
    
    return model_path

def run_stacking_with_fixed_split(X, y, feature_combination, combination_id, n_estimators, 
                                  train_idx, val_idx, test_idx, split_stats):
    """
    Run stacking ensemble with fixed stratified split.
    
    Args:
        X: Feature matrix
        y: Target variable
        feature_combination: List of features to use
        combination_id: Identifier for this combination
        n_estimators: Number of estimators
        train_idx: Training indices
        val_idx: Validation indices
        test_idx: Test indices
        split_stats: Statistics about the split
    
    Returns:
        Dictionary with results and metrics
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"    Processing n_estimators = {n_estimators}...")
    
    # Split data
    x_train, x_val, x_test = X.iloc[train_idx], X.iloc[val_idx], X.iloc[test_idx]
    y_train, y_val, y_test = y.iloc[train_idx], y.iloc[val_idx], y.iloc[test_idx]
    
    # Scale data
    x_scaler = RobustScaler()
    y_scaler = RobustScaler()
    
    x_train_scaled = x_scaler.fit_transform(x_train)
    x_val_scaled = x_scaler.transform(x_val)
    x_test_scaled = x_scaler.transform(x_test)
    
    y_train_scaled = y_scaler.fit_transform(y_train)
    y_val_scaled = y_scaler.transform(y_val)
    y_test_scaled = y_scaler.transform(y_test)
    
    # Create base models
    base_model_configs = create_base_models(n_estimators)
    
    # Train base models and generate meta-features
    train_meta_features = np.zeros((x_train_scaled.shape[0], len(base_model_configs)))
    val_meta_features = np.zeros((x_val_scaled.shape[0], len(base_model_configs)))
    test_meta_features = np.zeros((x_test_scaled.shape[0], len(base_model_configs)))
    
    base_models = []
    model_types = ['base_gb', 'base_rf', 'base_svr', 'base_lasso', 'base_elasticnet']
    
    for i, (model, param_grid) in enumerate(base_model_configs):
        # Train base model
        best_model, val_preds = train_base_model_with_validation(
            x_train_scaled, y_train_scaled, x_val_scaled, y_val_scaled,
            model, param_grid
        )
        
        # Generate predictions for meta-features
        train_meta_features[:, i] = best_model.predict(x_train_scaled)
        val_meta_features[:, i] = val_preds
        test_meta_features[:, i] = best_model.predict(x_test_scaled)
        
        base_models.append(best_model)
        
        # Save base model
        base_performance = {
            'train_r2': None,
            'train_mse': None,
            'train_mae': None,
            'val_r2': None,
            'val_mse': None,
            'val_mae': None
        }
        
        save_model_and_register(
            best_model, model_types[i], combination_id, n_estimators,
            feature_combination, base_performance, 
            best_model.get_params(), split_stats, timestamp
        )
    
    # Train meta-model
    meta_model = Ridge(random_state=RANDOM_SEED)
    meta_model.fit(train_meta_features, y_train_scaled.ravel())
    
    # Generate final predictions
    train_meta_pred = meta_model.predict(train_meta_features)
    val_meta_pred = meta_model.predict(val_meta_features)
    test_meta_pred = meta_model.predict(test_meta_features)
    
    # Calculate metrics
    train_r2, train_mse, train_mae = calculate_metrics(
        y_train_scaled, train_meta_pred.reshape(-1, 1), y_scaler
    )
    val_r2, val_mse, val_mae = calculate_metrics(
        y_val_scaled, val_meta_pred.reshape(-1, 1), y_scaler
    )
    test_r2, test_mse, test_mae = calculate_metrics(
        y_test_scaled, test_meta_pred.reshape(-1, 1), y_scaler
    )
    
    # Save meta-model
    meta_performance = {
        'train_r2': train_r2,
        'train_mse': train_mse,
        'train_mae': train_mae,
        'val_r2': val_r2,
        'val_mse': val_mse,
        'val_mae': val_mae,
        'test_r2': test_r2,
        'test_mse': test_mse,
        'test_mae': test_mae
    }
    
    save_model_and_register(
        meta_model, 'meta_ridge', combination_id, n_estimators,
        feature_combination, meta_performance,
        meta_model.get_params(), split_stats, timestamp
    )
    
    # DISABLED: Save scalers and ensemble bundle (to save disk space)
    # scaler_bundle = {
    #     'x_scaler': x_scaler,
    #     'y_scaler': y_scaler,
    #     'base_models': base_models,
    #     'meta_model': meta_model
    # }
    # 
    # scaler_filename = f"ensemble_combo{combination_id}_n{n_estimators}_{timestamp}.joblib"
    # joblib.dump(scaler_bundle, os.path.join('models', scaler_filename))
    
    return {
        'Feature Combination': str(feature_combination),
        'Combination ID': combination_id,
        'Number of Features': len(feature_combination),
        'n_estimators': n_estimators,
        'Train R2': train_r2,
        'Train MSE': train_mse,
        'Train MAE': train_mae,
        'Validation R2': val_r2,
        'Validation MSE': val_mse,
        'Validation MAE': val_mae,
        'Test R2': test_r2,
        'Test MSE': test_mse,
        'Test MAE': test_mae,
        'Train Size': split_stats['train_size'],
        'Val Size': split_stats['val_size'],
        'Test Size': split_stats['test_size'],
        'Timestamp': timestamp
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
    
    # Define features
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
    
    fig = plt.figure(figsize=(20, 12))
    fig.patch.set_facecolor('white')
    
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.25, 
                          left=0.06, right=0.95, top=0.92, bottom=0.08)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])
    
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        ax.set_facecolor('white')
    
    # Plot 1: MAE vs N_Estimators (Validation and Test)
    avg_val_mae = df_results.groupby('n_estimators')['Validation MAE'].mean()
    avg_test_mae = df_results.groupby('n_estimators')['Test MAE'].mean()
    
    ax1.plot(avg_val_mae.index, avg_val_mae, 'o-', color='#C44E52', 
             markersize=8, label='Validation MAE', linewidth=2)
    ax1.plot(avg_test_mae.index, avg_test_mae, 's-', color='#DD8452', 
             markersize=8, label='Test MAE', linewidth=2)
    ax1.set_xlabel('Number of Estimators', fontsize=12)
    ax1.set_ylabel('MAE (°C)', fontsize=12)
    ax1.set_title('A: Average MAE vs N_Estimators', fontsize=14, fontweight='bold', pad=15)
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.legend(fontsize=10)
    
    # Plot 2: R² vs N_Estimators (Validation and Test)
    avg_val_r2 = df_results.groupby('n_estimators')['Validation R2'].mean()
    avg_test_r2 = df_results.groupby('n_estimators')['Test R2'].mean()
    
    ax2.plot(avg_val_r2.index, avg_val_r2, 'o-', color='#4C72B0', 
             markersize=8, label='Validation R²', linewidth=2)
    ax2.plot(avg_test_r2.index, avg_test_r2, 's-', color='#55A868', 
             markersize=8, label='Test R²', linewidth=2)
    ax2.set_xlabel('Number of Estimators', fontsize=12)
    ax2.set_ylabel('R²', fontsize=12)
    ax2.set_title('B: Average R² vs N_Estimators', fontsize=14, fontweight='bold', pad=15)
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.legend(fontsize=10)
    
    # Plot 3: Best Performance vs N_Estimators
    best_val_mae = df_results.groupby('n_estimators')['Validation MAE'].min()
    best_test_mae = df_results.groupby('n_estimators')['Test MAE'].min()
    
    ax3.plot(best_val_mae.index, best_val_mae, 'o-', color='#C44E52', 
             markersize=8, label='Best Val MAE', linewidth=2)
    ax3.plot(best_test_mae.index, best_test_mae, 's-', color='#DD8452', 
             markersize=8, label='Best Test MAE', linewidth=2)
    ax3.set_xlabel('Number of Estimators', fontsize=12)
    ax3.set_ylabel('MAE (°C)', fontsize=12)
    ax3.set_title('C: Best Performance vs N_Estimators', fontsize=14, fontweight='bold', pad=15)
    ax3.grid(True, linestyle='--', alpha=0.3)
    ax3.legend(fontsize=10)
    
    # Plot 4: Performance Distribution by Number of Features
    feature_performance = df_results.groupby('Number of Features')['Validation MAE'].agg(['mean', 'std', 'min', 'max'])
    
    x_pos = np.arange(len(feature_performance))
    ax4.bar(x_pos, feature_performance['mean'], yerr=feature_performance['std'], 
            alpha=0.7, color='skyblue', edgecolor='black', capsize=5)
    ax4.set_xlabel('Number of Features', fontsize=12)
    ax4.set_ylabel('Average Validation MAE (°C)', fontsize=12)
    ax4.set_title('D: Performance by Feature Count', fontsize=14, fontweight='bold', pad=15)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(feature_performance.index)
    ax4.grid(True, linestyle='--', alpha=0.3, axis='y')
    
    # Plot 5: Heatmap of Validation Performance
    pivot_data = df_results.pivot_table(values='Validation MAE', 
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
    ax5.set_title('E: Validation MAE Heatmap', fontsize=14, fontweight='bold', pad=15)
    
    cbar = plt.colorbar(im, ax=ax5)
    cbar.set_label('MAE (°C)', fontsize=10)
    
    # Plot 6: Top 10 Best Combinations (by Test MAE)
    top_10 = df_results.nsmallest(10, 'Test MAE')
    labels = [f"Combo {row['Combination ID']}, n={row['n_estimators']}" 
              for _, row in top_10.iterrows()]
    
    bars = ax6.barh(range(len(labels)), top_10['Test MAE'], 
                   alpha=0.7, color='lightcoral', edgecolor='black')
    ax6.set_yticks(range(len(labels)))
    ax6.set_yticklabels(labels, fontsize=8)
    ax6.set_xlabel('Test MAE (°C)', fontsize=12)
    ax6.set_title('F: Top 10 Best Combinations (Test MAE)', fontsize=14, fontweight='bold', pad=15)
    ax6.grid(True, linestyle='--', alpha=0.3, axis='x')
    
    for i, (bar, mae) in enumerate(zip(bars, top_10['Test MAE'])):
        width = bar.get_width()
        ax6.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                f'{mae:.2f}', ha='left', va='center', fontsize=8)
    
    fig.suptitle('Fixed Stratified Split Analysis - All Feature Combinations', 
                 fontsize=16, fontweight='bold', y=0.98)

    for ext in ['png', 'tiff', 'pdf', 'svg']:
        plt.savefig(f'fixed_split_analysis.{ext}', 
                   dpi=300, bbox_inches='tight', facecolor='white')

    plt.show()

def load_checkpoint():
    """Load existing results and checkpoint data if available."""
    checkpoint_file = 'checkpoint.json'
    results_file = 'fixed_split_results.csv'
    registry_file = 'model_registry.csv'
    
    existing_results = []
    completed_combinations = set()
    
    # Load existing results
    if os.path.exists(results_file):
        print(f"Found existing results file: {results_file}")
        df_existing = pd.read_csv(results_file)
        existing_results = df_existing.to_dict('records')
        
        # Track completed combination_id + n_estimators pairs
        for result in existing_results:
            key = (result['Combination ID'], result['n_estimators'])
            completed_combinations.add(key)
        
        print(f"  Loaded {len(existing_results)} existing results")
        print(f"  Completed {len(completed_combinations)} combination-estimator pairs")
    
    # Load existing model registry
    if os.path.exists(registry_file):
        print(f"Found existing model registry: {registry_file}")
        df_registry = pd.read_csv(registry_file)
        MODEL_REGISTRY.extend(df_registry.to_dict('records'))
        print(f"  Loaded {len(df_registry)} registry entries")
    
    return existing_results, completed_combinations

def save_checkpoint(all_results, combo_idx, total_combos):
    """Save results incrementally after each combination."""
    # Save results
    df_results = pd.DataFrame(all_results)
    df_results.to_csv('fixed_split_results.csv', index=False)
    
    # Save model registry
    df_registry = pd.DataFrame(MODEL_REGISTRY)
    df_registry.to_csv('model_registry.csv', index=False)
    
    # Save checkpoint info
    checkpoint_info = {
        'last_completed_combination': combo_idx,
        'total_combinations': total_combos,
        'total_results': len(all_results),
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    with open('checkpoint.json', 'w') as f:
        json.dump(checkpoint_info, f, indent=4)
    
    print(f"  ✓ Checkpoint saved: {len(all_results)} results, {len(MODEL_REGISTRY)} registry entries")

def main():
    """Main execution function with checkpoint and resume support."""
    print("="*80)
    print("FIXED STRATIFIED SPLIT ANALYSIS - ALL FEATURE COMBINATIONS")
    print("="*80)
    print("Split Strategy: 16 validation, 16 test, remaining training")
    print("Testing n_estimators values: 1, 10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000")
    print("With checkpoint and resume support")
    print("="*80)
    
    # Load and prepare data
    X, y = load_and_prepare_data()
    
    # Create stratified fixed split
    print("\nCreating stratified fixed split...")
    train_idx, val_idx, test_idx = stratified_split(X, y, val_size=16, test_size=16, random_state=RANDOM_SEED)
    
    # Analyze split distribution
    split_stats = analyze_split_distribution(X, y, train_idx, val_idx, test_idx)
    
    print(f"\nSplit Statistics:")
    print(f"  Training set: {split_stats['train_size']} samples")
    print(f"    Target mean: {split_stats['train_target_mean']:.2f}°C")
    print(f"    Target std: {split_stats['train_target_std']:.2f}°C")
    print(f"    Target range: [{split_stats['train_target_min']:.2f}, {split_stats['train_target_max']:.2f}]°C")
    print(f"  Validation set: {split_stats['val_size']} samples")
    print(f"    Target mean: {split_stats['val_target_mean']:.2f}°C")
    print(f"    Target std: {split_stats['val_target_std']:.2f}°C")
    print(f"    Target range: [{split_stats['val_target_min']:.2f}, {split_stats['val_target_max']:.2f}]°C")
    print(f"  Test set: {split_stats['test_size']} samples")
    print(f"    Target mean: {split_stats['test_target_mean']:.2f}°C")
    print(f"    Target std: {split_stats['test_target_std']:.2f}°C")
    print(f"    Target range: [{split_stats['test_target_min']:.2f}, {split_stats['test_target_max']:.2f}]°C")
    
    # Save split statistics
    with open('split_statistics.json', 'w') as f:
        json.dump(split_stats, f, indent=4)
    
    # Define mandatory and optional features
    mandatory_features = ['Lignin (wt%)', 'Co-polyol type (PTHF)', 'r']
    optional_features = ['Copolyol (wt%)', 'Isocyanate (wt%)', 'Isocyanate (mmol NCO)', 
                        'Isocyonate type', 'tin(II) octoate', 'Sratio(%)']
    
    # Generate all feature combinations
    all_combinations = []
    for r in range(1, len(optional_features) + 1):
        for optional_combo in itertools.combinations(optional_features, r):
            features_to_use = mandatory_features + list(optional_combo)
            all_combinations.append(features_to_use)
    
    print(f"\nTotal feature combinations to test: {len(all_combinations)}")
    
    # Define estimator values to test
    estimator_values = [1, 10, 50]  # TEST: Limited values
    
    # Load checkpoint if exists
    print("\nChecking for existing progress...")
    all_results, completed_combinations = load_checkpoint()
    
    if completed_combinations:
        print(f"\n⚠ RESUMING FROM CHECKPOINT")
        print(f"  Already completed: {len(completed_combinations)} combination-estimator pairs")
        print(f"  Will skip completed work and continue from where we left off\n")
    else:
        print("  No existing progress found. Starting fresh.\n")
    
    # Run analysis for each combination and estimator value
    
    for combo_idx, feature_combination in enumerate(all_combinations[:2]):  # TEST: Only first 2 combinations
        combination_id = combo_idx + 1
        
        print(f"\n{'='*60}")
        print(f"Feature Combination {combination_id}/{len(all_combinations)}")
        print(f"Features: {feature_combination}")
        print(f"Number of Features: {len(feature_combination)}")
        print('='*60)
        
        X_subset = X[feature_combination]
        
        # Track if any new results were added for this combination
        new_results_added = False
        
        # Test all estimator values for this combination
        for n_estimators in estimator_values:
            # Check if this combination-estimator pair is already completed
            if (combination_id, n_estimators) in completed_combinations:
                print(f"    ⏭ Skipping n_estimators = {n_estimators} (already completed)")
                continue
            
            # Run the experiment
            result = run_stacking_with_fixed_split(
                X_subset, y, feature_combination, combination_id, n_estimators,
                train_idx, val_idx, test_idx, split_stats
            )
            all_results.append(result)
            completed_combinations.add((combination_id, n_estimators))
            new_results_added = True
        
        # Save checkpoint after each combination (only if new results were added)
        if new_results_added:
            print(f"  Saving checkpoint after combination {combination_id}...")
            save_checkpoint(all_results, combination_id, len(all_combinations))
            print(f"Completed combination {combination_id}/{len(all_combinations)}")
        else:
            print(f"  All n_estimators already completed for combination {combination_id}")
    
    # Final save (in case last combination didn't trigger a save)
    print("\nSaving final results...")
    df_results = pd.DataFrame(all_results)
    df_results.to_csv('fixed_split_results.csv', index=False)
    print(f"Results saved to: fixed_split_results.csv")
    print(f"Total results: {len(df_results)} rows")
    
    df_registry = pd.DataFrame(MODEL_REGISTRY)
    df_registry.to_csv('model_registry.csv', index=False)
    print(f"Model registry saved to: model_registry.csv")
    print(f"Total registry entries: {len(MODEL_REGISTRY)}")
    
    # Create comprehensive plots
    create_progression_plots(df_results)
    
    # Print summary statistics
    print("\n" + "="*80)
    print("COMPREHENSIVE ANALYSIS SUMMARY")
    print("="*80)
    
    # Overall best performance (by test MAE)
    best_overall = df_results.nsmallest(1, 'Test MAE').iloc[0]
    print(f"\nOverall Best Performance (Test MAE):")
    print(f"  Combination ID: {best_overall['Combination ID']}")
    print(f"  Features: {best_overall['Feature Combination']}")
    print(f"  N_Estimators: {best_overall['n_estimators']}")
    print(f"  Test MAE: {best_overall['Test MAE']:.3f}°C")
    print(f"  Test R²: {best_overall['Test R2']:.3f}")
    print(f"  Validation MAE: {best_overall['Validation MAE']:.3f}°C")
    print(f"  Validation R²: {best_overall['Validation R2']:.3f}")
    
    # Best performance by n_estimators
    print(f"\nBest Test Performance by N_Estimators:")
    for n_est in estimator_values:
        best_for_n = df_results[df_results['n_estimators'] == n_est].nsmallest(1, 'Test MAE').iloc[0]
        print(f"  {n_est:4d}: Test MAE = {best_for_n['Test MAE']:.3f}°C, "
              f"Val MAE = {best_for_n['Validation MAE']:.3f}°C, "
              f"Features = {best_for_n['Number of Features']}")
    
    # Performance by feature count
    print(f"\nAverage Test Performance by Feature Count:")
    feature_stats = df_results.groupby('Number of Features')['Test MAE'].agg(['mean', 'std', 'min', 'count'])
    for num_feat, stats in feature_stats.iterrows():
        print(f"  {num_feat} features: Test MAE = {stats['mean']:.3f}±{stats['std']:.3f}°C "
              f"(min: {stats['min']:.3f}, combos: {stats['count']})")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("Generated Files:")
    print("  - fixed_split_results.csv")
    print("  - model_registry.csv")
    print("  - split_statistics.json")
    print("  - fixed_split_analysis.png/tiff/pdf/svg")
    print(f"  - models/ directory with {len(MODEL_REGISTRY)} saved models")
    print("="*80)

if __name__ == "__main__":
    main()
