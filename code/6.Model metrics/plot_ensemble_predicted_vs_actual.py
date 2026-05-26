# -*- coding: utf-8 -*-
"""
Plot predicted vs actual values for Stratified Stacked Ensemble
Uses the best performing ensemble model with confidence intervals
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Set random seed
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def stratified_split(X, y, val_size=16, test_size=16, random_state=42):
    """Perform stratified splitting to ensure representative distribution."""
    np.random.seed(random_state)
    
    # Create DataFrame with target for sorting
    data_with_target = X.copy()
    data_with_target['target'] = y.values.ravel()
    
    # Sort by target to enable stratified sampling
    sorted_indices = data_with_target.sort_values('target').index.values
    
    n_samples = len(sorted_indices)
    
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
    """Train a base model with hyperparameter tuning using validation set."""
    from sklearn.linear_model import Lasso, ElasticNet
    
    # Combine train and validation for GridSearchCV
    x_combined = np.vstack([x_train, x_val])
    y_combined = np.concatenate([y_train.ravel(), y_val.ravel()])
    
    # Create validation split indices for GridSearchCV
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

def load_and_prepare_data():
    """Load and prepare the dataset."""
    print("Loading and preparing data...")

    # Load dataset
    df = pd.read_excel('../4.Wrapper/Fixed_Stacking_Ensemble/dataset.xlsx')

    # Remove rows with NaN values in target variable
    df_clean = df.dropna(subset=['Tg(deg C)'])
    print(f"Dataset shape after cleaning: {df_clean.shape}")

    # Map categorical values
    isocyanate_mapping = {'N3600': 1, 'HDI': 0, 0: np.nan}
    if 'Isocyonate type' in df_clean.columns:
        df_clean['Isocyonate type'] = df_clean['Isocyonate type'].map(isocyanate_mapping)
        df_clean = df_clean.fillna(0)

    # Define features (best performing combination)
    X = df_clean[['Lignin (wt%)', 'Co-polyol type (PTHF)', 'r', 'Copolyol (wt%)', 'Isocyanate (wt%)']]
    y = df_clean[['Tg(deg C)']]

    print(f"Feature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")

    return X, y

def plot_ensemble_results():
    """Plot ensemble predicted vs actual values with confidence intervals."""
    
    print("="*80)
    print("ENSEMBLE PREDICTED VS ACTUAL PLOTS")
    print("="*80)
    
    # Load data
    X, y = load_and_prepare_data()
    
    # Best performing feature combination
    best_features = ['Lignin (wt%)', 'Co-polyol type (PTHF)', 'r', 'Copolyol (wt%)', 'Isocyanate (wt%)']
    
    # Use best n_estimators (10) based on previous analysis
    best_n_estimators = 10
    
    # Create stratified split
    train_idx, val_idx, test_idx = stratified_split(X, y, val_size=16, test_size=16, random_state=RANDOM_SEED)
    
    # Select features
    X_subset = X[best_features]
    
    # Split data
    x_train = X_subset.loc[train_idx]
    x_val = X_subset.loc[val_idx]
    x_test = X_subset.loc[test_idx]
    y_train = y.loc[train_idx]
    y_val = y.loc[val_idx]
    y_test = y.loc[test_idx]
    
    # Scale features
    x_scaler = RobustScaler()
    x_train_scaled = x_scaler.fit_transform(x_train)
    x_val_scaled = x_scaler.transform(x_val)
    x_test_scaled = x_scaler.transform(x_test)
    
    # Scale target
    y_scaler = RobustScaler()
    y_train_scaled = y_scaler.fit_transform(y_train)
    y_val_scaled = y_scaler.transform(y_val)
    y_test_scaled = y_scaler.transform(y_test)
    
    # Train ensemble with best n_estimators
    print(f"Training ensemble with {best_n_estimators} estimators...")
    
    # Create base models
    base_model_configs = create_base_models(best_n_estimators)
    
    # Train base models
    base_models = []
    train_meta_features = np.zeros((len(x_train_scaled), len(base_model_configs)))
    val_meta_features = np.zeros((len(x_val_scaled), len(base_model_configs)))
    test_meta_features = np.zeros((len(x_test_scaled), len(base_model_configs)))
    
    for i, (model, param_grid) in enumerate(base_model_configs):
        print(f"  Training base model {i+1}/5...")
        
        # Train base model
        best_model, val_preds = train_base_model_with_validation(
            x_train_scaled, y_train_scaled, x_val_scaled, y_val_scaled,
            model, param_grid
        )
        
        # Generate predictions
        train_preds = best_model.predict(x_train_scaled)
        val_preds = best_model.predict(x_val_scaled)
        test_preds = best_model.predict(x_test_scaled)
        
        # Store meta-features
        train_meta_features[:, i] = train_preds
        val_meta_features[:, i] = val_preds
        test_meta_features[:, i] = test_preds
        
        base_models.append(best_model)
    
    # Train meta-model
    print("  Training meta-model...")
    meta_model = Ridge(random_state=RANDOM_SEED)
    meta_model.fit(train_meta_features, y_train_scaled.ravel())
    
    # Generate predictions
    train_meta_pred = meta_model.predict(train_meta_features)
    val_meta_pred = meta_model.predict(val_meta_features)
    test_meta_pred = meta_model.predict(test_meta_features)
    
    # Unscale predictions
    train_pred = y_scaler.inverse_transform(train_meta_pred.reshape(-1, 1))
    val_pred = y_scaler.inverse_transform(val_meta_pred.reshape(-1, 1))
    test_pred = y_scaler.inverse_transform(test_meta_pred.reshape(-1, 1))
    
    # Get actual values
    y_train_true = y_scaler.inverse_transform(y_train_scaled)
    y_val_true = y_scaler.inverse_transform(y_val_scaled)
    y_test_true = y_scaler.inverse_transform(y_test_scaled)
    
    # Combine all data for plotting
    y_true_all = np.concatenate([y_train_true.ravel(), y_val_true.ravel(), y_test_true.ravel()])
    y_pred_all = np.concatenate([train_pred.ravel(), val_pred.ravel(), test_pred.ravel()])
    
    # Calculate separate metrics for each split
    train_correlation = np.corrcoef(y_train_true.ravel(), train_pred.ravel())[0, 1]
    val_correlation = np.corrcoef(y_val_true.ravel(), val_pred.ravel())[0, 1]
    test_correlation = np.corrcoef(y_test_true.ravel(), test_pred.ravel())[0, 1]
    
    train_r2 = r2_score(y_train_true, train_pred)
    val_r2 = r2_score(y_val_true, val_pred)
    test_r2 = r2_score(y_test_true, test_pred)
    
    train_mae = mean_absolute_error(y_train_true, train_pred)
    val_mae = mean_absolute_error(y_val_true, val_pred)
    test_mae = mean_absolute_error(y_test_true, test_pred)
    
    # Combined metrics (for overall correlation display)
    correlation_coef = np.corrcoef(y_true_all, y_pred_all)[0, 1]
    r2 = r2_score(y_true_all, y_pred_all)
    mae = mean_absolute_error(y_true_all, y_pred_all)
    
    print(f"\nPerformance Metrics by Split:")
    print(f"Train: r = {train_correlation:.4f}, R² = {train_r2:.4f}, MAE = {train_mae:.2f}°C")
    print(f"Validation: r = {val_correlation:.4f}, R² = {val_r2:.4f}, MAE = {val_mae:.2f}°C")
    print(f"Test: r = {test_correlation:.4f}, R² = {test_r2:.4f}, MAE = {test_mae:.2f}°C")
    print(f"\nOverall (Combined): r = {correlation_coef:.4f}, R² = {r2:.4f}, MAE = {mae:.2f}°C")
    
    # Use validation + test for plotting (more realistic performance)
    y_val_test_true = np.concatenate([y_val_true.ravel(), y_test_true.ravel()])
    y_val_test_pred = np.concatenate([val_pred.ravel(), test_pred.ravel()])
    val_test_correlation = np.corrcoef(y_val_test_true, y_val_test_pred)[0, 1]
    val_test_mae = mean_absolute_error(y_val_test_true, y_val_test_pred)
    
    # Create plots with exact original style
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.patch.set_facecolor('white')
    ax1.set_facecolor('white')
    ax2.set_facecolor('white')

    # Define common tick intervals (same as original)
    x_interval = 10  # Interval for x-axis
    y_interval = 10  # Interval for y-axis
    residual_y_interval = 10  # Interval for y-axis on the residual plot (changed from 5 to 10)

    # Plot A: Predicted vs Actual (validation + test only - more realistic)
    # Add grid first so it's behind the points
    ax1.grid(True, linestyle='--', alpha=0.7, color='#E0E0E0')
    
    # Original scatter plot style - blue actual, red predicted
    ax1.scatter(y_val_test_true, y_val_test_true, color='blue', alpha=0.6, label='Actual Values')
    ax1.scatter(y_val_test_true, y_val_test_pred, color='red', alpha=0.6, label='Predicted Values')
    ax1.plot([min(y_val_test_true), max(y_val_test_true)], [min(y_val_test_true), max(y_val_test_true)], 'k--', lw=2, label='Ideal Fit')
    
    ax1.set_xlabel('Actual Values', fontsize=14)
    ax1.set_ylabel('Predicted Values', fontsize=14)
    ax1.tick_params(axis='x', labelsize=14)
    ax1.tick_params(axis='y', labelsize=14)
    ax1.xaxis.set_major_locator(plt.MultipleLocator(x_interval))
    ax1.yaxis.set_major_locator(plt.MultipleLocator(y_interval))
    
    # Add correlation coefficient and MAE for validation+test (more meaningful)
    ax1.text(0.05, 0.88, f'r = {val_test_correlation:.4f}\nMAE = {val_test_mae:.2f}°C', 
             transform=ax1.transAxes, fontsize=12, 
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    
    # Legend with white background for plot A (original style)
    legend1 = ax1.legend(prop={'size': 10}, 
                        facecolor='white', 
                        framealpha=1,
                        edgecolor='#666666')
    
    ax1.set_xlim(min(y_val_test_true), max(y_val_test_true))
    ax1.set_ylim(min(y_val_test_true), max(y_val_test_true))
    ax1.text(0.85, 0.95, 'A', transform=ax1.transAxes, fontsize=16, 
             color='black', fontweight='bold', ha='center', va='center')

    # Plot B: Residuals (validation + test only)
    # Add grid first so it's behind the points
    ax2.grid(True, linestyle='--', alpha=0.7, color='#E0E0E0')
    
    residuals = y_val_test_true - y_val_test_pred
    ax2.scatter(y_val_test_pred, residuals, alpha=0.6, color='green', label='Residuals')
    ax2.axhline(y=0, color='k', linestyle='--', label='Zero Residual Line')
    ax2.set_xlabel('Predicted Values', fontsize=14)
    ax2.set_ylabel('Residuals', fontsize=14)
    ax2.tick_params(axis='x', labelsize=14)
    ax2.tick_params(axis='y', labelsize=14)
    
    # Set major tick intervals
    ax2.xaxis.set_major_locator(plt.MultipleLocator(x_interval))
    
    # Calculate appropriate y-axis limits and ticks for residual plot (original method)
    max_abs_residual = max(abs(max(residuals)), abs(min(residuals)))
    y_max = np.ceil(max_abs_residual / residual_y_interval) * residual_y_interval
    y_min = -y_max
    
    ax2.set_ylim(y_min, y_max)
    ax2.yaxis.set_major_locator(plt.MultipleLocator(residual_y_interval))
    
    # Add horizontal grid lines
    ax2.yaxis.grid(True, linestyle='--', alpha=0.7, color='#E0E0E0')
    
    # Legend with white background for plot B (original style)
    legend2 = ax2.legend(prop={'size': 10}, 
                        facecolor='white', 
                        framealpha=1,
                        edgecolor='#666666')
    
    ax2.set_xlim(min(y_val_test_pred), max(y_val_test_pred))
    ax2.text(0.85, 0.95, 'B', transform=ax2.transAxes, fontsize=16, 
             color='black', fontweight='bold', ha='center', va='center')

    # Make spines (plot borders) slightly grey (original style)
    for ax in [ax1, ax2]:
        for spine in ax.spines.values():
            spine.set_edgecolor('#666666')

    # Adjust the spacing between subplots (original style)
    plt.subplots_adjust(wspace=0.2)
    
    # Save in multiple formats
    for ext in ['tiff', 'pdf', 'png', 'svg', 'jpg']:
        plt.savefig(f'Ensemble_Predicted_vs_Actual.{ext}', 
                   dpi=600, bbox_inches='tight', facecolor='white')
    
    plt.show()
    
    return correlation_coef, r2, mae

if __name__ == "__main__":
    correlation, r2, mae = plot_ensemble_results()
    print(f"\nPlot saved as 'Ensemble_Predicted_vs_Actual.*'")
