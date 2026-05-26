# -*- coding: utf-8 -*-
"""
Analyze Individual Model Performance for Best Feature Combination
Tracks performance of each base model (GB, RF, SVR, Lasso, ElasticNet) and meta-model
across different n_estimators values for the best performing feature combination.
"""

import numpy as np
import pandas as pd
import sys
import os
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.utils import resample

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

def calculate_metrics_with_ci(y_true, y_pred, y_scaler, n_bootstrap=1000, confidence_level=0.95):
    """Calculate R2, MSE, and MAE metrics with confidence intervals using bootstrap."""
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    y_true_unscaled = y_scaler.inverse_transform(y_true)
    y_pred_unscaled = y_scaler.inverse_transform(y_pred)

    # Calculate point estimates
    r2 = r2_score(y_true_unscaled, y_pred_unscaled)
    mse = mean_squared_error(y_true_unscaled, y_pred_unscaled)
    mae = mean_absolute_error(y_true_unscaled, y_pred_unscaled)

    # Bootstrap for confidence intervals
    n_samples = len(y_true_unscaled)
    bootstrap_r2 = []
    bootstrap_mse = []
    bootstrap_mae = []
    
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = resample(np.arange(n_samples), replace=True)
        y_true_boot = y_true_unscaled[indices]
        y_pred_boot = y_pred_unscaled[indices]
        
        # Calculate metrics on bootstrap sample
        if len(np.unique(y_true_boot)) > 1:  # Avoid division by zero in R2
            bootstrap_r2.append(r2_score(y_true_boot, y_pred_boot))
        else:
            bootstrap_r2.append(r2)  # Use original value if can't compute
        
        bootstrap_mse.append(mean_squared_error(y_true_boot, y_pred_boot))
        bootstrap_mae.append(mean_absolute_error(y_true_boot, y_pred_boot))

    # Calculate confidence intervals
    r2_ci_lower = np.percentile(bootstrap_r2, lower_percentile)
    r2_ci_upper = np.percentile(bootstrap_r2, upper_percentile)
    mse_ci_lower = np.percentile(bootstrap_mse, lower_percentile)
    mse_ci_upper = np.percentile(bootstrap_mse, upper_percentile)
    mae_ci_lower = np.percentile(bootstrap_mae, lower_percentile)
    mae_ci_upper = np.percentile(bootstrap_mae, upper_percentile)

    return (r2, r2_ci_lower, r2_ci_upper), (mse, mse_ci_lower, mse_ci_upper), (mae, mae_ci_lower, mae_ci_upper)

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

def load_and_prepare_data():
    """Load and prepare the dataset."""
    print("Loading and preparing data...")

    # Load dataset
    df = pd.read_excel('../../4.Wrapper/Fixed_Stacking_Ensemble/dataset.xlsx')

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

def analyze_individual_models(features, n_estimators_values):
    """
    Analyze individual model performance across different n_estimators values.

    Args:
        features: List of features to use
        n_estimators_values: List of n_estimators values to test

    Returns:
        DataFrame with individual model performances
    """
    print(f"\nAnalyzing individual models for features: {features}")
    print(f"Testing n_estimators values: {n_estimators_values}")

    # Load data
    X, y = load_and_prepare_data()

    # Drop 'Sample name' if it exists
    if 'Sample name' in X.columns:
        X = X.drop('Sample name', axis=1)

    # Create stratified split
    train_idx, val_idx, test_idx = stratified_split(X, y, val_size=16, test_size=16, random_state=RANDOM_SEED)

    # Select features
    X_subset = X[features]

    # Split data using .loc since indices are from the original dataframe
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

    # Store results
    results = []

    model_names = ['GradientBoosting', 'RandomForest', 'SVR', 'Lasso', 'ElasticNet']

    for n_est in n_estimators_values:
        print(f"\n{'='*60}")
        print(f"Testing n_estimators = {n_est}")
        print(f"{'='*60}")

        # Create base models
        base_model_configs = create_base_models(n_est)

        # Train each base model and collect predictions
        base_models = []
        train_meta_features = np.zeros((len(x_train_scaled), len(base_model_configs)))
        val_meta_features = np.zeros((len(x_val_scaled), len(base_model_configs)))
        test_meta_features = np.zeros((len(x_test_scaled), len(base_model_configs)))

        for i, (model, param_grid) in enumerate(base_model_configs):
            print(f"  Training {model_names[i]}...")

            # Train base model
            best_model, val_preds = train_base_model_with_validation(
                x_train_scaled, y_train_scaled, x_val_scaled, y_val_scaled,
                model, param_grid
            )

            # Generate predictions
            train_preds = best_model.predict(x_train_scaled)
            test_preds = best_model.predict(x_test_scaled)

            # Store meta-features
            train_meta_features[:, i] = train_preds
            val_meta_features[:, i] = val_preds
            test_meta_features[:, i] = test_preds

            base_models.append(best_model)

            # Calculate metrics with confidence intervals for this individual model
            train_r2_metrics, train_mse_metrics, train_mae_metrics = calculate_metrics_with_ci(
                y_train_scaled, train_preds.reshape(-1, 1), y_scaler
            )
            val_r2_metrics, val_mse_metrics, val_mae_metrics = calculate_metrics_with_ci(
                y_val_scaled, val_preds.reshape(-1, 1), y_scaler
            )
            test_r2_metrics, test_mse_metrics, test_mae_metrics = calculate_metrics_with_ci(
                y_test_scaled, test_preds.reshape(-1, 1), y_scaler
            )

            # Store individual model results with confidence intervals
            results.append({
                'n_estimators': n_est,
                'Model': model_names[i],
                'Train_R2': train_r2_metrics[0], 'Train_R2_CI_Lower': train_r2_metrics[1], 'Train_R2_CI_Upper': train_r2_metrics[2],
                'Train_MSE': train_mse_metrics[0], 'Train_MSE_CI_Lower': train_mse_metrics[1], 'Train_MSE_CI_Upper': train_mse_metrics[2],
                'Train_MAE': train_mae_metrics[0], 'Train_MAE_CI_Lower': train_mae_metrics[1], 'Train_MAE_CI_Upper': train_mae_metrics[2],
                'Val_R2': val_r2_metrics[0], 'Val_R2_CI_Lower': val_r2_metrics[1], 'Val_R2_CI_Upper': val_r2_metrics[2],
                'Val_MSE': val_mse_metrics[0], 'Val_MSE_CI_Lower': val_mse_metrics[1], 'Val_MSE_CI_Upper': val_mse_metrics[2],
                'Val_MAE': val_mae_metrics[0], 'Val_MAE_CI_Lower': val_mae_metrics[1], 'Val_MAE_CI_Upper': val_mae_metrics[2],
                'Test_R2': test_r2_metrics[0], 'Test_R2_CI_Lower': test_r2_metrics[1], 'Test_R2_CI_Upper': test_r2_metrics[2],
                'Test_MSE': test_mse_metrics[0], 'Test_MSE_CI_Lower': test_mse_metrics[1], 'Test_MSE_CI_Upper': test_mse_metrics[2],
                'Test_MAE': test_mae_metrics[0], 'Test_MAE_CI_Lower': test_mae_metrics[1], 'Test_MAE_CI_Upper': test_mae_metrics[2]
            })

            print(f"    Test MAE: {test_mae_metrics[0]:.3f}°C (±{test_mae_metrics[2]-test_mae_metrics[0]:.3f}), Test R2: {test_r2_metrics[0]:.4f} (±{test_r2_metrics[2]-test_r2_metrics[0]:.4f})")

        # Train meta-model
        print(f"  Training Meta-Model (Ridge)...")
        meta_model = Ridge(random_state=RANDOM_SEED)
        meta_model.fit(train_meta_features, y_train_scaled.ravel())

        # Generate final ensemble predictions
        train_meta_pred = meta_model.predict(train_meta_features)
        val_meta_pred = meta_model.predict(val_meta_features)
        test_meta_pred = meta_model.predict(test_meta_features)

        # Calculate ensemble metrics with confidence intervals
        train_r2_metrics, train_mse_metrics, train_mae_metrics = calculate_metrics_with_ci(
            y_train_scaled, train_meta_pred.reshape(-1, 1), y_scaler
        )
        val_r2_metrics, val_mse_metrics, val_mae_metrics = calculate_metrics_with_ci(
            y_val_scaled, val_meta_pred.reshape(-1, 1), y_scaler
        )
        test_r2_metrics, test_mse_metrics, test_mae_metrics = calculate_metrics_with_ci(
            y_test_scaled, test_meta_pred.reshape(-1, 1), y_scaler
        )

        # Store ensemble results with confidence intervals
        results.append({
            'n_estimators': n_est,
            'Model': 'Ensemble',
            'Train_R2': train_r2_metrics[0], 'Train_R2_CI_Lower': train_r2_metrics[1], 'Train_R2_CI_Upper': train_r2_metrics[2],
            'Train_MSE': train_mse_metrics[0], 'Train_MSE_CI_Lower': train_mse_metrics[1], 'Train_MSE_CI_Upper': train_mse_metrics[2],
            'Train_MAE': train_mae_metrics[0], 'Train_MAE_CI_Lower': train_mae_metrics[1], 'Train_MAE_CI_Upper': train_mae_metrics[2],
            'Val_R2': val_r2_metrics[0], 'Val_R2_CI_Lower': val_r2_metrics[1], 'Val_R2_CI_Upper': val_r2_metrics[2],
            'Val_MSE': val_mse_metrics[0], 'Val_MSE_CI_Lower': val_mse_metrics[1], 'Val_MSE_CI_Upper': val_mse_metrics[2],
            'Val_MAE': val_mae_metrics[0], 'Val_MAE_CI_Lower': val_mae_metrics[1], 'Val_MAE_CI_Upper': val_mae_metrics[2],
            'Test_R2': test_r2_metrics[0], 'Test_R2_CI_Lower': test_r2_metrics[1], 'Test_R2_CI_Upper': test_r2_metrics[2],
            'Test_MSE': test_mse_metrics[0], 'Test_MSE_CI_Lower': test_mse_metrics[1], 'Test_MSE_CI_Upper': test_mse_metrics[2],
            'Test_MAE': test_mae_metrics[0], 'Test_MAE_CI_Lower': test_mae_metrics[1], 'Test_MAE_CI_Upper': test_mae_metrics[2]
        })

        print(f"  Ensemble Test MAE: {test_mae_metrics[0]:.3f}°C (±{test_mae_metrics[2]-test_mae_metrics[0]:.3f}), Test R2: {test_r2_metrics[0]:.4f} (±{test_r2_metrics[2]-test_r2_metrics[0]:.4f})")

    return pd.DataFrame(results)

def main():
    """Main execution function."""
    print("="*80)
    print("INDIVIDUAL MODEL PERFORMANCE ANALYSIS")
    print("Best Feature Combination from Stratified Fixed Split")
    print("="*80)

    # Best performing feature combination (from top_10_models_validation_with_test.csv)
    best_features = ['Lignin (wt%)', 'Co-polyol type (PTHF)', 'r', 'Copolyol (wt%)', 'Isocyanate (wt%)']

    # Test n_estimators values
    n_estimators_values = [1, 10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

    # Analyze individual models
    df_results = analyze_individual_models(best_features, n_estimators_values)

    # Save results
    output_file = 'individual_models_performance.csv'
    df_results.to_csv(output_file, index=False)
    print(f"\n{'='*80}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*80}")

    # Display summary
    print(f"\nSummary Statistics:")
    print(f"\nTest MAE by Model:")
    summary = df_results.groupby('Model')['Test_MAE'].agg(['mean', 'std', 'min', 'max'])
    print(summary)

    print(f"\nTest R2 by Model:")
    summary_r2 = df_results.groupby('Model')['Test_R2'].agg(['mean', 'std', 'min', 'max'])
    print(summary_r2)

if __name__ == "__main__":
    main()

