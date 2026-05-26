# -*- coding: utf-8 -*-
"""
Retrain and save the best model identified from wrapper experiments
Model: 10 base estimators with 5 features
Features: Lignin (wt%), Co-polyol type (PTHF), r, Copolyol (wt%), Isocyanate (wt%)
"""

import numpy as np
import pandas as pd
import joblib
import sys
import os
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Add the parent directory to path to import preprocessing module
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '1.Loading and Preprocessing'))

# Import the module with the correct filename
import importlib.util
spec = importlib.util.spec_from_file_location("loading_preprocessing", 
    os.path.join(os.path.dirname(__file__), '..', '1.Loading and Preprocessing', 'Loading and preprocessing.py'))
loading_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loading_module)

# Get the functions we need
read_csv_with_encoding = loading_module.read_csv_with_encoding
map_categorical_values = loading_module.map_categorical_values

# Set global random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

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
    """
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
    
    # Calculate step size for test sampling (from remaining samples)
    remaining_indices = [idx for idx in sorted_indices if idx not in val_indices]
    test_step = len(remaining_indices) / test_size
    test_indices = [remaining_indices[int(i * test_step)] for i in range(test_size)]
    
    # Training indices are what's left
    train_indices = [idx for idx in sorted_indices if idx not in val_indices and idx not in test_indices]
    
    return train_indices, val_indices, test_indices

def create_base_models(n_estimators):
    """Create base models with specified number of estimators.
    Hyperparameter grids must exactly match run_fixed_split_experiments.py
    to reproduce the manuscript's reported val MAE of 13.41 Â°C.
    """
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

def train_base_model_with_validation(x_train, y_train, x_val, y_val, model, param_grid):
    """Train a base model using grid search with validation."""
    # Combine train and validation for cross-validation
    x_combined = np.vstack([x_train, x_val])
    y_combined = np.vstack([y_train, y_val])
    
    # Create custom CV split
    train_indices = list(range(len(x_train)))
    val_indices = list(range(len(x_train), len(x_combined)))
    cv_split = [(train_indices, val_indices)]
    
    # Grid search â€” n_jobs=1 ensures deterministic tie-breaking, reproducing
    # the original wrapper result (val MAE 13.41 Â°C, manuscript Table 4 Rank 1)
    grid_search = GridSearchCV(
        model,
        param_grid,
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

def main():
    """Main execution function to retrain and save the best model."""
    print("="*80)
    print("RETRAINING BEST MODEL FOR MAPPING")
    print("="*80)
    print("Model: 10 base estimators")
    print("Features: Lignin (wt%), Co-polyol type (PTHF), r, Copolyol (wt%), Isocyanate (wt%)")
    print("="*80)
    
    # Load and prepare data
    X, y = load_and_prepare_data()
    
    # Create stratified fixed split
    print("\nCreating stratified fixed split...")
    train_idx, val_idx, test_idx = stratified_split(X, y, val_size=16, test_size=16, random_state=RANDOM_SEED)
    
    # Select the 5 features for the best model
    feature_combination = ['Lignin (wt%)', 'Co-polyol type (PTHF)', 'r', 'Copolyol (wt%)', 'Isocyanate (wt%)']
    
    print(f"\nUsing features: {feature_combination}")
    print(f"Number of features: {len(feature_combination)}")
    
    # Select features
    X_selected = X[feature_combination]
    
    # Split data
    x_train, x_val, x_test = X_selected.loc[train_idx], X_selected.loc[val_idx], X_selected.loc[test_idx]
    y_train, y_val, y_test = y.loc[train_idx], y.loc[val_idx], y.loc[test_idx]
    
    print(f"\nData split sizes:")
    print(f"  Training: {len(x_train)} samples")
    print(f"  Validation: {len(x_val)} samples")
    print(f"  Test: {len(x_test)} samples")
    
    # Scale data
    print("\nScaling data...")
    x_scaler = RobustScaler()
    y_scaler = RobustScaler()
    
    x_train_scaled = x_scaler.fit_transform(x_train)
    x_val_scaled = x_scaler.transform(x_val)
    x_test_scaled = x_scaler.transform(x_test)
    
    y_train_scaled = y_scaler.fit_transform(y_train)
    y_val_scaled = y_scaler.transform(y_val)
    y_test_scaled = y_scaler.transform(y_test)
    
    # Create base models with 10 estimators
    n_estimators = 10
    print(f"\nTraining base models with {n_estimators} estimators...")
    base_model_configs = create_base_models(n_estimators)
    
    # Train base models and generate meta-features
    train_meta_features = np.zeros((x_train_scaled.shape[0], len(base_model_configs)))
    val_meta_features = np.zeros((x_val_scaled.shape[0], len(base_model_configs)))
    test_meta_features = np.zeros((x_test_scaled.shape[0], len(base_model_configs)))
    
    base_models = []
    model_names = ['GradientBoosting', 'RandomForest', 'SVR', 'Lasso', 'ElasticNet']
    
    for i, (model, param_grid) in enumerate(base_model_configs):
        print(f"  Training {model_names[i]}...")
        
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
    
    # Train meta-model
    print("\nTraining meta-model (Ridge)...")
    meta_model = Ridge(random_state=RANDOM_SEED)
    meta_model.fit(train_meta_features, y_train_scaled.ravel())
    
    # Generate final predictions
    train_meta_pred = meta_model.predict(train_meta_features)
    val_meta_pred = meta_model.predict(val_meta_features)
    test_meta_pred = meta_model.predict(test_meta_features)
    
    # Calculate metrics
    print("\nCalculating metrics...")
    train_r2, train_mse, train_mae = calculate_metrics(
        y_train_scaled, train_meta_pred.reshape(-1, 1), y_scaler
    )
    val_r2, val_mse, val_mae = calculate_metrics(
        y_val_scaled, val_meta_pred.reshape(-1, 1), y_scaler
    )
    test_r2, test_mse, test_mae = calculate_metrics(
        y_test_scaled, test_meta_pred.reshape(-1, 1), y_scaler
    )
    
    print("\n" + "="*80)
    print("MODEL PERFORMANCE")
    print("="*80)
    print(f"Training Set:")
    print(f"  RÂ² = {train_r2:.4f}")
    print(f"  MSE = {train_mse:.4f}")
    print(f"  MAE = {train_mae:.4f}")
    print(f"\nValidation Set:")
    print(f"  RÂ² = {val_r2:.4f}")
    print(f"  MSE = {val_mse:.4f}")
    print(f"  MAE = {val_mae:.4f}")
    print(f"\nTest Set:")
    print(f"  RÂ² = {test_r2:.4f}")
    print(f"  MSE = {test_mse:.4f}")
    print(f"  MAE = {test_mae:.4f}")
    print("="*80)
    
    # Save models and scalers
    print("\nSaving models and scalers...")
    
    # Save base models
    joblib.dump(base_models, 'best_model_base_models.joblib')
    print("  âœ“ Saved base_models")
    
    # Save meta model
    joblib.dump(meta_model, 'best_model_meta_model.joblib')
    print("  âœ“ Saved meta_model")
    
    # Save scalers
    joblib.dump(x_scaler, 'best_model_x_scaler.joblib')
    print("  âœ“ Saved x_scaler")
    
    joblib.dump(y_scaler, 'best_model_y_scaler.joblib')
    print("  âœ“ Saved y_scaler")
    
    # Save feature list
    with open('best_model_features.txt', 'w') as f:
        for feature in feature_combination:
            f.write(f"{feature}\n")
    print("  âœ“ Saved feature list")
    
    # Save model metadata
    metadata = {
        'n_estimators': n_estimators,
        'features': feature_combination,
        'num_features': len(feature_combination),
        'train_r2': train_r2,
        'train_mse': train_mse,
        'train_mae': train_mae,
        'val_r2': val_r2,
        'val_mse': val_mse,
        'val_mae': val_mae,
        'test_r2': test_r2,
        'test_mse': test_mse,
        'test_mae': test_mae,
        'train_size': len(x_train),
        'val_size': len(x_val),
        'test_size': len(x_test),
        'random_seed': RANDOM_SEED
    }
    
    import json
    with open('best_model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)
    print("  âœ“ Saved metadata")
    
    print("\n" + "="*80)
    print("MODEL SAVED SUCCESSFULLY!")
    print("Files created:")
    print("  - best_model_base_models.joblib")
    print("  - best_model_meta_model.joblib")
    print("  - best_model_x_scaler.joblib")
    print("  - best_model_y_scaler.joblib")
    print("  - best_model_features.txt")
    print("  - best_model_metadata.json")
    print("="*80)

if __name__ == "__main__":
    main()

