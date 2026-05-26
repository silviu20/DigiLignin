# -*- coding: utf-8 -*-
"""
Individual Models and Stacked Ensemble Performance Analysis for Model #6 at 700 Base Estimators
Extracts performance metrics for each base model and the final stacked ensemble
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import RepeatedKFold, GridSearchCV, cross_val_predict
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import scipy.stats as stats

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

# Set global random seed
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def create_base_models(n_estimators):
    """Create base model configurations with hyperparameter grids."""
    return [
        ('GBR', GradientBoostingRegressor(random_state=RANDOM_SEED), {
            'n_estimators': [n_estimators],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }),
        ('RF', RandomForestRegressor(random_state=RANDOM_SEED), {
            'n_estimators': [n_estimators],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }),
        ('SVR', SVR(), {
            'C': [0.1, 1, 10],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto']
        }),
        ('Lasso', Lasso(random_state=RANDOM_SEED), {
            'alpha': [0.1, 1, 10],
            'max_iter': [1000, 5000]
        }),
        ('ElasticNet', ElasticNet(random_state=RANDOM_SEED), {
            'alpha': [0.1, 1, 10],
            'l1_ratio': [0.1, 0.5, 0.9],
            'max_iter': [1000, 5000]
        })
    ]

def evaluate_model_performance(X, y, model_name, model, param_grid, cv_splits):
    """Evaluate individual model performance on training and validation sets."""
    training_metrics = {'r2': [], 'mse': [], 'mae': []}
    validation_metrics = {'r2': [], 'mse': [], 'mae': []}
    
    print(f"Evaluating {model_name}...")
    
    for fold_idx, (train_index, val_index) in enumerate(cv_splits):
        # Split data
        x_train, x_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        
        # Scale data
        x_scaler = RobustScaler()
        y_scaler = RobustScaler()
        x_train_scaled = x_scaler.fit_transform(x_train)
        x_val_scaled = x_scaler.transform(x_val)
        y_train_scaled = y_scaler.fit_transform(y_train)
        y_val_scaled = y_scaler.transform(y_val)
        
        # Tune hyperparameters
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        grid_search.fit(x_train_scaled, y_train_scaled.ravel())
        best_model = grid_search.best_estimator_
        
        # Training predictions
        y_train_pred_scaled = best_model.predict(x_train_scaled)
        y_train_pred = y_scaler.inverse_transform(y_train_pred_scaled.reshape(-1, 1))
        y_train_true = y_scaler.inverse_transform(y_train_scaled)
        
        # Validation predictions
        y_val_pred_scaled = best_model.predict(x_val_scaled)
        y_val_pred = y_scaler.inverse_transform(y_val_pred_scaled.reshape(-1, 1))
        y_val_true = y_scaler.inverse_transform(y_val_scaled)
        
        # Calculate training metrics
        train_r2 = r2_score(y_train_true, y_train_pred)
        train_mse = mean_squared_error(y_train_true, y_train_pred)
        train_mae = mean_absolute_error(y_train_true, y_train_pred)
        
        # Calculate validation metrics
        val_r2 = r2_score(y_val_true, y_val_pred)
        val_mse = mean_squared_error(y_val_true, y_val_pred)
        val_mae = mean_absolute_error(y_val_true, y_val_pred)
        
        training_metrics['r2'].append(train_r2)
        training_metrics['mse'].append(train_mse)
        training_metrics['mae'].append(train_mae)
        
        validation_metrics['r2'].append(val_r2)
        validation_metrics['mse'].append(val_mse)
        validation_metrics['mae'].append(val_mae)
    
    # Calculate average metrics
    avg_train_r2 = np.mean(training_metrics['r2'])
    avg_train_mse = np.mean(training_metrics['mse'])
    avg_train_mae = np.mean(training_metrics['mae'])
    
    avg_val_r2 = np.mean(validation_metrics['r2'])
    avg_val_mse = np.mean(validation_metrics['mse'])
    avg_val_mae = np.mean(validation_metrics['mae'])
    
    generalizability = avg_val_mae - avg_train_mae
    
    return {
        'model_name': model_name,
        'train_r2': avg_train_r2,
        'train_mse': avg_train_mse,
        'train_mae': avg_train_mae,
        'val_r2': avg_val_r2,
        'val_mse': avg_val_mse,
        'val_mae': avg_val_mae,
        'generalizability': generalizability
    }

def evaluate_stacked_ensemble(X, y, feature_combination, n_estimators, cv_splits):
    """Evaluate stacked ensemble performance using OOF predictions."""
    base_model_configs = create_base_models(n_estimators)
    
    validation_metrics = {'r2': [], 'mse': [], 'mae': []}
    training_metrics = {'r2': [], 'mse': [], 'mae': []}
    
    print(f"Evaluating Stacked Ensemble...")
    
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
        
        for i, (model_name, model, param_grid) in enumerate(base_model_configs):
            # Generate OOF predictions
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=5,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            grid_search.fit(x_train_scaled, y_train_scaled.ravel())
            best_model = grid_search.best_estimator_
            
            # OOF predictions for meta-training
            oof_preds = cross_val_predict(
                best_model,
                x_train_scaled,
                y_train_scaled.ravel(),
                cv=5,
                n_jobs=-1
            )
            oof_meta_features[:, i] = oof_preds
            
            # Validation predictions
            val_meta_features[:, i] = best_model.predict(x_val_scaled)
        
        # Train meta-model on OOF predictions
        meta_model = Ridge(random_state=RANDOM_SEED)
        meta_model.fit(oof_meta_features, y_train_scaled.ravel())
        
        # Generate predictions
        train_meta_pred = meta_model.predict(oof_meta_features)
        val_meta_pred = meta_model.predict(val_meta_features)
        
        # Calculate training metrics
        train_r2, train_mse, train_mae = calculate_metrics(y_train_scaled, train_meta_pred.reshape(-1, 1), y_scaler)
        
        # Calculate validation metrics
        val_r2, val_mse, val_mae = calculate_metrics(y_val_scaled, val_meta_pred.reshape(-1, 1), y_scaler)
        
        training_metrics['r2'].append(train_r2)
        training_metrics['mse'].append(train_mse)
        training_metrics['mae'].append(train_mae)
        
        validation_metrics['r2'].append(val_r2)
        validation_metrics['mse'].append(val_mse)
        validation_metrics['mae'].append(val_mae)
    
    # Calculate average metrics
    avg_train_r2 = np.mean(training_metrics['r2'])
    avg_train_mse = np.mean(training_metrics['mse'])
    avg_train_mae = np.mean(training_metrics['mae'])
    
    avg_val_r2 = np.mean(validation_metrics['r2'])
    avg_val_mse = np.mean(validation_metrics['mse'])
    avg_val_mae = np.mean(validation_metrics['mae'])
    
    generalizability = avg_val_mae - avg_train_mae
    
    return {
        'model_name': 'Stacking Ensemble',
        'train_r2': avg_train_r2,
        'train_mse': avg_train_mse,
        'train_mae': avg_train_mae,
        'val_r2': avg_val_r2,
        'val_mse': avg_val_mse,
        'val_mae': avg_val_mae,
        'generalizability': generalizability
    }

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

def load_and_prepare_data():
    """Load and prepare the dataset using Model #6 feature combination."""
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
    
    # Model #6 feature combination (best without Swelling ratio)
    model6_features = ['Lignin (wt%)', 'Co-polyol type (PTHF)', 'r', 'Copolyol (wt%)', 'Isocyonate type']
    
    X = df_clean[model6_features]
    y = df_clean[['Tg(deg C)']]
    
    print(f"Using Model #6 features: {model6_features}")
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")
    
    return X, y, model6_features

def main():
    """Main execution function."""
    
    print("="*80)
    print("MODEL #6 INDIVIDUAL MODELS PERFORMANCE ANALYSIS (700 ESTIMATORS)")
    print("="*80)
    
    # Load and prepare data
    X, y, model6_features = load_and_prepare_data()
    
    # Create CV splits
    cv = RepeatedKFold(n_splits=5, n_repeats=2, random_state=RANDOM_SEED)
    cv_splits = list(cv.split(X))
    
    # Evaluate individual models
    base_model_configs = create_base_models(700)
    results = []
    
    for model_name, model, param_grid in base_model_configs:
        result = evaluate_model_performance(X, y, model_name, model, param_grid, cv_splits)
        results.append(result)
    
    # Evaluate stacked ensemble
    ensemble_result = evaluate_stacked_ensemble(X, y, model6_features, 700, cv_splits)
    results.append(ensemble_result)
    
    # Create performance table
    performance_df = pd.DataFrame(results)
    
    # Format the table
    formatted_table = pd.DataFrame({
        'Model': performance_df['model_name'],
        'Training R2': performance_df['train_r2'].round(3),
        'Training MSE': performance_df['train_mse'].round(2),
        'Training MAE': performance_df['train_mae'].round(2),
        'Validation R2': performance_df['val_r2'].round(3),
        'Validation MSE': performance_df['val_mse'].round(2),
        'Validation MAE': performance_df['val_mae'].round(2),
        'Generalizability': performance_df['generalizability'].round(2)
    })
    
    # Save to CSV
    formatted_table.to_csv('Model6_700_Individual_Models_Performance.csv', index=False)
    
    # Display results
    print("\n" + "="*80)
    print("INDIVIDUAL MODELS AND STACKED ENSEMBLE PERFORMANCE")
    print("="*80)
    print(formatted_table.to_string(index=False))
    
    # Analysis
    print("\n" + "="*80)
    print("PERFORMANCE ANALYSIS")
    print("="*80)
    
    best_train_model = formatted_table.loc[formatted_table['Training R2'].idxmax(), 'Model']
    best_val_model = formatted_table.loc[formatted_table['Validation R2'].idxmax(), 'Model']
    lowest_mae_model = formatted_table.loc[formatted_table['Validation MAE'].idxmin(), 'Model']
    best_generalizability = formatted_table.loc[formatted_table['Generalizability'].abs().idxmin(), 'Model']
    
    print(f"Best Training Performance (R²): {best_train_model}")
    print(f"Best Validation Performance (R²): {best_val_model}")
    print(f"Lowest Validation MAE: {lowest_mae_model}")
    print(f"Best Generalizability (closest to 0): {best_generalizability}")
    
    # Generalizability analysis
    print(f"\nGeneralizability Analysis:")
    print(f"  Negative values indicate underfitting")
    print(f"  Positive values indicate overfitting")
    print(f"  Values closer to 0 indicate better generalizability")
    
    print("\n" + "="*80)
    print("Generated Files:")
    print("  - Model6_700_Individual_Models_Performance.csv")
    print("="*80)

if __name__ == "__main__":
    main()
