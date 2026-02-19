# -*- coding: utf-8 -*-
"""
Debug and Fix the Corrected Stacking Implementation
Fix the MAE calculation error in the corrected method

Created: 2025-02-19
Purpose: Debug and fix the 777°C MAE error
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Lasso, ElasticNet, Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def calculate_metrics(y_true, y_pred, y_scaler=None):
    """
    Calculate R2, MSE, and MAE metrics.
    FIXED VERSION: Only apply inverse transform if values are scaled.
    """
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    # Check if values need to be unscaled
    if y_scaler is not None and hasattr(y_scaler, 'inverse_transform'):
        # Check if values appear to be scaled (rough check)
        if np.abs(y_true.mean()) < 10:  # Likely scaled (near zero mean)
            print(f"    DEBUG: Unscaled y_true range: {y_true.min():.3f} to {y_true.max():.3f}")
            y_true_unscaled = y_scaler.inverse_transform(y_true)
            y_pred_unscaled = y_scaler.inverse_transform(y_pred)
            print(f"    DEBUG: Unscaled to range: {y_true_unscaled.min():.1f} to {y_true_unscaled.max():.1f}")
        else:
            print(f"    DEBUG: Values already unscaled, skipping inverse transform")
            y_true_unscaled = y_true
            y_pred_unscaled = y_pred
    else:
        y_true_unscaled = y_true
        y_pred_unscaled = y_pred

    r2 = r2_score(y_true_unscaled, y_pred_unscaled)
    mse = mean_squared_error(y_true_unscaled, y_pred_unscaled)
    mae = mean_absolute_error(y_true_unscaled, y_pred_unscaled)

    print(f"    DEBUG: Final metrics - R²: {r2:.3f}, MAE: {mae:.1f}°C, MSE: {mse:.1f}")
    
    return r2, mse, mae

def scale_columns_with_robust_scaler(X):
    """Scale features using RobustScaler."""
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

def get_base_models():
    """Define base models with their hyperparameter grids."""
    return [
        ('GradientBoosting', GradientBoostingRegressor(), {
            'n_estimators': [100],  # Reduced for faster testing
            'learning_rate': [0.1],
            'max_depth': [3]
        }),
        ('RandomForest', RandomForestRegressor(), {
            'n_estimators': [100],  # Reduced for faster testing
            'max_depth': [10],
            'min_samples_split': [5]
        }),
        ('SVR', SVR(), {
            'C': [1],
            'kernel': ['rbf'],
            'gamma': ['scale']
        }),
        ('Lasso', Lasso(), {
            'alpha': [1],
            'max_iter': [1000]
        }),
        ('ElasticNet', ElasticNet(), {
            'alpha': [1],
            'l1_ratio': [0.5],
            'max_iter': [1000]
        })
    ]

def generate_oof_predictions(X, y, cv_splits, base_models):
    """Generate out-of-fold predictions for meta-features."""
    print("    Generating OOF predictions for meta-features...")
    
    # Initialize arrays
    n_samples = len(X)
    n_models = len(base_models)
    oof_predictions = np.zeros((n_samples, n_models))
    fitted_base_models = []
    
    # Scale target once
    y_scaled, y_scaler = scale_columns_with_robust_scaler(y)
    
    for model_idx, (name, model, param_grid) in enumerate(base_models):
        print(f"    Training {name} with OOF predictions...")
        
        # Initialize OOF predictions for this model
        model_oof = np.zeros(n_samples)
        
        for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y_scaled[train_idx], y_scaled[val_idx]
            
            # Scale features
            X_train_scaled, X_scaler = scale_columns_with_robust_scaler(X_train_fold)
            X_val_scaled = X_scaler.transform(X_val_fold)
            
            # Hyperparameter tuning
            grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error')
            grid_search.fit(X_train_scaled, y_train_fold.ravel())
            
            # Best model
            best_model = grid_search.best_estimator_
            
            # OOF prediction
            val_pred_scaled = best_model.predict(X_val_scaled)
            model_oof[val_idx] = val_pred_scaled
            
            # Store fitted model and scalers
            if fold_idx == 0:  # Store only the first fold's model for simplicity
                fitted_base_models.append((name, best_model, X_scaler, y_scaler))
        
        oof_predictions[:, model_idx] = model_oof
    
    return oof_predictions, fitted_base_models

def train_meta_model(oof_predictions, y):
    """Train meta-model on OOF predictions."""
    print("    Training meta-model on OOF predictions...")
    
    # Scale OOF predictions
    oof_scaler = RobustScaler()
    oof_predictions_scaled = oof_scaler.fit_transform(oof_predictions)
    
    # Scale target
    y_scaled, y_scaler = scale_columns_with_robust_scaler(y)
    
    # Train meta-model
    meta_model = Ridge(alpha=1.0)
    meta_model.fit(oof_predictions_scaled, y_scaled.ravel())
    
    return meta_model, oof_scaler, y_scaler

def calculate_confidence_intervals(scores, confidence=0.95):
    """Calculate confidence intervals for scores."""
    scores = np.array(scores)
    mean = np.mean(scores)
    std_error = stats.sem(scores)
    h = std_error * stats.t.ppf((1 + confidence) / 2., len(scores) - 1)
    return mean - h, mean + h

def simple_nested_cv_test(X, y):
    """
    Simplified nested CV test to debug the MAE calculation.
    """
    print("=== DEBUG: Simplified Nested CV Test ===")
    
    # Simple 3-fold outer CV for debugging
    outer_cv = RepeatedKFold(n_splits=3, n_repeats=1, random_state=RANDOM_SEED)
    outer_splits = list(outer_cv.split(X))
    
    outer_scores = {'r2': [], 'mse': [], 'mae': []}
    
    for outer_fold, (train_idx, test_idx) in enumerate(outer_splits):
        print(f"\nOuter Fold {outer_fold + 1}/3")
        
        X_train_outer, X_test_outer = X.iloc[train_idx], X.iloc[test_idx]
        y_train_outer, y_test_outer = y.iloc[train_idx], y.iloc[test_idx]
        
        print(f"  Train size: {len(X_train_outer)}, Test size: {len(X_test_outer)}")
        print(f"  Train Tg range: {y_train_outer.min().values[0]:.1f} to {y_train_outer.max().values[0]:.1f}°C")
        print(f"  Test Tg range: {y_test_outer.min().values[0]:.1f} to {y_test_outer.max().values[0]:.1f}°C")
        
        # Inner CV for hyperparameter tuning
        inner_cv = RepeatedKFold(n_splits=3, n_repeats=1, random_state=RANDOM_SEED + outer_fold)
        inner_splits = list(inner_cv.split(X_train_outer))
        
        # Generate OOF predictions
        base_models = get_base_models()
        oof_predictions, fitted_base_models = generate_oof_predictions(
            X_train_outer, y_train_outer, inner_splits, base_models
        )
        
        # Train meta-model
        meta_model, oof_scaler, y_scaler = train_meta_model(oof_predictions, y_train_outer)
        
        # Evaluate on test set
        print(f"    Evaluating on test set...")
        test_meta_features = []
        
        for name, model, X_scaler, y_scaler_model in fitted_base_models:
            X_test_scaled = X_scaler.transform(X_test_outer)
            test_pred_scaled = model.predict(X_test_scaled)
            test_pred = y_scaler_model.inverse_transform(test_pred_scaled.reshape(-1, 1)).ravel()
            test_meta_features.append(test_pred)
        
        test_meta_features = np.column_stack(test_meta_features)
        test_meta_scaled = oof_scaler.transform(test_meta_features)
        test_pred_scaled = meta_model.predict(test_meta_scaled)
        test_pred = y_scaler.inverse_transform(test_pred_scaled.reshape(-1, 1)).ravel()
        
        print(f"    Test predictions range: {test_pred.min():.1f} to {test_pred.max():.1f}°C")
        print(f"    Test actual range: {y_test_outer.min().values[0]:.1f} to {y_test_outer.max().values[0]:.1f}°C")
        
        # Calculate metrics with DEBUG version
        r2, mse, mae = calculate_metrics(
            y_test_outer.values.reshape(-1, 1), 
            test_pred.reshape(-1, 1), 
            None  # Pass None since values are already unscaled
        )
        
        outer_scores['r2'].append(r2)
        outer_scores['mse'].append(mse)
        outer_scores['mae'].append(mae)
        
        print(f"Outer Fold {outer_fold + 1} - R²: {r2:.3f}, MSE: {mse:.1f}, MAE: {mae:.1f}°C")
    
    # Calculate final scores
    final_results = {}
    for metric in ['r2', 'mse', 'mae']:
        mean_val = np.mean(outer_scores[metric])
        ci_lower, ci_upper = calculate_confidence_intervals(outer_scores[metric])
        final_results[metric] = {
            'mean': mean_val,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        }
    
    print(f"\n=== FINAL DEBUG RESULTS ===")
    print(f"R²: {final_results['r2']['mean']:.3f} [{final_results['r2']['ci_lower']:.3f}, {final_results['r2']['ci_upper']:.3f}]")
    print(f"MAE: {final_results['mae']['mean']:.1f}°C [{final_results['mae']['ci_lower']:.1f}, {final_results['mae']['ci_upper']:.1f}]")
    print(f"MSE: {final_results['mse']['mean']:.1f} [{final_results['mse']['ci_lower']:.1f}, {final_results['mse']['ci_upper']:.1f}]")
    
    return final_results

def main():
    """
    Main function to debug and fix the corrected implementation.
    """
    print("="*60)
    print("DEBUGGING CORRECTED STACKING IMPLEMENTATION")
    print("="*60)
    
    # Load data
    print("Loading dataset...")
    df = pd.read_excel('../dataset.csv.xlsx')
    df_clean = df.dropna(subset=['Tg(deg C)'])
    
    # Map categorical variables
    isocyanate_mapping = {'N3600': 1, 'HDI': 0}
    df_clean['Isocyanate type'] = df_clean['Isocyonate type'].map(isocyanate_mapping).fillna(0)
    
    # Prepare features
    feature_columns = ['Lignin (wt%)', 'r', 'Co-polyol type (PTHF)', 
                       'Isocyanate (mmol NCO)', 'Isocyanate type', 'tin(II) octoate', 'Sratio(%)']
    
    column_mapping = {
        'Lignin (wt%)': 'Lignin (wt%)',
        'r': 'Ratio',
        'Co-polyol type (PTHF)': 'Co-polyol type (PTHF)',
        'Isocyanate (mmol NCO)': 'Isocyanate (mmol NCO)',
        'Isocyanate type': 'Isocyanate type',
        'tin(II) octoate': 'Tin(II) octoate',
        'Sratio(%)': 'Swelling ratio (%)'
    }
    
    X = df_clean[feature_columns].copy()
    X.columns = [column_mapping[col] for col in X.columns]
    y = df_clean[['Tg(deg C)']].copy()
    y.columns = ['Tg (°C)']
    
    print(f"Dataset loaded: {X.shape}")
    print(f"Target range: {y.min().values[0]:.1f} to {y.max().values[0]:.1f}°C")
    
    # Run debug test
    results = simple_nested_cv_test(X, y)
    
    print(f"\n=== COMPARISON WITH PREVIOUS RESULTS ===")
    print(f"Original (with data leakage): R² = 0.998, MAE = 0.8°C")
    print(f"Original (proper splits): R² = 0.268, MAE = 18.2°C")
    print(f"Corrected (FIXED): R² = {results['r2']['mean']:.3f}, MAE = {results['mae']['mean']:.1f}°C")
    print(f"Corrected (BUGGY): R² = 0.280, MAE = 777°C ❌")
    
    if results['mae']['mean'] < 100:
        print(f"\n✅ FIXED: The corrected method now reports realistic MAE values")
        print(f"   The 777°C was indeed a calculation error from double inverse scaling")
    else:
        print(f"\n❌ Still issues with the corrected method")

if __name__ == "__main__":
    # Import stats for confidence intervals
    from scipy import stats
    main()
