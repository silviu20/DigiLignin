# -*- coding: utf-8 -*-
"""
Corrected Stacking Ensemble Implementation with OOF Predictions and Nested CV
Addresses data leakage issues in original implementation

Created: 2025-02-19
Purpose: Proper stacking with out-of-fold predictions and unbiased validation
"""

import numpy as np
import pandas as pd
import random
import scipy.stats as stats
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split, RepeatedKFold, GridSearchCV, cross_val_predict
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Set global random seed for reproducibility
RANDOM_SEED = 42

def set_global_random_seed(seed):
    """Set random seed for numpy and random modules."""
    np.random.seed(seed)
    random.seed(seed)

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

def get_base_models():
    """Define base models with their hyperparameter grids."""
    return [
        ('GradientBoosting', GradientBoostingRegressor(), {
            'n_estimators': [1000],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }),
        ('RandomForest', RandomForestRegressor(), {
            'n_estimators': [1000],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }),
        ('SVR', SVR(), {
            'C': [0.1, 1, 10],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto']
        }),
        ('Lasso', Lasso(), {
            'alpha': [0.1, 1, 10],
            'max_iter': [1000, 5000]
        }),
        ('ElasticNet', ElasticNet(), {
            'alpha': [0.1, 1, 10],
            'l1_ratio': [0.1, 0.5, 0.9],
            'max_iter': [1000, 5000]
        })
    ]

def generate_oof_predictions(X, y, cv_splits, base_models):
    """
    Generate out-of-fold predictions for meta-features.
    This prevents data leakage in stacking.
    """
    oof_predictions = np.zeros((len(X), len(base_models)))
    fitted_base_models = []
    
    print("Generating OOF predictions for meta-features...")
    
    for i, (name, model, param_grid) in enumerate(base_models):
        print(f"Training {name} with OOF predictions...")
        
        # Store predictions for each fold
        fold_predictions = np.zeros(len(X))
        
        for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train = y.iloc[train_idx]
            
            # Scale data
            X_train_scaled, X_scaler = scale_columns_with_robust_scaler(X_train)
            X_val_scaled = X_scaler.transform(X_val)
            y_train_scaled, y_scaler = scale_columns_with_robust_scaler(y_train)
            
            # Hyperparameter tuning on training fold only
            grid_search = GridSearchCV(
                estimator=model, 
                param_grid=param_grid, 
                cv=3,  # Inner CV for hyperparameter tuning
                scoring='neg_mean_squared_error'
            )
            grid_search.fit(X_train_scaled, y_train_scaled.ravel())
            
            # Predict on validation fold (OOF)
            val_pred_scaled = grid_search.best_estimator_.predict(X_val_scaled)
            val_pred = y_scaler.inverse_transform(val_pred_scaled.reshape(-1, 1)).ravel()
            fold_predictions[val_idx] = val_pred
        
        oof_predictions[:, i] = fold_predictions
        
        # Fit final model on full dataset for later use
        X_scaled, X_scaler = scale_columns_with_robust_scaler(X)
        y_scaled, y_scaler = scale_columns_with_robust_scaler(y)
        
        grid_search = GridSearchCV(
            estimator=model, 
            param_grid=param_grid, 
            cv=5, 
            scoring='neg_mean_squared_error'
        )
        grid_search.fit(X_scaled, y_scaled.ravel())
        fitted_base_models.append((name, grid_search.best_estimator_, X_scaler, y_scaler))
        
    return oof_predictions, fitted_base_models

def train_meta_model(oof_predictions, y):
    """Train meta-model on OOF predictions only."""
    print("Training meta-model on OOF predictions...")
    
    # Scale OOF predictions and target
    oof_scaled, oof_scaler = scale_columns_with_robust_scaler(oof_predictions)
    y_scaled, y_scaler = scale_columns_with_robust_scaler(y)
    
    # Train meta-model
    meta_model = Ridge()
    meta_model.fit(oof_scaled, y_scaled.ravel())
    
    return meta_model, oof_scaler, y_scaler

def nested_cross_validation(X, y, outer_cv_splits=5, inner_cv_splits=3):
    """
    Perform nested cross-validation for unbiased performance estimation.
    Outer CV: Performance estimation
    Inner CV: Hyperparameter tuning
    """
    print("Performing nested cross-validation...")
    
    # Outer CV splits
    outer_cv = RepeatedKFold(n_splits=outer_cv_splits, n_repeats=1, random_state=RANDOM_SEED)
    outer_splits = list(outer_cv.split(X))
    
    outer_scores = {'r2': [], 'mse': [], 'mae': []}
    
    for outer_fold, (train_idx, test_idx) in enumerate(outer_splits):
        print(f"\nOuter Fold {outer_fold + 1}/{len(outer_splits)}")
        
        X_train_outer, X_test_outer = X.iloc[train_idx], X.iloc[test_idx]
        y_train_outer, y_test_outer = y.iloc[train_idx], y.iloc[test_idx]
        
        # Inner CV for hyperparameter tuning
        inner_cv = RepeatedKFold(n_splits=inner_cv_splits, n_repeats=1, random_state=RANDOM_SEED + outer_fold)
        inner_splits = list(inner_cv.split(X_train_outer))
        
        # Generate OOF predictions on outer training data
        base_models = get_base_models()
        oof_predictions, fitted_base_models = generate_oof_predictions(
            X_train_outer, y_train_outer, inner_splits, base_models
        )
        
        # Train meta-model on OOF predictions
        meta_model, oof_scaler, y_scaler = train_meta_model(oof_predictions, y_train_outer)
        
        # Evaluate on outer test set (never seen during training)
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
        
        # Calculate metrics
        r2, mse, mae = calculate_metrics(
            y_test_outer.values.reshape(-1, 1), 
            test_pred.reshape(-1, 1), 
            y_scaler
        )
        
        outer_scores['r2'].append(r2)
        outer_scores['mse'].append(mse)
        outer_scores['mae'].append(mae)
        
        print(f"Outer Fold {outer_fold + 1} - R²: {r2:.3f}, MSE: {mse:.3f}, MAE: {mae:.3f}")
    
    # Calculate final scores with confidence intervals
    final_results = {}
    for metric in ['r2', 'mse', 'mae']:
        mean_val = np.mean(outer_scores[metric])
        ci_lower, ci_upper = calculate_confidence_intervals(outer_scores[metric])
        final_results[metric] = {
            'mean': mean_val,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'raw_scores': outer_scores[metric]
        }
    
    return final_results

def train_final_model_with_held_out_test(X, y, test_size=0.2):
    """
    Train final model with strict held-out test set.
    Returns model trained on training data and evaluation on untouched test data.
    """
    print(f"Training final model with {test_size*100}% held-out test set...")
    
    # Split into train/test (never use test data during training)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_SEED
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Generate CV splits for training data only
    cv_splits = RepeatedKFold(n_splits=5, n_repeats=2, random_state=RANDOM_SEED)
    train_cv_splits = list(cv_splits.split(X_train))
    
    # Generate OOF predictions on training data
    base_models = get_base_models()
    oof_predictions, fitted_base_models = generate_oof_predictions(
        X_train, y_train, train_cv_splits, base_models
    )
    
    # Train meta-model on OOF predictions
    meta_model, oof_scaler, y_scaler = train_meta_model(oof_predictions, y_train)
    
    # Evaluate on held-out test set (completely unbiased)
    test_meta_features = []
    for name, model, X_scaler, y_scaler_model in fitted_base_models:
        X_test_scaled = X_scaler.transform(X_test)
        test_pred_scaled = model.predict(X_test_scaled)
        test_pred = y_scaler_model.inverse_transform(test_pred_scaled.reshape(-1, 1)).ravel()
        test_meta_features.append(test_pred)
    
    test_meta_features = np.column_stack(test_meta_features)
    test_meta_scaled = oof_scaler.transform(test_meta_features)
    test_pred_scaled = meta_model.predict(test_meta_scaled)
    test_pred = y_scaler.inverse_transform(test_pred_scaled.reshape(-1, 1)).ravel()
    
    # Calculate test metrics
    test_r2, test_mse, test_mae = calculate_metrics(
        y_test.values.reshape(-1, 1), 
        test_pred.reshape(-1, 1), 
        y_scaler
    )
    
    test_results = {
        'r2': test_r2,
        'mse': test_mse,
        'mae': test_mae,
        'y_true': y_test.values,
        'y_pred': test_pred,
        'X_test': X_test
    }
    
    # Save final models
    final_models = {
        'base_models': fitted_base_models,
        'meta_model': meta_model,
        'oof_scaler': oof_scaler,
        'y_scaler': y_scaler
    }
    
    return final_models, test_results

def plot_unbiased_results(test_results, save_prefix='unbiased'):
    """
    Plot results using only held-out test predictions (no data leakage).
    """
    y_true = test_results['y_true']
    y_pred = test_results['y_pred']
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Regression plot
    ax1.scatter(y_true, y_true, color='blue', alpha=0.6, label='Actual Values')
    ax1.scatter(y_true, y_pred, color='red', alpha=0.6, label='Test Predictions')
    ax1.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'k--', lw=2, label='Ideal Fit')
    ax1.set_xlabel('Actual Values', fontsize=14)
    ax1.set_ylabel('Predicted Values', fontsize=14)
    ax1.set_title('Test Set: Actual vs Predicted', fontsize=16)
    ax1.grid(linestyle='--', alpha=0.7)
    ax1.legend(prop={'size': 10})
    
    # Add metrics text
    r2, mse, mae = test_results['r2'], test_results['mse'], test_results['mae']
    ax1.text(0.05, 0.95, f'R² = {r2:.3f}\nMSE = {mse:.3f}\nMAE = {mae:.3f}', 
             transform=ax1.transAxes, fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Residual plot
    residuals = y_true - y_pred
    ax2.scatter(y_pred, residuals, alpha=0.6, color='green', label='Residuals')
    ax2.axhline(y=0, color='k', linestyle='--', label='Zero Residual Line')
    ax2.set_xlabel('Predicted Values', fontsize=14)
    ax2.set_ylabel('Residuals', fontsize=14)
    ax2.set_title('Test Set: Residual Plot', fontsize=16)
    ax2.grid(linestyle='--', alpha=0.7)
    ax2.legend(prop={'size': 10})
    
    plt.tight_layout()
    
    # Save plots
    for ext in ['tiff', 'pdf', 'eps', 'svg', 'jpg']:
        plt.savefig(f'{save_prefix}_test_set_performance.{ext}', dpi=600, bbox_inches='tight')
    
    plt.show()
    
    print(f"\n=== UNBIASED TEST SET PERFORMANCE ===")
    print(f"R²: {r2:.3f}")
    print(f"MSE: {mse:.3f}")
    print(f"MAE: {mae:.3f}")
    print(f"Note: These are unbiased estimates using held-out test data only")

def main_analysis(X, y):
    """
    Main analysis pipeline with both nested CV and held-out test set evaluation.
    """
    print("=== CORRECTED STACKING ENSEMBLE ANALYSIS ===")
    print("Addressing data leakage with OOF predictions and unbiased validation\n")
    
    # 1. Nested Cross-Validation (most unbiased performance estimate)
    print("1. NESTED CROSS-VALIDATION RESULTS:")
    nested_results = nested_cross_validation(X, y)
    
    print("\nNested CV Performance (95% CI):")
    for metric in ['r2', 'mse', 'mae']:
        mean_val = nested_results[metric]['mean']
        ci_lower = nested_results[metric]['ci_lower']
        ci_upper = nested_results[metric]['ci_upper']
        print(f"{metric.upper()}: {mean_val:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]")
    
    # 2. Held-out Test Set Evaluation
    print("\n2. HELD-OUT TEST SET RESULTS:")
    final_models, test_results = train_final_model_with_held_out_test(X, y)
    
    # 3. Plot unbiased results
    plot_unbiased_results(test_results, 'corrected_stacking')
    
    # 4. Save final models
    joblib.dump(final_models, 'corrected_stacked_models.joblib')
    print("\nFinal models saved to 'corrected_stacked_models.joblib'")
    
    return nested_results, test_results, final_models

if __name__ == "__main__":
    # Load your data here
    # Example:
    # df = pd.read_csv('your_data.csv')
    # X = df[['Lignin (wt%)', 'Ratio', 'Co-polyol type (PTHF)', 'Isocyanate (mmol NCO)', 
    #         'Isocyanate type', 'Tin(II) octoate', 'Swelling ratio (%)']]
    # y = df[['Tg (°C)']]
    
    print("Please load your data and uncomment the example above to run the analysis.")
    print("This implementation provides:")
    print("1. Proper OOF predictions for meta-features")
    print("2. Nested cross-validation for unbiased performance estimation")
    print("3. Strict held-out test set evaluation")
    print("4. Plots showing only true test predictions (no data leakage)")
