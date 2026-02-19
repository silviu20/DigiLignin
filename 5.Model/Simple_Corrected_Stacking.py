# -*- coding: utf-8 -*-
"""
Simple Fix for Corrected Stacking Implementation
Fix the scaling issues causing unrealistic predictions

Created: 2025-02-19
Purpose: Provide a working corrected stacking method
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Lasso, ElasticNet, Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

def simple_corrected_stacking(X, y, test_size=0.2, random_state=42):
    """
    Simple corrected stacking implementation that avoids scaling issues.
    """
    print("=== SIMPLE CORRECTED STACKING ===")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Target range: {y.min().values[0]:.1f} to {y.max().values[0]:.1f}°C")
    
    # Define base models (simplified)
    base_models = [
        ('GradientBoosting', GradientBoostingRegressor(n_estimators=100, random_state=42)),
        ('RandomForest', RandomForestRegressor(n_estimators=100, random_state=42)),
        ('SVR', SVR(C=1.0, kernel='rbf')),
        ('Lasso', Lasso(alpha=1.0, max_iter=1000)),
        ('ElasticNet', ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=1000))
    ]
    
    # Train base models and generate OOF predictions using CV
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Initialize OOF predictions
    oof_predictions = np.zeros((len(X_train), len(base_models)))
    test_predictions = np.zeros((len(X_test), len(base_models)))
    
    print("\nTraining base models with OOF predictions...")
    
    for i, (name, model) in enumerate(base_models):
        print(f"  Training {name}...")
        
        # OOF predictions for training set
        oof_pred = np.zeros(len(X_train))
        test_pred = np.zeros(len(X_test))
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            # Train model
            model.fit(X_fold_train, y_fold_train.values.ravel())
            
            # OOF prediction
            oof_pred[val_idx] = model.predict(X_fold_val)
        
        # Train on full training data for test predictions
        model.fit(X_train, y_train.values.ravel())
        test_pred = model.predict(X_test)
        
        oof_predictions[:, i] = oof_pred
        test_predictions[:, i] = test_pred
        
        # Check individual model performance
        train_r2 = r2_score(y_train, oof_pred)
        train_mae = mean_absolute_error(y_train, oof_pred)
        print(f"    {name} - R²: {train_r2:.3f}, MAE: {train_mae:.1f}°C")
    
    # Train meta-model on OOF predictions
    print("\nTraining meta-model on OOF predictions...")
    meta_model = Ridge(alpha=1.0)
    meta_model.fit(oof_predictions, y_train.values.ravel())
    
    # Make final predictions
    final_predictions = meta_model.predict(test_predictions)
    
    # Calculate metrics
    test_r2 = r2_score(y_test, final_predictions)
    test_mae = mean_absolute_error(y_test, final_predictions)
    test_mse = mean_squared_error(y_test, final_predictions)
    
    print(f"\nFinal Test Performance:")
    print(f"  R²: {test_r2:.3f}")
    print(f"  MAE: {test_mae:.1f}°C")
    print(f"  MSE: {test_mse:.1f}")
    
    # Check prediction ranges
    print(f"\nPrediction ranges:")
    print(f"  Actual: {y_test.min().values[0]:.1f} to {y_test.max().values[0]:.1f}°C")
    print(f"  Predicted: {final_predictions.min():.1f} to {final_predictions.max():.1f}°C")
    
    return {
        'predictions': final_predictions,
        'actual': y_test.values.ravel(),
        'metrics': {
            'r2': test_r2,
            'mae': test_mae,
            'mse': test_mse
        }
    }

def main():
    """
    Main function to test the simple corrected stacking.
    """
    print("="*60)
    print("SIMPLE CORRECTED STACKING TEST")
    print("="*60)
    
    # Load data
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
    
    # Run simple corrected stacking
    results = simple_corrected_stacking(X, y)
    
    # Compare with other methods
    print(f"\n" + "="*60)
    print("COMPARISON OF ALL METHODS")
    print("="*60)
    
    print(f"1. Original (with data leakage):")
    print(f"   R²: 0.998, MAE: 0.8°C ❌ (Severely biased)")
    print(f"")
    print(f"2. Original (proper splits):")
    print(f"   R²: 0.268, MAE: 18.2°C ✅ (Realistic)")
    print(f"")
    print(f"3. Corrected (nested CV - BUGGY):")
    print(f"   R²: 0.280, MAE: 777°C ❌ (Scaling error)")
    print(f"")
    print(f"4. Corrected (simple version):")
    print(f"   R²: {results['metrics']['r2']:.3f}, MAE: {results['metrics']['mae']:.1f}°C ✅ (Fixed)")
    
    # Assessment
    print(f"\nASSESSMENT:")
    if results['metrics']['mae'] < 50:
        print(f"✅ The simple corrected method works properly!")
        print(f"   MAE of {results['metrics']['mae']:.1f}°C is realistic for Tg prediction")
        print(f"   R² of {results['metrics']['r2']:.3f} shows modest predictive power")
    else:
        print(f"❌ Still issues with the corrected method")
    
    print(f"\nRECOMMENDATION:")
    print(f"• Use the simple corrected method for unbiased evaluation")
    print(f"• The original method with proper splits (R²=0.268, MAE=18.2°C) is also valid")
    print(f"• Avoid the nested CV version until scaling issues are fixed")
    print(f"• Never use the original leaky version (R²=0.998, MAE=0.8°C)")

if __name__ == "__main__":
    main()
