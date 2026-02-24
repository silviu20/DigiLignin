# -*- coding: utf-8 -*-
"""
Test script to train a single model and measure actual file size
"""

import numpy as np
import pandas as pd
import joblib
import os
import sys
from datetime import datetime
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Lasso, ElasticNet, Ridge
from sklearn.model_selection import GridSearchCV

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
    
    # Define features (use a simple combination)
    features = ['Lignin (wt%)', 'Co-polyol type (PTHF)', 'r', 'Copolyol (wt%)', 'Isocyanate (wt%)']
    X = df_clean[features]
    y = df_clean[['Tg(deg C)']]
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")
    
    return X, y

def stratified_split(X, y, val_size=16, test_size=16, random_state=42):
    """Perform stratified splitting."""
    np.random.seed(random_state)
    
    # Create DataFrame with target for sorting
    data_with_target = X.copy()
    data_with_target['target'] = y.values.ravel()
    
    # Sort by target to enable stratified sampling
    sorted_indices = data_with_target.sort_values('target').index.values
    
    n_samples = len(sorted_indices)
    
    # Use systematic sampling for stratification
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

def get_file_size_mb(filepath):
    """Get file size in MB."""
    size_bytes = os.path.getsize(filepath)
    return size_bytes / (1024 * 1024)

def main():
    """Train one model and measure actual size."""
    print("="*60)
    print("SINGLE MODEL SIZE TEST")
    print("="*60)
    
    # Load data
    X, y = load_and_prepare_data()
    
    # Create split
    train_idx, val_idx, test_idx = stratified_split(X, y, val_size=16, test_size=16, random_state=42)
    
    # Split data
    x_train, x_val, x_test = X.loc[train_idx], X.loc[val_idx], X.loc[test_idx]
    y_train, y_val, y_test = y.loc[train_idx], y.loc[val_idx], y.loc[test_idx]
    
    # Scale data
    x_scaler = RobustScaler()
    y_scaler = RobustScaler()
    
    x_train_scaled = x_scaler.fit_transform(x_train)
    x_val_scaled = x_scaler.transform(x_val)
    y_train_scaled = y_scaler.fit_transform(y_train)
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Test different model types with n_estimators=100
    models_to_test = [
        ('GradientBoosting', GradientBoostingRegressor(n_estimators=100, random_state=42)),
        ('RandomForest', RandomForestRegressor(n_estimators=100, random_state=42)),
        ('SVR', SVR(C=1.0, kernel='rbf')),
        ('Lasso', Lasso(alpha=1.0, random_state=42)),
        ('ElasticNet', ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42)),
        ('Ridge', Ridge(random_state=42))
    ]
    
    print(f"\nTraining and saving models...")
    print("-" * 60)
    
    for model_name, model in models_to_test:
        print(f"Training {model_name}...")
        
        # Train model
        if model_name == 'SVR':
            model.fit(x_train_scaled, y_train_scaled.ravel())
        else:
            model.fit(x_train_scaled, y_train_scaled.ravel())
        
        # Save model
        filename = f"test_{model_name.lower()}_{timestamp}.joblib"
        filepath = os.path.join('models', filename)
        joblib.dump(model, filepath)
        
        # Get file size
        size_mb = get_file_size_mb(filepath)
        
        print(f"  {model_name}: {size_mb:.4f} MB ({size_mb*1024:.2f} KB)")
        
        # Clean up
        os.remove(filepath)
    
    # Test ensemble bundle
    print(f"\nTesting ensemble bundle...")
    print("-" * 60)
    
    # Train a small ensemble
    base_models = []
    model_configs = [
        GradientBoostingRegressor(n_estimators=50, random_state=42),
        RandomForestRegressor(n_estimators=50, random_state=42),
        SVR(C=1.0, kernel='rbf')
    ]
    
    for model in model_configs:
        model.fit(x_train_scaled, y_train_scaled.ravel())
        base_models.append(model)
    
    # Create meta-model
    meta_model = Ridge(random_state=42)
    
    # Generate meta-features
    train_meta = np.zeros((x_train_scaled.shape[0], len(base_models)))
    for i, model in enumerate(base_models):
        train_meta[:, i] = model.predict(x_train_scaled)
    
    meta_model.fit(train_meta, y_train_scaled.ravel())
    
    # Create ensemble bundle
    ensemble_bundle = {
        'x_scaler': x_scaler,
        'y_scaler': y_scaler,
        'base_models': base_models,
        'meta_model': meta_model
    }
    
    # Save ensemble
    ensemble_filename = f"test_ensemble_{timestamp}.joblib"
    ensemble_filepath = os.path.join('models', ensemble_filename)
    joblib.dump(ensemble_bundle, ensemble_filepath)
    
    ensemble_size_mb = get_file_size_mb(ensemble_filepath)
    print(f"Ensemble bundle: {ensemble_size_mb:.4f} MB ({ensemble_size_mb*1024:.2f} KB)")
    
    # Clean up
    os.remove(ensemble_filepath)
    
    print("\n" + "="*60)
    print("SIZE TEST COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
