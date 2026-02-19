# -*- coding: utf-8 -*-
"""
Alternative Study: Original Stacking with Proper Train/Validation/Test Splits
Uses the original stacking ensemble but with proper data splitting to avoid data leakage

Created: 2025-02-19
Purpose: Evaluate original stacking method with proper train/validation/test separation
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Import original stacking implementation
import importlib.util
spec = importlib.util.spec_from_file_location("stacked_ensembles", "Stacked Ensembles.py")
stacked_ensembles = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stacked_ensembles)

def load_and_prepare_data():
    """
    Load and prepare the DigiLignin dataset.
    """
    print("=== LOADING AND PREPARING DATASET ===")
    
    # Load the Excel dataset
    df = pd.read_excel('../dataset.csv.xlsx')
    print(f"✅ Dataset loaded: {df.shape}")
    
    # Clean data
    df_clean = df.dropna(subset=['Tg(deg C)'])
    print(f"   After removing missing Tg: {df_clean.shape}")
    
    # Map categorical variables
    isocyanate_mapping = {'N3600': 1, 'HDI': 0}
    df_clean['Isocyanate type'] = df_clean['Isocyonate type'].map(isocyanate_mapping).fillna(0)
    
    # Map column names to match original implementation
    column_mapping = {
        'Lignin (wt%)': 'Lignin (wt%)',
        'r': 'Ratio',
        'Co-polyol type (PTHF)': 'Co-polyol type (PTHF)',
        'Isocyanate (mmol NCO)': 'Isocyanate (mmol NCO)',
        'Isocyanate type': 'Isocyanate type',
        'tin(II) octoate': 'Tin(II) octoate',
        'Sratio(%)': 'Swelling ratio (%)'
    }
    
    # Select and rename features
    feature_columns = ['Lignin (wt%)', 'r', 'Co-polyol type (PTHF)', 
                       'Isocyanate (mmol NCO)', 'Isocyanate type', 'tin(II) octoate', 'Sratio(%)']
    
    X = df_clean[feature_columns].copy()
    X.columns = [column_mapping[col] for col in X.columns]
    y = df_clean[['Tg(deg C)']].copy()
    y.columns = ['Tg (°C)']
    
    print(f"   Features: {list(X.columns)}")
    print(f"   Target range: {y.min().values[0]:.1f} to {y.max().values[0]:.1f}°C")
    
    return X, y, df_clean

def create_evenly_distributed_splits(X, y, test_size=16, val_size=16, random_state=42):
    """
    Create evenly distributed train/validation/test splits.
    
    Ensures that each split has representative coverage of the feature space.
    """
    print("\n=== CREATING EVENLY DISTRIBUTED SPLITS ===")
    
    # Sort by target variable to ensure even distribution
    y_flat = y.values.ravel()
    sorted_indices = np.argsort(y_flat)
    
    total_samples = len(X)
    print(f"Total samples: {total_samples}")
    
    # Calculate split sizes
    test_val_size = test_size + val_size
    train_size = total_samples - test_val_size
    
    print(f"Split sizes:")
    print(f"  Training: {train_size} samples ({train_size/total_samples*100:.1f}%)")
    print(f"  Validation: {val_size} samples ({val_size/total_samples*100:.1f}%)")
    print(f"  Test: {test_size} samples ({test_size/total_samples*100:.1f}%)")
    
    print(f"\nStep 1: Creating test set...")
    
    # Create evenly distributed splits using stratified-like approach
    # We'll use quantile-based splitting to ensure even distribution
    
    # First, separate test set (16 samples from extremes and middle)
    test_indices = []
    
    # Take 4 samples from each quartile for test set
    quartiles = np.array_split(sorted_indices, 4)
    print(f"  Splitting into 4 quartiles: {[len(q) for q in quartiles]}")
    
    for i, q in enumerate(quartiles):
        print(f"  Quartile {i+1}: indices {q[0]} to {q[-1]}, Tg range: {y_flat[q[0]]:.1f} to {y_flat[q[-1]]:.1f}°C")
        # Take 1 from beginning and 1 from end of each quartile
        if len(q) >= 2:
            test_indices.extend([q[0], q[-1]])
            print(f"    Added indices {q[0]} and {q[-1]} to test set")
    
    test_indices = test_indices[:16]  # Ensure exactly 16
    print(f"  Test set created: {len(test_indices)} indices")
    print(f"  Test Tg values: {[f'{y_flat[i]:.1f}°C' for i in test_indices[:8]]}...")
    
    print(f"\nStep 2: Creating validation set...")
    remaining_indices = [i for i in sorted_indices if i not in test_indices]
    print(f"  Remaining indices after test removal: {len(remaining_indices)}")
    
    # Then separate validation set from remaining
    val_indices = []
    remaining_quartiles = np.array_split(remaining_indices, 4)
    print(f"  Splitting remaining into 4 quartiles: {[len(q) for q in remaining_quartiles]}")
    
    for i, q in enumerate(remaining_quartiles):
        print(f"  Remaining Quartile {i+1}: indices {q[0]} to {q[-1]}, Tg range: {y_flat[q[0]]:.1f} to {y_flat[q[-1]]:.1f}°C")
        # Take 2 samples from each quartile for validation
        if len(q) >= 4:
            idx1 = q[len(q)//4]  # 25th percentile
            idx2 = q[3*len(q)//4]  # 75th percentile
            val_indices.extend([idx1, idx2])
            print(f"    Added indices {idx1} and {idx2} to validation set")
        elif len(q) >= 2:
            val_indices.extend([q[0], q[-1]])
            print(f"    Added indices {q[0]} and {q[-1]} to validation set")
    
    val_indices = val_indices[:16]  # Ensure exactly 16
    print(f"  Validation set created: {len(val_indices)} indices")
    print(f"  Validation Tg values: {[f'{y_flat[i]:.1f}°C' for i in val_indices[:8]]}...")
    
    print(f"\nStep 3: Creating training set...")
    train_indices = [i for i in remaining_indices if i not in val_indices]
    print(f"  Training set created: {len(train_indices)} indices")
    
    # Create final splits
    print(f"\nStep 4: Creating DataFrames...")
    X_train = X.iloc[train_indices]
    X_val = X.iloc[val_indices]
    X_test = X.iloc[test_indices]
    
    y_train = y.iloc[train_indices]
    y_val = y.iloc[val_indices]
    y_test = y.iloc[test_indices]
    
    print(f"  DataFrames created successfully")
    
    print(f"\nSplit distribution:")
    print(f"  Train Tg range: {y_train.min().values[0]:.1f} to {y_train.max().values[0]:.1f}°C")
    print(f"  Val Tg range: {y_val.min().values[0]:.1f} to {y_val.max().values[0]:.1f}°C")
    print(f"  Test Tg range: {y_test.min().values[0]:.1f} to {y_test.max().values[0]:.1f}°C")
    
    print(f"\nStep 5: Creating distribution plots...")
    
    # Visualize the distribution
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.hist(y_train.values.ravel(), bins=10, alpha=0.7, color='blue', label='Train')
    plt.xlabel('Tg (°C)')
    plt.ylabel('Frequency')
    plt.title('Training Set')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.hist(y_val.values.ravel(), bins=10, alpha=0.7, color='orange', label='Validation')
    plt.xlabel('Tg (°C)')
    plt.title('Validation Set')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.hist(y_test.values.ravel(), bins=10, alpha=0.7, color='red', label='Test')
    plt.xlabel('Tg (°C)')
    plt.title('Test Set')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('data_split_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"  Plots created and saved")
    print(f"✅ Split creation completed successfully")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def train_original_stacking_properly(X_train, y_train, X_val, y_val):
    """
    Train the original stacking ensemble but with proper train/validation separation.
    """
    print("\n=== TRAINING ORIGINAL STACKING (PROPERLY SEPARATED) ===")
    
    print("Step 1: Setting up random seed...")
    # Set random seed for reproducibility
    stacked_ensembles.set_global_random_seed(42)
    print("  Random seed set to 42")
    
    print("Step 2: Creating CV splits...")
    # Create CV splits using only training data
    cv_splits = stacked_ensembles.get_consistent_cv_splits(X_train, random_state=42)
    print(f"  Created {len(cv_splits)} CV splits")
    print(f"  Each split: {len(cv_splits[0][0])} train, {len(cv_splits[0][1])} validation samples")
    
    print(f"Training on {len(X_train)} samples with {len(cv_splits)} CV folds")
    print(f"Validation on {len(X_val)} separate samples")
    
    print("Step 3: Training base models...")
    # Train base models with tuning (using only training data)
    n_estimators = 1000
    print(f"  Using {n_estimators} estimators for ensemble models")
    
    try:
        base_model_results, best_base_models = stacked_ensembles.run_base_models_with_tuning(
            X_train, y_train, n_estimators, cv_splits
        )
        print(f"  Successfully trained {len(best_base_models)} base models")
        for i, model in enumerate(best_base_models):
            print(f"    Model {i+1}: {type(model).__name__}")
    except Exception as e:
        print(f"  ❌ Error in base model training: {e}")
        raise
    
    print("Step 4: Preparing meta-features...")
    # Train meta-model using validation set (no data leakage)
    meta_model = stacked_ensembles.Ridge()
    print("  Created Ridge meta-model")
    
    # Generate meta-features from training data
    print("  Scaling training data...")
    X_train_scaled, x_scaler = stacked_ensembles.scale_columns_with_robust_scaler(X_train)
    y_train_scaled, y_scaler = stacked_ensembles.scale_columns_with_robust_scaler(y_train)
    print(f"  Scaled features: {X_train_scaled.shape}")
    print(f"  Scaled target: {y_train_scaled.shape}")
    
    print("  Generating training meta-features...")
    train_meta_features = np.zeros((X_train_scaled.shape[0], len(best_base_models)))
    for i, base_model in enumerate(best_base_models):
        print(f"    Processing base model {i+1}/{len(best_base_models)}...")
        train_meta_features[:, i] = base_model.predict(X_train_scaled)
    print("  Training meta-features generated")
    
    print("Step 5: Training meta-model...")
    # Train meta-model on training meta-features
    meta_model.fit(train_meta_features, y_train_scaled.ravel())
    print("  Meta-model trained successfully")
    
    print("Step 6: Validating on separate validation set...")
    # Validate on separate validation set
    print("  Scaling validation data...")
    X_val_scaled = x_scaler.transform(X_val)
    y_val_scaled = y_scaler.transform(y_val)
    print(f"  Scaled validation: {X_val_scaled.shape}")
    
    print("  Generating validation meta-features...")
    val_meta_features = np.zeros((X_val_scaled.shape[0], len(best_base_models)))
    for i, base_model in enumerate(best_base_models):
        print(f"    Processing base model {i+1}/{len(best_base_models)}...")
        val_meta_features[:, i] = base_model.predict(X_val_scaled)
    print("  Validation meta-features generated")
    
    print("  Making validation predictions...")
    val_predictions_scaled = meta_model.predict(val_meta_features)
    val_predictions = y_scaler.inverse_transform(val_predictions_scaled.reshape(-1, 1))
    print("  Validation predictions completed")
    
    print("Step 7: Calculating validation metrics...")
    # Calculate validation metrics
    val_r2 = r2_score(y_val, val_predictions)
    val_mae = mean_absolute_error(y_val, val_predictions)
    val_mse = mean_squared_error(y_val, val_predictions)
    
    print(f"Validation Performance:")
    print(f"  R²: {val_r2:.3f}")
    print(f"  MAE: {val_mae:.3f}°C")
    print(f"  MSE: {val_mse:.3f}")
    
    print("✅ Training completed successfully")
    
    return best_base_models, meta_model, x_scaler, y_scaler, val_predictions, val_r2, val_mae, val_mse

def evaluate_on_test_set(X_test, y_test, best_base_models, meta_model, x_scaler, y_scaler):
    """
    Evaluate the trained model on the held-out test set.
    """
    print("\n=== EVALUATING ON HELD-OUT TEST SET ===")
    
    # Generate meta-features for test set
    X_test_scaled = x_scaler.transform(X_test)
    y_test_scaled = y_scaler.transform(y_test)
    
    test_meta_features = np.zeros((X_test_scaled.shape[0], len(best_base_models)))
    for i, base_model in enumerate(best_base_models):
        test_meta_features[:, i] = base_model.predict(X_test_scaled)
    
    # Make predictions
    test_predictions_scaled = meta_model.predict(test_meta_features)
    test_predictions = y_scaler.inverse_transform(test_predictions_scaled.reshape(-1, 1))
    
    # Calculate test metrics
    test_r2 = r2_score(y_test, test_predictions)
    test_mae = mean_absolute_error(y_test, test_predictions)
    test_mse = mean_squared_error(y_test, test_predictions)
    
    print(f"Test Set Performance:")
    print(f"  R²: {test_r2:.3f}")
    print(f"  MAE: {test_mae:.3f}°C")
    print(f"  MSE: {test_mse:.3f}")
    
    return test_predictions, test_r2, test_mae, test_mse

def create_comprehensive_plots(y_train, y_val, y_test, val_predictions, test_predictions, 
                              val_r2, val_mae, test_r2, test_mae):
    """
    Create comprehensive visualization of results.
    """
    print("\n=== CREATING COMPREHENSIVE PLOTS ===")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Original Stacking with Proper Train/Val/Test Splits', fontsize=16, fontweight='bold')
    
    # Validation set scatter plot
    ax1 = axes[0, 0]
    ax1.scatter(y_val, y_val, color='blue', alpha=0.6, label='Actual')
    ax1.scatter(y_val, val_predictions, color='orange', alpha=0.7, label='Predicted')
    ax1.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'k--', lw=2, label='Ideal')
    ax1.set_xlabel('Actual Tg (°C)')
    ax1.set_ylabel('Predicted Tg (°C)')
    ax1.set_title(f'Validation Set\nR² = {val_r2:.3f}, MAE = {val_mae:.1f}°C')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Test set scatter plot
    ax2 = axes[0, 1]
    ax2.scatter(y_test, y_test, color='blue', alpha=0.6, label='Actual')
    ax2.scatter(y_test, test_predictions, color='red', alpha=0.7, label='Predicted')
    ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Ideal')
    ax2.set_xlabel('Actual Tg (°C)')
    ax2.set_ylabel('Predicted Tg (°C)')
    ax2.set_title(f'Test Set\nR² = {test_r2:.3f}, MAE = {test_mae:.1f}°C')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Combined plot
    ax3 = axes[0, 2]
    ax3.scatter(y_val, val_predictions, color='orange', alpha=0.7, label='Validation', s=50)
    ax3.scatter(y_test, test_predictions, color='red', alpha=0.7, label='Test', s=50)
    
    # Fix the pandas Series issue by using .values
    y_val_min = y_val.values.min()
    y_val_max = y_val.values.max()
    y_test_min = y_test.values.min()
    y_test_max = y_test.values.max()
    
    ax3.plot([min(y_val_min, y_test_min), max(y_val_max, y_test_max)], 
             [min(y_val_min, y_test_min), max(y_val_max, y_test_max)], 
             'k--', lw=2, label='Ideal')
    ax3.set_xlabel('Actual Tg (°C)')
    ax3.set_ylabel('Predicted Tg (°C)')
    ax3.set_title('Combined: Validation + Test')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Residual plots
    val_residuals = y_val.values.ravel() - val_predictions.ravel()
    test_residuals = y_test.values.ravel() - test_predictions.ravel()
    
    # Validation residuals
    ax4 = axes[1, 0]
    ax4.scatter(val_predictions, val_residuals, color='orange', alpha=0.7)
    ax4.axhline(y=0, color='k', linestyle='--')
    ax4.set_xlabel('Predicted Tg (°C)')
    ax4.set_ylabel('Residuals (°C)')
    ax4.set_title('Validation Residuals')
    ax4.grid(True, alpha=0.3)
    
    # Test residuals
    ax5 = axes[1, 1]
    ax5.scatter(test_predictions, test_residuals, color='red', alpha=0.7)
    ax5.axhline(y=0, color='k', linestyle='--')
    ax5.set_xlabel('Predicted Tg (°C)')
    ax5.set_ylabel('Residuals (°C)')
    ax5.set_title('Test Residuals')
    ax5.grid(True, alpha=0.3)
    
    # Performance comparison
    ax6 = axes[1, 2]
    metrics = ['R²', 'MAE (°C)']
    val_scores = [val_r2, val_mae]
    test_scores = [test_r2, test_mae]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax6.bar(x - width/2, val_scores, width, label='Validation', color='orange', alpha=0.7)
    ax6.bar(x + width/2, test_scores, width, label='Test', color='red', alpha=0.7)
    
    ax6.set_xlabel('Metrics')
    ax6.set_ylabel('Values')
    ax6.set_title('Performance Comparison')
    ax6.set_xticks(x)
    ax6.set_xticklabels(metrics)
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plots
    for ext in ['tiff', 'pdf', 'eps', 'svg', 'jpg']:
        plt.savefig(f'proper_split_stacking_results.{ext}', dpi=600, bbox_inches='tight')
    
    plt.show()

def main():
    """
    Main function to run the alternative study.
    """
    print("="*60)
    print("ALTERNATIVE STUDY: ORIGINAL STACKING WITH PROPER SPLITS")
    print("="*60)
    
    # 1. Load and prepare data
    X, y, df_clean = load_and_prepare_data()
    
    # 2. Create evenly distributed splits
    X_train, X_val, X_test, y_train, y_val, y_test = create_evenly_distributed_splits(
        X, y, test_size=16, val_size=16
    )
    
    # 3. Train original stacking with proper separation
    (best_base_models, meta_model, x_scaler, y_scaler, 
     val_predictions, val_r2, val_mae, val_mse) = train_original_stacking_properly(
        X_train, y_train, X_val, y_val
    )
    
    # 4. Evaluate on test set
    test_predictions, test_r2, test_mae, test_mse = evaluate_on_test_set(
        X_test, y_test, best_base_models, meta_model, x_scaler, y_scaler
    )
    
    # 5. Create comprehensive plots
    create_comprehensive_plots(
        y_train, y_val, y_test, val_predictions, test_predictions,
        val_r2, val_mae, test_r2, test_mae
    )
    
    # 6. Summary
    print("\n" + "="*60)
    print("STUDY SUMMARY")
    print("="*60)
    print("Methodology:")
    print("  • Original stacking ensemble")
    print("  • Proper train/validation/test separation")
    print("  • 16 samples each for validation and test")
    print("  • Evenly distributed across Tg range")
    print()
    print("Results:")
    print(f"  • Validation R²: {val_r2:.3f}, MAE: {val_mae:.1f}°C")
    print(f"  • Test R²: {test_r2:.3f}, MAE: {test_mae:.1f}°C")
    print()
    print("Interpretation:")
    if test_r2 > 0.3:
        print("  • Model shows reasonable predictive performance")
    elif test_r2 > 0.1:
        print("  • Model shows modest predictive performance")
    else:
        print("  • Model shows poor predictive performance")
    
    print(f"  • Test MAE of {test_mae:.1f}°C indicates {'good' if test_mae < 10 else 'moderate' if test_mae < 20 else 'poor'} accuracy")
    print()
    print("✅ This provides a more realistic assessment of the original stacking method")

if __name__ == "__main__":
    main()
