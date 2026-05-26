# -*- coding: utf-8 -*-
"""
Performance Analysis for the Best Performing Stacking Ensemble Model
Detailed analysis of base models and meta-model contributions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import sys
import os
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Add path to import preprocessing functions
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '1.Loading and Preprocessing'))
import importlib.util

# Import preprocessing module
spec = importlib.util.spec_from_file_location("loading_preprocessing", 
    os.path.join(os.path.dirname(__file__), '..', '..', '1.Loading and Preprocessing', 'Loading and preprocessing.py'))
loading_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loading_module)

read_csv_with_encoding = loading_module.read_csv_with_encoding
map_categorical_values = loading_module.map_categorical_values

def load_data_and_models():
    """Load data and the best performing models."""
    print("Loading data and models...")
    
    # Load dataset
    df = pd.read_excel(os.path.join(os.path.dirname(__file__), '..', '..', '4.Wrapper', 'Fixed_Stacking_Ensemble', 'dataset.xlsx'))
    
    # Remove rows with NaN values in target variable
    df_clean = df.dropna(subset=['Tg(deg C)'])
    
    # Map categorical values
    isocyanate_mapping = {'N3600': 1, 'HDI': 0, 0: np.nan}
    if 'Isocyonate type' in df_clean.columns:
        df_clean = map_categorical_values(df_clean, 'Isocyonate type', isocyanate_mapping)
        df_clean = df_clean.fillna(0)
    
    # Define the best performing feature combination
    best_features = ['Lignin (wt%)', 'Co-polyol type (PTHF)', 'r', 'Copolyol (wt%)', 'Sratio(%)']
    
    X = df_clean[best_features]
    y = df_clean[['Tg(deg C)']]
    
    # Load models
    feature_str = '_'.join(best_features).replace(' ', '_').replace('(', '').replace(')', '').replace('%', 'pct')
    model_path = os.path.join(os.path.dirname(__file__), '..', '..', '4.Wrapper', 'Fixed_Stacking_Ensemble')
    
    base_models = joblib.load(os.path.join(model_path, f'base_models_fixed_run_1_{feature_str}.joblib'))
    meta_model = joblib.load(os.path.join(model_path, f'meta_model_fixed_run_1_{feature_str}.joblib'))
    X_scaler = joblib.load(os.path.join(model_path, f'x_scaler_fixed_run_1_{feature_str}.joblib'))
    y_scaler = joblib.load(os.path.join(model_path, f'y_scaler_fixed_run_1_{feature_str}.joblib'))
    
    print(f"Loaded {len(base_models)} base models and 1 meta-model")
    return X, y, base_models, meta_model, X_scaler, y_scaler, best_features

def analyze_base_models(X, y, base_models, X_scaler, y_scaler):
    """Analyze individual base model performance."""
    print("\nAnalyzing base models...")
    
    X_scaled = X_scaler.transform(X)
    y_scaled = y_scaler.transform(y)
    y_true = y_scaler.inverse_transform(y_scaled)
    
    base_model_results = []
    model_names = []
    
    for i, model in enumerate(base_models):
        # Get model name
        model_name = model.__class__.__name__
        model_names.append(model_name)
        
        # Generate predictions
        y_pred_scaled = model.predict(X_scaled)
        y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))
        
        # Calculate metrics
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        base_model_results.append({
            'Model': model_name,
            'MAE': mae,
            'MSE': mse,
            'R²': r2,
            'RMSE': np.sqrt(mse)
        })
        
        print(f"  {model_name}: MAE = {mae:.3f}, R² = {r2:.3f}")
    
    return pd.DataFrame(base_model_results), model_names

def analyze_meta_model(X, y, base_models, meta_model, X_scaler, y_scaler):
    """Analyze meta-model performance and contributions."""
    print("\nAnalyzing meta-model...")
    
    X_scaled = X_scaler.transform(X)
    y_scaled = y_scaler.transform(y)
    y_true = y_scaler.inverse_transform(y_scaled)
    
    # Generate meta-features from base models
    meta_features = np.zeros((X_scaled.shape[0], len(base_models)))
    for i, model in enumerate(base_models):
        meta_features[:, i] = model.predict(X_scaled)
    
    # Generate final predictions
    y_pred_scaled = meta_model.predict(meta_features)
    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))
    
    # Calculate metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"  Meta-model: MAE = {mae:.3f}, R² = {r2:.3f}")
    
    # Analyze meta-model coefficients (feature importance)
    if hasattr(meta_model, 'coef_'):
        coefficients = meta_model.coef_
        print(f"  Meta-model coefficients: {coefficients}")
    
    return {
        'mae': mae,
        'mse': mse,
        'r2': r2,
        'rmse': np.sqrt(mse),
        'meta_features': meta_features,
        'y_true': y_true,
        'y_pred': y_pred,
        'coefficients': coefficients if hasattr(meta_model, 'coef_') else None
    }

def create_performance_plots(base_model_df, meta_results, model_names):
    """Create comprehensive performance visualization."""
    print("\nCreating performance plots...")
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with better spacing using gridspec
    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor('white')
    
    # Create subplot layout with better spacing
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3, 
                          left=0.08, right=0.95, top=0.92, bottom=0.08)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Set background colors
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_facecolor('white')
    
    # Plot 1: Base Model Performance Comparison
    ax1.grid(True, linestyle='--', alpha=0.7, color='#E0E0E0')
    bars = ax1.bar(model_names, base_model_df['MAE'], alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_ylabel('MAE (°C)', fontsize=12)
    ax1.set_title('A: Base Model Performance (MAE)', fontsize=14, fontweight='bold', pad=15)
    ax1.tick_params(axis='x', rotation=45, labelsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, mae in zip(bars, base_model_df['MAE']):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{mae:.2f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: R² Comparison
    ax2.grid(True, linestyle='--', alpha=0.7, color='#E0E0E0')
    bars2 = ax2.bar(model_names, base_model_df['R²'], alpha=0.7, color='lightcoral', edgecolor='black')
    ax2.set_ylabel('R²', fontsize=12)
    ax2.set_title('B: Base Model Performance (R²)', fontsize=14, fontweight='bold', pad=15)
    ax2.tick_params(axis='x', rotation=45, labelsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, r2 in zip(bars2, base_model_df['R²']):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{r2:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 3: Meta-model Coefficients (if available)
    ax3.grid(True, linestyle='--', alpha=0.7, color='#E0E0E0')
    if meta_results['coefficients'] is not None:
        coeffs = meta_results['coefficients']
        bars3 = ax3.bar(model_names, coeffs, alpha=0.7, color='lightgreen', edgecolor='black')
        ax3.set_ylabel('Coefficient Value', fontsize=12)
        ax3.set_title('C: Meta-model Coefficients', fontsize=14, fontweight='bold', pad=15)
        ax3.tick_params(axis='x', rotation=45, labelsize=10)
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # Add value labels
        for bar, coeff in zip(bars3, coeffs):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height >= 0 else -0.01),
                    f'{coeff:.3f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)
    else:
        ax3.text(0.5, 0.5, 'Coefficients not available', ha='center', va='center', 
                 transform=ax3.transAxes, fontsize=12)
        ax3.set_title('C: Meta-model Coefficients', fontsize=14, fontweight='bold', pad=15)
    
    # Plot 4: Performance Summary
    ax4.grid(True, linestyle='--', alpha=0.7, color='#E0E0E0')
    
    # Combine base models and meta-model for comparison
    all_models = model_names + ['Meta-Model']
    all_mae = list(base_model_df['MAE']) + [meta_results['mae']]
    all_r2 = list(base_model_df['R²']) + [meta_results['r2']]
    
    # Create scatter plot
    colors = ['blue'] * len(model_names) + ['red']
    sizes = [60] * len(model_names) + [100]
    
    ax4.scatter(all_r2, all_mae, c=colors, s=sizes, alpha=0.7, edgecolors='black')
    ax4.set_xlabel('R²', fontsize=12)
    ax4.set_ylabel('MAE (°C)', fontsize=12)
    ax4.set_title('D: Performance Summary (Lower MAE, Higher R² is Better)', fontsize=14, fontweight='bold', pad=15)
    ax4.grid(True, alpha=0.3)
    
    # Add labels for each point
    for i, (model, r2, mae) in enumerate(zip(all_models, all_r2, all_mae)):
        ax4.annotate(model, (r2, mae), xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Highlight meta-model
    ax4.scatter(meta_results['r2'], meta_results['mae'], c='red', s=150, 
               edgecolors='black', linewidth=2, marker='*', label='Meta-Model')
    ax4.legend(fontsize=9)

    # Make spines slightly grey
    for ax in [ax1, ax2, ax3, ax4]:
        for spine in ax.spines.values():
            spine.set_edgecolor('#666666')

    # Adjust layout - remove tight_layout since we're using gridspec
    # plt.tight_layout()  # Commented out as we're using gridspec spacing
    
    # Add main title with better positioning
    fig.suptitle('Best Performing Stacking Ensemble - Detailed Performance Analysis', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Save plots
    for ext in ['png', 'tiff', 'pdf', 'svg']:
        plt.savefig(f'Best_Model_Performance_Analysis.{ext}', 
                   dpi=300, bbox_inches='tight', facecolor='white')
    
    plt.show()

def create_detailed_report(base_model_df, meta_results, model_names):
    """Create a detailed performance report."""
    print("\n" + "="*80)
    print("DETAILED PERFORMANCE REPORT")
    print("="*80)
    
    print(f"\nBASE MODELS PERFORMANCE:")
    print("-" * 50)
    for idx, row in base_model_df.iterrows():
        print(f"{row['Model']}:")
        print(f"  MAE: {row['MAE']:.3f}°C")
        print(f"  MSE: {row['MSE']:.3f}")
        print(f"  R²: {row['R²']:.3f}")
        print(f"  RMSE: {row['RMSE']:.3f}°C")
        print()
    
    print(f"\nMETA-MODEL PERFORMANCE:")
    print("-" * 50)
    print(f"MAE: {meta_results['mae']:.3f}°C")
    print(f"MSE: {meta_results['mse']:.3f}")
    print(f"R²: {meta_results['r2']:.3f}")
    print(f"RMSE: {meta_results['rmse']:.3f}°C")
    
    if meta_results['coefficients'] is not None:
        print(f"\nMETA-MODEL COEFFICIENTS:")
        print("-" * 50)
        for name, coeff in zip(model_names, meta_results['coefficients']):
            print(f"{name}: {coeff:.4f}")
    
    # Calculate improvement
    best_base_mae = base_model_df['MAE'].min()
    best_base_r2 = base_model_df['R²'].max()
    
    mae_improvement = ((best_base_mae - meta_results['mae']) / best_base_mae) * 100
    r2_improvement = ((meta_results['r2'] - best_base_r2) / best_base_r2) * 100
    
    print(f"\nIMPROVEMENT OVER BEST BASE MODEL:")
    print("-" * 50)
    print(f"MAE Improvement: {mae_improvement:.2f}%")
    print(f"R² Improvement: {r2_improvement:.2f}%")
    
    # Save detailed report to CSV
    report_data = []
    for idx, row in base_model_df.iterrows():
        report_data.append({
            'Model_Type': 'Base',
            'Model_Name': row['Model'],
            'MAE': row['MAE'],
            'MSE': row['MSE'],
            'R2': row['R²'],
            'RMSE': row['RMSE']
        })
    
    report_data.append({
        'Model_Type': 'Meta',
        'Model_Name': 'Stacking Ensemble',
        'MAE': meta_results['mae'],
        'MSE': meta_results['mse'],
        'R2': meta_results['r2'],
        'RMSE': meta_results['rmse']
    })
    
    report_df = pd.DataFrame(report_data)
    report_df.to_csv('Best_Model_Detailed_Report.csv', index=False)
    
    print(f"\nDetailed report saved to: Best_Model_Detailed_Report.csv")

def main():
    """Main execution function."""
    print("="*80)
    print("BEST PERFORMING STACKING ENSEMBLE - PERFORMANCE ANALYSIS")
    print("="*80)
    
    # Load data and models
    X, y, base_models, meta_model, X_scaler, y_scaler, best_features = load_data_and_models()
    
    # Analyze base models
    base_model_df, model_names = analyze_base_models(X, y, base_models, X_scaler, y_scaler)
    
    # Analyze meta-model
    meta_results = analyze_meta_model(X, y, base_models, meta_model, X_scaler, y_scaler)
    
    # Create plots
    create_performance_plots(base_model_df, meta_results, model_names)
    
    # Create detailed report
    create_detailed_report(base_model_df, meta_results, model_names)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("Generated Files:")
    print("  - Best_Model_Performance_Analysis.png")
    print("  - Best_Model_Performance_Analysis.tiff")
    print("  - Best_Model_Performance_Analysis.pdf")
    print("  - Best_Model_Performance_Analysis.svg")
    print("  - Best_Model_Detailed_Report.csv")
    print("="*80)

if __name__ == "__main__":
    main()
