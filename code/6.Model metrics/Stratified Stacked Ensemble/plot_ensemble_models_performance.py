import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_metric(ax, x, y_train, y_train_lower, y_train_upper, y_val, y_val_lower, y_val_upper, y_test, y_test_lower, y_test_upper, title, ylabel, color_train, color_val, color_test, model_name):
    """Plot metric for a specific model with train, validation, and test data with confidence intervals"""
    # Plot with very subtle error bars but original dot appearance
    ax.errorbar(x, y_train, yerr=[y_train - y_train_lower, y_train_upper - y_train], fmt='o-', capsize=2, color=color_train, 
                ecolor=color_train, alpha=0.3, markersize=8, linewidth=2, label='Train')
    ax.errorbar(x, y_val, yerr=[y_val - y_val_lower, y_val_upper - y_val], fmt='s-', capsize=2, color=color_val, 
                ecolor=color_val, alpha=0.3, markersize=8, linewidth=2, label='Validation')
    ax.errorbar(x, y_test, yerr=[y_test - y_test_lower, y_test_upper - y_test], fmt='^--', capsize=2, color=color_test, 
                ecolor=color_test, alpha=0.2, markersize=6, linewidth=1.5, label='Test')
    
    # Replot the dots on top with full opacity to maintain original appearance
    ax.plot(x, y_train, 'o', color=color_train, markersize=8, alpha=1.0)
    ax.plot(x, y_val, 's', color=color_val, markersize=8, alpha=1.0)
    ax.plot(x, y_test, '^', color=color_test, markersize=6, alpha=1.0)
    
    ax.set_title(f'{title}', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Base estimators', fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # Set x-axis with specific tick labels
    ax.set_xticks([0, 200, 400, 600, 800, 1000])
    ax.set_xticklabels(['0', '200', '400', '600', '800', '1000'], fontsize=12, rotation=45, ha='right')
    
    # Add legend
    ax.legend(fontsize=12, loc='best')
    
    sns.despine(ax=ax, offset=10, trim=True)
    
    # Annotate key points (10 and 1000 estimators)
    if len(x) > 0:
        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
        offset = y_range * 0.04  # 4% of the y-axis range for better spacing

        # Find indices for 10 and 1000 estimators
        for target_est in [10, 1000]:
            for j, x_val in enumerate(x):
                if x_val == target_est:
                    # Train annotation - positioned higher to avoid overlap
                    ax.text(x_val - 20, y_train.iloc[j] + offset * 1.5, 
                            f'{y_train.iloc[j]:.3f}', 
                            fontsize=10, ha='center', va='bottom', color=color_train, fontweight='bold')
                    # Validation annotation - centered position
                    ax.text(x_val, y_val.iloc[j] + offset, 
                            f'{y_val.iloc[j]:.3f}', 
                            fontsize=10, ha='center', va='bottom', color=color_val, fontweight='bold')
                    # Test annotation - positioned lower to avoid overlap
                    ax.text(x_val + 20, y_test.iloc[j] - offset * 1.5, 
                            f'{y_test.iloc[j]:.3f}', 
                            fontsize=10, ha='center', va='top', color=color_test, fontweight='bold')
                    break

def plot_ensemble_comparison():
    """Create comparison plots for ensemble models only"""
    
    # Load the ensemble data
    print("Loading ensemble models performance data...")
    df = pd.read_csv('Stratified Stacked Ensemble/individual_models_performance.csv')
    
    # Filter for Ensemble model only
    ensemble_df = df[df['Model'] == 'Ensemble'].copy()
    
    print(f"Ensemble data loaded successfully. Shape: {ensemble_df.shape}")
    print(f"N_estimators range: {ensemble_df['n_estimators'].min()} to {ensemble_df['n_estimators'].max()}")
    
    # Create figure with three subplots
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor('white')
    
    # Create x positions (n_estimators values)
    x = pd.Series(ensemble_df['n_estimators'].values)
    
    # Plot 1: R-squared
    plot_metric(axs[0], x,
                y_train=pd.Series(ensemble_df['Train_R2'].values),
                y_train_lower=pd.Series(ensemble_df['Train_R2_CI_Lower'].values),
                y_train_upper=pd.Series(ensemble_df['Train_R2_CI_Upper'].values),
                y_val=pd.Series(ensemble_df['Val_R2'].values),
                y_val_lower=pd.Series(ensemble_df['Val_R2_CI_Lower'].values),
                y_val_upper=pd.Series(ensemble_df['Val_R2_CI_Upper'].values),
                y_test=pd.Series(ensemble_df['Test_R2'].values),
                y_test_lower=pd.Series(ensemble_df['Test_R2_CI_Lower'].values),
                y_test_upper=pd.Series(ensemble_df['Test_R2_CI_Upper'].values),
                title='A', ylabel='R-squared',
                color_train='#2E8B57', color_val='#4C72B0', color_test='#D55E00', model_name='Ensemble')

    # Plot 2: MSE
    plot_metric(axs[1], x,
                y_train=pd.Series(ensemble_df['Train_MSE'].values),
                y_train_lower=pd.Series(ensemble_df['Train_MSE_CI_Lower'].values),
                y_train_upper=pd.Series(ensemble_df['Train_MSE_CI_Upper'].values),
                y_val=pd.Series(ensemble_df['Val_MSE'].values),
                y_val_lower=pd.Series(ensemble_df['Val_MSE_CI_Lower'].values),
                y_val_upper=pd.Series(ensemble_df['Val_MSE_CI_Upper'].values),
                y_test=pd.Series(ensemble_df['Test_MSE'].values),
                y_test_lower=pd.Series(ensemble_df['Test_MSE_CI_Lower'].values),
                y_test_upper=pd.Series(ensemble_df['Test_MSE_CI_Upper'].values),
                title='B', ylabel='MSE',
                color_train='#2E8B57', color_val='#55A868', color_test='#CC79A7', model_name='Ensemble')

    # Plot 3: MAE
    plot_metric(axs[2], x,
                y_train=pd.Series(ensemble_df['Train_MAE'].values),
                y_train_lower=pd.Series(ensemble_df['Train_MAE_CI_Lower'].values),
                y_train_upper=pd.Series(ensemble_df['Train_MAE_CI_Upper'].values),
                y_val=pd.Series(ensemble_df['Val_MAE'].values),
                y_val_lower=pd.Series(ensemble_df['Val_MAE_CI_Lower'].values),
                y_val_upper=pd.Series(ensemble_df['Val_MAE_CI_Upper'].values),
                y_test=pd.Series(ensemble_df['Test_MAE'].values),
                y_test_lower=pd.Series(ensemble_df['Test_MAE_CI_Lower'].values),
                y_test_upper=pd.Series(ensemble_df['Test_MAE_CI_Upper'].values),
                title='C', ylabel='MAE (°C)',
                color_train='#2E8B57', color_val='#C44E52', color_test='#0072B2', model_name='Ensemble')

    # Adjust layout
    plt.tight_layout()

    # Save the figure in multiple formats
    for ext in ['jpg', 'png', 'pdf', 'tiff', 'svg']:
        plt.savefig(f'Ensemble_n_estimators_performance.{ext}', dpi=600, bbox_inches='tight', facecolor='white')

    # Show the plot
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*80)
    print("ENSEMBLE MODEL PERFORMANCE SUMMARY")
    print("="*80)
    
    # Find best performance
    best_mae_idx = ensemble_df['Val_MAE'].idxmin()
    best_r2_idx = ensemble_df['Val_R2'].idxmax()
    
    print(f"\nBest MAE Performance (Validation):")
    best_mae_row = ensemble_df.loc[best_mae_idx]
    print(f"  Estimators: {best_mae_row['n_estimators']}")
    print(f"  MAE: {best_mae_row['Val_MAE']:.3f}°C")
    print(f"  R²: {best_mae_row['Val_R2']:.3f}")
    
    print(f"\nBest R² Performance (Validation):")
    best_r2_row = ensemble_df.loc[best_r2_idx]
    print(f"  Estimators: {best_r2_row['n_estimators']}")
    print(f"  R²: {best_r2_row['Val_R2']:.3f}")
    print(f"  MAE: {best_r2_row['Val_MAE']:.3f}°C")
    
    print(f"\nPerformance at key estimator values:")
    key_estimators = [1, 10, 50, 100, 500, 700, 1000]
    for n_est in key_estimators:
        row = ensemble_df[ensemble_df['n_estimators'] == n_est].iloc[0]
        print(f"  {n_est:4d}: MAE = {row['Val_MAE']:.3f}°C, R² = {row['Val_R2']:.3f}")
    
    print("="*80)

def plot_all_models_comparison():
    """Create comparison plots for all models"""
    
    # Load the data
    print("Loading all models performance data...")
    df = pd.read_csv('Stratified Stacked Ensemble/individual_models_performance.csv')
    
    print(f"Data loaded successfully. Shape: {df.shape}")
    print(f"Models: {df['Model'].unique()}")
    print(f"N_estimators range: {df['n_estimators'].min()} to {df['n_estimators'].max()}")
    
    # Get unique models
    models = df['Model'].unique()
    
    # Create figure with subplots for each model and metric
    fig, axes = plt.subplots(len(models), 3, figsize=(18, 4*len(models)))
    fig.patch.set_facecolor('white')
    
    # Color scheme for different models
    colors = {
        'GradientBoosting': ('#2E8B57', '#4C72B0', '#D55E00'),
        'RandomForest': ('#2E8B57', '#55A868', '#CC79A7'),
        'SVR': ('#2E8B57', '#C44E52', '#0072B2'),
        'Lasso': ('#2E8B57', '#8172B2', '#D55E00'),
        'ElasticNet': ('#2E8B57', '#55A868', '#C44E52'),
        'Ensemble': ('#2E8B57', '#CC79A7', '#4C72B0')
    }
    
    for i, model in enumerate(models):
        model_df = df[df['Model'] == model].copy()
        x = pd.Series(model_df['n_estimators'].values)
        color_train, color_val, color_test = colors[model]
        
        # Plot 1: R-squared
        plot_metric(axes[i, 0], x,
                    y_train=pd.Series(model_df['Train_R2'].values),
                    y_train_lower=pd.Series(model_df['Train_R2_CI_Lower'].values),
                    y_train_upper=pd.Series(model_df['Train_R2_CI_Upper'].values),
                    y_val=pd.Series(model_df['Val_R2'].values),
                    y_val_lower=pd.Series(model_df['Val_R2_CI_Lower'].values),
                    y_val_upper=pd.Series(model_df['Val_R2_CI_Upper'].values),
                    y_test=pd.Series(model_df['Test_R2'].values),
                    y_test_lower=pd.Series(model_df['Test_R2_CI_Lower'].values),
                    y_test_upper=pd.Series(model_df['Test_R2_CI_Upper'].values),
                    title='A', ylabel='R-squared',
                    color_train=color_train, color_val=color_val, color_test=color_test, model_name=model)
        
        # Plot 2: MSE
        plot_metric(axes[i, 1], x,
                    y_train=pd.Series(model_df['Train_MSE'].values),
                    y_train_lower=pd.Series(model_df['Train_MSE_CI_Lower'].values),
                    y_train_upper=pd.Series(model_df['Train_MSE_CI_Upper'].values),
                    y_val=pd.Series(model_df['Val_MSE'].values),
                    y_val_lower=pd.Series(model_df['Val_MSE_CI_Lower'].values),
                    y_val_upper=pd.Series(model_df['Val_MSE_CI_Upper'].values),
                    y_test=pd.Series(model_df['Test_MSE'].values),
                    y_test_lower=pd.Series(model_df['Test_MSE_CI_Lower'].values),
                    y_test_upper=pd.Series(model_df['Test_MSE_CI_Upper'].values),
                    title='B', ylabel='MSE',
                    color_train=color_train, color_val=color_val, color_test=color_test, model_name=model)
        
        # Plot 3: MAE
        plot_metric(axes[i, 2], x,
                    y_train=pd.Series(model_df['Train_MAE'].values),
                    y_train_lower=pd.Series(model_df['Train_MAE_CI_Lower'].values),
                    y_train_upper=pd.Series(model_df['Train_MAE_CI_Upper'].values),
                    y_val=pd.Series(model_df['Val_MAE'].values),
                    y_val_lower=pd.Series(model_df['Val_MAE_CI_Lower'].values),
                    y_val_upper=pd.Series(model_df['Val_MAE_CI_Upper'].values),
                    y_test=pd.Series(model_df['Test_MAE'].values),
                    y_test_lower=pd.Series(model_df['Test_MAE_CI_Lower'].values),
                    y_test_upper=pd.Series(model_df['Test_MAE_CI_Upper'].values),
                    title='C', ylabel='MAE (°C)',
                    color_train=color_train, color_val=color_val, color_test=color_test, model_name=model)
    
    # Adjust layout
    plt.tight_layout()

    # Save the figure in multiple formats
    for ext in ['jpg', 'png', 'pdf', 'tiff', 'svg']:
        plt.savefig(f'All_Models_n_estimators_performance.{ext}', dpi=600, bbox_inches='tight', facecolor='white')

    # Show the plot
    plt.show()

def main():
    """Main function to plot ensemble models performance."""
    
    print("Choose plotting option:")
    print("1. Plot Ensemble model only")
    print("2. Plot all models comparison")
    
    choice = input("Enter choice (1 or 2, default=1): ").strip()
    
    if choice == '2':
        plot_all_models_comparison()
    else:
        plot_ensemble_comparison()

if __name__ == "__main__":
    main()
