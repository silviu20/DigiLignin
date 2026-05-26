import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_metric(ax, x, y_val, yerr_val, y_test, yerr_test, title, ylabel, color_val, color_test):
    ax.errorbar(x, y_val, yerr=yerr_val, fmt='o', capsize=5, color=color_val, ecolor=color_val, alpha=0.7, markersize=10, label='Validation')
    ax.fill_between(x, y_val - yerr_val[0], y_val + yerr_val[1], alpha=0.2, color=color_val)

    ax.errorbar(x, y_test, yerr=yerr_test, fmt='s', capsize=3, color=color_test, ecolor=color_test, alpha=0.5, markersize=5, linestyle='--', label='Train')
    ax.fill_between(x, y_test - yerr_test[0], y_test + yerr_test[1], alpha=0.1, color=color_test)

    ax.set_title(title, fontsize=26, fontweight='bold', pad=20)
    ax.set_xlabel('Base estimators', fontsize=20)
    ax.set_ylabel(ylabel, fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=18)
    
    # Set x-axis with specific tick labels while keeping data points at original positions
    ax.set_xticks([0, 200, 400, 600, 800, 1000])
    ax.set_xticklabels(['0', '200', '400', '600', '800', '1000'], fontsize=16, rotation=45, ha='right')
    
    # Increase legend font size and move below x-axis
    ax.legend(fontsize=18, loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=2)
    
    sns.despine(ax=ax, offset=10, trim=True)

    # Annotate only base estimators 700 and 1000
    if len(x) > 0:
        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
        offset = y_range * 0.05  # 5% of the y-axis range

        # Find indices for 700 and 1000 estimators
        for target_est in [700, 1000]:
            for j, x_val in enumerate(x):
                if x_val == target_est:
                    # Validation annotation - use actual x value for positioning
                    ax.text(x_val, y_val.iloc[j] + yerr_val[1].iloc[j] + offset, 
                            f'{y_val.iloc[j]:.2f}', 
                            fontsize=18, ha='center', va='bottom', color=color_val, fontweight='bold')
                    # Train annotation - use actual x value for positioning
                    ax.text(x_val, y_test.iloc[j] - yerr_test[0].iloc[j] - offset, 
                            f'{y_test.iloc[j]:.2f}', 
                            fontsize=18, ha='center', va='top', color=color_test, fontweight='bold')
                    break

def main():
    """Main function to plot Model #6 n_estimators performance."""
    
    # Load the plot data
    print("Loading Model #6 n_estimators data...")
    df = pd.read_csv('../4.Wrapper/Fixed_stacking_ensemble_with_n_estimators/model6_plot_data.csv')
    
    print(f"Data loaded successfully. Shape: {df.shape}")
    print(f"N_estimators range: {df['n_estimators'].min()} to {df['n_estimators'].max()}")
    
    # Create figure with three subplots (matching original size)
    fig, axs = plt.subplots(1, 3, figsize=(20, 6))
    fig.patch.set_facecolor('white')
    
    # Create x positions (n_estimators values) - convert to pandas Series for .iloc access
    x = pd.Series(df['n_estimators'].values)
    
    # Plot 1: R-squared
    plot_metric(axs[0], x,
                y_val=pd.Series(df['R2 Validation'].values),
                yerr_val=[pd.Series(df['R2 Validation'].values - df['R2 Validation CI Lower'].values),
                         pd.Series(df['R2 Validation CI Upper'].values - df['R2 Validation'].values)],
                y_test=pd.Series(df['Train R2'].values),
                yerr_test=[pd.Series(df['Train R2'].values - df['Train R2 CI Lower'].values),
                         pd.Series(df['Train R2 CI Upper'].values - df['Train R2'].values)],
                title='A', ylabel='R-squared',
                color_val='#4C72B0', color_test='#D55E00')

    # Plot 2: MSE
    plot_metric(axs[1], x,
                y_val=pd.Series(df['MSE Validation'].values),
                yerr_val=[pd.Series(df['MSE Validation'].values - df['MSE Validation CI Lower'].values),
                         pd.Series(df['MSE Validation CI Upper'].values - df['MSE Validation'].values)],
                y_test=pd.Series(df['Train MSE'].values),
                yerr_test=[pd.Series(df['Train MSE'].values - df['Train MSE CI Lower'].values),
                         pd.Series(df['Train MSE CI Upper'].values - df['Train MSE'].values)],
                title='B', ylabel='MSE',
                color_val='#55A868', color_test='#CC79A7')

    # Plot 3: MAE
    plot_metric(axs[2], x,
                y_val=pd.Series(df['MAE Validation'].values),
                yerr_val=[pd.Series(df['MAE Validation'].values - df['MAE Validation CI Lower'].values),
                         pd.Series(df['MAE Validation CI Upper'].values - df['MAE Validation'].values)],
                y_test=pd.Series(df['Train MAE'].values),
                yerr_test=[pd.Series(df['Train MAE'].values - df['Train MAE CI Lower'].values),
                         pd.Series(df['Train MAE CI Upper'].values - df['Train MAE'].values)],
                title='C', ylabel='MAE (°C)',
                color_val='#C44E52', color_test='#0072B2')

    # Add main title
    # fig.suptitle('Model #6 Performance vs Number of Estimators\n(Best Model without Swelling Ratio)', 
    #              fontsize=28, fontweight='bold', y=0.95)

    # Adjust layout to make room for bottom legend
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # Make room for legend at the bottom

    # Save the figure in multiple formats
    for ext in ['jpg', 'png', 'pdf', 'tiff', 'svg']:
        plt.savefig(f'Model6_n_estimators_performance.{ext}', dpi=600, bbox_inches='tight', facecolor='white')

    # Show the plot
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*80)
    print("MODEL #6 PERFORMANCE SUMMARY")
    print("="*80)
    
    # Find best performance
    best_mae_idx = df['MAE Validation'].idxmin()
    best_r2_idx = df['R2 Validation'].idxmax()
    
    print(f"\nBest MAE Performance:")
    best_mae_row = df.loc[best_mae_idx]
    print(f"  Estimators: {best_mae_row['n_estimators']}")
    print(f"  MAE: {best_mae_row['MAE Validation']:.3f}°C")
    print(f"  R²: {best_mae_row['R2 Validation']:.3f}")
    
    print(f"\nBest R² Performance:")
    best_r2_row = df.loc[best_r2_idx]
    print(f"  Estimators: {best_r2_row['n_estimators']}")
    print(f"  R²: {best_r2_row['R2 Validation']:.3f}")
    print(f"  MAE: {best_r2_row['MAE Validation']:.3f}°C")
    
    print(f"\nPerformance at key estimator values:")
    key_estimators = [1, 10, 50, 100, 500, 700, 1000]
    for n_est in key_estimators:
        row = df[df['n_estimators'] == n_est].iloc[0]
        print(f"  {n_est:4d}: MAE = {row['MAE Validation']:.3f}°C, R² = {row['R2 Validation']:.3f}")
    
    print("="*80)

if __name__ == "__main__":
    main()
