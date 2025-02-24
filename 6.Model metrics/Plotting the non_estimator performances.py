import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_metric(ax, x, y_val, yerr_val, y_test, yerr_test, title, ylabel, color_val, color_test):
    # Create scatter plot with error bars
    ax.errorbar(x - 0.1, y_val, yerr=yerr_val, fmt='o', capsize=5, 
                color=color_val, ecolor=color_val, alpha=0.7, markersize=10, 
                label='Validation')
    # ax.fill_between(x - 0.1, y_val - yerr_val[0], y_val + yerr_val[1], 
    #                 alpha=0.2, color=color_val)

    ax.errorbar(x + 0.1, y_test, yerr=yerr_test, fmt='s', capsize=3, 
                color=color_test, ecolor=color_test, alpha=0.5, markersize=5, 
                label='Train')
    # ax.fill_between(x + 0.1, y_test - yerr_test[0], y_test + yerr_test[1], 
    #                 alpha=0.1, color=color_test)

    ax.set_title(title, fontsize=26, fontweight='bold', pad=20)
    ax.set_ylabel(ylabel, fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=18)
    
    # Set x-axis labels
    ax.set_xticks(x)
    ax.set_xticklabels(['SVR', 'Lasso', 'ElasticNet'], fontsize=18, rotation=45, ha='right')
    
    # Increase legend font size
    ax.legend(fontsize=18)
    
    sns.despine(ax=ax, offset=10, trim=True)

    # Annotate points
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    offset = y_range * 0.05

    for i in range(len(x)):
        # Validation value annotation
        ax.text(x[i] - 0.1, y_val[i] + yerr_val[1][i] + offset,
                f'{y_val[i]:.3f}',
                fontsize=18, ha='center', va='bottom',
                color=color_val, fontweight='bold')
        
        # Train value annotation
        ax.text(x[i] + 0.1, y_test[i] - yerr_test[0][i] - offset,
                f'{y_test[i]:.3f}',
                fontsize=18, ha='center', va='top',
                color=color_test, fontweight='bold')

    # Adjust y-axis limits to accommodate annotations
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min - y_range * 0.1, y_max + y_range * 0.1)

# Create figure with three subplots
fig, axs = plt.subplots(1, 3, figsize=(20, 6))

# Create x positions for the three algorithms
x = np.array([1, 2, 3])

# Plot 1: R-squared
plot_metric(axs[0], x,
            y_val=modified_df['R-squared Validation'],
            yerr_val=[modified_df['R-squared Validation'] - modified_df['Validation R-squared CI Lower'],
                     modified_df['Validation R-squared CI Upper'] - modified_df['R-squared Validation']],
            y_test=modified_df['Train R-squared'],
            yerr_test=[modified_df['Train R-squared'] - modified_df['Train R-squared CI Lower'],
                     modified_df['Train R-squared CI Upper'] - modified_df['Train R-squared']],
            title='A', ylabel='R-squared',
            color_val='#4C72B0', color_test='#D55E00')

# Plot 2: MSE
plot_metric(axs[1], x,
            y_val=modified_df['MSE Validation'],
            yerr_val=[modified_df['MSE Validation'] - modified_df['Validation MSE CI Lower'],
                     modified_df['Validation MSE CI Upper'] - modified_df['MSE Validation']],
            y_test=modified_df['Train MSE'],
            yerr_test=[modified_df['Train MSE'] - modified_df['Train MSE CI Lower'],
                     modified_df['Train MSE CI Upper'] - modified_df['Train MSE']],
            title='B', ylabel='MSE',
            color_val='#55A868', color_test='#CC79A7')

# Plot 3: MAE
plot_metric(axs[2], x,
            y_val=modified_df['MAE Validation'],
            yerr_val=[modified_df['MAE Validation'] - modified_df['Validation MAE CI Lower'],
                     modified_df['Validation MAE CI Upper'] - modified_df['MAE Validation']],
            y_test=modified_df['Train MAE'],
            yerr_test=[modified_df['Train MAE'] - modified_df['Train MAE CI Lower'],
                     modified_df['Train MAE CI Upper'] - modified_df['Train MAE']],
            title='C', ylabel='MAE',
            color_val='#C44E52', color_test='#0072B2')

# Adjust layout
plt.tight_layout()

# Save the figure
plt.savefig('Algorithm_comparison_training_validation.jpg', dpi=600, bbox_inches='tight')

# Show the plot
plt.show()