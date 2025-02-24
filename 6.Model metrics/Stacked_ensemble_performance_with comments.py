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
    
    # Increase legend font size
    ax.legend(fontsize=18)
    
    sns.despine(ax=ax, offset=10, trim=True)


    # Annotate only the last point (most recent base estimator)
    if len(x) > 0:
        i = -1  # Index of the last element
        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
        offset = y_range * 0.05  # 5% of the y-axis range

        ax.text(x.iloc[i], y_val.iloc[i] + yerr_val[1].iloc[i] + offset, 
                f'{y_val.iloc[i]:.2f}', 
                fontsize=18, ha='center', va='bottom', color=color_val, fontweight='bold')
        ax.text(x.iloc[i], y_test.iloc[i] - yerr_test[0].iloc[i] - offset, 
                f'{y_test.iloc[i]:.2f}', 
                fontsize=18, ha='center', va='top', color=color_test, fontweight='bold')

        # Add a vertical line to highlight the latest base estimator
        ax.axvline(x=x.iloc[i], color='gray', linestyle=':', alpha=0.5)

    # Adjust y-axis limits to accommodate annotations
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min - y_range * 0.1, y_max + y_range * 0.1)

# Loop through each row in modified_df and create a plot
for i in range(len(modified_df)):
    n_estimators = modified_df['Number of Estimators'].iloc[:i+1]

    # Create a figure with three subplots
    fig, axs = plt.subplots(1, 3, figsize=(20, 6))

    # Plot 1: R-squared
    plot_metric(axs[0], n_estimators, modified_df['R-squared Validation'].iloc[:i+1],
                yerr_val=[modified_df['R-squared Validation'].iloc[:i+1] - modified_df['Validation R-squared CI Lower'].iloc[:i+1],
                          modified_df['Validation R-squared CI Upper'].iloc[:i+1] - modified_df['R-squared Validation'].iloc[:i+1]],
                y_test=modified_df['Train R-squared'].iloc[:i+1],
                yerr_test=[modified_df['Train R-squared'].iloc[:i+1] - modified_df['Train R-squared CI Lower'].iloc[:i+1],
                           modified_df['Train R-squared CI Upper'].iloc[:i+1] - modified_df['Train R-squared'].iloc[:i+1]],
                title='A', ylabel='R-squared', color_val='#4C72B0', color_test='#D55E00')

    # Plot 2: MSE
    plot_metric(axs[1], n_estimators, modified_df['MSE Validation'].iloc[:i+1],
                yerr_val=[modified_df['MSE Validation'].iloc[:i+1] - modified_df['Validation MSE CI Lower'].iloc[:i+1],
                          modified_df['Validation MSE CI Upper'].iloc[:i+1] - modified_df['MSE Validation'].iloc[:i+1]],
                y_test=modified_df['Train MSE'].iloc[:i+1],
                yerr_test=[modified_df['Train MSE'].iloc[:i+1] - modified_df['Train MSE CI Lower'].iloc[:i+1],
                           modified_df['Train MSE CI Upper'].iloc[:i+1] - modified_df['Train MSE'].iloc[:i+1]],
                title='B', ylabel='MSE', color_val='#55A868', color_test='#CC79A7')

    # Plot 3: MAE
    plot_metric(axs[2], n_estimators, modified_df['MAE Validation'].iloc[:i+1],
                yerr_val=[modified_df['MAE Validation'].iloc[:i+1] - modified_df['Validation MAE CI Lower'].iloc[:i+1],
                          modified_df['Validation MAE CI Upper'].iloc[:i+1] - modified_df['MAE Validation'].iloc[:i+1]],
                y_test=modified_df['Train MAE'].iloc[:i+1],
                yerr_test=[modified_df['Train MAE'].iloc[:i+1] - modified_df['Train MAE CI Lower'].iloc[:i+1],
                           modified_df['Train MAE CI Upper'].iloc[:i+1] - modified_df['Train MAE'].iloc[:i+1]],
                title='C', ylabel='MAE', color_val='#C44E52', color_test='#0072B2')

    # Add a main title
    # fig.suptitle(f'Model Validation and Test Metrics for {n_estimators.iloc[-1]} Estimators', fontsize=20, fontweight='bold', y=1.05)

    # Adjust layout and add some breathing room
    plt.tight_layout()
    fig.subplots_adjust(top=0.85)

    # Save the figure in multiple formats
    for ext in ['jpg']:
        plt.savefig(f'RF_Stacked_ensembles_v2_{n_estimators.iloc[-1]}.{ext}', dpi=600, bbox_inches='tight')

    # Show the plot
    plt.show()
