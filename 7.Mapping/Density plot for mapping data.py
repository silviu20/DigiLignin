# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 14:53:38 2024

@author: P70090917
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def create_density_plot(data, x_feature, y_feature, save_path=None, data_save_path=None):
    """
    Create and save a density plot for given features, along with the data used.
    
    Args:
        data: DataFrame containing the features
        x_feature: String name of the x-axis feature
        y_feature: String name of the y-axis feature
        save_path: String path where to save the plot (optional)
        data_save_path: String path where to save the data (optional)
    """
    # Create new figure
    plt.figure(figsize=(12, 8))
    
    # Set style parameters
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['grid.color'] = 'gray'
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.alpha'] = 0.3
    
    # Calculate KDE data
    kde = sns.kdeplot(data=data,
                      x=x_feature,
                      y=y_feature,
                      cmap='YlGnBu',
                      fill=True,
                      levels=10,
                      bw_adjust=0.8,
                      thresh=0.05,
                      alpha=1)
    
    # Extract and save the KDE data if data_save_path is provided
    if data_save_path:
        # Get the density data
        density_data = {
            x_feature: data[x_feature],
            y_feature: data[y_feature]
        }
        
        # Create DataFrame with the raw data used
        density_df = pd.DataFrame(density_data)
        
        # Save to CSV
        density_df.to_csv(data_save_path, index=False)
    
    # Rest of the plotting code remains the same
    plt.grid(True)
    plt.xlabel(x_feature, fontsize=36)
    plt.ylabel(y_feature, fontsize=36)
    plt.xticks(fontsize=32)
    plt.yticks(fontsize=32)
    
    x_min, x_max = data[x_feature].min(), data[x_feature].max()
    y_min, y_max = data[y_feature].min(), data[y_feature].max()
    
    x_padding = (x_max - x_min) * 0.1
    y_padding = (y_max - y_min) * 0.1
    
    plt.xlim(x_min - x_padding, x_max + x_padding)
    plt.ylim(y_min - y_padding, y_max + y_padding)
    
    ax = plt.gca()
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# Read and prepare the data
mapped_results = pd.read_csv('C:/Users/P70090917/Desktop/Polyuerthane Lignin/Experiments/dataset2/rework 21_Nov_2024/testing fesature combinations/mapped_results_tg.csv')
# reduced_results = mapped_results.iloc[::4].reset_index(drop=True)# reducing the size of the mapping results

reduced_results = mapped_results

# Define features to plot against Tg
features = [
    'Lignin (wt%)',
    'Co-polyol type (PTHF)',
    'Ratio',
    'Co-polyol (wt%)',
    'Isocyanate (mmol NCO)',
    'Tin(II) octoate',
    'Swelling ratio (%)'
]
target_feature = 'Tg (°C)'

# Create output directories
output_dir = 'C:/Users/P70090917/Desktop/Polyuerthane Lignin/Experiments/dataset2/rework 21_Nov_2024/testing fesature combinations/density plots/density plots larger labels'
data_output_dir = os.path.join(output_dir, 'plot_data')
os.makedirs(output_dir, exist_ok=True)
os.makedirs(data_output_dir, exist_ok=True)

# Create and save plots and data for each feature against Tg
for feature in features:
    # Create filenames for plot and data
    base_filename = f'{feature.replace(" ", "_").replace("(", "").replace(")", "")}_{target_feature.replace(" ", "_").replace("(", "").replace(")", "")}'
    plot_filename = f'density_plot_{base_filename}.png'
    data_filename = f'density_data_{base_filename}.csv'
    
    save_path = os.path.join(output_dir, plot_filename)
    data_save_path = os.path.join(data_output_dir, data_filename)
    
    # Create and save the plot and data
    create_density_plot(reduced_results, feature, target_feature, save_path, data_save_path)
    print(f'Created plot and saved data: {base_filename}')

print(f'\nAll plots have been saved in the "{output_dir}" directory.')
print(f'All plot data has been saved in the "{data_output_dir}" directory.')