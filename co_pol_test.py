import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

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

# Load the mapped results data
try:
    mapped_results = pd.read_csv('mapped_results_tg_best_model.csv')
    print(f"Loaded data with shape: {mapped_results.shape}")
    print(f"Columns: {list(mapped_results.columns)}")
except FileNotFoundError:
    print("Error: mapped_results_tg_best_model.csv not found in current directory")
    print("Please make sure the file is in the correct location")
    exit()

# Check if the required columns exist
x_feature = 'Copolyol (wt%)'
y_feature = 'Tg (°C)'

if x_feature not in mapped_results.columns:
    print(f"Error: Column '{x_feature}' not found in data")
    print(f"Available columns: {list(mapped_results.columns)}")
    exit()

if y_feature not in mapped_results.columns:
    print(f"Error: Column '{y_feature}' not found in data")
    print(f"Available columns: {list(mapped_results.columns)}")
    exit()

# Create the density plot for Co-polyol wt% vs Tg
print(f"Creating density plot: {x_feature} vs {y_feature}")
create_density_plot(mapped_results, x_feature, y_feature, save_path='co-pol_test.png', data_save_path='co-pol_test_data.csv')

print("Density plot saved as 'co-pol_test.png'")
print("Plot data saved as 'co-pol_test_data.csv'")
