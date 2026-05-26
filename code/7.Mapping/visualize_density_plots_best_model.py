# -*- coding: utf-8 -*-
"""
Density plot visualization for the best model mapping results
Creates 2D density plots showing relationships between features and predicted Tg
Adapted for the 5-feature model: Lignin (wt%), Co-polyol type (PTHF), r, Copolyol (wt%), Isocyanate (wt%)
"""

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
    
    # Customize plot
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

def main():
    """Main function to create all density plots."""
    print("="*80)
    print("CREATING DENSITY PLOTS - BEST MODEL")
    print("="*80)
    
    # Read and prepare the data
    # Try fast version first, then full version
    csv_filename = None
    for filename in ['mapped_results_tg_best_model_fast.csv', 'mapped_results_tg_best_model.csv']:
        if os.path.exists(filename):
            csv_filename = filename
            break
    
    if csv_filename is None:
        print(f"\n❌ ERROR: Could not find mapping results file")
        print("Please run 'mapping_best_model_fast.py' or 'mapping_best_model.py' first.")
        return
    
    print(f"\nLoading data from {csv_filename}...")
    mapped_results = pd.read_csv(csv_filename)
    print(f"Loaded {len(mapped_results):,} predictions")
    
    # Option to reduce dataset size for faster plotting (if needed)
    # Uncomment the following line to use every 4th row
    # reduced_results = mapped_results.iloc[::4].reset_index(drop=True)
    reduced_results = mapped_results
    
    print(f"Using {len(reduced_results):,} data points for plotting")
    
    # Define features to plot against Tg (based on the 5-feature model)
    features = [
        'Lignin (wt%)',
        'Co-polyol type (PTHF)',
        'r',
        'Copolyol (wt%)',
        'Isocyanate (wt%)'
    ]
    target_feature = 'Tg (°C)'
    
    # Create output directories
    output_dir = 'density_plots_best_model'
    data_output_dir = os.path.join(output_dir, 'plot_data')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(data_output_dir, exist_ok=True)
    
    print(f"\nCreating density plots...")
    print(f"Output directory: {output_dir}")
    
    # Create and save plots and data for each feature against Tg
    for i, feature in enumerate(features, 1):
        print(f"  [{i}/{len(features)}] Creating plot for {feature} vs {target_feature}...")
        
        # Create filenames for plot and data
        base_filename = f'{feature.replace(" ", "_").replace("(", "").replace(")", "").replace("%", "pct")}_{target_feature.replace(" ", "_").replace("(", "").replace(")", "").replace("°", "deg")}'
        plot_filename = f'density_plot_{base_filename}.png'
        data_filename = f'density_data_{base_filename}.csv'
        
        save_path = os.path.join(output_dir, plot_filename)
        data_save_path = os.path.join(data_output_dir, data_filename)
        
        # Create and save the plot and data
        create_density_plot(reduced_results, feature, target_feature, save_path, data_save_path)
        print(f"      ✓ Saved plot: {plot_filename}")
        print(f"      ✓ Saved data: {data_filename}")
    
    print("\n" + "="*80)
    print("DENSITY PLOTS COMPLETED!")
    print("="*80)
    print(f"All plots saved in: {output_dir}/")
    print(f"All plot data saved in: {data_output_dir}/")
    print("="*80)

if __name__ == "__main__":
    main()
