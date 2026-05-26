# -*- coding: utf-8 -*-
"""
Distribution visualization for predicted Tg values from the best model
Adapted for the 5-feature model: Lignin (wt%), Co-polyol type (PTHF), r, Copolyol (wt%), Isocyanate (wt%)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import inferno
import os

def plot_tg_distribution(csv_filename='mapped_results_tg_best_model.csv', save_plots=True):
    """
    Create a distribution plot of predicted Tg values.
    
    Args:
        csv_filename: Name of the CSV file with mapping results
        save_plots: Whether to save plots to disk
    """
    
    # Load the dataset
    print(f"Loading data from {csv_filename}...")
    df = pd.read_csv(csv_filename)
    print(f"Loaded {len(df):,} predictions")
    
    # Prepare the x values from -10 to 80
    x_values = np.arange(-10, 81, 1)
    
    # Calculate the frequency for each value in x_values
    counts, _ = np.histogram(df['Tg (°C)'], bins=x_values)
    
    # Set up figure
    plt.figure(figsize=(18, 10), facecolor='white')
    plt.style.use('default')
    
    # Create color gradient
    colors = inferno(np.linspace(0.2, 0.8, len(counts)))
    
    # Set bar width and x positions with offset for increased distance
    bar_width = 0.8
    x_positions = x_values[:-1] + 0.1
    
    # Plot a bar for each integer in x_values
    plt.bar(x_positions, counts, color=colors, edgecolor='white', width=bar_width, linewidth=1.5)
    
    # Add count labels for the 5 highest and 5 lowest bars
    sorted_indices = np.argsort(counts)
    lowest_indices = sorted_indices[:7]
    highest_indices = sorted_indices[-5:]
    
    # Calculate the median values (considering count frequencies)
    median_indices = np.argsort(np.abs(counts - np.median(counts)))[:5]
    
    # Combine the indices to annotate
    indices_to_annotate = np.concatenate((lowest_indices, highest_indices, median_indices))
    
    # Add annotations with increased font size
    for i in indices_to_annotate:
        count = counts[i]
        if count > 0:
            plt.text(x_positions[i], count + (max(counts) * 0.01),
                     f'{count:,}', ha='center', va='bottom', 
                     fontsize=16, color='black', rotation=90)
    
    # Customize the plot
    plt.xlabel('Predicted Tg / °C', fontsize=28, color='black', labelpad=15)
    plt.ylabel('Frequency', fontsize=28, color='black', labelpad=15)
    
    # Customize grid and spines - only horizontal grid
    plt.grid(axis='y', alpha=0.3, linestyle='--', color='#666666')
    for spine in plt.gca().spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.5)
    
    # Set x-ticks with increased distance
    tick_positions = x_values
    offset = 0.3
    plt.xticks(tick_positions + offset, [f'{x:d}' for x in tick_positions], rotation=90, ha='center', fontsize=16)
    
    # Set y-tick font size
    plt.yticks(fontsize=20)
    
    # Adjust y-axis for headroom
    plt.margins(x=0.01, y=0.2)
    
    # Customize ticks
    plt.tick_params(colors='black', width=1.5, length=8)
    
    # Prevent label cutoff
    plt.tight_layout()
    
    # Save the figure in multiple formats
    if save_plots:
        for ext in ['png', 'svg', 'pdf']:
            filename = f'distribution_tg_best_model.{ext}'
            plt.savefig(filename, dpi=600, bbox_inches='tight')
            print(f"  ✓ Saved {filename}")
    
    # Show plot
    plt.show()
    
    # Print statistics
    print("\nDistribution Statistics:")
    print(f"  Total predictions: {len(df):,}")
    print(f"  Tg range: [{df['Tg (°C)'].min():.2f}, {df['Tg (°C)'].max():.2f}]°C")
    print(f"  Mean Tg: {df['Tg (°C)'].mean():.2f}°C")
    print(f"  Median Tg: {df['Tg (°C)'].median():.2f}°C")
    print(f"  Std Tg: {df['Tg (°C)'].std():.2f}°C")

if __name__ == "__main__":
    print("="*80)
    print("VISUALIZING TG DISTRIBUTION - BEST MODEL")
    print("="*80)
    plot_tg_distribution()
    print("="*80)
