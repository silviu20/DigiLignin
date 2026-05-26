# -*- coding: utf-8 -*-
"""
Extrapolation Plot for Best Model (5 features)
Visualizes target vs predicted Tg values with inset zoom regions
Shows extrapolated data points beyond training range
"""

import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from adjustText import adjust_text
import os

def create_extrapolation_plot(csv_filename='closest_inputs_best_model.csv', 
                               training_range=(-8, 96),
                               save_plots=True):
    """
    Create extrapolation plot showing target vs predicted Tg values.
    
    Args:
        csv_filename: Name of CSV file with closest inputs results
        training_range: Tuple of (min, max) Tg values in training data
        save_plots: Whether to save plots to disk
    """
    
    print("="*80)
    print("CREATING EXTRAPOLATION PLOT - BEST MODEL")
    print("="*80)
    
    # Load data
    print(f"\nLoading data from {csv_filename}...")
    if not os.path.exists(csv_filename):
        print(f"❌ ERROR: File not found: {csv_filename}")
        print("Please run 'adaptive_grid_search_best_model.py' first.")
        return
    
    closest_inputs = pd.read_csv(csv_filename)
    print(f"Loaded {len(closest_inputs)} data points")
    
    # Identify extrapolated points (outside training range)
    extrapolated_points = closest_inputs[
        (closest_inputs['Target_Tg'] < training_range[0]) | 
        (closest_inputs['Target_Tg'] > training_range[1])
    ]
    print(f"Extrapolated points: {len(extrapolated_points)}")
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Define common tick intervals
    x_interval = 20
    y_interval = 20
    
    # Plot the regression plot
    scatter = ax.scatter(closest_inputs['Target_Tg'], closest_inputs['Predicted_Tg'], 
                         c=closest_inputs['Predicted_Tg'], cmap='inferno', 
                         label='Data', alpha=0.7, s=60)
    
    # Highlight extrapolated points
    ax.scatter(extrapolated_points['Target_Tg'], extrapolated_points['Predicted_Tg'], 
               color='red', label='Extrapolated Data', alpha=0.7, s=60, edgecolors='black')
    
    # Plot the perfect prediction line
    ax.plot([-30, 130], [-30, 130], color='black', linestyle='--', label='Perfect Prediction', linewidth=2)
    
    # Customize the plot
    ax.set_xlabel('Target Tg / °C', fontsize=26, labelpad=15)
    ax.set_ylabel('Predicted Tg / °C', fontsize=26)
    ax.set_axisbelow(True)
    ax.grid(linestyle='--', alpha=0.7)
    ax.tick_params(axis='both', which='major', labelsize=22, pad=10)
    ax.xaxis.set_major_locator(plt.MultipleLocator(x_interval))
    ax.yaxis.set_major_locator(plt.MultipleLocator(y_interval))
    
    # Adjust axis spines to move ticks away from the plot
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 40))
    ax.spines['top'].set_position(('outward', 15))
    
    # Set axis limits
    ax.set_xlim(-30, 130)
    ax.set_ylim(-30, 130)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, shrink=1.0, pad=0.02)
    cbar.set_label('Predicted Tg / °C', rotation=270, labelpad=30, fontsize=26)
    cbar.ax.tick_params(labelsize=22)
    
    # Add insets for zoomed areas
    # Inset 1: Low temperature extrapolation region
    ax_inset1 = inset_axes(ax, width="50%", height="35%", loc='lower right')
    ax_inset1.scatter(closest_inputs['Target_Tg'], closest_inputs['Predicted_Tg'], 
                      c=closest_inputs['Predicted_Tg'], cmap='viridis', alpha=0.7, s=60)
    ax_inset1.scatter(extrapolated_points['Target_Tg'], extrapolated_points['Predicted_Tg'], 
                      color='red', alpha=0.7, s=60, edgecolors='black')
    ax_inset1.plot([-20, 0], [-20, 0], color='black', linestyle='--', linewidth=1.5)
    ax_inset1.set_xlim(-20, 0)
    ax_inset1.set_ylim(-20, 0)
    ax_inset1.grid(linestyle='--', alpha=0.7)
    ax_inset1.tick_params(axis='both', which='major', labelsize=18)
    
    # Annotate points in the first inset
    texts1 = []
    for i, row in extrapolated_points.iterrows():
        if -20 <= row['Target_Tg'] <= 0 and -20 <= row['Predicted_Tg'] <= 0:
            texts1.append(ax_inset1.text(row['Target_Tg'], row['Predicted_Tg'], 
                                      f"({row['Target_Tg']:.1f}, {row['Predicted_Tg']:.1f})", 
                                      fontsize=16))
    
    # Adjust text to avoid overlap
    if texts1:
        adjust_text(texts1, 
                   ax=ax_inset1,
                   force_points=(0.5, 0.5),
                   expand_points=(2, 2),
                   force_text=(0.5, 0.5),
                   arrowprops=dict(arrowstyle='->', color='gray', lw=1, alpha=0.6),
                   ha='right',
                   va='bottom')
    
    # Inset 2: High temperature extrapolation region
    ax_inset2 = inset_axes(ax, width="50%", height="30%", loc='upper center')
    ax_inset2.scatter(closest_inputs['Target_Tg'], closest_inputs['Predicted_Tg'], 
                      c=closest_inputs['Predicted_Tg'], cmap='viridis', alpha=0.7, s=60)
    ax_inset2.scatter(extrapolated_points['Target_Tg'], extrapolated_points['Predicted_Tg'], 
                      color='red', alpha=0.7, s=60, edgecolors='black')
    ax_inset2.plot([95, 105], [95, 105], color='black', linestyle='--', linewidth=1.5)
    ax_inset2.set_xlim(95, 105)
    ax_inset2.set_ylim(95, 105)
    ax_inset2.grid(linestyle='--', alpha=0.7)
    ax_inset2.tick_params(axis='both', which='major', labelsize=18)
    
    # Annotate points in the second inset
    texts2 = []
    for i, row in extrapolated_points.iterrows():
        if 95 <= row['Target_Tg'] <= 105 and 95 <= row['Predicted_Tg'] <= 105:
            texts2.append(ax_inset2.text(row['Target_Tg'], row['Predicted_Tg'], 
                                       f"({row['Target_Tg']:.1f}, {row['Predicted_Tg']:.1f})", 
                                       fontsize=16))
    
    if texts2:
        adjust_text(texts2, ax=ax_inset2, arrowprops=dict(arrowstyle='->', color='gray', lw=1.0))
    
    # Mark connections between insets and the main plot
    mark_inset(ax, ax_inset1, loc1=2, loc2=3, fc="none", ec="0.5")
    mark_inset(ax, ax_inset2, loc1=1, loc2=4, fc="none", ec="0.5")
    
    # Adjust layout
    plt.tight_layout()
    
    # Get the current position of the main axis
    bbox = ax.get_position()
    
    # Create a new axis for the legend below the main plot
    legend_distance = 0.19
    legend_height = 0.03
    legend_ax = fig.add_axes([bbox.x0, bbox.y0 - legend_distance - legend_height, 
                              bbox.width, legend_height])
    legend_ax.axis('off')
    
    # Add the legend to the new axis
    legend = legend_ax.legend(*ax.get_legend_handles_labels(), loc='center', ncol=3, 
                             frameon=True, fontsize=22)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('gray')
    legend.get_frame().set_alpha(0.9)
    
    # Save the figure in multiple formats
    if save_plots:
        for ext in ['png', 'svg', 'pdf']:
            filename = f'extrapolation_plot_best_model.{ext}'
            plt.savefig(filename, dpi=600, bbox_inches='tight')
            print(f"  ✓ Saved {filename}")
    
    # Show the figure
    plt.show()
    
    print("\n" + "="*80)
    print("EXTRAPOLATION PLOT COMPLETED!")
    print("="*80)

if __name__ == "__main__":
    create_extrapolation_plot()
