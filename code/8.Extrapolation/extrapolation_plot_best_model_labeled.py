# -*- coding: utf-8 -*-
"""
Extrapolation Plot for Best Model (5 features) - With Labeled Points
Visualizes target vs predicted Tg values with inset zoom regions
Shows extrapolated data points with labels and arrows
"""

import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import os

# Try to import adjustText, if not available, use basic text placement
try:
    from adjustText import adjust_text
    HAS_ADJUST_TEXT = True
except ImportError:
    HAS_ADJUST_TEXT = False
    print("Note: adjustText not available, using basic text placement")

def create_extrapolation_plot(csv_filename='closest_inputs_best_model.csv', 
                               training_range=(-8, 96),
                               save_plots=True):
    """
    Create extrapolation plot showing target vs predicted Tg values with labeled points.
    """
    
    print("="*80)
    print("CREATING EXTRAPOLATION PLOT WITH LABELS - BEST MODEL")
    print("="*80)
    
    # Load data
    print(f"\nLoading data from {csv_filename}...")
    if not os.path.exists(csv_filename):
        print(f"❌ ERROR: File not found: {csv_filename}")
        print("Please run 'adaptive_grid_search_best_model.py' first.")
        return
    
    closest_inputs = pd.read_csv(csv_filename)
    print(f"Loaded {len(closest_inputs)} data points")
    
    # Identify points with significant deviation from ideal line
    closest_inputs['Deviation'] = abs(closest_inputs['Predicted_Tg'] - closest_inputs['Target_Tg'])
    
    # Identify extrapolated points (outside training range)
    extrapolated_points = closest_inputs[
        (closest_inputs['Target_Tg'] < training_range[0]) | 
        (closest_inputs['Target_Tg'] > training_range[1])
    ].copy()
    print(f"Extrapolated points: {len(extrapolated_points)}")
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 11))
    
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
    
    # Adjust axis spines
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
    
    # Inset 1: Low temperature extrapolation region (where deviation occurs)
    # Focus on the area where extrapolated points deviate
    low_extrap = extrapolated_points[extrapolated_points['Target_Tg'] < training_range[0]]
    if len(low_extrap) > 0:
        # Determine inset bounds based on actual data
        x_min_inset = low_extrap['Target_Tg'].min() - 2
        x_max_inset = max(low_extrap['Target_Tg'].max() + 2, training_range[0] + 2)
        y_min_inset = min(low_extrap['Predicted_Tg'].min() - 2, x_min_inset)
        y_max_inset = max(low_extrap['Predicted_Tg'].max() + 2, x_max_inset)
        
        ax_inset1 = inset_axes(ax, width="45%", height="35%", loc='lower right')
        ax_inset1.scatter(closest_inputs['Target_Tg'], closest_inputs['Predicted_Tg'], 
                          c=closest_inputs['Predicted_Tg'], cmap='viridis', alpha=0.7, s=80)
        ax_inset1.scatter(extrapolated_points['Target_Tg'], extrapolated_points['Predicted_Tg'], 
                          color='red', alpha=0.7, s=80, edgecolors='black', linewidths=2)
        ax_inset1.plot([x_min_inset, x_max_inset], [x_min_inset, x_max_inset], 
                       color='black', linestyle='--', linewidth=2)
        ax_inset1.set_xlim(x_min_inset, x_max_inset)
        ax_inset1.set_ylim(y_min_inset, y_max_inset)
        ax_inset1.grid(linestyle='--', alpha=0.7)
        ax_inset1.tick_params(axis='both', which='major', labelsize=18)
        ax_inset1.set_title('Low Tg Extrapolation', fontsize=18, fontweight='bold', pad=10)
        
        # Add labels to extrapolated points in inset 1
        texts1 = []
        for i, row in low_extrap.iterrows():
            if x_min_inset <= row['Target_Tg'] <= x_max_inset and y_min_inset <= row['Predicted_Tg'] <= y_max_inset:
                label = f"({row['Target_Tg']:.1f}, {row['Predicted_Tg']:.1f})"
                texts1.append(ax_inset1.text(row['Target_Tg'], row['Predicted_Tg'], 
                                          label, fontsize=16, fontweight='bold'))
        
        # Adjust text to avoid overlap
        if texts1:
            if HAS_ADJUST_TEXT:
                adjust_text(texts1, 
                           ax=ax_inset1,
                           force_points=(0.5, 0.5),
                           expand_points=(2.5, 2.5),
                           force_text=(0.5, 0.5),
                           expand_text=(1.5, 1.5),
                           arrowprops=dict(arrowstyle='->', color='darkred', lw=1.5, alpha=0.8),
                           ha='right',
                           va='bottom')
            else:
                # Basic placement without overlap detection
                for text in texts1:
                    text.set_bbox(dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='gray'))
        
        # Mark connection to main plot
        mark_inset(ax, ax_inset1, loc1=2, loc2=3, fc="none", ec="0.5", linewidth=2)
    
    # Inset 2: High temperature extrapolation region (where deviation occurs)
    high_extrap = extrapolated_points[extrapolated_points['Target_Tg'] > training_range[1]]
    if len(high_extrap) > 0:
        # Determine inset bounds based on actual data
        x_min_inset = min(high_extrap['Target_Tg'].min() - 2, training_range[1] - 2)
        x_max_inset = high_extrap['Target_Tg'].max() + 2
        y_min_inset = min(high_extrap['Predicted_Tg'].min() - 2, x_min_inset)
        y_max_inset = max(high_extrap['Predicted_Tg'].max() + 2, x_max_inset)
        
        ax_inset2 = inset_axes(ax, width="45%", height="32%", loc='upper left', 
                               bbox_to_anchor=(0.05, 0.95, 1, 1), bbox_transform=ax.transAxes)
        ax_inset2.scatter(closest_inputs['Target_Tg'], closest_inputs['Predicted_Tg'], 
                          c=closest_inputs['Predicted_Tg'], cmap='viridis', alpha=0.7, s=80)
        ax_inset2.scatter(extrapolated_points['Target_Tg'], extrapolated_points['Predicted_Tg'], 
                          color='red', alpha=0.7, s=80, edgecolors='black', linewidths=2)
        ax_inset2.plot([x_min_inset, x_max_inset], [x_min_inset, x_max_inset], 
                       color='black', linestyle='--', linewidth=2)
        ax_inset2.set_xlim(x_min_inset, x_max_inset)
        ax_inset2.set_ylim(y_min_inset, y_max_inset)
        ax_inset2.grid(linestyle='--', alpha=0.7)
        ax_inset2.tick_params(axis='both', which='major', labelsize=18)
        ax_inset2.set_title('High Tg Extrapolation', fontsize=18, fontweight='bold', pad=10)
        
        # Add labels to extrapolated points in inset 2
        texts2 = []
        for i, row in high_extrap.iterrows():
            if x_min_inset <= row['Target_Tg'] <= x_max_inset and y_min_inset <= row['Predicted_Tg'] <= y_max_inset:
                label = f"({row['Target_Tg']:.1f}, {row['Predicted_Tg']:.1f})"
                texts2.append(ax_inset2.text(row['Target_Tg'], row['Predicted_Tg'], 
                                           label, fontsize=16, fontweight='bold'))
        
        # Adjust text to avoid overlap
        if texts2:
            if HAS_ADJUST_TEXT:
                adjust_text(texts2, 
                           ax=ax_inset2,
                           force_points=(0.5, 0.5),
                           expand_points=(2.5, 2.5),
                           force_text=(0.5, 0.5),
                           expand_text=(1.5, 1.5),
                           arrowprops=dict(arrowstyle='->', color='darkred', lw=1.5, alpha=0.8),
                           ha='left',
                           va='top')
            else:
                # Basic placement without overlap detection
                for text in texts2:
                    text.set_bbox(dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='gray'))
        
        # Mark connection to main plot
        mark_inset(ax, ax_inset2, loc1=1, loc2=4, fc="none", ec="0.5", linewidth=2)
    
    # Adjust layout
    plt.tight_layout()
    
    # Get the current position of the main axis
    bbox = ax.get_position()
    
    # Create a new axis for the legend below the main plot
    legend_distance = 0.15
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
            filename = f'extrapolation_plot_best_model_labeled.{ext}'
            plt.savefig(filename, dpi=600, bbox_inches='tight')
            print(f"  ✓ Saved {filename}")
    
    # Show the figure
    plt.show()
    
    # Print summary of deviations
    print("\n" + "="*80)
    print("DEVIATION SUMMARY")
    print("="*80)
    print(f"\nExtrapolated points with largest deviations:")
    top_deviations = extrapolated_points.nlargest(5, 'Deviation')[['Target_Tg', 'Predicted_Tg', 'Deviation']]
    print(top_deviations.to_string(index=False))
    
    print("\n" + "="*80)
    print("EXTRAPOLATION PLOT COMPLETED!")
    print("="*80)

if __name__ == "__main__":
    create_extrapolation_plot()
