# -*- coding: utf-8 -*-
"""
Replica of Extrapolation Plot Code - Adapted for New Data
Based on 'Extrapolation of the closes_inputs_plot_v2_2.py'
Modified to work with 'closest_inputs_best_model.csv'
"""

import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import os

# Try to import adjustText, fall back to basic placement if not available
try:
    from adjustText import adjust_text
    HAS_ADJUST_TEXT = True
except ImportError:
    HAS_ADJUST_TEXT = False
    print("Note: adjustText not available, using basic text placement with backgrounds")

def create_extrapolation_replica(csv_file='closest_inputs_best_model.csv'):
    """
    Create a replica of the original extrapolation plot using the new data structure.
    """
    
    print("="*80)
    print("CREATING EXTRAPOLATION PLOT REPLICA - NEW DATA")
    print("="*80)
    
    # Load the new data
    print(f"\nLoading data from {csv_file}...")
    if not os.path.exists(csv_file):
        print(f"❌ ERROR: File not found: {csv_file}")
        return
    
    closest_inputs = pd.read_csv(csv_file)
    print(f"✓ Loaded {len(closest_inputs)} data points")
    print(f"Columns: {list(closest_inputs.columns)}")
    
    # Identify points with significant deviation (>1.0°C)
    closest_inputs['Deviation'] = abs(closest_inputs['Predicted_Tg'] - closest_inputs['Target_Tg'])
    significant_deviation_all = closest_inputs[closest_inputs['Deviation'] > 1.0]

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 10))

    # Define common tick intervals
    x_interval = 20
    y_interval = 20

    # Plot the regression plot
    scatter = ax.scatter(closest_inputs['Target_Tg'], closest_inputs['Predicted_Tg'], 
                         c=closest_inputs['Predicted_Tg'], cmap='inferno', 
                         label='Data', alpha=0.7, s=60)

    # Highlight extrapolated points (outside training range -8 to 96)
    extrapolated_points = closest_inputs[(closest_inputs['Target_Tg'] < -8) | (closest_inputs['Target_Tg'] > 96)]
    print(f"Extrapolated points identified: {len(extrapolated_points)}")
    
    ax.scatter(extrapolated_points['Target_Tg'], extrapolated_points['Predicted_Tg'], 
               color='red', label='Extrapolated Data', alpha=0.7, s=60, edgecolors='black')

    # Highlight significant deviating points in red on main plot (no legend entry)
    ax.scatter(significant_deviation_all['Target_Tg'], significant_deviation_all['Predicted_Tg'], 
               color='red', alpha=0.9, s=80, edgecolors='darkred', linewidths=2)

    # Plot the perfect prediction line
    ax.plot([-30, 130], [-30, 130], color='black', linestyle='--', label='Perfect Prediction', linewidth=2)

    # Customize the plot
    ax.set_xlabel('Actual Target Values', fontsize=26, labelpad=15)
    ax.set_ylabel('Predicted Target Values', fontsize=26)
    ax.set_axisbelow(True)
    ax.grid(linestyle='--', alpha=0.7)
    ax.tick_params(axis='both', which='major', labelsize=22, pad=10)  # Add 'pad' to increase tick distance
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
    cbar.set_label('Predicted Values', rotation=270, labelpad=30, fontsize=26)

    # Adjust colorbar tick label font size
    cbar.ax.tick_params(labelsize=22)

    # Add insets for zoomed areas
    # Inset for the first region with updated limits
    ax_inset1 = inset_axes(ax, width="50%", height="35%", loc='lower right') # the size of the zoomed box
    # Identify points with significant deviation in the bottom inset region
    bottom_region = closest_inputs[(closest_inputs['Target_Tg'] >= -5) & (closest_inputs['Target_Tg'] <= 15) & 
                                   (closest_inputs['Predicted_Tg'] >= -5) & (closest_inputs['Predicted_Tg'] <= 15)].copy()
    bottom_region['Deviation'] = abs(bottom_region['Predicted_Tg'] - bottom_region['Target_Tg'])
    significant_deviation_bottom = bottom_region[bottom_region['Deviation'] > 1.0]
    
    # Plot the bottom inset with highlighted deviating points
    ax_inset1.scatter(closest_inputs['Target_Tg'], closest_inputs['Predicted_Tg'], 
                      c=closest_inputs['Predicted_Tg'], cmap='viridis', alpha=0.7, s=60)
    ax_inset1.scatter(extrapolated_points['Target_Tg'], extrapolated_points['Predicted_Tg'], 
                      color='red', alpha=0.7, s=60, edgecolors='black')
    # Highlight significant deviating points in red
    ax_inset1.scatter(significant_deviation_bottom['Target_Tg'], significant_deviation_bottom['Predicted_Tg'], 
                      color='red', alpha=0.9, s=80, edgecolors='darkred', linewidths=2)
    ax_inset1.plot([-5, 15], [-5, 15], color='black', linestyle='--', linewidth=1.5)  # Updated perfect prediction line
    ax_inset1.set_xlim(-5, 15)  # Wider x limits to capture more points
    ax_inset1.set_ylim(-5, 15)  # Wider y limits to match
    ax_inset1.grid(linestyle='--', alpha=0.7)
    ax_inset1.tick_params(axis='both', which='major', labelsize=18)

    # Annotate selected points in the first inset with connecting lines
    texts1 = []
    arrows1 = []
    
    # Get all points in the bottom inset region that need labeling
    bottom_region_points = closest_inputs[(closest_inputs['Target_Tg'] >= -5) & (closest_inputs['Target_Tg'] <= 15) & 
                                         (closest_inputs['Predicted_Tg'] >= -5) & (closest_inputs['Predicted_Tg'] <= 15)].copy()
    bottom_region_points['Deviation'] = abs(bottom_region_points['Predicted_Tg'] - bottom_region_points['Target_Tg'])
    points_to_label = bottom_region_points[(bottom_region_points['Deviation'] > 1.0) | 
                                          (bottom_region_points['Target_Tg'].isin([-2.67, 14.04]))]
    
    # Sort points by deviation to prioritize most important ones
    points_to_label = points_to_label.sort_values('Deviation', ascending=False).head(4)
    
    # Define label positions in a grid layout away from data points
    label_positions = [
        (-3, 12),   # Top-left
        (12, 12),   # Top-right  
        (-3, -3),   # Bottom-left
        (12, -3)    # Bottom-right
    ]
    
    for i, (idx, row) in enumerate(points_to_label.iterrows()):
        if i >= len(label_positions):
            break
            
        # Get predefined label position
        label_x, label_y = label_positions[i]
        
        # Create the label at the grid position
        text = ax_inset1.text(label_x, label_y, 
                            f"({row['Target_Tg']:.1f}, {row['Predicted_Tg']:.1f})", 
                            fontsize=11, fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                                    alpha=0.95, edgecolor='darkgray'))
        texts1.append(text)
        
        # Draw connecting line from label to data point
        arrow = ax_inset1.annotate('', xy=(row['Target_Tg'], row['Predicted_Tg']), 
                                xytext=(label_x, label_y),
                                arrowprops=dict(arrowstyle='->', color='gray', 
                                              lw=1.0, alpha=0.7))
        arrows1.append(arrow)

    # Enhanced text adjustment parameters for first inset
    if texts1:
        if HAS_ADJUST_TEXT:
            adjust_text(texts1, 
                       ax=ax_inset1,
                       force_points=(0.5, 0.5),
                       expand_points=(2, 2),
                       force_text=(0.5, 0.5),
                       arrowprops=dict(arrowstyle='->', color='gray', lw=1, alpha=0.6),
                       ha='right',
                       va='bottom')
        else:
            # Basic placement with backgrounds for readability
            for text in texts1:
                text.set_bbox(dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))
                text.set_fontweight('bold')

    # Inset for values between 80 and 85 (second inset remains unchanged)
    ax_inset2 = inset_axes(ax, width="50%", height="30%", loc='upper right')
    # Identify points with significant deviation in the top inset region
    top_region = closest_inputs[(closest_inputs['Target_Tg'] >= 65) & (closest_inputs['Target_Tg'] <= 85) & 
                                (closest_inputs['Predicted_Tg'] >= 65) & (closest_inputs['Predicted_Tg'] <= 85)].copy()
    top_region['Deviation'] = abs(top_region['Predicted_Tg'] - top_region['Target_Tg'])
    significant_deviation_top = top_region[top_region['Deviation'] > 1.0]
    
    # Plot the top inset with highlighted deviating points
    ax_inset2.scatter(closest_inputs['Target_Tg'], closest_inputs['Predicted_Tg'], 
                      c=closest_inputs['Predicted_Tg'], cmap='viridis', alpha=0.7, s=60)
    ax_inset2.scatter(extrapolated_points['Target_Tg'], extrapolated_points['Predicted_Tg'], 
                      color='red', alpha=0.7, s=60, edgecolors='black')
    # Highlight significant deviating points in red
    ax_inset2.scatter(significant_deviation_top['Target_Tg'], significant_deviation_top['Predicted_Tg'], 
                      color='red', alpha=0.9, s=80, edgecolors='darkred', linewidths=2)
    ax_inset2.plot([65, 85], [65, 85], color='black', linestyle='--', linewidth=1.5)
    ax_inset2.set_xlim(65, 85)  # Expanded x limits to capture deviation vicinity
    ax_inset2.set_ylim(65, 85)  # Expanded y limits to match
    ax_inset2.grid(linestyle='--', alpha=0.7)
    ax_inset2.tick_params(axis='both', which='major', labelsize=18)

    # Annotate selected points in the second inset with connecting lines
    texts2 = []
    arrows2 = []
    
    # Get all points in the top inset region that need labeling
    top_region_points = closest_inputs[(closest_inputs['Target_Tg'] >= 65) & (closest_inputs['Target_Tg'] <= 85) & 
                                      (closest_inputs['Predicted_Tg'] >= 65) & (closest_inputs['Predicted_Tg'] <= 85)].copy()
    top_region_points['Deviation'] = abs(top_region_points['Predicted_Tg'] - top_region_points['Target_Tg'])
    points_to_label_top = top_region_points[(top_region_points['Deviation'] > 1.0) | 
                                           (top_region_points['Target_Tg'].isin([71.35, 78.51]))]
    
    # Sort points by deviation to prioritize most important ones
    points_to_label_top = points_to_label_top.sort_values('Deviation', ascending=False).head(3)
    
    # Define label positions in a grid layout away from data points for top inset
    label_positions_top = [
        (67, 82),   # Top-left
        (82, 82),   # Top-right  
        (67, 67)    # Bottom-left
    ]
    
    for i, (idx, row) in enumerate(points_to_label_top.iterrows()):
        if i >= len(label_positions_top):
            break
            
        # Get predefined label position
        label_x, label_y = label_positions_top[i]
        
        # Create the label at the grid position
        text = ax_inset2.text(label_x, label_y, 
                            f"({row['Target_Tg']:.1f}, {row['Predicted_Tg']:.1f})", 
                            fontsize=11, fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                                    alpha=0.95, edgecolor='darkgray'))
        texts2.append(text)
        
        # Draw connecting line from label to data point
        arrow = ax_inset2.annotate('', xy=(row['Target_Tg'], row['Predicted_Tg']), 
                                xytext=(label_x, label_y),
                                arrowprops=dict(arrowstyle='->', color='gray', 
                                              lw=1.0, alpha=0.7))
        arrows2.append(arrow)

    if texts2:
        if HAS_ADJUST_TEXT:
            adjust_text(texts2, ax=ax_inset2, arrowprops=dict(arrowstyle='->', color='gray', lw=1.0))
        else:
            # Basic placement with backgrounds for readability
            for text in texts2:
                text.set_bbox(dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))
                text.set_fontweight('bold')

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
    for ext in ['png', 'svg', 'pdf']:
        filename = f'Target_Predicted_Regression_Plot_Replica.{ext}'
        plt.savefig(filename, dpi=600, bbox_inches='tight')
        print(f"  ✓ Saved {filename}")

    # Show the figure
    plt.show()
    
    print("\n" + "="*80)
    print("EXTRAPOLATION REPLICA PLOT COMPLETED!")
    print("="*80)
    print("\nFIGURE CAPTION:")
    print("Figure 1. Model performance analysis showing target vs predicted Tg values with extrapolation behavior.")
    print("Red points indicate >1.0°C deviation from ideal prediction. Zoom boxes highlight low (-5 to 15°C) and")
    print("high temperature (65 to 85°C) regions where systematic prediction errors occur outside the training range.")
    print("="*80)

if __name__ == "__main__":
    create_extrapolation_replica()
