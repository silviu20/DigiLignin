import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib as mpl
from math import ceil
import textwrap

# Set style parameters for a more scientific look
plt.style.use('default')
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 18
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['figure.constrained_layout.use'] = False

def get_title_width(ax):
    """Calculate the width of the subplot in characters based on figure size"""
    bbox = ax.get_window_extent().transformed(ax.figure.dpi_scale_trans.inverted())
    width_inches = bbox.width
    # Approximate characters that fit in the width (assuming average char width)
    chars_per_inch = 8  # This can be adjusted based on font size and style
    return int(width_inches * chars_per_inch)

def wrap_title(title, ax):
    """Wrap title text to fit subplot width dynamically"""
    width = get_title_width(ax)
    return '\n'.join(textwrap.wrap(title, width=width))

def create_subplot(ax, data, feature_combination):
    """Create a single subplot for a feature combination"""
    
    # Add light grid
    ax.grid(True, linestyle='--', color='#E0E0E0', zorder=0)
    
    # Get data for this feature combination
    feature_data = data[data["Feature Combination"] == feature_combination]
    
    # Sort by number of features for better visualization
    feature_data = feature_data.sort_values('Number of Features')
    
    # Get y-axis limits
    y_min = data['MAE Validation'].min()
    y_max = data['MAE Validation'].max()
    y_range = y_max - y_min
    
    # Create bar plot
    x_pos = np.arange(len(feature_data))
    bars = ax.bar(x_pos, feature_data['MAE Validation'], 
                  color='#0b53c1', alpha=0.8, edgecolor='white', linewidth=1.0)
    
    # Add error bars if confidence intervals are available
    if 'Validation MAE CI Lower' in feature_data.columns and 'Validation MAE CI Upper' in feature_data.columns:
        lower_err = np.abs(feature_data['MAE Validation'] - feature_data['Validation MAE CI Lower'])
        upper_err = np.abs(feature_data['Validation MAE CI Upper'] - feature_data['MAE Validation'])
        yerr = [lower_err, upper_err]
        ax.errorbar(x_pos, feature_data['MAE Validation'], yerr=yerr, 
                   fmt='none', ecolor='black', capsize=3, capthick=1.0, zorder=5)
    
    # Customize subplot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['bottom'].set_linewidth(1.0)
    
    # Set x-axis labels as number of features
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{int(n)}' for n in feature_data['Number of Features']], rotation=0)
    
    # Set x-axis limits with padding
    ax.set_xlim(-0.5, len(x_pos) - 0.5)
    
    # Set labels
    ax.set_xlabel("Number of Features", fontsize=14)
    ax.set_ylabel("MAE Validation (°C)", fontsize=14)
    
    # Create wrapped title from feature combination
    title = str(feature_combination).replace("'", "").replace("[", "").replace("]", "")
    
    # Draw the figure once to ensure correct window extent calculations
    plt.draw()
    
    # Get wrapped title with dynamic width
    wrapped_title = wrap_title(title, ax)
    
    # Set title with adjusted parameters
    ax.set_title(wrapped_title, fontsize=12, pad=10, wrap=True,
                bbox=dict(facecolor='white', edgecolor='none', pad=3.0, alpha=0.9))
    
    # Add tick parameters
    ax.tick_params(axis='both', which='major', labelsize=10, width=1.0, length=4)
    
    # Add value labels on top of bars
    for i, (bar, mae) in enumerate(zip(bars, feature_data['MAE Validation'])):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01 * y_range,
                f'{mae:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Adjust y-axis limits
    ax.set_ylim(y_min - 0.05 * y_range, y_max + 0.15 * y_range)
    
    return ax

# Read and process data from our stacking ensemble results
df1 = pd.read_csv('Fixed_Stacking_Ensemble/fixed_stacking_results_all_combinations.csv')

# Add number of features column for better visualization
df1['Number of Features'] = df1['Feature Combination'].apply(lambda x: len(eval(x)))

# Retain only necessary columns
columns = ['Feature Combination', 'MAE Validation', 'Validation MAE CI Lower', 'Validation MAE CI Upper', 'Number of Features']
df1 = df1[columns]

# Get unique feature combinations
feature_combinations = df1["Feature Combination"].unique()

# Calculate number of figures needed (8 subplots per figure: 4 columns x 2 rows)
n_figures = ceil(len(feature_combinations) / 8)

print(f"Creating {n_figures} figures with {len(feature_combinations)} feature combinations...")

# Create figures
for fig_num in range(n_figures):
    # Create figure with wider aspect ratio for 4x2 layout
    fig = plt.figure(figsize=(24, 12), dpi=300, facecolor='white')
    
    # Create grid with carefully adjusted spacing
    plt.subplots_adjust(left=0.06, right=0.94,
                       bottom=0.08, top=0.92,
                       wspace=0.25, hspace=0.4)
    
    # Process 8 feature combinations per figure
    start_idx = fig_num * 8
    end_idx = min(start_idx + 8, len(feature_combinations))
    
    for i, feature_comb in enumerate(feature_combinations[start_idx:end_idx]):
        # Create subplot (2 rows instead of 3)
        ax = plt.subplot(2, 4, i + 1)
        ax.set_facecolor('white')
        
        # Create subplot
        create_subplot(ax, df1, feature_comb)
    
    # Add figure title
    fig.suptitle(f'Fixed Stacking Ensemble - Feature Combinations (Fig {fig_num+1}/{n_figures})', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Save figure
    plt.savefig(f'stacking_feature_combinations_fig_{fig_num+1}.png', 
                bbox_inches='tight',
                dpi=300,
                pad_inches=0.2)
    plt.show()
    plt.close()

print(f"Successfully created {n_figures} figures!")
print("Files saved as: stacking_feature_combinations_fig_1.png to stacking_feature_combinations_fig_N.png")
