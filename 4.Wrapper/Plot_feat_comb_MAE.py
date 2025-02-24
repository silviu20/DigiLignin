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

def create_subplot(ax, data, feature_combination, MODELS):
    """Create a single subplot for a feature combination"""
    # Find best model for this feature combination
    MODEL = data.loc[data['MAE Validation'].idxmin()]['Model']
    
    # Add light grid
    ax.grid(True, linestyle='--', color='#E0E0E0', zorder=0)
    
    # Get y-axis limits to help with legend placement
    y_min = data['MAE Validation'].min()
    y_max = data['MAE Validation'].max()
    y_range = y_max - y_min
    
    # Plot each model
    lines = []
    labels = []
    
    # Get unique x values and create mapping to equally spaced positions
    all_x_values = sorted(data["Number of Estimators"].unique())
    x_positions = np.arange(len(all_x_values))
    x_mapping = dict(zip(all_x_values, x_positions))
    
    for i, model in enumerate(MODELS):
        d = data[data["Model"] == model]
        x = np.array([x_mapping[val] for val in d["Number of Estimators"].values])
        y = d["MAE Validation"].values
        
        lower_err = np.abs(y - d["Validation MAE CI Lower"].values)
        upper_err = np.abs(d["Validation MAE CI Upper"].values - y)
        yerr = [lower_err, upper_err]
        
        if model == MODEL:
            color = colors[0]
            lw = 2.0
            zorder = 10
            alpha = 1.0
        else:
            color = '#BFBFBF'
            lw = 1.2
            zorder = 1
            alpha = 0.7
            
        # Plot with enhanced error bars
        line = ax.errorbar(x, y, yerr=yerr, color=color, lw=lw, zorder=zorder,
                         fmt=markers[i % len(markers)], capsize=3, capthick=1.0,
                         markersize=marker_size, alpha=alpha, markeredgewidth=1.0,
                         markeredgecolor='white', ecolor=color)
        
        lines.append(line)
        labels.append(model)
    
    # Customize subplot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['bottom'].set_linewidth(1.0)
    
    # Set equally spaced x-axis ticks and their labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels(all_x_values, rotation=45)
   
    # Set x-axis limits with padding
    ax.set_xlim(-0.5, len(all_x_values) - 0.5)
    
    
    
    # Set labels
    ax.set_xlabel("Number of Estimators", fontsize=18)
    ax.set_ylabel("MAE Validation", fontsize=18)
    
    # Create wrapped title from feature combination
    title = str(feature_combination).replace("'", "").replace("[", "").replace("]", "")
    
    # Draw the figure once to ensure correct window extent calculations
    plt.draw()
    
    # Get wrapped title with dynamic width
    wrapped_title = wrap_title(title, ax)
    
    # Set title with adjusted parameters
    ax.set_title(wrapped_title, fontsize=14, pad=10, wrap=True,
                bbox=dict(facecolor='white', edgecolor='none', pad=3.0, alpha=0.9))
    
    # Add tick parameters
    ax.tick_params(axis='both', which='major', labelsize=9, width=1.0, length=4)
    
    # Create legend inside the plot at the top
    legend = ax.legend(lines, labels, 
                      loc='upper center',
                      ncol=2,
                      fontsize=10,
                      frameon=True,
                      edgecolor='black',
                      columnspacing=1,
                      handlelength=1.5,
                      handletextpad=0.5,
                      bbox_to_anchor=(0.5, 0.98))
    legend.get_frame().set_linewidth(0.5)
    legend.get_frame().set_alpha(0.9)
    
    # Adjust y-axis limits to accommodate legend
    ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.25 * y_range)
    
    return ax

# Define color palette and markers
colors = ['#0b53c1', '#e31a1c', '#33a02c', '#6a3d9a', '#ff7f00', '#1f78b4', '#b15928']
markers = ['o', 's', '^', 'D', 'v', 'p', '*']
marker_size = 6

# Read and process data
df1 = pd.read_csv('C:/Users/P70090917/Desktop/Polyuerthane Lignin/Experiments/dataset2/rework 21_Nov_2024/testing fesature combinations/stacking_results_all_combinations.csv')

# Extract model names
n_models = df1['Model'].str.extract(r'^(.+?)(?: \(n=\d+\))?$')
df1['Model'] = df1['Model'].replace(dict.fromkeys(df1['Model'].unique(), n_models.values.flatten()))

# Retain only necessary columns
columns = ['Number of Estimators', 'Feature Combination', 'Model', 'MAE Validation', 
          'Validation MAE CI Lower', 'Validation MAE CI Upper']
df1 = df1[columns]

# Get unique feature combinations and models
feature_combinations = df1["Feature Combination"].unique()
MODELS = df1["Model"].unique()

# Calculate number of figures needed (8 subplots per figure: 4 columns x 2 rows)
n_figures = ceil(len(feature_combinations) / 8)

# Create figures
for fig_num in range(n_figures):
    # Create figure with wider aspect ratio for 4x2 layout
    # Adjusted figure height from 16 to 12 to maintain proper aspect ratio with 2 rows
    fig = plt.figure(figsize=(24, 12), dpi=300, facecolor='white')
    
    # Create grid with carefully adjusted spacing
    # Adjusted top margin since we have fewer rows
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
        
        # Get data for this feature combination
        feature_data = df1[df1["Feature Combination"] == feature_comb]
        
        # Create subplot
        create_subplot(ax, feature_data, feature_comb, MODELS)
    
    Save figure
    plt.savefig(f'feature_combinations_fig_{fig_num+1}.png', 
                bbox_inches='tight',
                dpi=300,
                pad_inches=0.2)
    plt.show()
    plt.close()