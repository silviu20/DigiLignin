# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 11:19:26 2024

@author: P70090917
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import inferno

# Load the dataset
df = pd.read_csv('C:/Users/P70090917/Desktop/Polyuerthane Lignin/Experiments/dataset2/rework 21_Nov_2024/testing fesature combinations/mapped_results_tg.csv')
# df=mapped_results
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
bar_width = 0.8  # Width of each bar
x_positions = x_values[:-1] + 0.1  # Shift x positions slightly to center bars

# Plot a bar for each integer in x_values
plt.bar(x_positions, counts, color=colors, edgecolor='white', width=bar_width, linewidth=1.5)

# Add count labels for the 5 highest and 5 lowest bars
sorted_indices = np.argsort(counts)  # Get indices of counts sorted in ascending order
lowest_indices = sorted_indices[:7]   # Indices of the 5 lowest counts
highest_indices = sorted_indices[-5:]  # Indices of the 5 highest counts

# Calculate the median values (considering count frequencies)
median_indices = np.argsort(np.abs(counts - np.median(counts)))[:5]  # Indices closest to the median counts

# Combine the indices to annotate
indices_to_annotate = np.concatenate((lowest_indices, highest_indices, median_indices))

# Add annotations with increased font size
for i in indices_to_annotate:
    count = counts[i]
    if count > 0:  # Add labels only for non-zero bars
        plt.text(x_positions[i], count + (max(counts) * 0.01),
                 f'{count:,}', ha='center', va='bottom', 
                 fontsize=16, color='black', rotation=90)  # Increased fontsize for annotations

# Customize the plot
# plt.title('Distribution of Predicted Tg Values', fontsize=30, pad=20, color='black')  # Uncomment if you want the title
plt.xlabel('Predicted Tg / °C', fontsize=28, color='black', labelpad=15)  # Increased xlabel fontsize
plt.ylabel('Frequency', fontsize=28, color='black', labelpad=15)  # Increased ylabel fontsize

# Customize grid and spines - only horizontal grid
plt.grid(axis='y', alpha=0.3, linestyle='--', color='#666666')  # Only horizontal grid lines
for spine in plt.gca().spines.values():
    spine.set_color('black')
    spine.set_linewidth(1.5)

# Set x-ticks with increased distance
tick_positions = x_values
offset = 0.3  # Adjust this value to increase/decrease the distance between ticks
plt.xticks(tick_positions + offset, [f'{x:d}' for x in tick_positions], rotation=90, ha='center', fontsize=16)  # Increased fontsize for ticks

# Set y-tick font size
plt.yticks(fontsize=20)  # Increased fontsize for y-ticks

# Adjust y-axis for headroom
plt.margins(x=0.01, y=0.2)

# Customize ticks
plt.tick_params(colors='black', width=1.5, length=8)

# Prevent label cutoff
plt.tight_layout()

# # Save the figure in multiple formats
# for ext in ['tiff','svg', 'jpg']:
#     plt.savefig(f'Distribution of Predicted Tg Values_mapped_results.{ext}', dpi=600, bbox_inches='tight')

# Show plot
plt.show()
