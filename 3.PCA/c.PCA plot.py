# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 11:49:47 2024

@author: P70090917
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Create DataFrame from the feature importance data
feature_importance = pd.DataFrame(
    pca_final.components_.T,
    columns=[f'PC{i+1}' for i in range(n_components)],
    index=numeric_columns
)

# Calculate average absolute loading across all PCs for each feature
avg_importance = np.abs(feature_importance).mean(axis=1)

# Calculate maximum absolute loading for each feature
max_importance = np.abs(feature_importance).max(axis=1)

# Create DataFrame for plotting
df_plot = pd.DataFrame({
    'Features': feature_importance.index,
    'Average': avg_importance,
    'Maximum': max_importance
})

# Sort by average importance
df_plot = df_plot.sort_values('Average', ascending=True)  # Ascending for horizontal bars

# Set the figure size and DPI
plt.figure(figsize=(10, 8), dpi=300)

# Create bar plots
bar_width = 0.35
index = np.arange(len(df_plot))

# Colors for the bars: vibrant blue and rich red
colors = ['#1f77b4', '#d62728']

# Create the grouped bar chart
average_bars = plt.barh(index, df_plot['Average'], bar_width, 
                       label='Average Absolute Loading', color=colors[0], alpha=0.8)
maximum_bars = plt.barh(index + bar_width, df_plot['Maximum'], bar_width,
                       label='Maximum Absolute Loading', color=colors[1], alpha=0.8)

# Customize the plot
plt.xlabel('Absolute Loading', fontsize=22)
plt.ylabel('Features', fontsize=22)

# Set y-tick labels
plt.yticks(index + bar_width / 2, df_plot['Features'], fontsize=20)

# Customize x-axis
plt.xlim(0, max(df_plot['Maximum']) * 1.1)
plt.xticks(fontsize=20)

# Add a left-aligned legend
plt.legend(fontsize=20, loc='upper left', bbox_to_anchor=(-0.02, -0.1), ncol=1)

# Add gridlines
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Add value labels on the bars
def add_value_labels(bars):
    for bar in bars:
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2, f'{width:.2f}', 
                ha='left', va='center', fontsize=16)

# Add value labels to both sets of bars
add_value_labels(average_bars)
add_value_labels(maximum_bars)

# Adjust layout to make space for the legend
plt.tight_layout()

# Save figures in various formats
formats = ['tiff']
for fmt in formats:
    plt.savefig(f'PCA_feature_Importance2.{fmt}', 
                dpi=600 if fmt in ['png', 'tiff'] else None, 
                bbox_inches='tight')

# Show plot
plt.show()