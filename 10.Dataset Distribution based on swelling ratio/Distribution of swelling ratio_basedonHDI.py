# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 17:16:15 2024
@author: P70090917
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# Assuming df is your DataFrame
# Unique values in the 'Ratio' column
unique_ratios = [1.4, 1.2, 1, 0.8, 0.6]
isocyanate_types = [0, 1]

# Create a custom colormap
colors = ['#FF9999', '#FF3333', '#990000', '#3333FF', '#9999FF']
n_bins = 100
cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=n_bins)

# Set the style for the plot
plt.style.use('default')
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.5)  # Increased font scale for better readability

# Create the figure and axes
fig, axes = plt.subplots(len(unique_ratios), 2, figsize=(18, 4 * len(unique_ratios)), sharex=True)
# fig.suptitle('Distribution of Swelling Ratio (%) for Different Ratios and Isocyanate Types', fontsize=24, y=1.02)

# Iterate through each unique ratio value and isocyanate type
for i, ratio in enumerate(unique_ratios):
    for j, isocyanate_type in enumerate(isocyanate_types):
        ax = axes[i, j]

        # Filter the dataset based on the current ratio value and isocyanate type
        filtered_df = df[(df['Ratio'] == ratio) & (df['Isocyanate type'] == isocyanate_type)]

        # Plot the histogram and KDE
        sns.histplot(filtered_df['Swelling ratio (%)'], kde=True, ax=ax, color=colors[j],
                     edgecolor='black', linewidth=0.5, alpha=0.7)

        # Customize the subplot
        isocyanate_label = 'HDI' if isocyanate_type == 0 else 'HDIt'
        ax.set_title(f'Ratio = {ratio}, Isocyanate type = {isocyanate_label}', fontsize=18)
        ax.set_xlabel('Swelling Ratio (%)', fontsize=18)
        ax.set_ylabel('Frequency', fontsize=18)

        # Increase tick label font size
        ax.tick_params(axis='both', which='major', labelsize=14)

        # Add text with percentage and range
        total_rows = len(df)
        filtered_rows = len(filtered_df)
        percentage = (filtered_rows / total_rows) * 100
        swelling_ratio_range = filtered_df['Swelling ratio (%)'].agg(['min', 'max'])

        ax.text(0.95, 0.95,
                f"Percentage: {percentage:.2f}%\nRange: [{swelling_ratio_range['min']:.2f}, {swelling_ratio_range['max']:.2f}]",
                transform=ax.transAxes, ha='right', va='top',
                bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8),
                fontsize=14)  # Increased font size for the text box

# Adjust the layout and save the figure
plt.tight_layout()
formats = ['tiff', 'pdf', 'png']
for fmt in formats:
    plt.savefig(f'Distribution_of_swelling_ratio_vs_Ratio_and_Isocyanate_type_without_title.{fmt}',
                dpi=600 if fmt in ['png', 'tiff'] else None,
                bbox_inches='tight')
plt.show()
