# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 11:34:20 2024

@author: P70090917
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Assuming your data is in a DataFrame called 'df'
# If it's not, you'll need to load your data first
# df = pd.read_csv('your_data.csv')  # Uncomment and modify this line if needed
# df2.fillna(0, inplace=True)
df_scaled=df_scaled.drop('Sample name', axis=1) # need to activate this when running for the first time
                                                # after loading and preprocessing code
# Calculate the correlation matrix
correlation_matrix = df_scaled.corr()

# Define column names (use your actual column names)
columns = ['Lignin (wt%)', 'Co-polyol (wt%)',
           'Co-polyol type (PTHF)', 'Isocyanate (wt%)', 'Isocyanate (mmol NCO)',
           'Isocyanate type', 'Ratio', 'Tin(II) octoate', 'Tg (°C)', 'Swelling ratio (%)']

# Create a mask to show only the lower triangle
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

# Set the font scale for all text elements
sns.set(font_scale=1.5)

# Create the heatmap
plt.figure(figsize=(16, 12.8))
heatmap = sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, square=True, 
            annot_kws={"size": 24}, fmt=".2f", cbar=True)

# Customize the color bar font size
colorbar = heatmap.collections[0].colorbar
colorbar.ax.tick_params(labelsize=24)  # Set the size of the color bar tick labels
colorbar.set_label("Correlation", fontsize=24)  # Set the color bar label font size

# Set labels
plt.xlabel('Features', fontsize=28)
plt.ylabel('Features', fontsize=28)

# Rotate the x-axis labels for better readability
plt.xticks(rotation=45, ha='right')
plt.xticks(fontsize=26)
plt.yticks(fontsize=26)

# Adjust the layout to prevent cutting off labels
plt.tight_layout()

# # Save the figure in multiple formats
# for ext in ['tiff', 'jpg']:
#     plt.savefig(f'Correlation matrix2 .{ext}', dpi=600, bbox_inches='tight')

# Show the plot
plt.show()
