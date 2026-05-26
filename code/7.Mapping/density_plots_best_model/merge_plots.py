import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import numpy as np

# Define the image paths and their labels
image_info = {
    'A': 'density_plot_Lignin_wtpct_Tg_degC.png',
    'B': 'density_plot_Co-polyol_type_PTHF_Tg_degC.png', 
    'C': 'density_plot_Co-polyol_type_PTHF_Tg_degC.png',  # Same as B per your request
    'D': 'density_plot_Copolyol_wtpct_Tg_degC.png',
    'E': 'density_plot_Isocyanate_wtpct_Tg_degC.png'
}

# Load images
images = {}
for label, filename in image_info.items():
    try:
        img = Image.open(filename)
        images[label] = img
        print(f"Loaded {label}: {filename} - Size: {img.size}")
    except Exception as e:
        print(f"Error loading {filename}: {e}")

# Create a figure to arrange the plots in 2x2 grid with one in middle at bottom
fig, axes = plt.subplots(3, 2, figsize=(12, 18))
# Remove title completely

# Define positions: 2x2 grid + one in middle at bottom
positions = {'A': (0, 0), 'B': (0, 1), 'C': (1, 0), 'D': (1, 1), 'E': (2, 1)}

for label, filename in image_info.items():
    row, col = positions[label]
    ax = axes[row, col]
    
    if label in images:
        ax.imshow(images[label])
        # Add label within the plot area, moved 2cm to the right from left corner
        # 2cm ≈ 0.79 inches, for a 12-inch wide figure this is ~0.066 of figure width
        ax.text(0.066, 0.95, f'{label}', transform=ax.transAxes, 
                fontsize=14, fontweight='bold', va='top', ha='left',
                color='black', bbox=dict(boxstyle='round,pad=0.3', facecolor='none', alpha=0, edgecolor='none'))
        ax.axis('off')
    else:
        ax.text(0.5, 0.5, f'Image {label}\nnot found', 
                ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')

# Hide the bottom-left subplot (position 2,0) since we only need one image in middle at bottom
axes[2, 0].axis('off')

# Reduce spacing between plots significantly
plt.subplots_adjust(wspace=0.05, hspace=0.001)

plt.tight_layout()
plt.savefig('merged_density_plots.png', dpi=300, bbox_inches='tight')
plt.show()

print("Merged plot saved as 'merged_density_plots.png'")
