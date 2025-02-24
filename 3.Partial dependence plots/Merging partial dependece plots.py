from PIL import Image, ImageDraw, ImageFont

# List of image filenames
image_filenames = ['image1.tiff', 'image2.tiff', 'image3.tiff', 'image4.tiff']

# Load the images
images = [Image.open(filename) for filename in image_filenames]

# Font size settings
LABEL_FONT_SIZE = 300 # Large font size for A, B, C, D labels
YAXIS_FONT_SIZE = 48  # Medium font size for y-axis label

# Font setup for labels
try:
    # Larger font for the A, B, C, D labels
    label_font = ImageFont.truetype("arial.ttf", LABEL_FONT_SIZE)
    # Smaller but still visible font for y-axis label
    y_axis_font = ImageFont.truetype("arial.ttf", YAXIS_FONT_SIZE)
except IOError:
    label_font = ImageFont.load_default()
    y_axis_font = ImageFont.load_default()
    print("Warning: Arial font not found, using default font")

# Padding settings
vertical_padding = 20
horizontal_padding = 50
label_padding = 30  # Increased padding for larger labels
background_padding = 10  # Padding for white background behind labels

# Calculate dimensions
max_width = max(img.width for img in images)
total_width = max_width * 2 + horizontal_padding

# Calculate heights for each column
column1_height = sum(img.height for idx, img in enumerate(images) if idx % 2 == 0) + (len(images) // 2) * vertical_padding
column2_height = sum(img.height for idx, img in enumerate(images) if idx % 2 == 1) + ((len(images) - 1) // 2) * vertical_padding
total_height = max(column1_height, column2_height)

# Create new blank image
merged_image = Image.new('RGB', (total_width, total_height), (255, 255, 255))
draw = ImageDraw.Draw(merged_image)

# Define labels
labels = ['A', 'B', 'C', 'D']
y_axis_label = " "

# Column x-offsets
x_offsets = [0, max_width + horizontal_padding]
y_offsets = [0, 0]

# Paste images and add labels
for idx, img in enumerate(images):
    column = idx % 2
    row = idx // 2
    
    # Paste the image
    current_x = x_offsets[column]
    current_y = y_offsets[column]
    merged_image.paste(img, (current_x, current_y))
    
    # Add A, B, C, D label
    label = labels[idx]
    # Get text size for centering
    text_bbox = draw.textbbox((0, 0), label, font=label_font)
    label_width = text_bbox[2] - text_bbox[0]
    label_height = text_bbox[3] - text_bbox[1]
    
    # Draw label with white background for better visibility
    label_x = current_x + label_padding
    label_y = current_y + label_padding
    
    # Draw white background rectangle with increased padding
    draw.rectangle([
        label_x - background_padding,
        label_y - background_padding,
        label_x + label_width + background_padding,
        label_y + label_height + background_padding
    ], fill='white')
    
    # Draw the label text
    draw.text((label_x, label_y), label, fill=(0, 0, 0), font=label_font)
    
    # Add y-axis label
    label_text = y_axis_label
    text_bbox = draw.textbbox((0, 0), label_text, font=y_axis_font)
    label_width = text_bbox[2] - text_bbox[0]
    label_height = text_bbox[3] - text_bbox[1]
    
    # Create vertical y-axis label
    label_image = Image.new('RGBA', (label_height, label_width), (255, 255, 255, 0))
    label_draw = ImageDraw.Draw(label_image)
    label_draw.text((0, 0), label_text, fill=(0, 0, 0), font=y_axis_font)
    label_image = label_image.rotate(90, expand=1)
    
    # Paste the y-axis label
    merged_image.paste(label_image, (current_x - 60, current_y), label_image)
    
    # Update Y-offset for the column
    y_offsets[column] += img.height + vertical_padding

# Save the final image
merged_image.save('merged_image_with_large_labels.tiff', format='TIFF', dpi=(300, 300))