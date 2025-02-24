from PIL import Image, ImageDraw, ImageFont

# List of image filenames
image_filenames = ['image1.png', 'image2.png', 'image3.png', 'image4.png','image5.png','image6.png' ]

# Load the images
images = [Image.open(filename) for filename in image_filenames]

# Font for labels
try:
    # Try to load Arial Bold font
    font = ImageFont.truetype("arialbd.ttf", 190)  # Arial Bold
except IOError:
    try:
        # Fallback to regular Arial but with increased weight
        font = ImageFont.truetype("arial.ttf", 190, font_weight='bold')
    except IOError:
        font = ImageFont.load_default()

# Set padding between images and between columns
vertical_padding = 20
horizontal_padding = 50

# Determine dimensions
max_width = max(img.width for img in images)
total_width = max_width * 2 + horizontal_padding

column1_height = sum(img.height for idx, img in enumerate(images) if idx % 2 == 0) + (len(images) // 2) * vertical_padding
column2_height = sum(img.height for idx, img in enumerate(images) if idx % 2 == 1) + ((len(images) - 1) // 2) * vertical_padding
total_height = max(column1_height, column2_height)

# Create merged image
merged_image = Image.new('RGB', (total_width, total_height), (255, 255, 255))
draw = ImageDraw.Draw(merged_image)

# Define y-axis label
y_axis_label = "Tg/°C"

# Define labels A through F
labels = ['A', 'B', 'C', 'D', 'E', 'F']

# Paste images and add labels
x_offsets = [0, max_width + horizontal_padding]
y_offsets = [0, 0]

for idx, img in enumerate(images):
    column = idx % 2
    
    # Create a copy of the image to draw the label on
    img_with_label = img.copy()
    draw_label = ImageDraw.Draw(img_with_label)
    
    # Add label to the image
    label = labels[idx]
    # Get text size for positioning
    text_bbox = draw_label.textbbox((0, 0), label, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    # Position label more towards the right
    label_x = 550 # Changed from 10 to 50 - adjust this value to move it more right
    label_y = 40
    
    # Draw label with black color
    draw_label.text((label_x, label_y), label, fill=(0, 0, 0), font=font)
    
    # Paste the image with label
    merged_image.paste(img_with_label, (x_offsets[column], y_offsets[column]))
    
    # Create vertical y-axis label
    label_text = y_axis_label
    text_bbox = draw.textbbox((0, 0), label_text, font=font)
    label_width = text_bbox[2] - text_bbox[0]
    label_height = text_bbox[3] - text_bbox[1]
    
    # Create and rotate vertical label
    label_image = Image.new('RGBA', (label_height, label_width), (255, 255, 255, 0))
    label_draw = ImageDraw.Draw(label_image)
    label_draw.text((0, 0), label_text, fill=(0, 0, 0), font=font)
    label_image = label_image.rotate(90, expand=1)
    
    # Update Y-offset for the column
    y_offsets[column] += img_with_label.height + vertical_padding

# Save the merged image
merged_image.save('merged_image_with_labels.png', format='PNG', dpi=(600, 600))