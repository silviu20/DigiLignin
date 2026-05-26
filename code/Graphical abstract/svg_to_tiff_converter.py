#!/usr/bin/env python3
"""
SVG to TIFF Converter - Standalone Python Script
Extracted from Jupyter Notebook for direct execution
"""

import os
import math
import io
import logging
from cairosvg import svg2png
from PIL import Image, ImageCms

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def convert_svg_to_tiff(svg_path, output_path=None, width_mm=55, height_mm=50, dpi=2400, 
                        color_mode='RGB', compression='tiff_lzw', quality=100, bg_color=(255, 255, 255)):
    """
    Convert SVG to high-quality TIFF with specified dimensions and settings.
    
    Args:
        svg_path (str): Path to the input SVG file
        output_path (str, optional): Path for the output TIFF file. 
            If None, will use the same name as input with .tiff extension.
        width_mm (float): Width in millimeters
        height_mm (float): Height in millimeters
        dpi (int): Resolution in dots per inch (higher for better quality)
        color_mode (str): Color mode for output ('RGB', 'CMYK', etc.)
        compression (str): TIFF compression method ('tiff_lzw', 'tiff_adobe_deflate', 'tiff_none')
        quality (int): Quality setting (1-100) for compression (higher = better quality)
        bg_color (tuple): Background color as RGB tuple
    
    Returns:
        str: Path to the created TIFF file
    """
    try:
        # Calculate dimensions in pixels based on DPI
        # 1 inch = 25.4 mm
        width_px = int(math.ceil((width_mm / 25.4) * dpi))
        height_px = int(math.ceil((height_mm / 25.4) * dpi))
        
        # If output path is not specified, create one based on input path
        if output_path is None:
            base_name = os.path.splitext(svg_path)[0]
            output_path = f"{base_name}.tiff"
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # Log conversion details
        logger.info(f"Converting {svg_path} to {output_path}")
        logger.info(f"Dimensions: {width_mm}mm × {height_mm}mm at {dpi} DPI")
        logger.info(f"Pixel dimensions: {width_px}px × {height_px}px")
        
        # Convert SVG to PNG in memory with high quality settings
        png_data = svg2png(url=svg_path, 
                          output_width=width_px, 
                          output_height=height_px,
                          scale=1.0,  # Maintain scale
                          unsafe=False)  # Safe mode
        
        # Load PNG from memory into a PIL Image
        png_image = Image.open(io.BytesIO(png_data))
        
        # Handle different color profiles
        if color_mode == 'CMYK' and png_image.mode != 'CMYK':
            # Convert to CMYK color space for print quality
            if png_image.mode == 'RGBA':
                # Create a background with specified color
                background = Image.new('RGB', png_image.size, bg_color)
                # Paste the image on the background using alpha channel
                background.paste(png_image, mask=png_image.split()[3])
                png_image = background
            
            # Convert RGB to CMYK using ICC profile if available
            try:
                rgb_profile = ImageCms.createProfile("sRGB")
                cmyk_profile = ImageCms.createProfile("USWebCoatedSWOP")
                png_image = ImageCms.profileToProfile(png_image, rgb_profile, cmyk_profile, outputMode='CMYK')
                logger.info("Converted to CMYK color space")
            except Exception as e:
                logger.warning(f"Could not convert to CMYK: {e}. Using standard conversion.")
                png_image = png_image.convert('CMYK')
        elif png_image.mode == 'RGBA':
            # Create a background with specified color
            background = Image.new('RGB', png_image.size, bg_color)
            # Paste the image on the background using alpha channel
            background.paste(png_image, mask=png_image.split()[3])
            png_image = background
        
        # Enhanced TIFF saving with optimal settings
        tiff_settings = {
            'format': 'TIFF',
            'dpi': (dpi, dpi),
            'compression': compression
        }
        
        # Add compression-specific parameters
        if compression == 'tiff_adobe_deflate':
            tiff_settings['adobe_deflate_level'] = int(quality / 10)  # Convert to 0-9 scale
        elif compression == 'jpeg':
            tiff_settings['quality'] = quality
        
        # Save as TIFF with specified settings
        png_image.save(output_path, **tiff_settings)
        
        logger.info(f"Conversion complete: {output_path}")
        return output_path
    
    except Exception as e:
        logger.error(f"Error converting {svg_path}: {str(e)}")
        raise

def batch_convert_svgs(svg_folder, output_folder=None, width_mm=55, height_mm=50, dpi=2400,
                       color_mode='RGB', compression='tiff_adobe_deflate', quality=100, bg_color=(255, 255, 255)):
    """
    Convert all SVG files in a given folder to high-quality TIFF.
    
    Args:
        svg_folder (str): Path to the folder containing SVG files
        output_folder (str, optional): Path to save TIFF files. 
            If None, saves in the same folder as SVGs.
        width_mm (float): Width in millimeters
        height_mm (float): Height in millimeters
        dpi (int): Resolution in dots per inch
        color_mode (str): Color mode for output ('RGB', 'CMYK', etc.)
        compression (str): TIFF compression method ('tiff_lzw', 'tiff_adobe_deflate', 'tiff_none')
        quality (int): Quality setting (1-100) for compression
        bg_color (tuple): Background color as RGB tuple
    
    Returns:
        list: Paths of converted TIFF files
    """
    # Create output folder if not specified
    if output_folder is None:
        output_folder = svg_folder
    
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # List to store converted file paths
    converted_files = []
    failed_files = []
    
    # Count total SVG files
    svg_files = [f for f in os.listdir(svg_folder) if f.lower().endswith('.svg')]
    total_files = len(svg_files)
    
    logger.info(f"Found {total_files} SVG files in {svg_folder}")
    
    # Iterate through SVG files in the folder
    for i, filename in enumerate(svg_files, 1):
        try:
            # Full path to input SVG
            svg_path = os.path.join(svg_folder, filename)
            
            # Generate output TIFF filename
            tiff_filename = os.path.splitext(filename)[0] + '.tiff'
            tiff_path = os.path.join(output_folder, tiff_filename)
            
            logger.info(f"Processing file {i}/{total_files}: {filename}")
            
            # Convert the SVG to TIFF
            converted_file = convert_svg_to_tiff(
                svg_path, 
                output_path=tiff_path, 
                width_mm=width_mm, 
                height_mm=height_mm, 
                dpi=dpi,
                color_mode=color_mode,
                compression=compression,
                quality=quality,
                bg_color=bg_color
            )
            converted_files.append(converted_file)
            
        except Exception as e:
            logger.error(f"Failed to convert {filename}: {str(e)}")
            failed_files.append(filename)
    
    # Summary
    logger.info(f"Conversion summary: {len(converted_files)} successful, {len(failed_files)} failed")
    if failed_files:
        logger.warning(f"Failed files: {', '.join(failed_files)}")
    
    return converted_files

if __name__ == "__main__":
    # Example usage when running as script
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python svg_to_tiff_converter.py <svg_path_or_folder> [output_path_or_folder]")
        print("Example: python svg_to_tiff_converter.py 'path/to/file.svg'")
        print("Example: python svg_to_tiff_converter.py 'path/to/svg/folder' 'path/to/output/folder'")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    if os.path.isfile(input_path) and input_path.lower().endswith('.svg'):
        # Convert single file
        convert_svg_to_tiff(input_path, output_path)
    elif os.path.isdir(input_path):
        # Batch convert folder
        batch_convert_svgs(input_path, output_path)
    else:
        print(f"Error: {input_path} is not a valid SVG file or directory")
        sys.exit(1)
