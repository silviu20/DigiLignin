#!/usr/bin/env python3
"""
SVG to TIFF Converter - Alternative version without Cairo dependencies
Uses Inkscape or SVG rendering libraries that don't require Cairo
"""

import os
import math
import io
import logging
import subprocess
import tempfile
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_inkscape():
    """Check if Inkscape is available in the system PATH."""
    try:
        result = subprocess.run(['inkscape', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            logger.info(f"Inkscape found: {result.stdout.strip()}")
            return True
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
        pass
    return False

def convert_svg_to_tiff_inkscape(svg_path, output_path=None, width_mm=55, height_mm=50, dpi=300):
    """
    Convert SVG to TIFF using Inkscape (requires Inkscape installation).
    
    Args:
        svg_path (str): Path to the input SVG file
        output_path (str, optional): Path for the output TIFF file
        width_mm (float): Width in millimeters
        height_mm (float): Height in millimeters
        dpi (int): Resolution in dots per inch
    
    Returns:
        str: Path to the created TIFF file
    """
    if not check_inkscape():
        raise OSError("Inkscape not found. Please install Inkscape and add it to your PATH.")
    
    # If output path is not specified, create one based on input path
    if output_path is None:
        base_name = os.path.splitext(svg_path)[0]
        output_path = f"{base_name}.tiff"
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Convert mm to pixels for Inkscape
    width_px = int((width_mm / 25.4) * dpi)
    height_px = int((height_mm / 25.4) * dpi)
    
    logger.info(f"Converting {svg_path} to {output_path} using Inkscape")
    logger.info(f"Dimensions: {width_mm}mm × {height_mm}mm at {dpi} DPI")
    logger.info(f"Pixel dimensions: {width_px}px × {height_px}px")
    
    # Use Inkscape to export PNG first
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_png:
        temp_png_path = temp_png.name
    
    try:
        # Inkscape command to export PNG
        cmd = [
            'inkscape',
            '--export-type=png',
            f'--export-filename={temp_png_path}',
            f'--export-width={width_px}',
            f'--export-height={height_px}',
            svg_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            raise RuntimeError(f"Inkscape failed: {result.stderr}")
        
        # Convert PNG to TIFF using PIL
        with Image.open(temp_png_path) as img:
            if img.mode == 'RGBA':
                # Create white background
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])
                img = background
            
            img.save(output_path, format='TIFF', dpi=(dpi, dpi), compression='tiff_lzw')
        
        logger.info(f"Conversion complete: {output_path}")
        return output_path
    
    finally:
        # Clean up temporary PNG file
        if os.path.exists(temp_png_path):
            os.unlink(temp_png_path)

def convert_svg_to_tiff_wand(svg_path, output_path=None, width_mm=55, height_mm=50, dpi=300):
    """
    Convert SVG to TIFF using Wand (ImageMagick Python binding).
    
    Args:
        svg_path (str): Path to the input SVG file
        output_path (str, optional): Path for the output TIFF file
        width_mm (float): Width in millimeters
        height_mm (float): Height in millimeters
        dpi (int): Resolution in dots per inch
    
    Returns:
        str: Path to the created TIFF file
    """
    try:
        from wand.image import Image
        from wand.color import Color
    except ImportError:
        raise ImportError("Wand not installed. Install with: pip install Wand")
    
    # If output path is not specified, create one based on input path
    if output_path is None:
        base_name = os.path.splitext(svg_path)[0]
        output_path = f"{base_name}.tiff"
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Convert mm to pixels
    width_px = int((width_mm / 25.4) * dpi)
    height_px = int((height_mm / 25.4) * dpi)
    
    logger.info(f"Converting {svg_path} to {output_path} using Wand")
    logger.info(f"Dimensions: {width_mm}mm × {height_mm}mm at {dpi} DPI")
    logger.info(f"Pixel dimensions: {width_px}px × {height_px}px")
    
    try:
        with Image(filename=svg_path, resolution=dpi) as img:
            img.resize(width_px, height_px)
            img.background_color = Color('white')
            img.alpha_channel = 'remove'
            
            # Save as TIFF
            with img.clone() as tiff_img:
                tiff_img.compression = 'lzw'
                tiff_img.save(filename=output_path)
        
        logger.info(f"Conversion complete: {output_path}")
        return output_path
    
    except Exception as e:
        raise RuntimeError(f"Wand conversion failed: {str(e)}")

def convert_svg_to_tiff(svg_path, output_path=None, width_mm=55, height_mm=50, dpi=300, method='auto'):
    """
    Convert SVG to TIFF using available methods.
    
    Args:
        svg_path (str): Path to the input SVG file
        output_path (str, optional): Path for the output TIFF file
        width_mm (float): Width in millimeters
        height_mm (float): Height in millimeters
        dpi (int): Resolution in dots per inch
        method (str): Conversion method ('auto', 'inkscape', 'wand')
    
    Returns:
        str: Path to the created TIFF file
    """
    if method == 'auto':
        # Try Inkscape first, then Wand
        if check_inkscape():
            return convert_svg_to_tiff_inkscape(svg_path, output_path, width_mm, height_mm, dpi)
        else:
            logger.info("Inkscape not found, trying Wand...")
            return convert_svg_to_tiff_wand(svg_path, output_path, width_mm, height_mm, dpi)
    elif method == 'inkscape':
        return convert_svg_to_tiff_inkscape(svg_path, output_path, width_mm, height_mm, dpi)
    elif method == 'wand':
        return convert_svg_to_tiff_wand(svg_path, output_path, width_mm, height_mm, dpi)
    else:
        raise ValueError(f"Unknown method: {method}")

def batch_convert_svgs(svg_folder, output_folder=None, width_mm=55, height_mm=50, dpi=300, method='auto'):
    """
    Convert all SVG files in a given folder to TIFF.
    
    Args:
        svg_folder (str): Path to the folder containing SVG files
        output_folder (str, optional): Path to save TIFF files
        width_mm (float): Width in millimeters
        height_mm (float): Height in millimeters
        dpi (int): Resolution in dots per inch
        method (str): Conversion method ('auto', 'inkscape', 'wand')
    
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
                method=method
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
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python svg_to_tiff_converter_no_cairo.py <svg_path_or_folder> [output_path_or_folder] [method]")
        print("Methods: auto (default), inkscape, wand")
        print("Example: python svg_to_tiff_converter_no_cairo.py 'path/to/file.svg'")
        print("Example: python svg_to_tiff_converter_no_cairo.py 'path/to/svg/folder' 'path/to/output/folder' inkscape")
        print("\nRequirements:")
        print("- For Inkscape method: Install Inkscape and add to PATH")
        print("- For Wand method: pip install Wand")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    method = sys.argv[3] if len(sys.argv) > 3 else 'auto'
    
    if os.path.isfile(input_path) and input_path.lower().endswith('.svg'):
        # Convert single file
        convert_svg_to_tiff(input_path, output_path, method=method)
    elif os.path.isdir(input_path):
        # Batch convert folder
        batch_convert_svgs(input_path, output_path, method=method)
    else:
        print(f"Error: {input_path} is not a valid SVG file or directory")
        sys.exit(1)
