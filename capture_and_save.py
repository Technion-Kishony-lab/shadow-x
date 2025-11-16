#!/usr/bin/env python
"""
Capture and save brightfield and darkfield images.

This script:
1. Takes a simple brightfield image (white backlight)
2. Captures multiple images with moving stripe patterns
3. Processes them into a synthetic darkfield image
4. Saves all results to disk

Usage:
    python capture_and_save.py
"""
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path

from dark_field import StripeRunner, StripesAnalyzer
from resources.camera_and_screens import get_overhead_camera, get_or_create_backlight_screen
from graphics.helpers import set_matplotlib_backend


def create_output_filename(base_name, suffix=""):
    """Create timestamped filename"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if suffix:
        return f"{timestamp}_{base_name}_{suffix}.jpg"
    return f"{timestamp}_{base_name}.jpg"


def capture_brightfield_image(output_dir="output"):
    """
    Capture a simple brightfield image with uniform white backlight.
    
    Args:
        output_dir: Directory to save the image
        
    Returns:
        tuple: (frame, filename)
    """
    print("=" * 60)
    print("Capturing Brightfield Image")
    print("=" * 60)
    
    Path(output_dir).mkdir(exist_ok=True)
    
    camera = get_overhead_camera()
    backlight_screen = get_or_create_backlight_screen()
    
    # Illuminate with white
    print("Setting white backlight...")
    backlight_screen.illuminate(color=(255, 255, 255), pause=0.5)
    
    # Capture frame
    print("Capturing frame...")
    frame = camera.take_picture()
    
    # Save
    filename = Path(output_dir) / create_output_filename("brightfield")
    cv2.imwrite(str(filename), frame)
    print(f"✓ Saved brightfield image: {filename}")
    
    # Turn off backlight
    backlight_screen.illuminate(color=(0, 0, 0), pause=0.1)
    
    return frame, str(filename)


def capture_darkfield_images(light_width=14, dark_width=17, num_images=None,
                             num_images_to_discard=11, 
                             dark_dist=0.4, max_dark_dist=1.0, smoothing=1,
                             output_dir="output"):
    """
    Capture and process darkfield images using stripe illumination.
    
    Args:
        light_width: Width of bright stripes in pixels
        dark_width: Width of dark stripes in pixels
        num_images: Number of images to capture (default: light_width + dark_width)
        num_images_to_discard: Number of initial images to discard
        dark_dist: Gaussian weighting parameter for averaging
        max_dark_dist: Maximum distance in dark phase to include
        smoothing: Gaussian smoothing sigma for preprocessing
        output_dir: Directory to save images
        
    Returns:
        dict: Dictionary with keys 'rgb', 'gray', 'rgb_file', 'gray_file', 'data_file'
    """
    print("\n" + "=" * 60)
    print("Capturing Darkfield Images")
    print("=" * 60)
    
    Path(output_dir).mkdir(exist_ok=True)
    
    # Step 1: Run stripe illumination capture
    print(f"\n1. Capturing {num_images or (light_width + dark_width)} images with moving stripes...")
    print(f"   - Light width: {light_width}")
    print(f"   - Dark width: {dark_width}")
    print(f"   - Discarding first: {num_images_to_discard}")
    
    runner = StripeRunner(
        light_width=light_width,
        dark_width=dark_width,
        num_images=num_images,
        num_images_to_discard=num_images_to_discard,
        show_camera=True  # Show live capture
    )
    runner.run()
    
    print(f"   ✓ Captured {len(runner.images)} images")
    
    # Step 2: Process into darkfield
    print("\n2. Processing images into darkfield...")
    analyzer = StripesAnalyzer.from_stripe_runner(runner)
    rgb, gray = analyzer.get_pseudo_darkfield_image(
        dark_dist=dark_dist,
        max_dark_dist=max_dark_dist,
        smoothing=smoothing,
        substract_background=False,
        normalize=False
    )
    
    print("   ✓ Darkfield processing complete")
    
    # Step 3: Save results
    print("\n3. Saving results...")
    
    # Save RGB darkfield
    rgb_filename = Path(output_dir) / create_output_filename("darkfield", "rgb")
    cv2.imwrite(str(rgb_filename), rgb)
    print(f"   ✓ Saved RGB darkfield: {rgb_filename}")
    
    # Save grayscale darkfield
    gray_filename = Path(output_dir) / create_output_filename("darkfield", "gray")
    cv2.imwrite(str(gray_filename), gray)
    print(f"   ✓ Saved grayscale darkfield: {gray_filename}")
    
    # Save raw data (pickle)
    data_filename = Path(output_dir) / create_output_filename("darkfield_data", "pkl").replace('.jpg', '')
    runner.to_pickle(str(data_filename))
    print(f"   ✓ Saved raw data: {data_filename}")
    
    # Also save reference images
    bright_ref_filename = Path(output_dir) / create_output_filename("reference", "bright")
    cv2.imwrite(str(bright_ref_filename), runner.bright_image)
    
    dark_ref_filename = Path(output_dir) / create_output_filename("reference", "dark")
    cv2.imwrite(str(dark_ref_filename), runner.dark_image)
    print(f"   ✓ Saved reference images")
    
    return {
        'rgb': rgb,
        'gray': gray,
        'rgb_file': str(rgb_filename),
        'gray_file': str(gray_filename),
        'data_file': str(data_filename),
        'bright_ref': str(bright_ref_filename),
        'dark_ref': str(dark_ref_filename),
        'analyzer': analyzer,
        'runner': runner
    }


def capture_full_session(output_dir="output"):
    """
    Capture a complete imaging session: brightfield + darkfield.
    
    Args:
        output_dir: Directory to save all images
        
    Returns:
        dict: Dictionary with all captured data and filenames
    """
    print("\n" + "=" * 60)
    print("CAPTURE AND SAVE SESSION")
    print("=" * 60)
    
    # Initialize matplotlib backend
    set_matplotlib_backend()
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    print(f"\nOutput directory: {output_dir}/\n")
    
    # Capture brightfield
    brightfield_frame, brightfield_file = capture_brightfield_image(output_dir)
    
    # Capture darkfield
    darkfield_results = capture_darkfield_images(
        light_width=14,
        dark_width=17,
        num_images_to_discard=11,
        dark_dist=0.4,
        max_dark_dist=1.0,
        smoothing=1,
        output_dir=output_dir
    )
    
    print("\n" + "=" * 60)
    print("SESSION COMPLETE!")
    print("=" * 60)
    print("\nSaved files:")
    print(f"  Brightfield:      {brightfield_file}")
    print(f"  Darkfield (RGB):  {darkfield_results['rgb_file']}")
    print(f"  Darkfield (gray): {darkfield_results['gray_file']}")
    print(f"  Raw data:         {darkfield_results['data_file']}")
    print(f"  References:       {darkfield_results['bright_ref']}, {darkfield_results['dark_ref']}")
    print("\n")
    
    return {
        'brightfield_frame': brightfield_frame,
        'brightfield_file': brightfield_file,
        **darkfield_results
    }


def main():
    """Run a complete capture session"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Capture brightfield and darkfield images")
    parser.add_argument('--display', action='store_true', 
                       help='Display results after capture (requires closing windows)')
    parser.add_argument('--output', default='output',
                       help='Output directory (default: output)')
    args = parser.parse_args()
    
    results = capture_full_session(output_dir=args.output)
    
    # Optionally display results
    if args.display:
        import matplotlib.pyplot as plt
        from resources.camera_and_screens import create_camera_figure
        
        print("\nDisplaying results (close windows to exit)...")
        print("Press Ctrl+C in terminal to skip display\n")
        
        try:
            plt.ion()
            
            fig1 = create_camera_figure(0)
            fig1.set_image(results['brightfield_frame'], allow_resize=True)
            fig1.set_text("Brightfield")
            
            fig2 = create_camera_figure(1)
            fig2.set_image(results['rgb'], allow_resize=True)
            fig2.set_text("Darkfield (RGB)")
            
            fig3 = create_camera_figure(2)
            fig3.set_image(results['gray'], allow_resize=True, cmap='gray', clim=(0, 160))
            fig3.set_text("Darkfield (Gray)")
            
            print("Close the matplotlib windows to finish.")
            plt.show(block=True)
        except KeyboardInterrupt:
            print("\nDisplay skipped.")
    else:
        print("\nCapture complete! Use --display flag to view results.")


if __name__ == "__main__":
    main()

