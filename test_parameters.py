#!/usr/bin/env python
"""
Test the effect of different parameters on darkfield image quality.

This script systematically tests:
1. Strip thickness (light_width and dark_width)
2. Sample rate (number of images captured)

Usage:
    python test_parameters.py
    python test_parameters.py --quick  # Run a quick test with fewer combinations
"""

import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
import argparse

from dark_field import StripeRunner, StripesAnalyzer
from graphics.helpers import set_matplotlib_backend


def test_parameter_combination(light_width, dark_width, num_images, 
                               num_images_to_discard=11,
                               dark_dist=0.4, max_dark_dist=1.0, smoothing=1,
                               output_dir="output/parameter_tests"):
    """
    Test a single combination of parameters.
    
    Args:
        light_width: Width of bright stripes
        dark_width: Width of dark stripes
        num_images: Number of images to capture (None = light_width + dark_width)
        num_images_to_discard: Initial images to skip
        dark_dist: Gaussian weighting parameter
        max_dark_dist: Maximum distance in dark phase
        smoothing: Gaussian smoothing sigma
        output_dir: Directory to save results
        
    Returns:
        dict: Results including images and filenames
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Determine actual number of images
    actual_num_images = num_images if num_images is not None else (light_width + dark_width)
    
    # Create descriptive filename base
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    param_str = f"lw{light_width}_dw{dark_width}_n{actual_num_images}"
    
    print(f"\n{'='*70}")
    print(f"Testing: light_width={light_width}, dark_width={dark_width}, num_images={actual_num_images}")
    print(f"{'='*70}")
    
    # Capture images with stripe illumination
    print(f"Capturing {actual_num_images} images (discarding first {num_images_to_discard})...")
    runner = StripeRunner(
        light_width=light_width,
        dark_width=dark_width,
        num_images=num_images,
        num_images_to_discard=num_images_to_discard,
        show_camera=True
    )
    runner.run()
    
    print(f"Captured {len(runner.images)} images")
    
    # Process into darkfield
    print("Processing darkfield...")
    analyzer = StripesAnalyzer.from_stripe_runner(runner)
    rgb, gray = analyzer.get_pseudo_darkfield_image(
        dark_dist=dark_dist,
        max_dark_dist=max_dark_dist,
        smoothing=smoothing,
        substract_background=False,
        normalize=False
    )
    
    # Save results with descriptive names
    rgb_file = Path(output_dir) / f"{timestamp}_{param_str}_rgb.jpg"
    gray_file = Path(output_dir) / f"{timestamp}_{param_str}_gray.jpg"
    bright_file = Path(output_dir) / f"{timestamp}_{param_str}_ref_bright.jpg"
    dark_file = Path(output_dir) / f"{timestamp}_{param_str}_ref_dark.jpg"
    
    cv2.imwrite(str(rgb_file), rgb)
    cv2.imwrite(str(gray_file), gray)
    cv2.imwrite(str(bright_file), runner.bright_image)
    cv2.imwrite(str(dark_file), runner.dark_image)
    
    print(f"✓ Saved: {param_str}")
    
    return {
        'light_width': light_width,
        'dark_width': dark_width,
        'num_images': actual_num_images,
        'rgb': rgb,
        'gray': gray,
        'rgb_file': str(rgb_file),
        'gray_file': str(gray_file),
        'param_str': param_str
    }


def run_strip_thickness_test(output_dir="output/parameter_tests", quick=False):
    """
    Test different strip thicknesses while keeping sample rate constant.
    
    Tests combinations of light_width and dark_width.
    """
    print("\n" + "="*70)
    print("STRIP THICKNESS TEST")
    print("="*70)
    
    if quick:
        # Quick test with extreme variations
        test_configs = [
            {'light_width': 1, 'dark_width': 1},      # Extremely thin
            {'light_width': 5, 'dark_width': 5},      # Very thin
            {'light_width': 10, 'dark_width': 10},    # Thin
            {'light_width': 50, 'dark_width': 50},    # Thick
            {'light_width': 100, 'dark_width': 100},  # Very thick
        ]
    else:
        # Full test with more combinations
        test_configs = [
            {'light_width': 5, 'dark_width': 5},    # Thin, equal
            {'light_width': 10, 'dark_width': 10},  # Medium, equal
            {'light_width': 14, 'dark_width': 17},  # Current default
            {'light_width': 15, 'dark_width': 15},  # Medium, equal
            {'light_width': 10, 'dark_width': 20},  # Thin light, thick dark
            {'light_width': 20, 'dark_width': 10},  # Thick light, thin dark
            {'light_width': 20, 'dark_width': 30},  # Both thick
            {'light_width': 25, 'dark_width': 25},  # Very thick, equal
        ]
    
    results = []
    for config in test_configs:
        result = test_parameter_combination(
            light_width=config['light_width'],
            dark_width=config['dark_width'],
            num_images=None,  # Auto: light_width + dark_width
            output_dir=output_dir
        )
        results.append(result)
    
    return results


def run_sample_rate_test(output_dir="output/parameter_tests", quick=False):
    """
    Test different sample rates (number of images) while keeping strip thickness constant.
    
    Uses fixed strip widths and varies the number of captured images.
    """
    print("\n" + "="*70)
    print("SAMPLE RATE TEST")
    print("="*70)
    
    # Use default strip widths
    light_width = 14
    dark_width = 17
    period = light_width + dark_width  # 31
    
    if quick:
        # Quick test with fewer samples
        sample_counts = [
            period // 2,      # Under-sampled (15-16 images)
            period,           # One full period (31 images)
            period * 2,       # Two periods (62 images)
        ]
    else:
        # Full test with more variations
        sample_counts = [
            period // 4,      # Very under-sampled
            period // 2,      # Under-sampled
            period,           # One full period
            period + period // 2,  # 1.5 periods
            period * 2,       # Two periods
            period * 3,       # Three periods
        ]
    
    results = []
    for num_images in sample_counts:
        result = test_parameter_combination(
            light_width=light_width,
            dark_width=dark_width,
            num_images=num_images,
            output_dir=output_dir
        )
        results.append(result)
    
    return results


def run_combined_test(output_dir="output/parameter_tests"):
    """
    Test specific combinations that might be interesting.
    """
    print("\n" + "="*70)
    print("COMBINED TEST - Interesting Combinations")
    print("="*70)
    
    test_configs = [
        # Format: (light_width, dark_width, num_images_multiplier)
        (10, 40, 1.0),   # Very thin light, thick dark, normal sampling
        (10, 40, 2.0),   # Very thin light, thick dark, double sampling
        (20, 20, 1.0),   # Equal medium thickness
        (20, 20, 1.5),   # Equal medium thickness, higher sampling
        (30, 10, 1.0),   # Thick light, thin dark
    ]
    
    results = []
    for light_width, dark_width, multiplier in test_configs:
        num_images = int((light_width + dark_width) * multiplier)
        result = test_parameter_combination(
            light_width=light_width,
            dark_width=dark_width,
            num_images=num_images,
            output_dir=output_dir
        )
        results.append(result)
    
    return results


def print_summary(results, test_name):
    """Print a summary of test results."""
    print("\n" + "="*70)
    print(f"{test_name} - SUMMARY")
    print("="*70)
    print(f"{'Config':<30} {'Gray File'}")
    print("-"*70)
    for r in results:
        config = f"L={r['light_width']}, D={r['dark_width']}, N={r['num_images']}"
        print(f"{config:<30} {Path(r['gray_file']).name}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Test darkfield parameters: strip thickness and sample rate"
    )
    parser.add_argument('--test', choices=['thickness', 'sample_rate', 'combined', 'all'],
                       default='all',
                       help='Which test to run (default: all)')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick test with fewer combinations')
    parser.add_argument('--output', default='output/parameter_tests',
                       help='Output directory (default: output/parameter_tests)')
    args = parser.parse_args()
    
    # Initialize matplotlib backend
    set_matplotlib_backend()
    
    print("\n" + "="*70)
    print("DARKFIELD PARAMETER TESTING")
    print("="*70)
    print(f"Output directory: {args.output}/")
    if args.quick:
        print("Mode: QUICK (fewer test cases)")
    else:
        print("Mode: FULL (comprehensive testing)")
    
    all_results = {}
    
    # Run selected tests
    if args.test in ['thickness', 'all']:
        results = run_strip_thickness_test(args.output, args.quick)
        print_summary(results, "STRIP THICKNESS TEST")
        all_results['thickness'] = results
    
    if args.test in ['sample_rate', 'all']:
        results = run_sample_rate_test(args.output, args.quick)
        print_summary(results, "SAMPLE RATE TEST")
        all_results['sample_rate'] = results
    
    if args.test in ['combined', 'all'] and not args.quick:
        results = run_combined_test(args.output)
        print_summary(results, "COMBINED TEST")
        all_results['combined'] = results
    
    # Final summary
    print("\n" + "="*70)
    print("ALL TESTS COMPLETE!")
    print("="*70)
    print(f"\nAll results saved to: {args.output}/")
    print("\nTo compare results, open the images and look for:")
    print("  - Image clarity and detail")
    print("  - Noise levels")
    print("  - Artifacts from stripe patterns")
    print("  - Edge definition")
    print("\nFilename format: TIMESTAMP_lw{light}_dw{dark}_n{samples}_[rgb|gray].jpg")
    print()


if __name__ == "__main__":
    main()

