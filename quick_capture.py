#!/usr/bin/env python
"""
Quick capture script - take single pictures with different backlight settings.

Usage:
    python quick_capture.py                    # Interactive mode
    python quick_capture.py --white            # Capture with white backlight
    python quick_capture.py --black            # Capture with black backlight
    python quick_capture.py --color 255 0 0    # Capture with custom RGB color
"""
import argparse
import cv2
from datetime import datetime
from pathlib import Path

from resources.camera_and_screens import get_overhead_camera, get_or_create_backlight_screen


def quick_capture(backlight_color=(255, 255, 255), pause=0.5, output_dir="output", suffix=""):
    """
    Quickly capture a single image with specified backlight.
    
    Args:
        backlight_color: RGB tuple for backlight color
        pause: Seconds to wait after setting backlight
        output_dir: Directory to save image
        suffix: Optional suffix for filename
        
    Returns:
        tuple: (frame, filename)
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    camera = get_overhead_camera()
    backlight_screen = get_or_create_backlight_screen()
    
    # Set backlight
    print(f"Setting backlight to RGB{backlight_color}...")
    backlight_screen.illuminate(color=backlight_color, pause=pause)
    
    # Capture
    print("Capturing...")
    frame = camera.take_picture()
    
    # Generate filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    color_str = f"RGB{'_'.join(map(str, backlight_color))}"
    if suffix:
        filename = Path(output_dir) / f"capture_{timestamp}_{suffix}.jpg"
    else:
        filename = Path(output_dir) / f"capture_{timestamp}_{color_str}.jpg"
    
    # Save
    cv2.imwrite(str(filename), frame)
    print(f"✓ Saved: {filename}")
    
    # Turn off backlight
    backlight_screen.illuminate(color=(0, 0, 0), pause=0.1)
    
    return frame, str(filename)


def interactive_mode():
    """Interactive capture mode"""
    print("=" * 60)
    print("Interactive Quick Capture")
    print("=" * 60)
    print("\nCommands:")
    print("  w - White backlight")
    print("  b - Black backlight")
    print("  r - Red backlight")
    print("  g - Green backlight")
    print("  rgb R G B - Custom RGB color (e.g., 'rgb 128 64 255')")
    print("  q - Quit")
    print()
    
    camera = get_overhead_camera()
    backlight_screen = get_or_create_backlight_screen()
    
    while True:
        cmd = input("Command: ").strip().lower()
        
        if cmd == 'q':
            print("Exiting...")
            backlight_screen.illuminate(color=(0, 0, 0), pause=0.1)
            break
        elif cmd == 'w':
            quick_capture((255, 255, 255), suffix="white")
        elif cmd == 'b':
            quick_capture((0, 0, 0), suffix="black")
        elif cmd == 'r':
            quick_capture((0, 0, 255), suffix="red")  # Note: BGR in OpenCV
        elif cmd == 'g':
            quick_capture((0, 255, 0), suffix="green")
        elif cmd.startswith('rgb'):
            try:
                parts = cmd.split()
                r, g, b = int(parts[1]), int(parts[2]), int(parts[3])
                # Convert RGB to BGR for OpenCV
                quick_capture((b, g, r), suffix=f"RGB{r}_{g}_{b}")
            except (IndexError, ValueError):
                print("Error: Use format 'rgb R G B' with values 0-255")
        else:
            print(f"Unknown command: {cmd}")


def main():
    parser = argparse.ArgumentParser(description="Quick image capture with backlight control")
    parser.add_argument('--white', action='store_true', help='Capture with white backlight')
    parser.add_argument('--black', action='store_true', help='Capture with black backlight')
    parser.add_argument('--color', nargs=3, type=int, metavar=('R', 'G', 'B'),
                       help='Capture with custom RGB color (0-255)')
    parser.add_argument('--output', default='output', help='Output directory (default: output)')
    parser.add_argument('--pause', type=float, default=0.5,
                       help='Seconds to wait after setting backlight (default: 0.5)')
    
    args = parser.parse_args()
    
    if args.white:
        quick_capture((255, 255, 255), pause=args.pause, output_dir=args.output, suffix="white")
    elif args.black:
        quick_capture((0, 0, 0), pause=args.pause, output_dir=args.output, suffix="black")
    elif args.color:
        r, g, b = args.color
        # Convert RGB to BGR for OpenCV
        quick_capture((b, g, r), pause=args.pause, output_dir=args.output, 
                     suffix=f"RGB{r}_{g}_{b}")
    else:
        # No arguments - run interactive mode
        interactive_mode()


if __name__ == "__main__":
    main()

