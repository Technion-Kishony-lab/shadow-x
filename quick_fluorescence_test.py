#!/usr/bin/env python
"""
Quick fluorescence test - minimal script for rapid experimentation.

REQUIREMENTS:
- Emission filter on camera (CRITICAL - blocks excitation, passes fluorescence)
- Fluorescent sample (GFP, RFP, dyes, etc.)
- Dark environment

COMMON EXCITATION COLORS:
- GFP/FITC: (0, 0, 255) - Blue
- RFP/mCherry: (0, 255, 0) - Green
- CFP: (200, 0, 255) - Violet
- YFP: (0, 100, 255) - Blue

Usage:
    1. Edit EXCITATION_RGB below
    2. Run: python quick_fluorescence_test.py
    3. If signal is weak, increase PAUSE_TIME or NUM_FRAMES_TO_AVERAGE
"""
import cv2
from datetime import datetime
from pathlib import Path
from resources.camera_and_screens import get_overhead_camera, get_or_create_backlight_screen

# ============ CONFIGURE HERE ============
EXCITATION_RGB = (0, 0, 255)  # Blue for GFP/FITC - adjust as needed
PAUSE_TIME = 1.0  # seconds to wait before capture
NUM_FRAMES_TO_AVERAGE = 3  # average multiple frames to reduce noise
OUTPUT_DIR = "output"
# ========================================

def quick_test():
    """Quick fluorescence capture test"""
    print("\n" + "="*60)
    print("QUICK FLUORESCENCE TEST")
    print("="*60)
    print(f"Excitation RGB: {EXCITATION_RGB}")
    print(f"Pause time: {PAUSE_TIME}s")
    print(f"Frame averaging: {NUM_FRAMES_TO_AVERAGE}")
    print("="*60 + "\n")
    
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    
    camera = get_overhead_camera()
    backlight = get_or_create_backlight_screen()
    
    # Capture with excitation
    print("1. Setting excitation light...")
    backlight.illuminate(color=EXCITATION_RGB, pause=PAUSE_TIME)
    
    print("2. Capturing fluorescence image...")
    if NUM_FRAMES_TO_AVERAGE == 1:
        fluor_frame = camera.take_picture()
    else:
        import numpy as np
        import time
        frames = []
        for i in range(NUM_FRAMES_TO_AVERAGE):
            frames.append(camera.take_picture().astype(np.float32))
            time.sleep(0.1)
        fluor_frame = np.mean(frames, axis=0).astype(np.uint8)
        print(f"   (averaged {NUM_FRAMES_TO_AVERAGE} frames)")
    
    # Capture dark reference
    print("3. Capturing dark reference...")
    backlight.illuminate(color=(0, 0, 0), pause=0.3)
    dark_frame = camera.take_picture()
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fluor_file = Path(OUTPUT_DIR) / f"{timestamp}_quick_fluor.jpg"
    dark_file = Path(OUTPUT_DIR) / f"{timestamp}_quick_dark.jpg"
    
    cv2.imwrite(str(fluor_file), fluor_frame)
    cv2.imwrite(str(dark_file), dark_frame)
    
    print(f"\n✓ Saved fluorescence: {fluor_file}")
    print(f"✓ Saved dark reference: {dark_file}")
    
    # Quick analysis
    import numpy as np
    signal = fluor_frame.astype(np.float32) - dark_frame.astype(np.float32)
    mean_signal = np.mean(signal)
    max_signal = np.max(signal)
    
    print(f"\nQuick Stats:")
    print(f"  Mean signal: {mean_signal:.1f}")
    print(f"  Max signal: {max_signal:.1f}")
    print(f"  Signal range: [{signal.min():.1f}, {signal.max():.1f}]")
    
    if mean_signal < 1.0:
        print("\n⚠ WARNING: Very weak signal detected!")
        print("  Suggestions:")
        print("  - Increase PAUSE_TIME")
        print("  - Increase NUM_FRAMES_TO_AVERAGE")
        print("  - Check that emission filter is installed")
        print("  - Ensure sample has fluorophores")
        print("  - Darken the room")
    elif mean_signal > 50:
        print("\n✓ Good signal detected!")
    else:
        print("\n⚠ Weak but detectable signal")
    
    print("\n" + "="*60)
    return fluor_frame, dark_frame

if __name__ == "__main__":
    quick_test()

