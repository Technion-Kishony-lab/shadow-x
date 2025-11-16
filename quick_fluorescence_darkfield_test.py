#!/usr/bin/env python
"""
Quick fluorescence darkfield test - colored stripes instead of white.

This combines:
- Darkfield technique (moving stripes, average in dark phase)
- Fluorescence excitation (colored stripes instead of white)

Result: Enhanced edge/particle detection with fluorescence!

REQUIREMENTS:
- Emission filter on camera (CRITICAL)
- Fluorescent sample
- Dark environment

COMMON STRIPE COLORS:
- GFP: (0, 0, 255) - Blue
- RFP: (0, 255, 0) - Green

Usage:
    1. Edit STRIPE_COLOR below
    2. Run: python quick_fluorescence_darkfield_test.py
    3. Wait ~30 seconds for capture + processing
"""
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
from matplotlib import pyplot as plt

from dark_field import StripesAnalyzer
from runners import CameraScreenRunner, timed
from graphics.helpers import set_matplotlib_backend
from graphics.patterns import get_stripes_image

# ============ CONFIGURE HERE ============
STRIPE_COLOR = (0, 0, 255)  # Blue for GFP - change as needed
LIGHT_WIDTH = 14  # Width of colored stripes
DARK_WIDTH = 17   # Width of dark stripes
NUM_IMAGES_TO_DISCARD = 11  # Initial frames to skip
OUTPUT_DIR = "output"
# ========================================


class ColoredStripeRunner(CameraScreenRunner):
    """
    Stripe runner with colored stripes instead of white.
    For fluorescence darkfield imaging.
    """
    BACKGROUND_COLOR = (0, 0, 0)
    ILLUMINATION_PAUSE = 0.01

    def __init__(self, stripes_color=(255, 255, 255), camera=None, 
                 backlight_screen=None, show_camera=True,
                 light_width=10, dark_width=40, num_images=None, 
                 num_images_to_discard=0):
        
        self.stripes_color = stripes_color
        num_images = num_images if num_images is not None else light_width + dark_width
        iterations = num_images + num_images_to_discard
        
        super().__init__(iterations, False, camera, backlight_screen,
                         show_camera, use_blitting=True, refresh_together=False)
        
        self.num_images_to_discard = num_images_to_discard
        self.num_images = num_images
        self.light_width = light_width
        self.dark_width = dark_width

    def _setup(self):
        super()._setup()
        self.images = []

    @timed
    def _illuminate_pattern(self, i):
        pattern = self._generate_illumination_pattern(i)
        self.update_backlight_image(pattern)
        plt.pause(self.ILLUMINATION_PAUSE)

    def _do_iteration(self, i):
        i = i - self.num_images_to_discard
        self._illuminate_pattern(i)
        image = self.take_picture()
        self.maybe_show_camera(image)
        if i >= 0:
            assert len(self.images) == i
            self.images.append(image)

    @timed
    def _generate_illumination_pattern(self, frame_index):
        phase = frame_index / self.num_images
        size = self.backlight_screen.get_recomended_image_size()
        return get_stripes_image(
            light_width=self.light_width, 
            dark_width=self.dark_width, 
            phase=phase, 
            size=size,
            stripes_color=self.stripes_color,
            background_color=self.BACKGROUND_COLOR
        )

    def take_uniform_image(self, color):
        self.backlight_screen.illuminate(color=color, pause=0.5)
        image = self.take_picture()
        self.maybe_show_camera(image)
        return image

    def _after_run(self):
        self.print_timers_report()
        self.dark_image = self.take_uniform_image(color=self.BACKGROUND_COLOR)
        self.bright_image = self.take_uniform_image(color=self.stripes_color)
        super()._after_run()

def quick_test():
    """Quick fluorescence darkfield test"""
    print("\n" + "="*60)
    print("QUICK FLUORESCENCE DARKFIELD TEST")
    print("="*60)
    print(f"Stripe color (excitation): RGB{STRIPE_COLOR}")
    print(f"Light width: {LIGHT_WIDTH}, Dark width: {DARK_WIDTH}")
    print("="*60 + "\n")
    
    set_matplotlib_backend()
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    
    # Step 1: Capture with colored stripes
    print("1. Capturing images with COLORED moving stripes...")
    runner = ColoredStripeRunner(
        stripes_color=STRIPE_COLOR,
        light_width=LIGHT_WIDTH,
        dark_width=DARK_WIDTH,
        num_images_to_discard=NUM_IMAGES_TO_DISCARD,
        show_camera=True
    )
    runner.run()
    print(f"   ✓ Captured {len(runner.images)} images\n")
    
    # Step 2: Process into darkfield
    print("2. Processing into fluorescence darkfield...")
    analyzer = StripesAnalyzer.from_stripe_runner(runner)
    rgb, gray = analyzer.get_pseudo_darkfield_image(
        dark_dist=0.4,
        max_dark_dist=1.0,
        smoothing=1,
        substract_background=False,
        normalize=False
    )
    print("   ✓ Processing complete\n")
    
    # Step 3: Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rgb_file = Path(OUTPUT_DIR) / f"{timestamp}_quick_fluor_df_rgb.jpg"
    gray_file = Path(OUTPUT_DIR) / f"{timestamp}_quick_fluor_df_gray.jpg"
    
    cv2.imwrite(str(rgb_file), rgb)
    cv2.imwrite(str(gray_file), gray)
    
    print(f"3. Saved results:")
    print(f"   ✓ RGB: {rgb_file}")
    print(f"   ✓ Gray: {gray_file}\n")
    
    # Quick stats
    signal = rgb.astype(np.float32) - runner.dark_image.astype(np.float32)
    mean_signal = np.mean(signal)
    max_signal = np.max(signal)
    
    print("Quick Stats:")
    print(f"  Mean signal: {mean_signal:.1f}")
    print(f"  Max signal: {max_signal:.1f}")
    
    if mean_signal < 1.0:
        print("\n⚠ WARNING: Very weak signal!")
        print("  - Check emission filter is installed")
        print("  - Ensure sample has fluorophores")
        print("  - Darken the room")
    elif mean_signal > 50:
        print("\n✓ Good signal detected!")
    
    print("\n" + "="*60)
    return rgb, gray

if __name__ == "__main__":
    quick_test()

