#!/usr/bin/env python
"""
Simple test script to verify the shadow-x setup is working correctly.
This tests imports and configuration without requiring actual hardware.
"""
import sys
import os

print("=" * 60)
print("Testing shadow-x / macroscope setup")
print("=" * 60)
print()

# Test 1: Environment loader
print("1. Testing environment loader...")
try:
    from env_loader import get_environment_name, BACKGROUND_SCREEN_INDEX, DISPLAY_SCREEN_INDEX, OVERHEAD_CAMERA, BACKEND
    env_name = get_environment_name()
    print(f"   [OK] Environment: {env_name}")
    print(f"   [OK] Background screen index: {BACKGROUND_SCREEN_INDEX}")
    print(f"   [OK] Display screen index: {DISPLAY_SCREEN_INDEX}")
    print(f"   [OK] Camera config: index={OVERHEAD_CAMERA['camera_index']}, fps={OVERHEAD_CAMERA['fps']}")
    print(f"   [OK] Backend: {BACKEND}")
except Exception as e:
    print(f"   [FAIL] Failed: {e}")
    sys.exit(1)

# Test 2: Core modules
print("\n2. Testing core module imports...")
try:
    from resources.camera import Camera
    from resources.camera_and_screens import get_overhead_camera, get_or_create_backlight_screen
    from graphics.helpers import set_matplotlib_backend, subtract_images
    from graphics.figures import SCREENS_TO_COORDS
    from runners import Runner, CameraScreenRunner, MappingRunner
    print("   [OK] All core modules imported successfully")
except Exception as e:
    print(f"   [FAIL] Failed: {e}")
    sys.exit(1)

# Test 3: Application modules
print("\n3. Testing application modules...")
try:
    from shadow import ShadowRunner
    from dark_field import StripeRunner, StripesAnalyzer
    print("   [OK] Application modules imported successfully")
except Exception as e:
    print(f"   [FAIL] Failed: {e}")
    sys.exit(1)

# Test 4: Screen configuration
print("\n4. Testing screen configuration...")
try:
    print(f"   [OK] Screens configured: {list(SCREENS_TO_COORDS.keys())}")
    for screen_idx, (w, h, x, y) in SCREENS_TO_COORDS.items():
        print(f"     Screen {screen_idx}: {w}x{h} at ({x}, {y})")
except Exception as e:
    print(f"   [FAIL] Failed: {e}")
    sys.exit(1)

# Test 5: Matplotlib backend
print("\n5. Testing matplotlib backend...")
try:
    set_matplotlib_backend()
    import matplotlib
    backend = matplotlib.get_backend()
    print(f"   [OK] Matplotlib backend: {backend}")
except Exception as e:
    print(f"   [FAIL] Failed: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("[OK] All tests passed! Setup is working correctly.")
print("=" * 60)
print("\nNote: This test doesn't verify camera/screen hardware access.")
print("To test with hardware, run: python shadow.py or python dark_field.py")

