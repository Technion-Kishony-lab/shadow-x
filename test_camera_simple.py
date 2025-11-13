#!/usr/bin/env python
"""
Simple camera test - captures a single frame and saves it.
This tests camera connection without requiring screen setup.
"""
import sys
import os
from datetime import datetime

print("=" * 60)
print("Simple Camera Test")
print("=" * 60)

# Test 1: Check environment
print("\n1. Checking environment configuration...")
try:
    from env_loader import get_environment_name, get_active_env
    env_name = get_environment_name()
    print(f"   Environment: {env_name}")
    
    env = get_active_env()
    use_digicam = getattr(env, 'USE_DIGICAM', False)
    print(f"   USE_DIGICAM: {use_digicam}")
    
    if use_digicam:
        print(f"   DigiCam URL: {env.DIGICAM_BASE_URL}")
except Exception as e:
    print(f"   [FAIL] Error loading environment: {e}")
    sys.exit(1)

# Test 2: Import camera modules
print("\n2. Importing camera modules...")
try:
    from resources.camera_and_screens import get_overhead_camera
    print("   [OK] Camera modules imported")
except Exception as e:
    print(f"   [FAIL] Failed to import: {e}")
    sys.exit(1)

# Test 3: Connect to camera
print("\n3. Connecting to camera...")
try:
    camera = get_overhead_camera()
    print(f"   [OK] Camera object created: {type(camera).__name__}")
except Exception as e:
    print(f"   [FAIL] Failed to get camera: {e}")
    sys.exit(1)

# Test 4: Get camera resolution
print("\n4. Getting camera resolution...")
try:
    resolution = camera.get_resolution()
    print(f"   [OK] Resolution: {resolution[0]}x{resolution[1]}")
except Exception as e:
    print(f"   [FAIL] Failed to get resolution: {e}")
    sys.exit(1)

# Test 5: Capture a frame
print("\n5. Capturing a test frame...")
try:
    frame = camera.take_picture()
    print(f"   [OK] Frame captured!")
    print(f"   Frame shape: {frame.shape}")
    print(f"   Frame dtype: {frame.dtype}")
    print(f"   Frame min/max: {frame.min()}/{frame.max()}")
except Exception as e:
    print(f"   [FAIL] Failed to capture frame: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Save the frame
print("\n6. Saving test frame...")
try:
    import cv2
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"test_frame_{timestamp}.jpg"
    cv2.imwrite(filename, frame)
    print(f"   [OK] Frame saved to: {filename}")
except Exception as e:
    print(f"   [FAIL] Failed to save frame: {e}")
    sys.exit(1)

# Test 7: Release camera
print("\n7. Releasing camera...")
try:
    if hasattr(camera, 'release'):
        camera.release()
    print("   [OK] Camera released")
except Exception as e:
    print(f"   [WARN] Error releasing camera: {e}")

print("\n" + "=" * 60)
print("[SUCCESS] Camera test completed!")
print("=" * 60)
print(f"\nTest frame saved to: {filename}")
print("If the frame looks correct, your camera setup is working!")

