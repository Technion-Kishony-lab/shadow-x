"""
Macroscope environment configuration.
Adapt this file to match your macroscope system's hardware setup.
"""
import cv2

# Screen configuration for macroscope
# Note: You have 2 screens available (indices 0 and 1)
BACKGROUND_SCREEN_INDEX = 1  # Screen used for backlight illumination (secondary monitor)
BACKGROUND_SCREEN_BIN_SIZE = 1  # binning factor for the background screen

DISPLAY_SCREEN_INDEX = 0  # Screen used for displaying camera images (primary monitor)

# Camera configuration for macroscope
# Set USE_DIGICAM to True to use CanonCameraController via digiCamControl
# Set to False to use OpenCV VideoCapture (webcam/USB camera)
USE_DIGICAM = True  # Use digiCamControl for Canon EOS cameras
DIGICAM_BASE_URL = "http://127.0.0.1:5514"  # digiCamControl web server URL

OVERHEAD_CAMERA = {
    'camera_index': 0,  # For OpenCV cameras only - adjust to match your camera index
    'fps': 30,
    'buffer_size': 1,
    'rotate': None,  # Adjust rotation as needed: None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180
    'color_order': 'BGR',  # or 'RGB'
    'exposure': None,  # Set to None for auto exposure, or a specific value (e.g., 20.0)
    # For digiCamControl:
    'base_url': DIGICAM_BASE_URL,  # digiCamControl web server URL
    'resolution': None,  # Optional: (width, height) if known, otherwise auto-detected
}

# Matplotlib backend for macroscope
BACKEND = 'qtagg'  # Options: 'tkagg', 'wxagg', 'qt5agg', 'qt6agg'

# Screen coordinate mappings for macroscope
# Format: {screen_index: (width, height, x_offset, y_offset)}
# If using TK backend, use get_screen_geometrys() from graphics.helpers to get the screen geometries
SCREENS_TO_COORDS = {
    # W, H, x0, y0
    # Update these values to match your actual display setup
    0: (1920, 1080, 0, 0),      # Primary display (camera view)
    1: (1920, 1080, 1920, 0),   # Secondary display (backlight screen)
}

