from typing import Optional

import numpy as np

from env_loader import BACKGROUND_SCREEN_INDEX, DISPLAY_SCREEN_INDEX, OVERHEAD_CAMERA, BACKGROUND_SCREEN_BIN_SIZE, get_active_env
from resources.camera import get_or_create_camera, Camera
from graphics.image_figure import ImageFigure

_backlight_screen: Optional[ImageFigure] = None


def get_overhead_camera() -> Camera:
    """
    Get overhead camera - uses digiCamControl if USE_DIGICAM is True in env_macroscope,
    otherwise uses OpenCV VideoCapture.
    """
    # Check if we should use digiCamControl
    try:
        env = get_active_env()
        use_digicam = getattr(env, 'USE_DIGICAM', False)
        
        if use_digicam:
            # Use digiCamControl camera adapter
            from resources.camera_digicam import get_or_create_digicam_camera
            base_url = OVERHEAD_CAMERA.get('base_url', 'http://127.0.0.1:5514')
            resolution = OVERHEAD_CAMERA.get('resolution', None)
            rotate = OVERHEAD_CAMERA.get('rotate', None)
            color_order = OVERHEAD_CAMERA.get('color_order', None)
            
            print(f"Using digiCamControl camera (base_url: {base_url})")
            return get_or_create_digicam_camera(
                base_url=base_url,
                rotate=rotate,
                color_order=color_order,
                resolution=resolution
            )
    except (AttributeError, ImportError) as e:
        # Fall back to OpenCV if digiCamControl not available
        print(f"Warning: digiCamControl not available ({e}), falling back to OpenCV")
        pass
    
    # Default: use OpenCV VideoCapture
    return get_or_create_camera(**OVERHEAD_CAMERA)


def get_or_create_backlight_screen() -> ImageFigure:
    global _backlight_screen
    if _backlight_screen is None:
        _backlight_screen = ImageFigure(screen_index=BACKGROUND_SCREEN_INDEX, bin_size=BACKGROUND_SCREEN_BIN_SIZE)
    return _backlight_screen


def create_camera_figure(index=0) -> ImageFigure:
    camera_resolution_pixels = get_overhead_camera().get_resolution()
    figure_position = np.array([100 + index * 100, 100, camera_resolution_pixels[0],
                                camera_resolution_pixels[1]]).astype(int)
    return ImageFigure(screen_index=DISPLAY_SCREEN_INDEX,
                       figure_position=figure_position,
                       axes_position="full")
