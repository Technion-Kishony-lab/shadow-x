from typing import Optional

import matplotlib
import numpy as np

from env import BACKGROUND_SCREEN_INDEX, DISPLAY_SCREEN_INDEX, OVERHEAD_CAMERA, BACKEND, BACKGROUND_SCREEN_BIN_SIZE
from camera import get_or_create_camera, Camera
from graphics.image_figure import ImageFigure

_backlight_screen: Optional[ImageFigure] = None


def set_matplotlib_backend():
    matplotlib.use(BACKEND)


def get_overhead_camera() -> Camera:
    return get_or_create_camera(**OVERHEAD_CAMERA)


def get_or_create_backlight_screen() -> ImageFigure:
    global _backlight_screen
    if _backlight_screen is None:
        _backlight_screen = ImageFigure(screen_index=BACKGROUND_SCREEN_INDEX, bin_size=BACKGROUND_SCREEN_BIN_SIZE)
    return _backlight_screen


def create_camera_figure(index=0) -> ImageFigure:
    camera_resolution_pixels = get_overhead_camera().get_resolution()
    axes_position = np.array([0, 0, 1, 0.9])
    figure_position = np.array([100 + index * 100, 100, camera_resolution_pixels[0] / axes_position[3],
                                camera_resolution_pixels[1] / axes_position[2]]).astype(int)
    return ImageFigure(screen_index=DISPLAY_SCREEN_INDEX,
                       figure_position=figure_position,
                       axes_position=axes_position)
