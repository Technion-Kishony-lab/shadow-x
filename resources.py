import matplotlib
from env import BACKGROUND_SCREEN, DISPLAY_SCREEN, OVERHEAD_CAMERA, BACKEND
from camera import get_or_create_camera, Camera

from graphics.figures import get_or_create_fullscreen_figure


def get_background_axes():
    fig, ax = get_or_create_fullscreen_figure(screen=BACKGROUND_SCREEN)
    return ax


def get_display_axes():
    fig, ax = get_or_create_fullscreen_figure(screen=DISPLAY_SCREEN)
    return ax


def get_overhead_camera() -> Camera:
    return get_or_create_camera(**OVERHEAD_CAMERA)


def set_matplotlib_backend():
    matplotlib.use(BACKEND)
