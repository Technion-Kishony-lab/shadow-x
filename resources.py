import matplotlib
from env import BACKGROUND_SCREEN, DISPLAY_SCREEN, OVERHEAD_CAMERA_INDEX, BACKEND
from camera import get_or_create_camera

from graphics.figures import get_or_create_fullscreen_figure


def get_background_axes():
    fig, ax = get_or_create_fullscreen_figure(screen=BACKGROUND_SCREEN)
    return ax


def get_display_axes():
    fig, ax = get_or_create_fullscreen_figure(screen=DISPLAY_SCREEN)
    return ax


def get_overhead_camera():
    return get_or_create_camera(OVERHEAD_CAMERA_INDEX)


def set_matplotlib_backend():
    matplotlib.use(BACKEND)
