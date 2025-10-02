import matplotlib
from matplotlib.axes import Axes

from env import BACKGROUND_SCREEN, DISPLAY_SCREEN, OVERHEAD_CAMERA, BACKEND
from camera import get_or_create_camera, Camera

from graphics.figures import get_or_create_named_figure


def get_background_axes() -> Axes:
    fig, ax = get_or_create_named_figure("background", screen=BACKGROUND_SCREEN, is_image=True)
    return ax


def get_display_axes() -> Axes:
    fig, ax = get_or_create_named_figure("display", screen=DISPLAY_SCREEN, figure_position=[200, 200, 1000, 800],
                                         axes_position=[0.1, 0.1, 0.8, 0.85], is_image=True)
    return ax


def get_overhead_camera() -> Camera:
    return get_or_create_camera(**OVERHEAD_CAMERA)


def set_matplotlib_backend():
    matplotlib.use(BACKEND)
