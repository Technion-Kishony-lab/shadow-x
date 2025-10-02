import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.image import AxesImage

from env import BACKGROUND_SCREEN, DISPLAY_SCREEN, OVERHEAD_CAMERA, BACKEND, BACKGROUND_SCREEN_BIN_SIZE
from camera import get_or_create_camera, Camera

from graphics.figures import get_or_create_named_figure
from graphics.patterns import get_size_in_pixels


def set_matplotlib_backend():
    matplotlib.use(BACKEND)


def get_overhead_camera() -> Camera:
    return get_or_create_camera(**OVERHEAD_CAMERA)


def get_background_axes() -> Axes:
    fig, ax = get_or_create_named_figure("background", screen=BACKGROUND_SCREEN, is_image=True, remove_toolbar=True)
    return ax


def get_camera_display_axes(index=0) -> Axes:
    camera_resolution_pixels = get_overhead_camera().get_resolution()
    axes_position = np.array([0.1, 0.1, 0.8, 0.85])
    figure_position = np.array([100 + index*100, 100, camera_resolution_pixels[0] / axes_position[3],
                                camera_resolution_pixels[1] / axes_position[2]]).astype(int)
    fig, ax = get_or_create_named_figure(f"display{index}", screen=DISPLAY_SCREEN, figure_position=figure_position,
                                         axes_position=axes_position, is_image=True)
    return ax


def get_axes_image(ax: Axes, bin_size=1):
    if not ax.images:
        ax_size = get_size_in_pixels(ax)
        if bin_size > 1:
            ax_size = ax_size // bin_size
        ax.imshow(255 * np.ones(ax_size, dtype=np.uint8), cmap='gray', vmin=0, vmax=255)
    return ax.images[0]


def get_background_image() -> AxesImage:
    ax = get_background_axes()
    return get_axes_image(ax, bin_size=BACKGROUND_SCREEN_BIN_SIZE)


def get_background_image_size() -> tuple[int, int]:
    img = get_background_image()
    return img.get_array().shape[:2]


def get_camera_display_image(index=0) -> AxesImage:
    ax = get_camera_display_axes(index)
    return get_axes_image(ax)


def set_axes_image(ax, image: np.ndarray):
    img = get_axes_image(ax)
    img.set_array(image)
    ax.figure.canvas.draw()
    # ax.figure.canvas.flush_events()


def set_background_image(image: np.ndarray, pause=0):
    ax = get_background_axes()
    set_axes_image(ax, image)
    if pause:
        plt.pause(pause)


def set_camera_display_image(image: np.ndarray, index=0):
    ax = get_camera_display_axes(index)
    set_axes_image(ax, image)


def illuminate(color=(255, 255, 255), pause=1):
    img = get_background_image()
    size = img.get_array().shape[:2]
    img.set_array(np.full((*size, 3), color, dtype=np.uint8))
    img.axes.figure.canvas.draw()
    img.axes.figure.canvas.flush_events()
    if pause:
        plt.pause(pause)
