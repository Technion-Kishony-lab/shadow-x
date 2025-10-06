import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from env import BACKGROUND_SCREEN, DISPLAY_SCREEN, OVERHEAD_CAMERA, BACKEND, BACKGROUND_SCREEN_BIN_SIZE
from camera import get_or_create_camera, Camera

from graphics.figures import get_or_create_named_figure, get_figure_if_existing
from graphics.helpers import get_axes_size_in_pixels


def set_matplotlib_backend():
    matplotlib.use(BACKEND)


def get_overhead_camera() -> Camera:
    return get_or_create_camera(**OVERHEAD_CAMERA)


def get_backlight_axes() -> Axes:
    fig, ax = get_or_create_named_figure("background", screen=BACKGROUND_SCREEN, is_image=True, remove_toolbar=True)
    return ax


def get_camera_display_axes(index=0) -> Axes:
    fig_name = f"display{index}"
    fig, ax = get_figure_if_existing(fig_name)
    if ax is not None:
        return ax

    camera_resolution_pixels = get_overhead_camera().get_resolution()
    axes_position = np.array([0.1, 0.1, 0.8, 0.85])
    figure_position = np.array([100 + index*100, 100, camera_resolution_pixels[0] / axes_position[3],
                                camera_resolution_pixels[1] / axes_position[2]]).astype(int)
    fig, ax = get_or_create_named_figure(fig_name, screen=DISPLAY_SCREEN, figure_position=figure_position,
                                         axes_position=axes_position, is_image=True)
    return ax


def get_backlight_image_size():
    ax = get_backlight_axes()
    ax_size = get_axes_size_in_pixels(ax)
    image_size = ax_size // BACKGROUND_SCREEN_BIN_SIZE
    return image_size


def set_axes_image(ax, image: np.ndarray):
    if ax.images:
        img = ax.images[0]
        assert np.all(img.get_array().shape[:2] == image.shape[:2])
        img.set_array(image)
    else:
        img = ax.imshow(image, cmap='gray', vmin=0, vmax=255)
    ax.figure.canvas.draw()
    return img


def set_background_image(image: np.ndarray, pause=0):
    ax = get_backlight_axes()
    img = set_axes_image(ax, image)
    if pause:
        plt.pause(pause)
    return ax, img


def set_camera_display_image(image: np.ndarray, index=0, pause=0):
    ax = get_camera_display_axes(index)
    img = set_axes_image(ax, image)
    if pause:
        plt.pause(pause)
    return ax, img


def illuminate(color=(255, 255, 255), pause=1):
    size = get_backlight_image_size()
    background_image = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    background_image[:, :] = color
    return set_background_image(background_image, pause=pause)
