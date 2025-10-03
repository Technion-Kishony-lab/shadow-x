import numpy as np
from matplotlib import pyplot as plt


def subtract_images(image1, image2, as_gray=False, as_uint8=False):
    diff = image1.astype(np.int32) - image2.astype(np.int32)
    if as_gray:
        diff = diff.mean(axis=2)
    if as_uint8:
        diff = np.clip(diff, 0, 255).astype(np.uint8)
    return diff


def get_axes_size_in_pixels(ax):
    bbox = ax.get_window_extent().transformed(ax.figure.dpi_scale_trans.inverted())
    width_in_pixels = bbox.width * ax.figure.dpi
    height_in_pixels = bbox.height * ax.figure.dpi
    return np.array([height_in_pixels, width_in_pixels], dtype=int)


def wait_for_keypress(fig, options=('y', 'n')) -> str:
    key_pressed = None

    def on_key_press(event):
        nonlocal key_pressed
        key_pressed = event.key.lower()

    fig.canvas.mpl_connect('key_press_event', on_key_press)

    while key_pressed not in options:
        plt.pause(0.1)
    return key_pressed
