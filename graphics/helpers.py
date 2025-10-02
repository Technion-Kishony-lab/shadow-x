import numpy as np


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
