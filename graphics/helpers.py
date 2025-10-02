import numpy as np


def subtract_images(image1, image2, as_gray=False, as_uint8=False):
    diff = image1.astype(np.int32) - image2.astype(np.int32)
    if as_gray:
        diff = diff.mean(axis=2)
    if as_uint8:
        diff = np.clip(diff, 0, 255).astype(np.uint8)
    return diff
