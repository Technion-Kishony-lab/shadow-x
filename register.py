"""
Register screen positions on the camera.

Our setup:
Screen lays flat on the table.
Camera is placed above the screen.
Both screen and camera are fixed.

We need a mapping from screen coordinates to camera coordinates.

We will use a chessboard pattern to register the screen positions on the camera.

"""

import cv2

from graphics.patterns import show_chessboard_pattern
from resources import get_background_axes, get_overhead_camera, set_matplotlib_backend
from matplotlib import pyplot as plt
from camera import take_picture

set_matplotlib_backend()


def _find_corners(gray, pattern_size):
    glag_sets = [
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FILTER_QUADS,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    ]
    for flags in glag_sets:
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
        if ret:
            return ret, corners
    return False, None


def register_screen_camera(size = 100):

    camera = get_overhead_camera()
    ax_bg = get_background_axes()

    chessboard, n_squares = show_chessboard_pattern(ax_bg, square_size=size)
    pattern_size = n_squares - 1  # number of inner corners
    plt.pause(1)  # give some time to display the pattern

    frame = take_picture(cap=camera)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, corners = _find_corners(gray, pattern_size=pattern_size)

    fig, ax = plt.subplots()
    ax.imshow(gray, cmap='gray')
    if ret:
        cv2.drawChessboardCorners(frame, pattern_size, corners, ret)
        plt.title("Chessboard Corners Detected")
    else:
        plt.title("Chessboard Corners NOT Detected")


if __name__ == "__main__":
    register_screen_camera(size=100)
    plt.show()
