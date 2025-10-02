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
import numpy as np

from env import BACKGROUND_SCREEN
from graphics.patterns import show_chessboard_pattern
from helpers import get_background_axes, get_display_axes

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from camera import get_camera, take_picture, get_or_create_camera

    size = 500
    from matplotlib import use

    use('qtagg')


    CAP = get_or_create_camera(0)

    # set a figure on the entire screen:
    ax_bg = get_background_axes()
    chessboard = show_chessboard_pattern(ax_bg, square_size=size)

    plt.pause(1)  # give some time to display the pattern

    frame = take_picture(cap=CAP)
    plt.figure()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    plt.imshow(gray, cmap='gray')
    # plt.show()

    # Find the chess board corners
    pattern_size = (chessboard.shape[1]//size - 1, chessboard.shape[0]//size - 1)  # number of inner corners per a chessboard row and column
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

    if ret:
        # If found, draw corners
        cv2.drawChessboardCorners(frame, pattern_size, corners, ret)
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.title("Detected Chessboard Corners")
        plt.show()

        # Save the corner positions for further processing
        np.save("chessboard_corners.npy", corners)
        print("Chessboard corners saved to chessboard_corners.npy")
    else:
        print("Chessboard not detected. Please adjust the camera or pattern and try again.")