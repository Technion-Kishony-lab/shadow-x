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

from graphics.patterns import show_chessboard_pattern
from resources import get_background_axes, get_overhead_camera, set_matplotlib_backend
from matplotlib import pyplot as plt

set_matplotlib_backend()


def _find_chessboard_vertices(gray, pattern_size):
    # Try different flag combinations - removed FAST_CHECK as it's strict with distortion
    flag_sets = [
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FILTER_QUADS,
        cv2.CALIB_CB_ADAPTIVE_THRESH,
        cv2.CALIB_CB_NORMALIZE_IMAGE,
        0  # No flags
    ]
    
    # Try different preprocessing approaches
    preprocessed_images = [
        ("original", gray),
        ("blurred", cv2.GaussianBlur(gray, (5, 5), 0)),
        ("equalized", cv2.equalizeHist(gray)),
        ("blurred+equalized", cv2.equalizeHist(cv2.GaussianBlur(gray, (5, 5), 0))),
        ("binary", cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2))
    ]

    for preproc_name, img in preprocessed_images:
        for flags in flag_sets:
            ret, vertices = cv2.findChessboardCorners(img, pattern_size, flags)
            if ret:
                return ret, vertices
    
    print("✗ Detection failed with all approaches")
    return None, None


def _map_from_camera_to_screen(size, n_squares, detected_vertices, camera_points):
    """
    Map points from camera coordinates to screen coordinates.

    camera_points: array of shape (N, 2)
    """
    screen_vertices = np.array([
        [i, j]
        for j in range(1, n_squares[1])
        for i in range(1, n_squares[0])
    ], dtype=np.float32) * size

    camera_vertices = detected_vertices.reshape(-1, 2).astype(np.float32)
    assert camera_vertices.shape[0] == screen_vertices.shape[0]
    assert camera_vertices.shape[1] == 2
    assert camera_points.shape[1] == 2

    # Find the homography matrix
    H, mask = cv2.findHomography(camera_vertices, screen_vertices, cv2.RANSAC)
    if H is None:
        raise ValueError("Could not find homography matrix")

    # Map the camera points to screen points
    camera_points_homogeneous = np.hstack([camera_points, np.ones((camera_points.shape[0], 1))])
    screen_points_homogeneous = camera_points_homogeneous @ H.T
    screen_points = screen_points_homogeneous[:, :2] / screen_points_homogeneous[:, 2:3]
    return screen_points


def register_screen_camera(size = 100):

    camera = get_overhead_camera()
    ax_bg = get_background_axes()

    chessboard, n_squares = show_chessboard_pattern(ax_bg, square_size=size)
    pattern_size = n_squares - 1  # number of inner corners
    print(f"pattern_size: {pattern_size}")
    plt.pause(1)  # give some time to display the pattern

    frame = camera.take_picture()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, vertices = _find_chessboard_vertices(gray, pattern_size=pattern_size)

    fig, ax = plt.subplots()
    if vertices is None:
        ax.imshow(gray, cmap='gray')
        plt.title("Chessboard Corners NOT Detected")
    else:
        cv2.drawChessboardCorners(frame, pattern_size, vertices, ret)
        ax.imshow(frame, cmap='gray')
        plt.title("Chessboard Corners Detected")

        # map the chessboard corners to the ax_bg coordinates:
        screen_points = _map_from_camera_to_screen(
            size=size,
            n_squares=n_squares,
            detected_vertices=vertices,
            camera_points=vertices.reshape(-1, 2)
        )

        # plot the mapped points on the screen chessboard:
        ax_bg.plot(screen_points[:, 1], screen_points[:, 0], 'rx', markersize=7)
        ax_bg.figure.canvas.draw()
        ax_bg.figure.canvas.flush_events()
        plt.pause(0.1)


if __name__ == "__main__":
    register_screen_camera(size=200)
    plt.show()
