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
from resources import get_background_axes, get_overhead_camera, set_matplotlib_backend, get_display_axes
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


def _get_mapping(size, n_squares, detected_vertices):
    screen_vertices = np.array([
        [i, j]
        for j in range(1, n_squares[1])
        for i in range(1, n_squares[0])
    ], dtype=np.float32) * size

    camera_vertices = detected_vertices.reshape(-1, 2).astype(np.float32)

    H, mask = cv2.findHomography(camera_vertices, screen_vertices, cv2.RANSAC)
    if H is None:
        raise ValueError("Could not find homography matrix")
    return H


def _map_from_camera_to_screen(mapping, camera_points):
    camera_points_homogeneous = np.hstack([camera_points, np.ones((camera_points.shape[0], 1))])
    screen_points_homogeneous = camera_points_homogeneous @ mapping.T
    screen_points = screen_points_homogeneous[:, :2] / screen_points_homogeneous[:, 2:3]
    return screen_points


def wait_for_keypress(fig, options=('y', 'n')) -> str:
    key_pressed = None

    def on_key_press(event):
        nonlocal key_pressed
        key_pressed = event.key.lower()

    fig.canvas.mpl_connect('key_press_event', on_key_press)

    while key_pressed not in options:
        plt.pause(0.1)
    return key_pressed


def register_screen_camera(size=100, display=True):
    """
    Register the screen positions on the camera using a chessboard pattern.
    Parameters
    ----------
    size : int
        Size of each square in the chessboard pattern in pixels.
    display : bool or None
        If True, display the mapping and ask the user to verify it.
        If False, do not display the mapping, unless fail to detect the pattern.
    """

    camera = get_overhead_camera()
    ax_bg = get_background_axes()

    chessboard, n_squares = show_chessboard_pattern(ax_bg, square_size=size)
    pattern_size = n_squares - 1  # number of inner corners
    plt.pause(1)  # give some time to display the pattern

    mapping = None
    while True:
        frame = camera.take_picture()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, vertices = _find_chessboard_vertices(gray, pattern_size=pattern_size)

        if ret:
            mapping = _get_mapping(
                size=size,
                n_squares=n_squares,
                detected_vertices=vertices
            )
            if display is False:
                break

        disp_ax = get_display_axes()
        if not ret:
            disp_ax.imshow(gray, cmap='gray')
            disp_ax.set_title("Chessboard Corners NOT Detected. Press 'y' to nreak, 'n' to adjust.")
        else:
            # map the chessboard corners to the ax_bg coordinates:
            cv2.drawChessboardCorners(frame, pattern_size, vertices, ret)
            disp_ax.imshow(frame, cmap='gray')
            disp_ax.set_title("Chessboard Corners Detected. Press 'y' to confirm, 'n' to adjust.")

            screen_points = _map_from_camera_to_screen(mapping=mapping, camera_points=vertices.reshape(-1, 2))

            # plot the mapped points on the screen chessboard:
            ax_bg.plot(screen_points[:, 1], screen_points[:, 0], 'rx', markersize=7)
            ax_bg.figure.canvas.draw()
            ax_bg.figure.canvas.flush_events()
            disp_ax.figure.canvas.draw()
            disp_ax.figure.canvas.flush_events()
        key = wait_for_keypress(disp_ax.figure)
        if key == 'y':
            break
    return mapping


if __name__ == "__main__":
    register_screen_camera(size=200)
