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

from graphics.helpers import subtract_images
from graphics.patterns import get_array_of_circles_image
from resources import get_background_axes, get_overhead_camera, set_matplotlib_backend, get_camera_display_axes, \
    set_background_image, illuminate, get_background_image_size
from matplotlib import pyplot as plt

set_matplotlib_backend()


def _get_mapping(xs, ys, detected_circles):
    screen_vertices = np.array(
        [
            [x, y]
            for x in xs
            for y in ys
        ]
    )
    camera_vertices = detected_circles.astype(np.float32)

    H, mask = cv2.findHomography(camera_vertices, screen_vertices, cv2.RANSAC)
    if H is None:
        raise ValueError("Could not find homography matrix")
    return H


def _map_from_camera_points_to_screen_points(mapping, camera_points):
    camera_points = camera_points.reshape(-1, camera_points.shape[-1])
    camera_points_homogeneous = np.hstack([camera_points, np.ones((camera_points.shape[0], 1))])
    screen_points_homogeneous = camera_points_homogeneous @ mapping.T
    screen_points = screen_points_homogeneous[:, :2] / screen_points_homogeneous[:, 2:3]
    return screen_points


def map_camera_image_to_screen_image(mapping, camera_image, output_size):
    screen_image = cv2.warpPerspective(camera_image, mapping, output_size)
    return screen_image


def wait_for_keypress(fig, options=('y', 'n')) -> str:
    key_pressed = None

    def on_key_press(event):
        nonlocal key_pressed
        key_pressed = event.key.lower()

    fig.canvas.mpl_connect('key_press_event', on_key_press)

    while key_pressed not in options:
        plt.pause(0.1)
    return key_pressed


def register_screen_camera(num_tile_rows=10, display=True):
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

    image_with_circles, xs, ys = get_array_of_circles_image(size=get_background_image_size(), num_rows=num_tile_rows)

    mapping = None
    while True:
        illuminate(color=(0, 0, 0), pause=0.1)
        image0 = camera.take_picture()
        set_background_image(image_with_circles, pause=0.1)
        image1 = camera.take_picture()

        diff_image = 255 - subtract_images(image1, image0, as_gray=True, as_uint8=True)
        # save:
        cv2.imwrite("diff_image.png", diff_image)
        ret, centers = cv2.findCirclesGrid(diff_image, (len(ys), len(xs)), cv2.CALIB_CB_SYMMETRIC_GRID)

        if ret:
            mapping = _get_mapping(xs, ys, centers)
            if display is False:
                break

        disp_ax = get_camera_display_axes()
        disp_ax.imshow(diff_image, cmap='gray')
        if not ret:
            disp_ax.set_title("Pattern NOT detected. Press 'y' to break, or adjust setup and press 'n' to try again.")
        else:
            # map the chessboard corners to the ax_bg coordinates:
            disp_ax.plot(centers[:, 0, 0], centers[:, 0, 1], 'rx', markersize=7)
            disp_ax.set_title("Pattern detected. Press 'y' to confirm, or adjust setup and press 'n' to try again.")

            screen_points = _map_from_camera_points_to_screen_points(mapping=mapping,
                                                                     camera_points=centers)

            # plot the mapped points on the screen chessboard:
            ax_bg.plot(screen_points[:, 0], screen_points[:, 1], 'rx', markersize=7)
            ax_bg.figure.canvas.draw()
            ax_bg.figure.canvas.flush_events()
            disp_ax.figure.canvas.draw()
            disp_ax.figure.canvas.flush_events()
        key = wait_for_keypress(disp_ax.figure)
        if key == 'y':
            ax_bg.cla()
            disp_ax.cla()
            break

    return mapping


if __name__ == "__main__":
    register_screen_camera(num_tile_rows=16)
