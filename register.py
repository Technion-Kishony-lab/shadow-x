import cv2
import numpy as np
import pickle

from graphics.helpers import subtract_images, wait_for_keypress
from graphics.patterns import get_array_of_circles_image
# from mapping import HomographyMapping as Mapping
from mapping import PolynomialWarpMapping as Mapping
from resources import get_background_axes, get_overhead_camera, set_matplotlib_backend, get_camera_display_axes, \
    set_background_image, illuminate, get_background_image_size, set_camera_display_image

set_matplotlib_backend()

def find_circles_grid(diff_image, size):
    ret, centers_on_camera = cv2.findCirclesGrid(255 - diff_image, size, cv2.CALIB_CB_SYMMETRIC_GRID)
    centers_on_camera = centers_on_camera.reshape(-1, 2)
    return ret, centers_on_camera


def get_diff_image(num_tile_rows):
    camera = get_overhead_camera()
    image_with_circles, xs, ys = get_array_of_circles_image(size=get_background_image_size(), num_rows=num_tile_rows)
    illuminate(color=(0, 0, 0), pause=0.5)
    image0 = camera.take_picture()
    set_background_image(image_with_circles, pause=0.5)
    image1 = camera.take_picture()

    diff_image = subtract_images(image1, image0, as_gray=True, as_uint8=True)
    return diff_image, xs, ys


def get_centers_on_screen_and_camera(num_tile_rows):
    diff_image, xs, ys = get_diff_image(num_tile_rows)
    centers_on_screen = np.array([[x, y] for x in xs for y in ys])
    ret, centers_on_camera = find_circles_grid(diff_image, (len(ys), len(xs)))
    image_size = diff_image.shape
    if not ret:
        centers_on_camera = None
    return centers_on_screen, centers_on_camera, image_size, diff_image


def register_screen_camera(num_tile_rows=10, display=True) -> Mapping:
    """
    Register the screen positions on the camera using a circles grid pattern.
    Parameters
    ----------
    num_tile_rows : int
        Number of tile rows in the pattern.
        The number of columns is determined by the aspect ratio of the background image.
    display : bool or None
        If True, display the mapping and ask the user to verify it.
        If False, do not display the mapping, unless fail to detect the pattern.
    """

    ax_bgd = get_background_axes()

    mapping = None
    while True:
        centers_on_screen, centers_on_camera, image_size, diff_image = get_centers_on_screen_and_camera(num_tile_rows)

        if centers_on_camera is not None:
            mapping = Mapping.from_matching_points(
                centers_on_screen, 
                centers_on_camera,
                image_size=image_size
            )
            if display is False:
                break

        disp_ax = get_camera_display_axes()

        set_camera_display_image(diff_image)
        if centers_on_camera is None:
            disp_ax.set_title("Pattern NOT detected. Press Enter to break, or adjust setup and press Space to retry.")
        else:
            # plot the detected circles on the camera image:
            disp_ax.plot(centers_on_camera[:, 0], centers_on_camera[:, 1], 'rx', markersize=7)
            disp_ax.set_title("Pattern detected. Press Enter to confirm, or adjust setup and press Space to try again.")

            # map the detected circles to screen coordinates:
            screen_points = mapping.map_camera_points_to_screen_points(centers_on_camera)

            # clac the mean of the squared distances between the mapped points and the screen points:
            distances = np.linalg.norm(screen_points - centers_on_screen, axis=1)
            print(f"Mean of squared distances: {distances.mean()}")

            # plot the mapped points on the screen chessboard:
            ax_bgd.plot(screen_points[:, 0], screen_points[:, 1], 'rx', markersize=7)
            ax_bgd.figure.canvas.draw()
            disp_ax.figure.canvas.draw()

        key = wait_for_keypress(disp_ax.figure, options=('enter', ' '))
        disp_ax.cla()
        if key == 'enter':
            ax_bgd.cla()
            break

    return mapping


if __name__ == "__main__":
    register_screen_camera(num_tile_rows=16)
