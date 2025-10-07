from typing import Type
import cv2
import numpy as np

from camera import Camera
from graphics.image_figure import ImageFigure
from graphics.helpers import subtract_images, wait_for_keypress
from graphics.patterns import get_array_of_circles_image
from mapping import HomographyMapping, Mapping
from resources import get_overhead_camera, set_matplotlib_backend, get_or_create_camera_figure, get_or_create_backlight_screen

set_matplotlib_backend()


def find_circles_grid(diff_image, size):
    ret, centers_on_camera = cv2.findCirclesGrid(255 - diff_image, size, cv2.CALIB_CB_SYMMETRIC_GRID)
    if ret:
        centers_on_camera = centers_on_camera.reshape(-1, 2)
    return ret, centers_on_camera


def get_diff_image(camera: Camera, screen: ImageFigure, num_tile_rows):
    image_with_circles, xs, ys = get_array_of_circles_image(size=screen.get_image_size(), num_rows=num_tile_rows)
    screen.illuminate(color=(0, 0, 0), pause=0.5)
    image0 = camera.take_picture()
    screen.set_image(image_with_circles, pause=0.5)
    image1 = camera.take_picture()

    diff_image = subtract_images(image1, image0, as_gray=True, as_uint8=True)
    centers_on_screen = np.array([[x, y] for x in xs for y in ys])
    return diff_image, xs, ys, centers_on_screen


def get_centers_on_screen_and_camera(camera: Camera, screen: ImageFigure, num_tile_rows):
    diff_image, xs, ys, centers_on_screen = get_diff_image(camera, screen, num_tile_rows)
    ret, centers_on_camera = find_circles_grid(diff_image, (len(ys), len(xs)))
    image_size = diff_image.shape
    if not ret:
        centers_on_camera = None
    return centers_on_screen, centers_on_camera, image_size, diff_image


def register_screen_camera(camera: Camera, backlight_screen: ImageFigure, num_tile_rows=10, display=True,
                           mapping_class: Type[Mapping] = HomographyMapping) -> Mapping:
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
    mapping_class : class
        The Mapping subclass to use for registration.
    """

    mapping = None
    while True:
        centers_on_screen, centers_on_camera, image_size, diff_image = get_centers_on_screen_and_camera(
            camera, backlight_screen, num_tile_rows)

        if centers_on_camera is not None:
            mapping = mapping_class.from_matching_points(
                centers_on_screen, 
                centers_on_camera,
                image_size=image_size
            )
            if display is False:
                break

        disp_ax = get_or_create_camera_figure().ax
        camera_display = get_or_create_camera_figure(index=0)
        camera_display.set_image(diff_image)
        if centers_on_camera is None:
            disp_ax.set_title("Pattern NOT detected. Press Enter to break, or adjust setup and press Space to retry.")
        else:
            # plot the detected circles on the camera image:
            disp_ax.plot(centers_on_camera[:, 0], centers_on_camera[:, 1], 'rx', markersize=7)
            disp_ax.set_title("Pattern detected. Press Enter to confirm, or adjust setup and press Space to try again.")

            # map the detected circles to screen coordinates:
            screen_points = mapping.map_camera_points_to_screen_points(centers_on_camera)

            # calculate mapping accuracy:
            mean_error = mapping.calculate_accuracy(centers_on_screen, screen_points)
            print(f"Mean mapping error: {mean_error:.2f} pixels")

            # plot the mapped points on the screen chessboard:
            backlight_screen.ax.plot(screen_points[:, 0], screen_points[:, 1], 'rx', markersize=7)
            backlight_screen.canvas.draw()
            disp_ax.figure.canvas.draw()

        key = wait_for_keypress(disp_ax.figure, options=('enter', ' '))
        disp_ax.cla()
        if key == 'enter':
            backlight_screen.clear()
            break

    return mapping


if __name__ == "__main__":
    register_screen_camera(num_tile_rows=16, camera=get_overhead_camera(),
                           backlight_screen=get_or_create_backlight_screen())
