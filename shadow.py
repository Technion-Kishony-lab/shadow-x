import numpy as np
from matplotlib import pyplot as plt
import timers

from graphics.helpers import subtract_images
from register import register_screen_camera
from resources import set_matplotlib_backend, get_overhead_camera, \
    illuminate, set_camera_display_image, set_background_image, get_background_image_size

BACKGROUND_COLOR = (255, 255, 255)
SHADOW_COLOR = (255, 0, 0)

SHOW_CAMERA_DISPLAY = True
SHOW_CAMERA_DISPLAY_SHADOW = True
PRINT_TIMERS = False

set_matplotlib_backend()

camera = get_overhead_camera()

mapping = register_screen_camera(12, display=True)

illuminate(color=BACKGROUND_COLOR, pause=1)
image0 = camera.take_picture()
bgd_image_size = get_background_image_size()


def detect_obstractions(bgd_img, frame):
    normalized_image = frame.astype(np.float32) / (bgd_img.astype(np.float32) + 1)
    is_red = normalized_image[:, :, 0] > 1.1 * normalized_image[:, :, 2]
    detected_mask = np.abs(subtract_images(bgd_img[:, :, 2:], frame[:, :, 2:], as_gray=True)) > 70

    return detected_mask & ~is_red


for i in range(5000):
    with timers.timeit("all"):
        with timers.timeit("take_picture"):
            image1 = camera.take_picture()

        if SHOW_CAMERA_DISPLAY:
            with timers.timeit("set_camera_display_image"):
                set_camera_display_image(image1, index=0)

        with timers.timeit("detect_obstractions"):
            detected_mask = detect_obstractions(image0, image1)
            shadow_image_on_camera = np.zeros_like(image1)
            shadow_image_on_camera[:, :] = BACKGROUND_COLOR
            shadow_image_on_camera[detected_mask] = SHADOW_COLOR

        if SHOW_CAMERA_DISPLAY_SHADOW:
            with timers.timeit("set_camera_display_image shadow"):
                set_camera_display_image(shadow_image_on_camera, index=1)

        with timers.timeit("map_camera_image_to_screen_image"):
            screen_image = mapping.map_camera_image_to_screen_image(shadow_image_on_camera, bgd_image_size[::-1])

        with timers.timeit("set_background_image"):
            set_background_image(screen_image)

        with timers.timeit("plt.pause"):
            plt.pause(0.001)

    if i % 100 == 0 and PRINT_TIMERS:
        print('\n')
        timers.print_all_timers()
