import cv2
import numpy as np

from matplotlib import pyplot as plt
import timers

from graphics.helpers import subtract_images
from register import register_screen_camera
from resources import set_matplotlib_backend, get_overhead_camera, \
    illuminate, set_camera_display_image, set_background_image, get_background_image_size

BACKGROUND_COLOR = (255, 255, 255)
SHADOW_COLOR = (255, 0, 0)

SHOW_CAMERA_DISPLAY = False
SHOW_CAMERA_DISPLAY_SHADOW = False
SMOOTHING = False
PRINT_TIMERS = False

set_matplotlib_backend()

camera = get_overhead_camera()

mapping = register_screen_camera(12, display=True)

ax, img = illuminate(color=BACKGROUND_COLOR, pause=1)
fig = ax.figure
image0 = camera.take_picture()
bgd_image_size = get_background_image_size()

# Setup matplotlib figure for blitting
plt.show(block=False)

# Create background for blitting
background = fig.canvas.copy_from_bbox(ax.bbox)


def detect_obstractions(bgd_img, frame):
    # normalized_image = frame.astype(np.float32) / (bgd_img.astype(np.float32) + 1)
    # is_red = normalized_image[:, :, 0] > 1.5 * normalized_image[:, :, 2]
    # detected_mask = np.abs(subtract_images(bgd_img[:, :, 2:], frame[:, :, 2:], as_gray=True)) > 30
    return bgd_img[:, :, 0].astype(float) - frame[:, :, 0].astype(float) > 60


for i in range(5000):
    with timers.timeit("all"):
        with timers.timeit("take_picture"):
            image1 = camera.take_picture()

        if SHOW_CAMERA_DISPLAY:
            with timers.timeit("set_camera_display_image"):
                set_camera_display_image(image1, index=0)

        with timers.timeit("detect_obstractions"):
            detected_mask = detect_obstractions(image0, image1)

        if SMOOTHING:
            with timers.timeit("smoothing"):
                # detected_mask = cv2.morphologyEx(detected_mask.astype(np.uint8), cv2.MORPH_OPEN,
                #                                  np.ones((3, 3), np.uint8)).astype(bool)
                detected_mask = cv2.blur(detected_mask.astype(np.float32), (5, 5)) > 0.1

        with timers.timeit("create shadow image"):
            shadow_image_on_camera = np.zeros_like(image1)
            shadow_image_on_camera[:, :] = BACKGROUND_COLOR
            shadow_image_on_camera[detected_mask] = SHADOW_COLOR

        if SHOW_CAMERA_DISPLAY_SHADOW:
            with timers.timeit("set_camera_display_image shadow"):
                set_camera_display_image(shadow_image_on_camera, index=1)

        with timers.timeit("map_camera_image_to_screen_image"):
            screen_image = mapping.map_camera_image_to_screen_image(shadow_image_on_camera, bgd_image_size[::-1])

        with timers.timeit("blit_update"):
            fig.canvas.restore_region(background)
            img.set_data(screen_image)
            ax.draw_artist(img)
            fig.canvas.blit(ax.bbox)
            fig.canvas.flush_events()

    if i % 100 == 0 and PRINT_TIMERS:
        print('\n')
        timers.print_all_timers()
