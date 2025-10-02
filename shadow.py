import numpy as np
from matplotlib import pyplot as plt

from graphics.helpers import subtract_images
from register import register_screen_camera, map_camera_image_to_screen_image
from resources import set_matplotlib_backend, get_overhead_camera, \
    illuminate, set_camera_display_image, set_background_image, get_background_image_size

BACKGROUND_COLOR = (255, 255, 255)
SHADOW_COLOR = (255, 0, 0)

set_matplotlib_backend()

camera = get_overhead_camera()

mapping = register_screen_camera(12, display=True)


illuminate(color=BACKGROUND_COLOR, pause=1)
image0 = camera.take_picture()
background_image_size = get_background_image_size()


def detect_obstractions(bgd_img, frame):
    normalized_image = frame.astype(np.float32) / (bgd_img.astype(np.float32) + 1)
    is_red = normalized_image[:,:,0] > 1.1 * normalized_image[:,:,2]
    detected_mask = np.abs(subtract_images(bgd_img[:,:,2:], frame[:,:,2:], as_gray=True)) > 70

    return detected_mask & ~is_red


for i in range(5000):
    image1 = camera.take_picture()
    set_camera_display_image(image1)

    detected_mask = detect_obstractions(image0, image1)

    # the shadow should be SHADOW_COLOR on BACKGROUND_COLOR
    shadow_image_on_camera = np.zeros_like(image1)
    shadow_image_on_camera[:,:] = BACKGROUND_COLOR
    shadow_image_on_camera[detected_mask] = SHADOW_COLOR

    # map detected points to screen
    screen_image = map_camera_image_to_screen_image(mapping, shadow_image_on_camera, background_image_size[::-1])
    set_background_image(screen_image)

    plt.pause(0.01)
