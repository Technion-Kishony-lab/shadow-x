import numpy as np
from matplotlib import pyplot as plt

from register import register_screen_camera, map_camera_image_to_screen_image
from resources import set_matplotlib_backend, get_overhead_camera, get_background_axes, get_camera_display_axes, \
    illuminate, set_camera_display_image, set_background_image

set_matplotlib_backend()

camera = get_overhead_camera()

ax_bg = get_background_axes()
ax_display = get_camera_display_axes()

mapping = register_screen_camera(100, display=True)

illuminate(color=(255, 255, 255))
bgd_img = camera.take_picture()

for i in range(50):
    frame = camera.take_picture()

    detected_mask = np.abs((bgd_img - frame).mean(axis=2)) > 20

    set_camera_display_image(frame)

    # map detected points to screen
    screen_image = map_camera_image_to_screen_image(mapping, (1-detected_mask.astype(np.uint8)) * 255, (ax_bg.images[0].get_array().shape[1], ax_bg.images[0].get_array().shape[0]))
    set_background_image(screen_image)

    plt.pause(0.5)
