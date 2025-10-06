import numpy as np

from matplotlib import pyplot as plt

from graphics.helpers import subtract_images, wait_for_keypress
from resources import set_matplotlib_backend, get_overhead_camera, \
    illuminate, set_camera_display_image, set_background_image, get_backlight_image_size, get_camera_display_axes

BACKGROUND_COLOR = (255, 255, 255)
SHADOW_COLOR = (255, 0, 0)

set_matplotlib_backend()

camera = get_overhead_camera()

bgd_image_size = get_backlight_image_size()

# find the screen pixels:
illuminate(color=(0, 0, 0), pause=1)
image_black = camera.take_picture()
illuminate(color=(255, 255, 255), pause=1)
image_white = camera.take_picture()
screen_mask = subtract_images(image_white, image_black, as_gray=True) > 100

# create image with stripes of BACKGROUND_COLOR and SHADOW_COLOR
stripe_height = bgd_image_size[1] // 10
stripe_image = np.zeros((bgd_image_size[0], bgd_image_size[1], 3), dtype=np.uint8)
x, y = np.indices((bgd_image_size[0], bgd_image_size[1]))
stripe_image[(y // stripe_height) % 2 == 0] = BACKGROUND_COLOR
stripe_image[(y // stripe_height) % 2 == 1] = SHADOW_COLOR

set_background_image(stripe_image, pause=1)
image0 = camera.take_picture()
ax, img = set_camera_display_image(image0, index=0)
# add countour of mask:
plt.contour(screen_mask, colors='y', linewidths=1.5)

disp_ax = get_camera_display_axes()
disp_ax.set_title('Place a hand over the stripes pattern and press Enter.')
wait_for_keypress(disp_ax.figure, options=('enter',))

image1 = camera.take_picture()

set_camera_display_image(image1, index=0)

# plot histograms:
plt.figure('Histograms')
plt.clf()
colors = ('r', 'g', 'b')
normalized_image = image1.astype(np.float32) / (image0.astype(np.float32) + 1)
for i, color in enumerate(colors):
    for j in (1, 2, 3):
        plt.subplot(3, 3, i*3 + j)
        if j == 1:
            plt.hist(image0[screen_mask, i].ravel(), bins=256, color=color, label='Foreground')
            plt.xlim([0, 256])
        elif j == 2:
            plt.hist(image1[screen_mask, i].ravel(), bins=256, color=color, label='Background')
            plt.xlim([0, 256])
        else:
            plt.hist(np.log2(normalized_image[screen_mask, i]+0.01).ravel(), bins=256, color=color, label='Normalized')

plt.show()
