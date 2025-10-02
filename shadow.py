from matplotlib import pyplot as plt

from resources import set_matplotlib_backend, get_overhead_camera, get_background_axes, get_display_axes

set_matplotlib_backend()

camera = get_overhead_camera()

ax_bg = get_background_axes()
ax_display = get_display_axes()

img = ax_display.imshow(camera.take_picture())

for i in range(50):
    frame = camera.take_picture()
    img.set_array(frame)
    plt.pause(0.01)

