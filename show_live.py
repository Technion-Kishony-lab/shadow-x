from matplotlib import pyplot as plt

from resources import set_matplotlib_backend, get_overhead_camera, set_camera_display_image

set_matplotlib_backend()

camera = get_overhead_camera()

for i in range(50):
    frame = camera.take_picture()
    set_camera_display_image(frame)
    plt.pause(0.01)
