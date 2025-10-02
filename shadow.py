from matplotlib import pyplot as plt

from register import register_screen_camera
from resources import set_matplotlib_backend, get_overhead_camera, get_background_axes, get_display_axes

set_matplotlib_backend()

camera = get_overhead_camera()

ax_bg = get_background_axes()
ax_display = get_display_axes()

mapping = register_screen_camera(100, display=True)
