import time
from matplotlib import pyplot as plt
from matplotlib import use

from env import BACKGROUND_SCREEN, DISPLAY_SCREEN, BACKEND
from graphics import set_figure_position

from camera import get_camera, take_picture

use(BACKEND)

CAP = get_camera(0)

# set a figure on the entire screen:
background_fig = plt.figure()
set_figure_position(background_fig, screen=BACKGROUND_SCREEN, position="full")

display_fig = plt.figure()
set_figure_position(display_fig, screen=DISPLAY_SCREEN, position="full")

ax = display_fig.add_subplot(1, 1, 1)
ax.set_xticks([])
ax.set_yticks([])
img = ax.imshow(take_picture(cap=CAP))
time_start = time.time()
for i in range(50):
    frame = take_picture(cap=CAP)
    img.set_array(frame)
    plt.pause(0.01)

time_end = time.time()

print('resolution: ', frame.shape)
print(f"Time taken: {time_end - time_start} seconds")
