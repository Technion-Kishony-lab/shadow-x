from matplotlib import pyplot as plt
from matplotlib import use

from graphics import set_figure_as_full_screen, set_figure_position

from camera import get_camera, take_picture

use('qtagg')

BACKGROUND_SCREEN = 1
DISPLAY_SCREEN = 2

CAP = get_camera(0)

# set a figure on the entire screen:
background_fig = plt.figure()
set_figure_position(background_fig, screen=BACKGROUND_SCREEN, position="full")

display_fig = plt.figure()
set_figure_position(display_fig, screen=DISPLAY_SCREEN, position="full")

ax = display_fig.add_subplot(1, 1, 1)
ax.set_xticks([])
ax.set_yticks([])

for i in range(25):
    frame = take_picture(cap=CAP)
    ax.imshow(frame)
    plt.pause(0.01)