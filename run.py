from matplotlib import pyplot as plt
from matplotlib import use
use('tkagg')

from camera import get_camera, take_picture_and_save, take_picture

cap = get_camera(0)

plt.figure()
ax = plt.gca()
ax.set_xticks([])
ax.set_yticks([])

for i in range(25):
    frame = take_picture(cap=cap)
    ax.imshow(frame)
    plt.pause(0.1)
