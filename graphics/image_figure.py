import numpy as np
from matplotlib import pyplot as plt

from graphics.figures import create_positioned_figure_and_axes
from graphics.helpers import get_axes_size_in_pixels


class ImageFigure:
    def __init__(self, screen_index=0, figure_position="full", axes_position="full", bin_size=1):
        self.fig, self.ax = create_positioned_figure_and_axes(screen=screen_index,
                                                              figure_position=figure_position,
                                                              axes_position=axes_position,
                                                              is_image=True, remove_toolbar=True)
        self.canvas = self.fig.canvas
        self.bin_size = bin_size
        self._background_for_bliting = None

    @property
    def image(self):
        return self.ax.images[0] if self.ax.images else None

    @property
    def image_array(self):
        image = self.image
        return image.get_array() if image is not None else None

    def set_image(self, arr: np.ndarray, pause=0, draw=True, allow_resize=False):
        img = self.image
        if img is not None:
            if not allow_resize:
                assert np.all(img.get_array().shape[:2] == arr.shape[:2])
            img.set_array(arr)
        else:
            img = self.ax.imshow(arr, cmap='gray', vmin=0, vmax=255)
        if draw:
            self.canvas.draw()
        if pause:
            plt.pause(pause)
        return img

    def get_recomended_image_size(self):
        return get_axes_size_in_pixels(self.ax) // self.bin_size

    def get_image_size(self):
        arr = self.image_array
        if arr is not None:
            return arr.shape[:2]
        else:
            return self.get_recomended_image_size()

    def illuminate(self, color=(255, 255, 255), pause=1):
        size = self.get_image_size()
        background_image = np.zeros((size[0], size[1], 3), dtype=np.uint8)
        background_image[:, :] = color
        return self.set_image(background_image, pause=pause)

    def capture_background_for_bliting(self):
        self._background_for_bliting = self.canvas.copy_from_bbox(self.fig.bbox)

    def update_image(self, arr: np.ndarray, blitting: bool, refresh_now: bool = True):
        """Update display with optional blitting for better performance"""
        refresh_func = None
        if blitting:
            self.canvas.restore_region(self._background_for_bliting)
            img = self.set_image(arr, draw=False)
            self.ax.draw_artist(img)
            self.canvas.blit(self.ax.bbox)
            if refresh_now:
                self.canvas.flush_events()
            else:
                refresh_func = self.canvas.flush_events
        else:
            self.set_image(arr, draw=refresh_now)
            if not refresh_now:
                refresh_func = self.canvas.draw_idle
        return refresh_func

    def clear(self, pause=0):
        self.ax.cla()
        if pause:
            plt.pause(pause)
