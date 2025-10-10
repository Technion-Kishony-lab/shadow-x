import numpy as np
from matplotlib import pyplot as plt

from graphics.figures import create_positioned_figure_and_axes
from graphics.helpers import get_axes_size_in_pixels


TEXT_KWARGS_DEFAULT = {'color': 'red', 'fontsize': 16, 'x': 0.5, 'y': 0.05, 'ha': 'center', 'va': 'center'}


class ImageFigure:
    def __init__(self, screen_index=0, figure_position="full", axes_position="full", bin_size=1,
                 text_kwargs=None):
        self.fig, self.ax = create_positioned_figure_and_axes(screen=screen_index,
                                                              figure_position=figure_position,
                                                              axes_position=axes_position,
                                                              is_image=True, remove_toolbar=True)
        self.canvas = self.fig.canvas
        self.bin_size = bin_size
        self.text_kwargs = {**TEXT_KWARGS_DEFAULT, **(text_kwargs or {})}
        self._background_for_bliting = None
        self.image = None
        self.text = None

    @property
    def image_array(self):
        return self.image.get_array() if self.image is not None else None

    def set_image(self, arr: np.ndarray, pause=0, draw=True, allow_resize=False,
                  cmap=None, clim=None):
        if self.image is not None:
            if not allow_resize:
                assert np.all(self.image_array.shape[:2] == arr.shape[:2])
            self.image.set_array(arr)
            if cmap is not None:
                self.image.set_cmap(cmap)
            if clim is not None:
                self.image.set_clim(*clim)
        else:
            cmap = cmap or 'gray' if arr.ndim == 2 else None
            clim = clim or (0, 255) if arr.dtype == np.uint8 else None
            self.image = self.ax.imshow(arr, cmap=cmap, vmin=clim[0], vmax=clim[1], zorder=0)
        if draw:
            self.canvas.draw()
        if pause:
            plt.pause(pause)
        return self.image

    def set_text(self, txt: str = "", pause=0, draw=True):
        if self.text is not None:
            self.text.set_text(txt)
        else:
            self.text = self.ax.text(**self.text_kwargs, s=txt, zorder=10, transform=self.ax.transAxes)
        if draw:
            self.canvas.draw()
        if pause:
            plt.pause(pause)

    def get_recomended_image_size(self):
        return get_axes_size_in_pixels(self.ax) // self.bin_size

    def get_image_size(self):
        arr = self.image_array
        if arr is not None:
            return arr.shape[:2]
        else:
            return self.get_recomended_image_size()

    def illuminate(self, color=(255, 255, 255), pause=1, clear=True):
        size = self.get_image_size()
        background_image = np.zeros((size[0], size[1], 3), dtype=np.uint8)
        background_image[:, :] = color
        if clear:
            self.clear()
        self.set_image(background_image, pause=pause)

    def capture_background_for_bliting(self):
        self._background_for_bliting = self.canvas.copy_from_bbox(self.fig.bbox)

    def update_image(self, arr: np.ndarray, blitting: bool, refresh_now: bool = True):
        """Update display with optional blitting for better performance"""
        refresh_func = None
        if blitting:
            if self._background_for_bliting is None:
                self.set_image(arr, draw=True)
                self.capture_background_for_bliting()
            else:
                self.canvas.restore_region(self._background_for_bliting)
                self.set_image(arr, draw=False)
                self.ax.draw_artist(self.image)
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
        # Remove all artists but keep axes properties
        self.text = None
        for artist in self.ax.get_children():
            if hasattr(artist, 'remove') and artist is not self.image:
                try:
                    artist.remove()
                except NotImplementedError:
                    pass
        if pause:
            plt.pause(pause)
