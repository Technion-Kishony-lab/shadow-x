import numpy as np
from matplotlib import pyplot as plt

from graphics.patterns import get_stripes_image
from resources.camera import Camera
from utils.find_phase import cyclic_true_center_last_axis
from graphics.helpers import subtract_images
from graphics.image_figure import ImageFigure
from runners import CameraScreenRunner
from resources.camera_and_screens import create_camera_figure

"""
Rationale.
We want to take an image with dark field illumination, i.e. with light coming from the side.
This is good for detecting scattered light from small particles.
We simulate dark field illumination by taking multiple images with a backlight screen
illuminated with moving stripes, and then combining the images such that each pixel is averaged
only over the images where it is in the dark part of the stripe pattern.


Schematic illustration of the illumination pattern over time:

=== light_width = 3
----- dark_width = 5

                                                          frame index
===-----===-----===-----===-----===-----===-----===-----  0
==-----===-----===-----===-----===-----===-----===-----=  1
=-----===-----===-----===-----===-----===-----===-----==  2
-----===-----===-----===-----===-----===-----===-----===  3
----===-----===-----===-----===-----===-----===-----===-  4
---===-----===-----===-----===-----===-----===-----===--  5
--===-----===-----===-----===-----===-----===-----===---  6
-===-----===-----===-----===-----===-----===-----===----  7
54321076543210765432107654321076543210765432107654321076  <-- "mid-dark index"

 "mid-dark"
     |
===-----

"""

ILLUMINATION_PAUSE = 0.01  # seconds to wait after setting illumination pattern


class StripeRunner(CameraScreenRunner):
    """
    Stitch the images taken by the camera using the backlight screen.
    The idea is that for each pixel we want to wight it more when it is far from a light stripe.
    So that effectively we get dark-illumation, which is good for detection of scatter light.
    """
    # we trash the first image to make sure all are consistent
    BACKGROUND_COLOR = (0, 0, 0)
    STRIPES_COLOR = (255, 255, 255)

    def __init__(self,
                 camera: Camera = None, backlight_screen: ImageFigure = None,
                 show_camera=True,
                 light_width: int = 10, dark_width: int = 40, num_images: int = None, num_images_to_trash: int = 0):
        num_images = num_images if num_images is not None else light_width + dark_width
        iterations = num_images + num_images_to_trash
        super().__init__(iterations, False, camera, backlight_screen,
                         show_camera, use_blitting=False, refresh_together=False)
        self.num_images_to_trash = num_images_to_trash
        self.num_images = num_images
        self.light_width = light_width
        self.dark_width = dark_width

    def _setup(self):
        super()._setup()
        self.images = []

    def _illuminate_pattern(self, i):
        pattern = self._generate_illumination_pattern(i)
        self.backlight_screen.set_image(pattern, pause=ILLUMINATION_PAUSE)
        return pattern

    def _do_iteration(self, i):
        i = i - self.num_images_to_trash
        self._illuminate_pattern(i)
        image = self.take_picture()
        self.maybe_show_camera(image)
        if i >= 0:
            assert len(self.images) == i
            self.images.append(image)

    def _generate_illumination_pattern(self, frame_index: int):
        phase = frame_index / self.num_images
        size = self.backlight_screen.get_recomended_image_size()
        return get_stripes_image(light_width=self.light_width, dark_width=self.dark_width, phase=phase, size=size,
                                 stripes_color=self.STRIPES_COLOR, background_color=self.BACKGROUND_COLOR)

    def take_uniform_image(self, color):
        self.backlight_screen.illuminate(color=color, pause=ILLUMINATION_PAUSE)
        image = self.take_picture()
        self.maybe_show_camera(image)
        return image

    def _after_run(self):
        self.dark_image = self.take_uniform_image(color=self.BACKGROUND_COLOR)
        self.bright_image = self.take_uniform_image(color=self.STRIPES_COLOR)
        super()._after_run()

    def save_to_pickle(self, filepath):
        import pickle
        data = {
            'images': np.array(self.images),
            'dark_image': self.dark_image,
            'bright_image': self.bright_image,
            'light_width': self.light_width,
            'dark_width': self.dark_width,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

    @classmethod
    def load_from_pickle(cls, filepath):
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        runner = cls(light_width=data['light_width'], dark_width=data['dark_width'],
                     num_images=len(data['images']), num_images_to_trash=0)
        runner.images = data['images']
        runner.dark_image = data['dark_image']
        runner.bright_image = data['bright_image']
        return runner



class StripesAnalyzer:
    NOT_LIGHTEN_COLOR = (255, 0, 0)

    def __init__(self, images, light_width: int = 2, dark_width: int = 50):
        self.images = np.asarray(images)
        self.grey_images = np.mean(images, axis=-1).astype(np.uint8)
        self.num_images = len(images)
        self.light_width = light_width
        self.dark_width = dark_width
        self._centers = None

    @classmethod
    def from_stripe_runner(cls, runner: StripeRunner):
        return cls(runner.images, light_width=runner.light_width, dark_width=runner.dark_width)

    def get_max_min_gray_images(self):
        max_image = np.max(self.grey_images, axis=0).astype(np.uint8)
        min_image = np.min(self.grey_images, axis=0).astype(np.uint8)
        return max_image, min_image

    def get_lightened_mask(self, threshold=100):
        im_max, im_min = self.get_max_min_gray_images()
        return subtract_images(im_max, im_min) > threshold

    def get_dark_cycle_index(self):
        if self._centers is None:
            im_max, im_min = self.get_max_min_gray_images()
            threshold = (im_max.astype(float) + im_min.astype(float)) // 2
            dark = self.grey_images < threshold
            dark = np.moveaxis(dark, 0, -1)
            self._centers = cyclic_true_center_last_axis(dark)
        return self._centers

    def _get_weights(self, sigma=0.25, max_sigma=0.5):
        dark_cycle_index = self.get_dark_cycle_index()
        frame_indices = np.arange(self.num_images)[None, None, :]
        num_iters_from_darkest = (dark_cycle_index[:, :, None] - frame_indices + self.num_images // 2) \
            % self.num_images - self.num_images // 2
        num_dark_iters = self.dark_width / (self.light_width + self.dark_width) * self.num_images
        norm_dist_in_dark = num_iters_from_darkest / (num_dark_iters * 0.5)
        weights = np.exp(-(norm_dist_in_dark * sigma) ** 2)
        weights[np.abs(norm_dist_in_dark) > max_sigma] = 0
        return weights

    def get_average_image(self, sigma=0.25, max_sigma=0.5):
        weights = self._get_weights(sigma, max_sigma)[:, :, None, :]
        imgs = np.moveaxis(self.images, 0, -1)
        avg_image_rgb = np.sum(imgs.astype(float) * weights, axis=-1) / np.sum(weights, axis=-1)
        avg_image_gray = np.mean(avg_image_rgb, axis=-1)
        avg_image_rgb[~self.get_lightened_mask()] = self.NOT_LIGHTEN_COLOR
        avg_image_gray[~self.get_lightened_mask()] = 0
        return avg_image_rgb.astype(np.uint8), avg_image_gray.astype(np.uint8)


def take_stripe_illuminated_image(light_width=15, dark_width=35, num_images=None,
                                  num_images_to_trash=1,
                                  sigma=0.1, max_sigma=0.8):
    num_images = num_images if num_images is not None else light_width + dark_width
    runner = StripeRunner(num_images=num_images, light_width=light_width, dark_width=dark_width,
                          num_images_to_trash=num_images_to_trash)
    runner.run()
    analyzer = StripesAnalyzer.from_stripe_runner(runner)
    return analyzer.get_average_image(sigma=sigma, max_sigma=max_sigma), analyzer


def main():
    (avg_rgb, avg_grey), analyzer = take_stripe_illuminated_image(
        light_width=10, dark_width=31, num_images=None, sigma=0.3, max_sigma=0.6, num_images_to_trash=11)
    img = create_camera_figure(1).set_image(avg_rgb, allow_resize=True)
    img = create_camera_figure(2).set_image(avg_grey, allow_resize=True)
    img.set_clim(0, 160)
    plt.show()


if __name__ == "__main__":
    main()
