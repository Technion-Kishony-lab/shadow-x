import numpy as np
from matplotlib import pyplot as plt

from scipy.ndimage import gaussian_filter

from graphics.patterns import get_stripes_image
from resources.camera import Camera
from utils.find_phase import cyclic_true_center_last_axis
from graphics.helpers import subtract_images, set_matplotlib_backend
from graphics.image_figure import ImageFigure
from runners import CameraScreenRunner, timed
from resources.camera_and_screens import create_camera_figure
from utils.persistence import with_file_cache

"""
Rationale.
We want to take an image with dark field illumination, i.e. with light coming from the side.
This is good for detecting scattered light from small particles.
We simulate dark field illumination by taking multiple images with a backlight screen
illuminated with moving stripes, and then combining the images such that each pixel is averaged
only over the images where it is in the dark part of the stripe pattern.


Schematic illustration.

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

 
     V "mid-dark"
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
                 light_width: int = 10, dark_width: int = 40, num_images: int = None, num_images_to_discard: int = 0):
        num_images = num_images if num_images is not None else light_width + dark_width
        iterations = num_images + num_images_to_discard
        super().__init__(iterations, False, camera, backlight_screen,
                         show_camera, use_blitting=True, refresh_together=False)
        self.num_images_to_discard = num_images_to_discard
        self.num_images = num_images
        self.light_width = light_width
        self.dark_width = dark_width

    def _setup(self):
        super()._setup()
        self.images = []

    @timed
    def _illuminate_pattern(self, i):
        pattern = self._generate_illumination_pattern(i)
        self.update_backlight_image(pattern)
        plt.pause(ILLUMINATION_PAUSE)

    def _do_iteration(self, i):
        i = i - self.num_images_to_discard
        self._illuminate_pattern(i)
        image = self.take_picture()
        self.maybe_show_camera(image)
        if i >= 0:
            assert len(self.images) == i
            self.images.append(image)

    @timed
    def _generate_illumination_pattern(self, frame_index: int):
        phase = frame_index / self.num_images
        size = self.backlight_screen.get_recomended_image_size()
        return get_stripes_image(light_width=self.light_width, dark_width=self.dark_width, phase=phase, size=size,
                                 stripes_color=self.STRIPES_COLOR, background_color=self.BACKGROUND_COLOR)

    def take_uniform_image(self, color):
        self.backlight_screen.illuminate(color=color, pause=0.5)
        image = self.take_picture()
        self.maybe_show_camera(image)
        return image

    def _after_run(self):
        self.print_timers_report()
        self.dark_image = self.take_uniform_image(color=self.BACKGROUND_COLOR)
        self.bright_image = self.take_uniform_image(color=self.STRIPES_COLOR)
        super()._after_run()

    def to_pickle(self, filepath):
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
    def from_pickle(cls, filepath):
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        runner = cls(light_width=data['light_width'], dark_width=data['dark_width'],
                     num_images=len(data['images']), num_images_to_discard=0)
        runner.images = data['images']
        runner.dark_image = data['dark_image']
        runner.bright_image = data['bright_image']
        return runner


class MidDarkAnalyzer:
    def __init__(self, gray):
        self.gray = gray
        self.num_images = gray.shape[0]

    def max_min_images(self):
        return np.max(self.gray, axis=0), np.min(self.gray, axis=0)

    def get_lightened_mask(self, relative_threshold=0.1):
        """
        For each pixel, determine if it is in the screen area (i.e. intensity varies when stripes move).
        """
        im_max, im_min = self.max_min_images()
        diff = subtract_images(im_max, im_min)
        max_diff = np.max(diff)
        return diff > relative_threshold * max_diff

    def get_mid_dark_index(self):
        """
        For each pixel, find the index of the frame where it is in mid-dark phase.
        """
        im_max, im_min = self.max_min_images()
        threshold = (im_max.astype(float) + im_min.astype(float)) // 2
        dark = self.gray < threshold
        return cyclic_true_center_last_axis(np.moveaxis(dark, 0, -1))


class StripesAnalyzer:
    NOT_LIGHTEN_COLOR = (255, 0, 0)

    # None: auto by observed dynamic range
    RGB_WEIGHTS = None  # np.array([0.2989, 0.5870, 0.1140])

    def __init__(self, images, dark_image, bright_image, light_width: int = 2, dark_width: int = 50):
        self.images = np.asarray(images)
        self.dark_image = dark_image
        self.bright_image = bright_image
        self.light_width = light_width
        self.dark_width = dark_width
        self._rgb_weights = None

    @classmethod
    def from_stripe_runner(cls, runner: StripeRunner):
        return cls(runner.images, runner.dark_image, runner.bright_image,
                   light_width=runner.light_width, dark_width=runner.dark_width)

    @property
    def num_images(self):
        return self.images.shape[0]

    def get_background_subtracted_images(self, substract=True, normalize=True):
        imgs = self.images
        if substract:
            imgs = subtract_images(imgs, self.dark_image)
        if normalize:
            mx = np.max(imgs, axis=(0, 1, 2))
            mn = np.min(imgs, axis=(0, 1, 2))
            imgs = (imgs - mn) / (mx - mn) * 255.0
        return np.clip(imgs, 0, 255).astype(np.uint8)

    def get_rgb_dynamic_range(self):
        imgs = self.get_background_subtracted_images()
        return np.max(imgs, axis=(0, 1, 2)) - np.min(imgs, axis=(0, 1, 2))

    def get_rgb_weights(self):
        if self._rgb_weights is None:
            self._rgb_weights = self._compute_rgb_weights()
        return self._rgb_weights

    def _compute_rgb_weights(self):
        if self.RGB_WEIGHTS is not None:
            return self.RGB_WEIGHTS
        dr = self.get_rgb_dynamic_range().astype(float)
        dr_sum = np.sum(dr)
        return dr / dr_sum

    def rgb_to_gray(self, images):
        return np.dot(images[..., :3], self.get_rgb_weights()).astype(np.uint8)

    def get_gray_images(self, smoothing: int = None):
        gray_images = self.rgb_to_gray(self.images)
        if smoothing is None:
            return gray_images
        return np.array([gaussian_filter(img, sigma=smoothing) for img in gray_images])

    def _get_weights(self, mid_dark_index, dark_dist, max_dark_dist):
        """
        For each pixel, compute the weights for each image based on the distance from the mid-dark index.
        """
        image_indices = np.arange(self.num_images)[None, None, :]
        num_frames_from_darkest = (mid_dark_index[:, :, None] - image_indices + self.num_images // 2) \
                                  % self.num_images - self.num_images // 2
        num_frames_in_dark_phase = self.dark_width / (self.light_width + self.dark_width) * self.num_images
        phase_in_dark = num_frames_from_darkest / (num_frames_in_dark_phase * 0.5)  # -1 to 1 in dark phase
        weights = np.exp(-(phase_in_dark / dark_dist) ** 2)
        weights[np.abs(phase_in_dark) > max_dark_dist] = 0
        return weights

    def get_pseudo_darkfield_image(self, dark_dist=0.25, max_dark_dist=0.5, smoothing=None,
                                   substract_background=True, normalize=True):
        mid_dark_analyzer = MidDarkAnalyzer(self.get_gray_images(smoothing=smoothing))
        mid_dark_index = mid_dark_analyzer.get_mid_dark_index()

        weights = self._get_weights(mid_dark_index, dark_dist, max_dark_dist)[:, :, None, :]
        img = self.get_background_subtracted_images(substract_background, normalize)
        imgs = np.moveaxis(img, 0, -1)
        avg_image_rgb = np.sum(imgs.astype(float) * weights, axis=-1) / np.sum(weights, axis=-1)
        avg_image_gray = self.rgb_to_gray(avg_image_rgb)

        mask = mid_dark_analyzer.get_lightened_mask()
        avg_image_rgb[~mask] = self.NOT_LIGHTEN_COLOR
        avg_image_gray[~mask] = 0
        return avg_image_rgb.astype(np.uint8), avg_image_gray.astype(np.uint8)


@with_file_cache('stripe_runner.pkl', 'calculate')
def run_stripes_illumination(light_width=15, dark_width=35, num_images=None,
                             num_images_to_discard=10) -> StripeRunner:
    runner = StripeRunner(num_images=num_images, light_width=light_width, dark_width=dark_width,
                          num_images_to_discard=num_images_to_discard)
    runner.run()
    return runner


def take_pseudo_darkfield_image(light_width=15, dark_width=35, num_images=None,
                                  num_images_to_discard=1,
                                  sigma=0.4, max_sigma=0.8,
                                  smoothing=None,
                                  substract_background=True,
                                  normalize=True,
                                  ):
    runner = run_stripes_illumination(light_width, dark_width, num_images, num_images_to_discard)
    analyzer = StripesAnalyzer.from_stripe_runner(runner)
    rgb, gray = analyzer.get_pseudo_darkfield_image(dark_dist=sigma, max_dark_dist=max_sigma, smoothing=smoothing,
                                               substract_background=substract_background, normalize=normalize)
    return rgb, gray, analyzer


def main():
    set_matplotlib_backend()
    plt.ion()
    rgb, gray, analyzer = take_pseudo_darkfield_image(light_width=14, dark_width=17, num_images=None,
                                                      num_images_to_discard=11, sigma=0.4, max_sigma=1, smoothing=1,
                                                      substract_background=False, normalize=False)
    create_camera_figure(1).set_image(rgb, allow_resize=True)
    create_camera_figure(2).set_image(gray, allow_resize=True, cmap='gray', clim=(0, 160))
    create_camera_figure(3).set_image(analyzer.bright_image, allow_resize=True)
    plt.show()


if __name__ == "__main__":
    main()
