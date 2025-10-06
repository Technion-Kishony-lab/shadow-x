import argparse
import time
import cv2
import numpy as np


from matplotlib import pyplot as plt
from graphics.helpers import beep, subtract_images
from mapping import HomographyMapping, Mapping
import timers

from register import register_screen_camera
from resources import set_matplotlib_backend, get_overhead_camera, \
    set_camera_display_image, get_backlight_image_size, set_backlight_image

from graphics.helpers import update_image, capture_background



def build_args_parser():
    parser = argparse.ArgumentParser(description="Shadow occlusion visualizer")
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--show-camera", action="store_true", default=False)
    parser.add_argument("--show-shadow", action="store_true", default=False)
    parser.add_argument("--smoothing", action="store_true", default=False)
    parser.add_argument("--print-timers", action="store_true", default=True)
    parser.add_argument("--registration-grid", type=int, default=12)
    parser.add_argument("--save-mapping", type=str, help="File path to save the mapping.")
    parser.add_argument("--load-mapping", type=str, help="File path to load the mapping from.")
    return parser


class ShadowRunner:
    MAPPING_CLASS = HomographyMapping
    BACKGROUND_COLOR = (255, 255, 255)
    SHADOW_COLOR = (255, 0, 0)
    TEXT_COLOR = (200, 200, 255)
    TEXT = "Shadow-X"

    def __init__(self, iterations=5000, show_camera=False, show_detection=False, smoothing=False,
                 print_timers=True, registration_grid=12, use_blitting=True,
                 mapping_filepath="mapping.pkl", load_mapping=None, save_mapping=None,
                 refresh_together=True):
        self.iterations = iterations
        self.show_camera = show_camera
        self.show_detection = show_detection
        self.smoothing = smoothing
        self.print_timers = print_timers
        self.registration_grid = registration_grid
        self.use_blitting = use_blitting
        self.mapping_filepath = mapping_filepath
        self.load_mapping = load_mapping
        self.save_mapping = save_mapping
        self.refresh_together = refresh_together

    def _setup_matplotlib(self):
        set_matplotlib_backend()

    def _setup_mapping(self):
        mapping = None
        if self.load_mapping is not False:
            try:
                mapping = Mapping.load_mapping(self.mapping_filepath)
            except FileNotFoundError:
                if self.load_mapping is True:
                    raise FileNotFoundError(f"File not found: {self.load_mapping}")
                else:
                    print(f"File not found: {self.load_mapping}, recreate mapping...")
        if mapping is None:
            mapping = register_screen_camera(self.registration_grid, display=True, mapping_class=self.MAPPING_CLASS)
            if self.save_mapping is not False:
                mapping.save_mapping(self.mapping_filepath)
        self.mapping = mapping

    def _get_bgd_image(self):
        img = np.zeros((self.backlight_image_size[0], self.backlight_image_size[1], 3), dtype=np.uint8)
        img[:, :] = self.BACKGROUND_COLOR
        cv2.putText(
            img=img,
            text=self.TEXT,
            org=(self.backlight_image_size[0] // 12, self.backlight_image_size[1] // 2),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=3,
            color=self.TEXT_COLOR,
            thickness=15,
            lineType=cv2.LINE_AA,
        )
        return img

    def _setup_display(self):
        self.backlight_image_size = get_backlight_image_size()
        self.backlight_bgd_image = self._get_bgd_image()
        self.backlight_ax, self.backlight_img = set_backlight_image(self.backlight_bgd_image, pause=1)
        self.backlight_fig = self.backlight_ax.figure
        self.backlight_background = capture_background(self.backlight_ax)  # Capture the initial background
        plt.show(block=False)

    def _setup_camera(self):
        self.camera = get_overhead_camera()
        self.camera_initial_frame = self.camera.take_picture()
        self.camera_detection_image = np.zeros((self.camera_initial_frame.shape[:2]), dtype=np.uint8)
        
        # Setup camera display axes and images for blitting
        self.camera_axs = {}
        self.camera_imgs = {}
        self.camera_backgrounds = {}
        
        if self.show_camera:
            ax, img = set_camera_display_image(self.camera_initial_frame, index=0)
            self.camera_axs[0] = ax
            self.camera_imgs[0] = img
            self.camera_backgrounds[0] = capture_background(ax)  # Capture the initial background
        if self.show_detection:
            ax, img = set_camera_display_image(
                np.zeros_like(self.camera_initial_frame[:, :, 0], dtype=np.uint8), index=1)
            self.camera_axs[1] = ax
            self.camera_imgs[1] = img
            self.camera_backgrounds[1] = capture_background(ax)  # Capture the initial background

    def _setup(self):
        self._setup_matplotlib()
        self._setup_mapping()
        self._setup_display()
        self._setup_camera()
        beep()


    def run(self):
        self._setup()
        t = time.time()
        for i in range(self.iterations):
            with timers.timeit("all"):
                self._refresh_funcs = []
                frame = self.take_picture()
                self.maybe_show_camera(frame)
                detection_mask = self.detect(frame)
                detection_mask = self.adjust_detection_mask(detection_mask)
                self.maybe_show_detection_mask(detection_mask)
                screen_image = self.map_to_screen(detection_mask)
                self.update_backlight_image(screen_image)
                self.refresh_all()

            self.maybe_print_timers(i)

        return time.time() - t

    # --- timed helpers ---

    @staticmethod
    def timed(func):
        def wrapper(self, *args, **kwargs):
            with timers.timeit(func.__name__):
                return func(self, *args, **kwargs)
        return wrapper

    @timed
    def refresh_all(self):
        for refresh_func in self._refresh_funcs:
            refresh_func()
        if not self.use_blitting:
            plt.pause(0.001)

    @timed
    def take_picture(self):
        return self.camera.take_picture()

    @timed
    def update_backlight_image(self, backlight_image):
        refresh_func = update_image(self.backlight_ax, self.backlight_img, backlight_image, self.backlight_background, self.use_blitting, refresh_now=not self.refresh_together)
        if refresh_func:
            self._refresh_funcs.append(refresh_func)

    @timed
    def maybe_show_camera(self, frame):
        if self.show_camera:
            refresh_func = update_image(self.camera_axs[0], self.camera_imgs[0], frame, self.camera_backgrounds[0], self.use_blitting, refresh_now=not self.refresh_together)
            if refresh_func:
                self._refresh_funcs.append(refresh_func)

    def _detect_from_diff(self, diff_image):
        return diff_image[:, :, 0] > 60

    @timed
    def detect(self, frame):
        return self._detect_from_diff(subtract_images(self.camera_initial_frame, frame))

    @timed
    def adjust_detection_mask(self, detection_mask):
        if self.smoothing:
            return cv2.blur(detection_mask.astype(np.float32), (5, 5)) > 0.1
        return detection_mask

    @timed
    def build_shadow_image(self, detected_mask):
        shadow_image_on_camera = self.camera_detection_image.copy()
        shadow_image_on_camera[detected_mask] = self.SHADOW_COLOR
        return shadow_image_on_camera

    @timed
    def maybe_show_detection_mask(self, detection_mask):
        if self.show_detection:
            refresh_func = update_image(self.camera_axs[1], self.camera_imgs[1], detection_mask * 255, self.camera_backgrounds[1], self.use_blitting, refresh_now=not self.refresh_together)
            if refresh_func:
                self._refresh_funcs.append(refresh_func)

    def _map_detection_mask_to_screen(self, detection_mask):
        return self.mapping.map_camera_image_to_screen_image(
            detection_mask.astype(np.uint8), self.backlight_image_size[::-1]).astype(bool)

    @timed
    def map_to_screen(self, detection_mask):
        detection_mask_on_screen = self._map_detection_mask_to_screen(detection_mask)
        backlight_image = self.backlight_bgd_image.copy()
        backlight_image[detection_mask_on_screen] = self.SHADOW_COLOR
        return backlight_image

    def maybe_print_timers(self, i):
        if (i + 1) % 100 == 0 and self.print_timers:
            print('\n')
            timers.print_all_timers()


def main():
    args = build_args_parser().parse_args()
    ShadowRunner(
        iterations=args.iterations,
        show_camera=args.show_camera,
        show_detection=args.show_shadow,
        smoothing=args.smoothing,
        print_timers=args.print_timers,
        registration_grid=args.registration_grid,
    ).run()


if __name__ == "__main__":
    # main()
    print(f"{'show_camera':<12} {'use_blitting':<13} {'refresh_together':<16} {'time':<8}")
    print("-" * 70)
    for show_camera in [True, False]:
            for use_blitting in [True, False]:
                for refresh_together in [True, False]:
                    t = ShadowRunner(show_camera=show_camera, show_detection=show_camera, use_blitting=use_blitting, refresh_together=refresh_together,
                                 iterations=101, save_mapping=None, load_mapping=None).run()
                    print(f"{str(show_camera):<12} {str(use_blitting):<13} {str(refresh_together):<16} {t:<8.3f}")
                    beep(frequency=1000, duration=0.1)
                    time.sleep(0.1)