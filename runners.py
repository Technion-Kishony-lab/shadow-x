import argparse
import time
import cv2
import numpy as np
from PIL.PdfParser import decode_text

from matplotlib import pyplot as plt
from graphics.helpers import beep, subtract_images
from mapping import HomographyMapping, Mapping

from register import register_screen_camera
from resources import set_matplotlib_backend, get_overhead_camera, \
    set_camera_display_image, get_backlight_image_size, set_backlight_image

from graphics.helpers import update_image, capture_background_for_bliting
from timers import Timer



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



class Runner:
    def __init__(self, iterations=5000, print_timers=True):
        self.iterations = iterations
        self.print_timers = print_timers
        self.timers = {}

    def get_timer(self, name):
        if name not in self.timers:
            self.timers[name] = Timer(name)
        return self.timers[name]

    def print_timers_report(self):
        for timer in self.timers.values():
            print(timer.report())

    @staticmethod
    def timed(func):
        def wrapper(self, *args, **kwargs):
            with self.get_timer(func.__name__):
                return func(self, *args, **kwargs)
        return wrapper

    def _setup(self):
        raise NotImplementedError("Subclasses should implement this!")

    def _after_setup(self):
        pass

    def run(self):
        self._setup()
        self._after_setup()
        with Timer("Entire run") as timer:
            for i in range(self.iterations):
                with self.get_timer("all"):
                    self._run_iteration(i)
        return timer.get_avg_time()

    @timed
    def _run_iteration(self, i):
        raise NotImplementedError("Subclasses should implement this!")

    def maybe_print_timers(self, i):
        if (i + 1) % 100 == 0 and self.print_timers:
            print('\n')
            self.print_timers_report()


class MappingRunner(Runner):

    MAPPING_CLASS = HomographyMapping
    def __init__(self, iterations=5000, print_timers=True, registration_grid=10,
                    mapping_filepath="mapping.pkl", load_mapping=None, save_mapping=None):
        super().__init__(iterations, print_timers)
        self.registration_grid = registration_grid
        self.mapping_filepath = mapping_filepath
        self.load_mapping = load_mapping
        self.save_mapping = save_mapping
        self.mapping = None

    def _setup_matplotlib(self):
        set_matplotlib_backend()

    def _setup_mapping(self):
        self.backlight_image_size = get_backlight_image_size()
        self.camera_image_size = get_overhead_camera().get_resolution()
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

    def _map_camera_image_to_screen_image(self, camera_image):
        return self.mapping.map_camera_image_to_screen_image(
            camera_image, self.backlight_image_size[::-1])

    def _setup(self):
        self._setup_matplotlib()
        self._setup_mapping()

    def _after_setup(self):
        beep()


class CameraScreenRunner(MappingRunner):
    BACKGROUND_COLOR = (255, 255, 255)

    def __init__(self, iterations=5000, print_timers=True, registration_grid=10,
                    mapping_filepath="mapping.pkl", load_mapping=None, save_mapping=None, show_camera=False,
                    use_blitting=True, refresh_together=True):
        super().__init__(iterations, print_timers, registration_grid, mapping_filepath, load_mapping, save_mapping)
        self.show_camera = show_camera
        self.use_blitting = use_blitting
        self.refresh_together = refresh_together

    def _setup_display(self):
        self.backlight_bgd_image = self._get_bgd_image()
        self.backlight_ax, self.backlight_img = set_backlight_image(self.backlight_bgd_image, pause=1)
        self.backlight_fig = self.backlight_ax.figure
        self.backlight_background = capture_background_for_bliting(self.backlight_fig)
        plt.show(block=False)

    def _get_bgd_image(self):
        img = np.zeros((self.backlight_image_size[0], self.backlight_image_size[1], 3), dtype=np.uint8)
        img[:, :] = self.BACKGROUND_COLOR
        return img

    def _get_initial_frames(self):
        initial_frames = []
        if self.show_camera:
            initial_frames.append(self.camera_initial_frame)
        return initial_frames

    def _setup_camera(self):
        self.camera = get_overhead_camera()
        self.camera_initial_frame = self.camera.take_picture()
        self.camera_detection_image = np.zeros((self.camera_initial_frame.shape[:2]), dtype=np.uint8)

        # Setup camera display axes and images for blitting
        self.camera_axs = {}
        self.camera_imgs = {}
        self.camera_backgrounds = {}

        for index, img in enumerate(self._get_initial_frames()):
            ax, img = set_camera_display_image(img, index=index, pause=0.1)
            self.camera_axs[index] = ax
            self.camera_imgs[index] = img
            self.camera_backgrounds[index] = capture_background_for_bliting(ax.figure)

    def _setup(self):
        super()._setup()
        self._setup_display()
        self._setup_camera()

    def _run_iteration(self, i):
        self._refresh_funcs = []

    @Runner.timed
    def refresh_all(self):
        for refresh_func in self._refresh_funcs:
            refresh_func()
        if not self.use_blitting:
            plt.draw()
            plt.pause(0.001)

    @Runner.timed
    def take_picture(self):
        return self.camera.take_picture()

    @staticmethod
    def collect_refresh(func):
        def wrapper(self, *args, **kwargs):
            result = func(self, *args, **kwargs)
            if result is not None:
                self._refresh_funcs.append(result)
            return result
        return wrapper

    @Runner.timed
    @collect_refresh
    def update_backlight_image(self, backlight_image):
        return update_image(self.backlight_ax, self.backlight_img, backlight_image, self.backlight_background,
                            self.use_blitting, refresh_now=not self.refresh_together)

    @Runner.timed
    @collect_refresh
    def maybe_show_camera(self, frame):
        if self.show_camera:
            return update_image(self.camera_axs[0], self.camera_imgs[0], frame, self.camera_backgrounds[0],
                                self.use_blitting, refresh_now=not self.refresh_together)


class ShadowRunner(CameraScreenRunner):
    SHADOW_COLOR = (255, 0, 0)
    TEXT_COLOR = (200, 200, 255)
    TEXT = "Shadow-X"

    def __init__(self, iterations=5000, show_camera=False, show_detection=False, smoothing=False,
                 print_timers=True, registration_grid=12, use_blitting=True,
                 mapping_filepath="mapping.pkl", load_mapping=None, save_mapping=None,
                 refresh_together=True):
        super().__init__(iterations, print_timers, registration_grid, mapping_filepath, load_mapping, save_mapping)
        self.show_camera = show_camera
        self.show_detection = show_detection
        self.smoothing = smoothing
        self.use_blitting = use_blitting
        self.refresh_together = refresh_together

    def _get_bgd_image(self):
        img = super()._get_bgd_image()
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

    def _get_initial_frames(self):
        initial_frames = super()._get_initial_frames()
        if self.show_detection:
            initial_frames.append(np.zeros_like(self.camera_initial_frame[:, :, 0], dtype=np.uint8))
        return initial_frames

    def _run_iteration(self, i):
        super()._run_iteration(i)
        frame = self.take_picture()
        self.maybe_show_camera(frame)
        detection_mask = self.detect(frame)
        detection_mask = np.random.rand(*detection_mask.shape) < 0.01
        detection_mask = self.adjust_detection_mask(detection_mask)
        self.maybe_show_detection_mask(detection_mask)
        screen_image = self.map_to_screen(detection_mask)
        self.update_backlight_image(screen_image)
        self.refresh_all()

        self.maybe_print_timers(i)

    # --- timed helpers ---

    def _detect_from_diff(self, diff_image):
        return diff_image[:, :, 0] > 60

    @Runner.timed
    def detect(self, frame):
        return self._detect_from_diff(subtract_images(self.camera_initial_frame, frame))

    @Runner.timed
    def adjust_detection_mask(self, detection_mask):
        if self.smoothing:
            return cv2.blur(detection_mask.astype(np.float32), (5, 5)) > 0.1
        return detection_mask

    @Runner.timed
    @CameraScreenRunner.collect_refresh
    def maybe_show_detection_mask(self, detection_mask):
        if self.show_detection:
            return update_image(self.camera_axs[1], self.camera_imgs[1], detection_mask * 255, self.camera_backgrounds[1], self.use_blitting, refresh_now=not self.refresh_together)

    @Runner.timed
    def map_to_screen(self, detection_mask):
        detection_mask_on_screen = self._map_camera_image_to_screen_image(detection_mask)
        backlight_image = self.backlight_bgd_image.copy()
        backlight_image[detection_mask_on_screen] = self.SHADOW_COLOR
        return backlight_image


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
    ShadowRunner(show_camera=False, show_detection=False, use_blitting=False,
                     refresh_together=False,
                     iterations=101, save_mapping=None, load_mapping=None).run()
    # print(f"{'show_camera':<12} {'use_blitting':<13} {'refresh_together':<16} {'time':<8}")
    # print("-" * 70)
    # for show_camera in [True, False]:
    #         for use_blitting in [True, False]:
    #             for refresh_together in [True, False]:
    #                 t = ShadowRunner(show_camera=show_camera, show_detection=show_camera, use_blitting=use_blitting, refresh_together=refresh_together,
    #                              iterations=101, save_mapping=None, load_mapping=None).run()
    #                 print(f"{str(show_camera):<12} {str(use_blitting):<13} {str(refresh_together):<16} {t:<8.3f}")
    #                 beep(frequency=1000, duration=0.1)
    #                 time.sleep(0.1)