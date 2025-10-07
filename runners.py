import argparse
import time
import cv2
import numpy as np

from matplotlib import pyplot as plt

from camera import Camera
from graphics.helpers import beep, subtract_images
from graphics.image_figure import ImageFigure
from mapping import HomographyMapping, Mapping

from register import register_screen_camera
from resources import get_or_create_backlight_screen, get_or_create_camera_figure, \
    set_matplotlib_backend, get_overhead_camera

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
        raise NotImplementedError

    def _after_setup(self):
        pass

    def run(self):
        self._setup()
        self._after_setup()
        with Timer("Entire run") as timer:
            for i in range(self.iterations):
                self._run_iteration(i)
        return timer.get_avg_time()

    @timed
    def _run_iteration(self, i):
        self._start_iteration(i)
        self._do_iteration(i)
        self._end_iteration(i)

    def _start_iteration(self, i):
        pass

    def _do_iteration(self, i):
        raise NotImplementedError

    def _end_iteration(self, i):
        self.maybe_print_timers(i)

    def maybe_print_timers(self, i):
        if (i + 1) % 100 == 0 and self.print_timers:
            print('\n')
            self.print_timers_report()


class MappingRunner(Runner):
    MAPPING_CLASS = HomographyMapping

    def __init__(self, iterations=5000, print_timers=True,
                 camera: Camera = None, backlight_screen: ImageFigure = None,
                 registration_grid=10,
                 mapping_filepath="mapping.pkl", load_mapping=None, save_mapping=None):
        super().__init__(iterations, print_timers)
        self.camera = camera if camera is not None else get_overhead_camera()
        self.backlight_screen = backlight_screen if backlight_screen is not None else get_or_create_backlight_screen()
        self.registration_grid = registration_grid
        self.mapping_filepath = mapping_filepath
        self.load_mapping = load_mapping
        self.save_mapping = save_mapping
        self.mapping = None

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
            mapping = register_screen_camera(
                self.camera, self.backlight_screen,
                self.registration_grid, display=True, mapping_class=self.MAPPING_CLASS)
            if self.save_mapping is not False:
                mapping.save_mapping(self.mapping_filepath)
        self.mapping = mapping

    def _map_camera_image_to_screen_image(self, camera_image):
        return self.mapping.map_camera_image_to_screen_image(
            camera_image, self.backlight_screen.get_image_size()[1::-1])

    def _setup(self):
        self._setup_matplotlib()
        self._setup_mapping()

    def _after_setup(self):
        beep()


class CameraScreenRunner(MappingRunner):
    BACKGROUND_COLOR = (255, 255, 255)

    def __init__(self,
                 iterations=5000, print_timers=True,
                 camera: Camera = None, backlight_screen: ImageFigure = None,
                 registration_grid=10,
                 mapping_filepath="mapping.pkl", load_mapping=None, save_mapping=None,
                 show_camera=False, use_blitting=True, refresh_together=True):
        super().__init__(iterations, print_timers,camera, backlight_screen, registration_grid,
                         mapping_filepath, load_mapping, save_mapping)
        self.show_camera = show_camera
        self.use_blitting = use_blitting
        self.refresh_together = refresh_together

    def _setup_display(self):
        self.backlight_bgd_image = self._get_bgd_image()
        self.backlight_screen.set_image(self.backlight_bgd_image, pause=1)
        self.backlight_screen.capture_background_for_bliting()

    def _get_bgd_image(self):
        size = self.backlight_screen.get_recomended_image_size()
        img = np.zeros((size[0], size[1], 3), dtype=np.uint8)
        img[:, :] = self.BACKGROUND_COLOR
        return img

    def _get_initial_frames(self):
        initial_frames = []
        if self.show_camera:
            initial_frames.append(self.camera_initial_frame)
        return initial_frames

    def _setup_camera(self):
        self.camera_initial_frame = self.camera.take_picture()

        # Setup camera display axes and images for blitting
        self.camera_figures = {}

        for index, img in enumerate(self._get_initial_frames()):
            cam_figure = get_or_create_camera_figure(index)
            cam_figure.set_image(img, allow_resize=True, pause=0.1)
            cam_figure.capture_background_for_bliting()
            self.camera_figures[index] = cam_figure

    def _setup(self):
        super()._setup()
        self._setup_display()
        self._setup_camera()

    def _start_iteration(self, i):
        super()._start_iteration(i)
        self._refresh_funcs = []

    def _end_iteration(self, i):
        super()._end_iteration(i)
        self.refresh_all()

    @Runner.timed
    def refresh_all(self):
        for refresh_func in self._refresh_funcs:
            refresh_func()
        if not self.use_blitting:
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
        return self.backlight_screen.update_image(backlight_image, self.use_blitting, refresh_now=not self.refresh_together)

    @Runner.timed
    @collect_refresh
    def maybe_show_camera(self, frame):
        if self.show_camera:
            return self.camera_figures[0].update_image(frame, self.use_blitting, refresh_now=not self.refresh_together)


class ShadowRunner(CameraScreenRunner):
    SHADOW_COLOR = (255, 0, 0)
    TEXT_COLOR = (200, 200, 255)
    TEXT = "Shadow-X"

    def __init__(self,
                 iterations=5000, print_timers=True,
                 camera: Camera = None, backlight_screen: ImageFigure = None,
                 registration_grid=10,
                 mapping_filepath="mapping.pkl", load_mapping=None, save_mapping=None,
                 show_camera=False, use_blitting=True, refresh_together=True,
                    show_detection=False, smoothing=False):
        super().__init__(
            iterations, print_timers, camera, backlight_screen, registration_grid,
            mapping_filepath, load_mapping, save_mapping,
            show_camera, use_blitting, refresh_together
        )
        self.show_detection = show_detection
        self.smoothing = smoothing

    def _get_bgd_image(self):
        img = super()._get_bgd_image()
        size = img.shape[1::-1]
        cv2.putText(
            img=img,
            text=self.TEXT,
            org=(size[0] // 12, size[1] // 2),
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

    def _do_iteration(self, i):
        frame = self.take_picture()
        self.maybe_show_camera(frame)
        detection_mask = self.detect(frame)
        detection_mask = self.adjust_detection_mask(detection_mask)
        self.maybe_show_detection_mask(detection_mask)
        screen_image = self.map_to_screen(detection_mask)
        self.update_backlight_image(screen_image)

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
            return self.camera_figures[1].update_image(
                (detection_mask * 255).astype(np.uint8), self.use_blitting, refresh_now=not self.refresh_together)

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
    print(f"{'show_camera':<12} {'use_blitting':<13} {'refresh_together':<16} {'time':<8}")
    print("-" * 70)
    for show_camera in [True, False]:
        for use_blitting in [True, False]:
            for refresh_together in [True, False]:
                t = ShadowRunner(show_camera=show_camera, show_detection=show_camera, use_blitting=use_blitting,
                                 refresh_together=refresh_together,
                                 camera=get_overhead_camera(),
                                 iterations=11, save_mapping=None, load_mapping=None).run()
                print(f"{str(show_camera):<12} {str(use_blitting):<13} {str(refresh_together):<16} {t:<8.3f}")
                beep(frequency=1000, duration=0.1)
                time.sleep(0.1)
