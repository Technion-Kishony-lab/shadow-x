import numpy as np

from matplotlib import pyplot as plt

from camera import Camera
from graphics.helpers import beep
from graphics.image_figure import ImageFigure
from mapping import HomographyMapping, Mapping

from register import register_screen_camera
from resources import get_or_create_backlight_screen, create_camera_figure, \
    set_matplotlib_backend, get_overhead_camera

from timers import Timer


def timed(func):
    def wrapper(self, *args, **kwargs):
        with self.get_timer(func.__name__):
            return func(self, *args, **kwargs)

    return wrapper


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


class CameraScreenRunner(Runner):
    BACKGROUND_COLOR = (255, 255, 255)

    def __init__(self,
                 iterations=5000, print_timers=True,
                 camera: Camera = None, backlight_screen: ImageFigure = None,
                 show_camera=False, use_blitting=True, refresh_together=True):
        super().__init__(iterations, print_timers)
        self.camera = camera if camera is not None else get_overhead_camera()
        self.backlight_screen = backlight_screen if backlight_screen is not None else get_or_create_backlight_screen()
        self.show_camera = show_camera
        self.use_blitting = use_blitting
        self.refresh_together = refresh_together
        self.camera_displays = {}

    def _setup_display(self):
        self.backlight_bgd_image = self._get_bgd_image()
        self.backlight_screen.set_image(self.backlight_bgd_image, pause=1)

    def _get_bgd_image(self):
        size = self.backlight_screen.get_recomended_image_size()
        img = np.zeros((size[0], size[1], 3), dtype=np.uint8)
        img[:, :] = self.BACKGROUND_COLOR
        return img

    def _setup_camera(self):
        self.camera_initial_frame = self.camera.take_picture()

    def get_camera_display(self, index=0):
        if index not in self.camera_displays:
            self.camera_displays[index] = create_camera_figure(index=index)
        return self.camera_displays[index]

    def _setup_matplotlib(self):
        set_matplotlib_backend()

    def _setup(self):
        self._setup_matplotlib()
        self._setup_display()
        self._setup_camera()

    def _start_iteration(self, i):
        super()._start_iteration(i)
        self._refresh_funcs = []

    def _end_iteration(self, i):
        super()._end_iteration(i)
        self.refresh_all()

    @timed
    def refresh_all(self):
        for refresh_func in self._refresh_funcs:
            refresh_func()
        if not self.use_blitting:
            plt.pause(0.001)

    @timed
    def take_picture(self):
        return self.camera.take_picture()

    @staticmethod
    def collect_refresh(func):
        def wrapper(self, *args, **kwargs):
            result = func(self, *args, **kwargs)
            if result is not None:
                self._refresh_funcs.append(result)
            return result
        wrapper.__name__ = func.__name__
        return wrapper

    @timed
    @collect_refresh
    def update_backlight_image(self, backlight_image):
        return self.backlight_screen.update_image(backlight_image, self.use_blitting,
                                                  refresh_now=not self.refresh_together)

    @timed
    @collect_refresh
    def maybe_show_camera(self, frame):
        if self.show_camera:
            return self.get_camera_display().update_image(frame, self.use_blitting, refresh_now=not self.refresh_together)


class MappingRunner(CameraScreenRunner):
    MAPPING_CLASS = HomographyMapping

    def __init__(self,
                 iterations=5000, print_timers=True,
                 camera: Camera = None, backlight_screen: ImageFigure = None,
                 show_camera=False, use_blitting=True, refresh_together=True,
                 registration_grid=10,
                 mapping_filepath="mapping.pkl", load_mapping=None, save_mapping=None):
        super().__init__(iterations, print_timers, camera, backlight_screen,
                         show_camera, use_blitting, refresh_together)
        self.registration_grid = registration_grid
        self.mapping_filepath = mapping_filepath
        self.load_mapping = load_mapping
        self.save_mapping = save_mapping
        self.mapping = None

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
                self.camera, self.backlight_screen, self.show_camera and self.get_camera_display(),
                self.registration_grid, mapping_class=self.MAPPING_CLASS)
            if mapping is None:
                raise RuntimeError("Mapping registration failed.")
            if self.save_mapping is not False:
                mapping.save_mapping(self.mapping_filepath)
        self.mapping = mapping

    def _map_camera_image_to_screen_image(self, camera_image):
        return self.mapping.map_camera_image_to_screen_image(
            camera_image, self.backlight_screen.get_image_size()[1::-1])

    def _setup(self):
        self._setup_matplotlib()
        self._setup_mapping()
        self._setup_display()
        self._setup_camera()

    def _after_setup(self):
        beep()
