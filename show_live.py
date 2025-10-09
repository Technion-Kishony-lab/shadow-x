from graphics.helpers import set_matplotlib_backend
from runners import CameraScreenRunner

set_matplotlib_backend()


class LiveRunner(CameraScreenRunner):
    def _do_iteration(self, i):
        frame = self.take_picture()
        self.maybe_show_camera(frame)


if __name__ == '__main__':
    LiveRunner(show_camera=True).run()
