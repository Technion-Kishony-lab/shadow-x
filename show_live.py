from matplotlib import pyplot as plt

from resources import set_matplotlib_backend, get_overhead_camera, set_camera_display_image


class LiveRunner(Runner):
    def _setup(self):
        set_matplotlib_backend()
        self.camera = get_overhead_camera()

    def _run_iteration(self, i):
        frame = self.camera.take_picture()
        set_camera_display_image(frame)
        plt.pause(0.01)


if __name__ == "__main__":
    runner = LiveRunner(iterations=5000)
    runner.run()
