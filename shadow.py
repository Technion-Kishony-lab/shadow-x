import time
import cv2
import numpy as np

from camera import Camera
from graphics.helpers import subtract_images, beep
from graphics.image_figure import ImageFigure
from runners import MappingRunner, timed


class ShadowRunner(MappingRunner):
    SHADOW_COLOR = (255, 0, 0)
    TEXT_COLOR = (200, 200, 255)
    TEXT = "Shadow-X"

    def __init__(self,
                 iterations=5000, print_timers=True,
                 camera: Camera = None, backlight_screen: ImageFigure = None,
                 show_camera=False, use_blitting=True, refresh_together=True,
                 registration_grid=10,
                 mapping_filepath="mapping.pkl", load_mapping=None, save_mapping=None,
                 show_detection=False, detection_kwargs=None, mask_adjustment_kwargs=None):
        super().__init__(iterations, print_timers, camera, backlight_screen,
                         show_camera, use_blitting, refresh_together,
                         registration_grid, mapping_filepath, load_mapping, save_mapping)
        self.show_detection = show_detection
        self.detection_kwargs = detection_kwargs if detection_kwargs is not None else {'threshold': 100}
        self.mask_adjustment_kwargs = mask_adjustment_kwargs if mask_adjustment_kwargs is not None else {}

    def _get_bgd_image(self):
        img = super()._get_bgd_image()
        size = img.shape[1::-1]
        cv2.putText(
            img=img,
            text=self.TEXT,
            org=(size[1] // 12, size[0] // 2),
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
        self.maybe_show_detection_mask(detection_mask, index=1)
        detection_mask = self.adjust_detection_mask(detection_mask)
        screen_image = self.map_to_screen(detection_mask)
        self.update_backlight_image(screen_image)

    # --- timed helpers ---

    def _detect_from_diff(self, diff_image):
        return diff_image[:, :, 0] > self.detection_kwargs['threshold']

    @timed
    def detect(self, frame):
        return self._detect_from_diff(subtract_images(self.camera_initial_frame, frame))

    @timed
    def adjust_detection_mask(self, detection_mask):
        return detection_mask

    @timed
    @MappingRunner.collect_refresh
    def maybe_show_detection_mask(self, detection_mask, index=1):
        if self.show_detection:
            return self.camera_figures[index].update_image(
                (detection_mask * 255).astype(np.uint8), self.use_blitting, refresh_now=not self.refresh_together)

    @timed
    def map_to_screen(self, detection_mask):
        detection_mask_on_screen = self._map_camera_image_to_screen_image(detection_mask)
        backlight_image = self.backlight_bgd_image.copy()
        backlight_image[detection_mask_on_screen] = self.SHADOW_COLOR
        return backlight_image


def run_options():
    print(f"{'show_camera':<12} {'use_blitting':<13} {'refresh_together':<16} {'time':<8}")
    print("-" * 70)
    for show_camera in [True, False]:
        for use_blitting in [True, False]:
            for refresh_together in [True, False]:
                t = ShadowRunner(show_camera=show_camera, show_detection=show_camera, use_blitting=use_blitting,
                                 refresh_together=refresh_together,
                                 iterations=11, save_mapping=None, load_mapping=None).run()
                print(f"{str(show_camera):<12} {str(use_blitting):<13} {str(refresh_together):<16} {t:<8.3f}")
                beep(frequency=1000, duration=0.1)
                time.sleep(0.1)


if __name__ == "__main__":
    ShadowRunner(show_camera=False, show_detection=False,
                 iterations=5000, save_mapping=None, load_mapping=None).run()
