import cv2
import time

from dataclasses import dataclass
from typing import Optional
from datetime import datetime


@dataclass
class Camera:
    index: int
    fps: int
    buffer_size: int
    rotate: int = None
    color_order: Optional[str] = None
    exposure: Optional[float] = None
    _cap: Optional[cv2.VideoCapture] = None

    @classmethod
    def create(cls, index=0, fps=30, buffer_size=1, rotate=None, color_order=None, exposure=None):
        self = cls(index=index, fps=fps, buffer_size=buffer_size, rotate=rotate, color_order=color_order,
                   exposure=exposure)
        self.get_capture()  # Initialize the capture
        return self

    def get_capture(self):
        if self._cap is not None:
            return self._cap
        cap = cv2.VideoCapture(self.index)
        time.sleep(1)
        if not cap.isOpened():
            raise Exception(f"Error: Could not open camera {self.index}")
        self._cap = cap
        self.set_properties()
        return cap

    def set_properties(self):
        cap = self.get_capture()
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
        if self.exposure is not None:
            cap.set(cv2.CAP_PROP_EXPOSURE, float(self.exposure))

    def take_picture(self):
        cap = self.get_capture()
        ret, frame = cap.read()
        if not ret:
            raise Exception("Error: Could not read frame from camera")
        if self.rotate:
            frame = cv2.rotate(frame, self.rotate)
        if self.color_order in ('RGB', 'BGR'):
            if self.color_order == 'BGR':
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def get_resolution(self):
        cap = self.get_capture()
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return width, height

    def take_picture_and_save(self, filename=None):
        frame = self.take_picture()
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"camera_{self.index}_{timestamp}.png"
        cv2.imwrite(filename, frame)
        return frame, filename

    def release(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None


INDICES_TO_CAMERAS: dict[int, Camera] = {}


def get_or_create_camera(camera_index=0, fps=30, buffer_size=1, rotate=0, color_order=None, exposure=None) -> Camera:
    if camera_index not in INDICES_TO_CAMERAS:
        INDICES_TO_CAMERAS[camera_index] = Camera.create(
            index=camera_index,
            fps=fps,
            buffer_size=buffer_size,
            rotate=rotate,
            color_order=color_order,
            exposure=exposure
        )

    return INDICES_TO_CAMERAS[camera_index]
