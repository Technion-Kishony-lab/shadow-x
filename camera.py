from dataclasses import dataclass
from typing import Optional

import cv2
import time
from datetime import datetime


@dataclass
class Camera:
    index: int
    fps: int
    buffer_size: int
    rotate: int = None
    cap: Optional[cv2.VideoCapture] = None

    @classmethod
    def create(cls, index=0, fps=30, buffer_size=1, rotate=None):
        self = cls(index=index, fps=fps, buffer_size=buffer_size, rotate=rotate)
        self.get_capture()  # Initialize the capture
        return self

    def get_capture(self):
        if self.cap is not None:
            return self.cap
        cap = cv2.VideoCapture(self.index)
        time.sleep(1)
        if not cap.isOpened():
            raise Exception(f"Error: Could not open camera {self.index}")
        self.cap = cap
        self.set_properties()
        return cap

    def set_properties(self):
        cap = self.get_capture()
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)

    def take_picture(self):
        cap = self.get_capture()
        ret, frame = cap.read()
        if not ret:
            raise Exception("Error: Could not read frame from camera")
        if self.rotate:
            frame = cv2.rotate(frame, self.rotate)
        return frame

    def get_resolution(self):
        cap = self.get_capture()
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return height, width

    def take_picture_and_save(self, filename=None):
        frame = self.take_picture()
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"camera_{self.index}_{timestamp}.png"
        cv2.imwrite(filename, frame)
        return frame, filename

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None


INDEX_TO_CAMERAS: dict[int, Camera] = {}


def get_or_create_camera(camera_index=0, fps=30, buffer_size=1, rotate=0) -> Camera:
    if camera_index not in INDEX_TO_CAMERAS:
        INDEX_TO_CAMERAS[camera_index] = Camera.create(
            index=camera_index,
            fps=fps,
            buffer_size=buffer_size,
            rotate=rotate
        )

    return INDEX_TO_CAMERAS[camera_index]
