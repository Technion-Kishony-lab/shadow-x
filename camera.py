import cv2
import time
from datetime import datetime


INDEX_TO_CAMERAS = {}


def get_camera(camera_index=0, fps=30, buffer_size=1):
    """
    Get a camera object.
    
    Args:
        camera_index (int): Index of the camera to use.
        fps (int): Frames per second.
        buffer_size (int): Buffer size.
    """
    cap = cv2.VideoCapture(camera_index)
    
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)
    
    time.sleep(1)  # Give camera time to initialize
    if not cap.isOpened():
        raise Exception(f"Error: Could not open camera {camera_index}")
    return cap


def get_or_create_camera(camera_index=0, fps=30, buffer_size=1):
    if camera_index not in INDEX_TO_CAMERAS:
        INDEX_TO_CAMERAS[camera_index] = get_camera(camera_index=camera_index, fps=fps, buffer_size=buffer_size)

    return INDEX_TO_CAMERAS[camera_index]


def take_picture(cap, exposure=None):
 
    if not cap.isOpened():
        raise Exception(f"Error: Could not open camera")
    
    # Capture frame
    ret, frame = cap.read()
    return frame


def take_picture_and_save(cap, filename=None):
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"photo_{timestamp}.jpg"

    frame = take_picture(cap=cap)
    success = cv2.imwrite(filename, frame)
    
    if not success:
        raise Exception("Error: Could not save image")
    return filename
