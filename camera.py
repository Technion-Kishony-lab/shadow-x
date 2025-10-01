import cv2
import time
from datetime import datetime


def get_camera(camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    time.sleep(1)  # Give camera time to initialize
    if not cap.isOpened():
        raise Exception(f"Error: Could not open camera {camera_index}")
    return cap


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
