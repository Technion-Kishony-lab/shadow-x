import cv2
import os
import time
from datetime import datetime

def take_picture(filename=None, camera_index=0):
 
    cap = cv2.VideoCapture(camera_index)
    
    # Give camera time to initialize
    time.sleep(1)
    
    if not cap.isOpened():
        raise Exception(f"Error: Could not open camera {camera_index}") 
    
    # Capture frame
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise Exception("Error: Could not capture frame")
    
    # Generate filename if not provided
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"photo_{timestamp}.jpg"
    
    # Save image
    success = cv2.imwrite(filename, frame)
    
    if not success:
        raise Exception("Error: Could not save image")
    return filename


def list_cameras():
    """List available cameras"""
    cameras = []
    for i in range(10):  # Check first 10 camera indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cameras.append(i)
            cap.release()
    return cameras


if __name__ == "__main__":
    # List available cameras
    cameras = list_cameras()
    print("Available cameras:", cameras)
    
    # Try each camera
    for cam_index in cameras:
        print(f"\nTrying camera {cam_index}...")
        result = take_picture(f"test_camera_{cam_index}.jpg", cam_index)
        if result:
            print(f"Success with camera {cam_index}! Image saved to: {result}")
            break
    else:
        print("No working cameras found")
