"""
Canon EOS Camera adapter for shadow-x using digiCamControl.
This adapter wraps CanonCameraController to provide the same interface as the Camera class.
"""
import cv2
import numpy as np
import time
from typing import Optional

try:
    import sys
    import os
    # Add NewMacroscope to path to import CanonCameraController
    newmacroscope_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'NewMacroscope')
    if os.path.exists(newmacroscope_path) and newmacroscope_path not in sys.path:
        sys.path.insert(0, newmacroscope_path)
    
    from camera_controller_digicam import CanonCameraController
    DIGICAM_AVAILABLE = True
except ImportError:
    DIGICAM_AVAILABLE = False
    CanonCameraController = None


class DigiCamCamera:
    """
    Camera adapter that uses CanonCameraController (digiCamControl) 
    but provides the same interface as shadow-x's Camera class.
    """
    
    def __init__(self, 
                 rotate=None, 
                 color_order=None,
                 base_url: str = "http://127.0.0.1:5514",
                 resolution: Optional[tuple] = None):
        """
        Initialize the DigiCamCamera adapter.
        
        Args:
            rotate: Rotation to apply (cv2.ROTATE_90_CLOCKWISE, etc.)
            color_order: Color order ('BGR' or 'RGB')
            base_url: digiCamControl web server URL
            resolution: Optional (width, height) tuple if known
        """
        if not DIGICAM_AVAILABLE:
            raise ImportError(
                "CanonCameraController not available. "
                "Make sure NewMacroscope/camera_controller_digicam.py is accessible."
            )
        
        self.rotate = rotate
        self.color_order = color_order
        self.base_url = base_url
        self._resolution = resolution
        self._canon_controller = None
        self._connected = False
        
        # These are kept for compatibility with Camera interface
        self.index = None  # Not used for digiCamControl
        self.fps = 30  # Not directly controllable via digiCamControl
        self.buffer_size = 1
        self.exposure = None
    
    def connect(self) -> bool:
        """Connect to camera via digiCamControl."""
        if self._connected:
            return True
        
        try:
            self._canon_controller = CanonCameraController()
            self._canon_controller.base_url = self.base_url
            if self._canon_controller.connect():
                self._connected = True
                # Start live view if available
                self._canon_controller.start_liveview()
                return True
            return False
        except Exception as e:
            print(f"Failed to connect to digiCamControl: {e}")
            return False
    
    def get_capture(self):
        """Compatibility method - returns self for digiCamControl."""
        if not self._connected:
            self.connect()
        return self
    
    def set_properties(self):
        """Set properties - not directly applicable to digiCamControl."""
        pass
    
    def take_picture(self) -> np.ndarray:
        """
        Get current frame from digiCamControl liveview.
        
        Returns:
            numpy array: Current camera frame
        """
        if not self._connected:
            self.connect()
        
        if not self._connected:
            raise Exception("Not connected to digiCamControl")
        
        try:
            # Get liveview image from digiCamControl
            import requests
            liveview_url = f"{self.base_url}/liveview.jpg"
            
            response = requests.get(liveview_url, timeout=5)
            if response.status_code != 200:
                raise Exception(f"Failed to get liveview: HTTP {response.status_code}")
            
            # Convert response to numpy array
            img_array = np.frombuffer(response.content, dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if frame is None:
                raise Exception("Failed to decode liveview image")
            
            # Apply rotation if needed
            if self.rotate:
                frame = cv2.rotate(frame, self.rotate)
            
            # Apply color order conversion if needed
            if self.color_order == 'RGB':
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            elif self.color_order == 'BGR':
                # Already BGR from cv2.imdecode
                pass
            
            return frame
            
        except Exception as e:
            raise Exception(f"Error getting frame from digiCamControl: {e}")
    
    def get_resolution(self) -> tuple:
        """
        Get camera resolution.
        
        Returns:
            tuple: (width, height)
        """
        if self._resolution:
            return self._resolution
        
        # Try to get resolution from a frame
        try:
            frame = self.take_picture()
            height, width = frame.shape[:2]
            self._resolution = (width, height)
            return self._resolution
        except:
            # Default resolution if we can't determine it
            return (1920, 1080)
    
    def take_picture_and_save(self, filename=None):
        """Take picture and save to file."""
        frame = self.take_picture()
        if filename is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"digicam_{timestamp}.png"
        cv2.imwrite(filename, frame)
        return frame, filename
    
    def release(self):
        """Release camera resources."""
        if self._canon_controller and self._connected:
            try:
                self._canon_controller.stop_liveview()
                self._canon_controller.disconnect()
            except:
                pass
            self._connected = False
            self._canon_controller = None
    
    @classmethod
    def create(cls, index=None, fps=30, buffer_size=1, rotate=None, 
               color_order=None, exposure=None, base_url="http://127.0.0.1:5514",
               resolution=None):
        """
        Create a DigiCamCamera instance.
        
        Note: index, fps, buffer_size, exposure are kept for compatibility
        but don't directly apply to digiCamControl.
        """
        return cls(rotate=rotate, color_order=color_order, 
                  base_url=base_url, resolution=resolution)


def get_or_create_digicam_camera(base_url="http://127.0.0.1:5514", 
                                 rotate=None, color_order=None,
                                 resolution=None):
    """
    Get or create a DigiCamCamera instance.
    
    Args:
        base_url: digiCamControl web server URL
        rotate: Rotation to apply
        color_order: Color order ('BGR' or 'RGB')
        resolution: Optional (width, height) tuple
        
    Returns:
        DigiCamCamera instance
    """
    # Use a simple cache key based on base_url
    cache_key = f"digicam_{base_url}"
    
    if not hasattr(get_or_create_digicam_camera, '_instances'):
        get_or_create_digicam_camera._instances = {}
    
    if cache_key not in get_or_create_digicam_camera._instances:
        get_or_create_digicam_camera._instances[cache_key] = DigiCamCamera.create(
            base_url=base_url, rotate=rotate, color_order=color_order,
            resolution=resolution
        )
    
    return get_or_create_digicam_camera._instances[cache_key]

