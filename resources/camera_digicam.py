"""
Canon EOS Camera adapter for shadow-x using digiCamControl.
This adapter wraps CanonCameraController to provide the same interface as the Camera class.
"""
import cv2
import numpy as np
import time
import atexit
from typing import Optional

try:
    import sys
    import os
    # Add NewMacroscope to path to import CanonCameraController
    newmacroscope_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'NewMacroscope')
    if os.path.exists(newmacroscope_path) and newmacroscope_path not in sys.path:
        sys.path.insert(0, newmacroscope_path)
    
    from camera_controller_digicam import CanonCameraController  # type: ignore
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
        self._liveview_just_started = False
        
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
                # Register cleanup handler to ensure liveview stops on exit
                atexit.register(self.release)
                # Start live view if available
                self._canon_controller.start_liveview()
                # Wait for liveview to be ready
                if self._wait_for_liveview_ready():
                    # Extra safety delay to ensure camera is fully ready
                    time.sleep(0.5)
                self._liveview_just_started = True
                return True
            return False
        except Exception as e:
            print(f"Failed to connect to digiCamControl: {e}")
            return False
    
    def _wait_for_liveview_ready(self, max_attempts=10, delay=0.5, min_brightness=10):
        """
        Wait for liveview to be ready by attempting to fetch properly exposed images.
        
        Args:
            max_attempts: Maximum number of attempts to check liveview
            delay: Delay between attempts in seconds
            min_brightness: Minimum mean brightness to consider image valid (0-255)
        """
        import requests
        liveview_url = f"{self.base_url}/liveview.jpg"
        
        print("Waiting for liveview to be ready...", end="", flush=True)
        for attempt in range(max_attempts):
            try:
                response = requests.get(liveview_url, timeout=2)
                if response.status_code == 200:
                    # Try to decode to verify it's a valid image
                    img_array = np.frombuffer(response.content, dtype=np.uint8)
                    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    if frame is not None and frame.size > 0:
                        # Check if image has reasonable brightness (not black)
                        mean_brightness = np.mean(frame)
                        if mean_brightness >= min_brightness:
                            print(f" ready! (brightness: {mean_brightness:.1f})")
                            return True
                        else:
                            print(f"[{mean_brightness:.0f}]", end="", flush=True)
            except Exception:
                pass
            
            print(".", end="", flush=True)
            time.sleep(delay)
        
        print(" timeout (continuing anyway)")
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
        
        # If liveview just started, discard first frame to avoid black images
        if self._liveview_just_started:
            print("Discarding first frame after liveview start...")
            self._fetch_frame()  # Discard this frame
            time.sleep(0.3)  # Give camera time to refresh
            self._liveview_just_started = False
        
        return self._fetch_frame()
    
    def _fetch_frame(self) -> np.ndarray:
        """
        Internal method to fetch a single frame from liveview.
        
        Returns:
            numpy array: Camera frame
        """
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
        """Release camera resources and stop liveview."""
        if self._canon_controller and self._connected:
            try:
                print("Stopping liveview and disconnecting camera...")
                self._canon_controller.stop_liveview()
                self._canon_controller.disconnect()
                print("Camera disconnected.")
            except Exception as e:
                print(f"Warning during camera release: {e}")
            finally:
                self._connected = False
                self._canon_controller = None
                # Unregister atexit handler to prevent double cleanup
                try:
                    atexit.unregister(self.release)
                except Exception:
                    pass
    
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

