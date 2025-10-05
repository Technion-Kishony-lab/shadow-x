import os
import sys
import pickle
import numpy as np
import pytest
import cv2
from unittest.mock import patch, MagicMock

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from register import get_centers_on_screen_and_camera, get_diff_image, find_circles_grid
from mapping import HomographyMapping, PolynomialWarpMapping


THIS_FILE_DIR = os.path.dirname(__file__)

class TestDataManager:
    """Helper class to manage test data capture and loading"""
    
    def __init__(self, test_data_dir=os.path.join(THIS_FILE_DIR, "test_data")):
        self.test_data_dir = test_data_dir
        os.makedirs(test_data_dir, exist_ok=True)
    
    @property
    def image_data_path(self):
        return os.path.join(self.test_data_dir, "image_data.pkl")
    
    @property
    def mapping_data_path(self):
        return os.path.join(self.test_data_dir, "mapping_data.pkl")
    
    def save_image_data(self, centers_on_screen, image_size, diff_image, xs, ys):
        """Save image and screen centers data for test_get_centers_on_screen_and_camera"""
        data = {
            'centers_on_screen': centers_on_screen,
            'image_size': image_size,
            'diff_image': diff_image,
            'xs': xs,
            'ys': ys
        }
        
        filepath = self.image_data_path
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Image test data saved to {filepath}")
    
    def load_image_data(self):
        """Load image and screen centers data"""
        filepath = self.image_data_path
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    def save_mapping_data(self, centers_on_screen, centers_on_camera, image_size):
        """Save mapping data for test_mapping"""
        data = {
            'centers_on_screen': centers_on_screen,
            'centers_on_camera': centers_on_camera,
            'image_size': image_size
        }
        
        filepath = self.mapping_data_path
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Mapping test data saved to {filepath}")
    
    def load_mapping_data(self):
        """Load mapping data"""
        filepath = self.mapping_data_path
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    def capture_image_data(self, num_tile_rows=10):
        """Capture new image data and save it"""
        print(f"Capturing new image data with {num_tile_rows} tile rows...")
        diff_image, xs, ys, centers_on_screen = get_diff_image(num_tile_rows)
        image_size = diff_image.shape
        self.save_image_data(centers_on_screen, image_size, diff_image, xs, ys)

    def get_or_capture_image_data(self, num_tile_rows=10):
        """Return image data; capture and save if missing."""
        # Ensure directory exists
        os.makedirs(self.test_data_dir, exist_ok=True)
        # Prefer a direct file existence check
        if not os.path.exists(self.image_data_path):
            # Create file by capturing, then load to return a consistent structure
            self.capture_image_data(num_tile_rows)
        image_data = self.load_image_data()
        return (
            image_data['centers_on_screen'],
            image_data['image_size'],
            image_data['diff_image'],
            image_data['xs'],
            image_data['ys']
        )
    
    def capture_mapping_data(self, num_tile_rows=10):
        """Capture new mapping data and save it"""
        print(f"Capturing new mapping data with {num_tile_rows} tile rows...")
        image_data = self.load_image_data()
        if image_data is not None:
            diff_image = image_data['diff_image']
            centers_on_screen = image_data['centers_on_screen']
            image_size = image_data['image_size']
            xs = image_data['xs']
            ys = image_data['ys']
        else:
            diff_image, xs, ys, centers_on_screen = get_diff_image(num_tile_rows)
            image_size = diff_image.shape

        ret, centers_on_camera = find_circles_grid(diff_image, (len(ys), len(xs)))
        if not ret:
            raise ValueError("Failed to detect pattern - no camera centers found")

        self.save_mapping_data(centers_on_screen, centers_on_camera, image_size)

    def get_or_capture_mapping_data(self, num_tile_rows=10):
        """Return mapping data; capture and save if missing."""
        # Ensure directory exists
        os.makedirs(self.test_data_dir, exist_ok=True)
        # Prefer a direct file existence check
        if not os.path.exists(self.mapping_data_path):
            # Create file by capturing, then load to return a consistent structure
            self.capture_mapping_data(num_tile_rows)
        mapping_data = self.load_mapping_data()
        return (
            mapping_data['centers_on_screen'],
            mapping_data['centers_on_camera'],
            mapping_data['image_size']
        )


# Global test data manager
test_manager = TestDataManager()


def test_get_centers_on_screen_and_camera():
    """Test the get_centers_on_screen_and_camera function with data capture and reuse"""
    
    # Try to load existing image data first
    centers_on_screen, image_size, diff_image, xs, ys = test_manager.get_or_capture_image_data(10)
    
    # Assertions - only test what this test needs
    assert centers_on_screen is not None, "Screen centers should not be None"
    assert len(centers_on_screen) > 0, "Should have detected screen centers"
    assert centers_on_screen.shape[1] == 2, "Screen centers should be 2D points"
    assert len(image_size) == 2, "Image size should be (width, height)"
    assert diff_image.ndim == 2, "Diff image should be 2D"
    
    assert len(centers_on_screen) == len(xs) * len(ys), "centers_on_screen should match grid size"
    print(f"✓ Screen centers shape: {centers_on_screen.shape}")
    print(f"✓ Image size: {image_size}")
    print(f"✓ Diff image shape: {diff_image.shape}")


@pytest.mark.parametrize(
    "MappingClass, kwargs",
    [
        (HomographyMapping, {}),
        (PolynomialWarpMapping, {"degree": 3}),
    ],
)
def test_mapping(MappingClass, kwargs):
    """Test mapping functionality using real data for both mapping subclasses"""
    
    centers_on_screen, centers_on_camera, image_size = test_manager.get_or_capture_mapping_data(10)
    
    mapping = MappingClass.from_matching_points(
        centers_on_screen, centers_on_camera, image_size, **kwargs
    )
    
    # Test point mapping accuracy
    mapped_points = mapping.map_camera_points_to_screen_points(centers_on_camera)
    mean_error = mapping.calculate_accuracy(centers_on_screen, mapped_points)
    
    print(f"✓ {MappingClass.__name__} mean error: {mean_error:.2f} pixels")
    assert mean_error < 10.0, f"{MappingClass.__name__} error too high: {mean_error}"


@pytest.mark.parametrize(
    "MappingClass, kwargs",
    [
        (HomographyMapping, {}),
        (PolynomialWarpMapping, {"degree": 3}),
    ],
)
def test_image_mapping(MappingClass, kwargs):
    """Basic shape test for image warping to screen space for both mapping subclasses"""
    centers_on_screen, centers_on_camera, image_size = test_manager.get_or_capture_mapping_data(10)

    mapping = MappingClass.from_matching_points(
        centers_on_screen, centers_on_camera, image_size, **kwargs
    )

    dummy_camera_image = np.random.randint(0, 255, (image_size[1], image_size[0], 3), dtype=np.uint8)
    output_size = (1920, 1080)
    warped = mapping.map_camera_image_to_screen_image(dummy_camera_image, output_size)
    assert warped.shape[:2] == output_size[::-1], "Warped image should have correct output size"


@pytest.mark.parametrize(
    "MappingClass, kwargs",
    [
        (HomographyMapping, {}),
        (PolynomialWarpMapping, {"degree": 3}),
    ],
)
def test_image_mapping_sparse_markers(MappingClass, kwargs):
    """Verify spatial correctness by warping sparse encoded markers and checking positions."""
    centers_on_screen, centers_on_camera, image_size = test_manager.get_or_capture_mapping_data(10)

    mapping = MappingClass.from_matching_points(
        centers_on_screen, centers_on_camera, image_size, **kwargs
    )

    h, w = image_size[1], image_size[0]
    camera_img = np.zeros((h, w, 3), dtype=np.uint8)

    # Choose a small, well-spread subset of points
    num_markers = min(20, len(centers_on_camera))
    idx = np.linspace(0, len(centers_on_camera) - 1, num_markers, dtype=int)
    cam_pts = centers_on_camera[idx]

    # Draw squares at camera locations with unique red-channel values (encode index)
    square = 5
    for k, pt in enumerate(cam_pts):
        val = int(min(250, (k + 1) * 10))  # unique <= 250
        cx, cy = int(round(pt[0])), int(round(pt[1]))
        x0, x1 = max(0, cx - square), min(w, cx + square + 1)
        y0, y1 = max(0, cy - square), min(h, cy + square + 1)
        camera_img[y0:y1, x0:x1, 2] = val  # BGR -> use R channel

    # Warp the image
    output_size = (1920, 1080)
    warped = mapping.map_camera_image_to_screen_image(camera_img, output_size)
    assert warped.shape[:2] == output_size[::-1], "Warped image should have correct output size"

    # Compute expected screen positions
    expected_screen = mapping.map_camera_points_to_screen_points(cam_pts)

    # For each marker, search a small neighborhood for the exact encoded value
    H, W = output_size[1], output_size[0]
    search_radius = 3
    red = warped[:, :, 2]
    for k, (sx, sy) in enumerate(expected_screen):
        val = int(min(250, (k + 1) * 10))
        ix, iy = int(round(sx)), int(round(sy))
        if not (0 <= ix < W and 0 <= iy < H):
            # Skip points mapped outside the output bounds
            continue
        x0, x1 = max(0, ix - search_radius), min(W, ix + search_radius + 1)
        y0, y1 = max(0, iy - search_radius), min(H, iy + search_radius + 1)
        patch = red[y0:y1, x0:x1]
        assert (patch == val).any(), f"Marker {k} value {val} not found near expected location {(ix, iy)}"
