import cv2
import numpy as np


class Mapping:
    @classmethod
    def from_matching_points(cls, screen_points, camera_points, image_size, **kwargs):
        """
        Create a mapping from matching points.
        screen_points, camera_points: numpy arrays of shape (n, 2)
        """
        pass

    def map_camera_points_to_screen_points(self, camera_points):
        pass

    def map_camera_image_to_screen_image(self, camera_image, output_size):
        pass
    
    def calculate_accuracy(self, expected_screen_points, mapped_screen_points):
        """
        Calculate mapping accuracy by measuring distances between mapped points and expected screen points.
        Returns mean distance in pixels.
        """
        distances = np.linalg.norm(mapped_screen_points - expected_screen_points, axis=1)
        return distances.mean()


class HomographyMapping(Mapping):
    def __init__(self, mapping):
        self.mapping = mapping

    @classmethod
    def from_matching_points(cls, screen_points, camera_points, image_size,
                             corners_only=False, num_rows=None):

        if corners_only:
            corners = [0, -1, -num_rows, num_rows - 1]
            screen_points = screen_points[corners]
            camera_points = camera_points[corners]

        camera_points = camera_points.astype(np.float32)
        screen_points = screen_points.astype(np.float32)

        H, mask = cv2.findHomography(camera_points, screen_points, cv2.RANSAC)
        if H is None:
            raise ValueError("Could not find homography matrix")
        return cls(H)

    def map_camera_points_to_screen_points(self, camera_points):
        camera_points_homogeneous = np.hstack([camera_points, np.ones((camera_points.shape[0], 1))])
        screen_points_homogeneous = camera_points_homogeneous @ self.mapping.T
        screen_points = screen_points_homogeneous[:, :2] / screen_points_homogeneous[:, 2:3]
        return screen_points

    def map_camera_image_to_screen_image(self, camera_image, output_size):
        screen_image = cv2.warpPerspective(camera_image, self.mapping, output_size)
        return screen_image


class PolynomialWarpMapping(Mapping):
    def __init__(self, coeffs_x, coeffs_y, degree, image_center):
        """
        coeffs_x, coeffs_y: coefficients for polynomial mapping
        degree: polynomial degree
        image_center: (cx, cy) center of the camera image
        """
        self.coeffs_x = coeffs_x
        self.coeffs_y = coeffs_y
        self.degree = degree
        self.image_center = image_center

    @classmethod
    def from_matching_points(cls, screen_points, camera_points, image_size, degree=3):
        """
        Fit polynomial warp of given degree mapping camera_points -> screen_points.
        """
        # Calculate image center
        cx, cy = image_size[0] / 2, image_size[1] / 2
        image_center = (cx, cy)
        
        # Convert camera points to center-relative coordinates
        camera_points_centered = camera_points - np.array([cx, cy])
        
        # Build polynomial design matrix
        X = cls._polynomial_terms(camera_points_centered, degree)

        # Solve least squares: screen = X @ coeffs
        coeffs_x, _, _, _ = np.linalg.lstsq(X, screen_points[:, 0], rcond=None)
        coeffs_y, _, _, _ = np.linalg.lstsq(X, screen_points[:, 1], rcond=None)

        return cls(coeffs_x, coeffs_y, degree, image_center)

    def map_camera_points_to_screen_points(self, camera_points):
        # Convert to center-relative coordinates
        cx, cy = self.image_center
        camera_points_centered = camera_points - np.array([cx, cy])
        
        terms = self._polynomial_terms(camera_points_centered, self.degree)
        x_mapped = terms @ self.coeffs_x
        y_mapped = terms @ self.coeffs_y
        return np.column_stack([x_mapped, y_mapped])

    def map_camera_image_to_screen_image(self, camera_image, output_size):
        """
        Warp image using inverse mapping + interpolation.
        For each screen pixel, find the corresponding camera pixel.
        """
        w, h = output_size
        # Create a grid of screen coordinates
        xv, yv = np.meshgrid(np.arange(w), np.arange(h))
        screen_coords = np.column_stack([xv.ravel(), yv.ravel()])

        # Find inverse mapping: screen -> camera
        # We need to solve: screen_coords = f(camera_coords)
        # For each screen coordinate, find the camera coordinate that maps to it
        camera_coords = self._inverse_map_screen_to_camera(screen_coords)

        map_x = camera_coords[:, 0].reshape(h, w).astype(np.float32)
        map_y = camera_coords[:, 1].reshape(h, w).astype(np.float32)

        warped = cv2.remap(camera_image, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT)
        return warped

    def _inverse_map_screen_to_camera(self, screen_coords):
        """
        Find camera coordinates that map to given screen coordinates.
        Uses iterative method to solve the inverse mapping.
        """
        # Start with screen coordinates as initial guess (use float for updates)
        camera_coords = screen_coords.astype(np.float64, copy=True)
        
        # Iterative refinement to find inverse mapping
        for _ in range(5):  # Usually converges in 3-5 iterations
            # Map current camera coordinates to screen
            mapped_screen = self.map_camera_points_to_screen_points(camera_coords)
            
            # Calculate error
            error = screen_coords.astype(np.float64) - mapped_screen
            
            # Update camera coordinates based on error
            # Simple gradient descent approach
            camera_coords += error * 0.5
            
        return camera_coords

    @staticmethod
    def _polynomial_terms(points, degree):
        """
        Construct polynomial basis up to given degree.
        E.g. degree=2 → [1, x, y, x^2, xy, y^2]
        """
        x, y = points[:, 0], points[:, 1]
        terms = []
        for d in range(0, degree + 1):
            for i in range(d + 1):
                j = d - i
                terms.append((x ** i) * (y ** j))
        return np.column_stack(terms)
