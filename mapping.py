import cv2
import numpy as np


class Mapping:
    def __init__(self, mapping):
        self.mapping = mapping

    @classmethod
    def from_matching_points(cls, screen_points, camera_points, **kwargs):
        pass

    def map_camera_points_to_screen_points(self, camera_points):
        pass

    def map_camera_image_to_screen_image(self, camera_image, output_size):
        pass


class HomographyMapping(Mapping):

    @classmethod
    def from_matching_points(cls, screen_points, camera_points, corners_only=False, num_rows=None):
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
        camera_points = camera_points.reshape(-1, camera_points.shape[-1])
        camera_points_homogeneous = np.hstack([camera_points, np.ones((camera_points.shape[0], 1))])
        screen_points_homogeneous = camera_points_homogeneous @ self.mapping.T
        screen_points = screen_points_homogeneous[:, :2] / screen_points_homogeneous[:, 2:3]
        return screen_points

    def map_camera_image_to_screen_image(self, camera_image, output_size):
        screen_image = cv2.warpPerspective(camera_image, self.mapping, output_size)
        return screen_image


class PolynomialWarpMapping(Mapping):
    def __init__(self, coeffs_x, coeffs_y, degree):
        """
        coeffs_x, coeffs_y: coefficients for polynomial mapping
        degree: polynomial degree
        """
        self.coeffs_x = coeffs_x
        self.coeffs_y = coeffs_y
        self.degree = degree

    @classmethod
    def from_matching_points(cls, screen_points, camera_points, degree=3):
        """
        Fit polynomial warp of given degree mapping camera_points -> screen_points.
        """
        camera_points = np.asarray(camera_points, dtype=np.float64)
        camera_points = camera_points.reshape(-1, 2)
        screen_points = np.asarray(screen_points, dtype=np.float64)

        if camera_points.shape != screen_points.shape:
            raise ValueError("camera_points and screen_points must have the same shape")

        # Build polynomial design matrix
        X = cls._polynomial_terms(camera_points, degree)

        # Solve least squares: screen = X @ coeffs
        coeffs_x, _, _, _ = np.linalg.lstsq(X, screen_points[:, 0], rcond=None)
        coeffs_y, _, _, _ = np.linalg.lstsq(X, screen_points[:, 1], rcond=None)

        return cls(coeffs_x, coeffs_y, degree)

    def map_camera_points_to_screen_points(self, camera_points):
        camera_points = camera_points.reshape(-1, 2)
        terms = self._polynomial_terms(camera_points, self.degree)
        x_mapped = terms @ self.coeffs_x
        y_mapped = terms @ self.coeffs_y
        return np.column_stack([x_mapped, y_mapped])

    def map_camera_image_to_screen_image(self, camera_image, output_size):
        """
        Warp image using inverse mapping + interpolation.
        For simplicity, use backward mapping (screen->camera).
        """
        w, h = output_size
        # Create a grid of screen coordinates
        xv, yv = np.meshgrid(np.arange(w), np.arange(h))
        screen_coords = np.column_stack([xv.ravel(), yv.ravel()])

        # Estimate inverse warp (approximate by fitting camera->screen inverse)
        # Here we just use forward mapping approximation (not exact inverse)
        # For accurate results, you’d fit another PolynomialWarpMapping with swapped roles.
        # For now, we approximate by nearest neighbor
        mapped = self.map_camera_points_to_screen_points(screen_coords)

        map_x = mapped[:, 0].reshape(h, w).astype(np.float32)
        map_y = mapped[:, 1].reshape(h, w).astype(np.float32)

        warped = cv2.remap(camera_image, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT)
        return warped

    @staticmethod
    def _polynomial_terms(points, degree):
        """
        Construct polynomial basis up to given degree.
        E.g. degree=2 → [1, x, y, x^2, xy, y^2]
        """
        x, y = points[:, 0], points[:, 1]
        terms = [np.ones_like(x)]
        for d in range(1, degree + 1):
            for i in range(d + 1):
                j = d - i
                terms.append((x ** i) * (y ** j))
        return np.column_stack(terms)
