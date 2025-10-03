import cv2
import numpy as np


class Mapping:
    def __init__(self, mapping):
        self.mapping = mapping

    @classmethod
    def from_matching_points(cls, screen_points, camera_points, **kwargs):
        pass

    def map_camera_points_to_screen_points(self, camera_points):
        if self.mapping is None:
            raise ValueError("Mapping has not been calculated yet.")
        return self._map_camera_points_to_screen_points(camera_points)

    def map_camera_image_to_screen_image(self, camera_image, output_size):
        if self.mapping is None:
            raise ValueError("Mapping has not been calculated yet.")
        return self._map_camera_image_to_screen_image(camera_image, output_size)

    def _map_camera_points_to_screen_points(self, camera_points):
        pass

    def _map_camera_image_to_screen_image(self, camera_image, output_size):
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


    def _map_camera_points_to_screen_points(self, camera_points):
        camera_points = camera_points.reshape(-1, camera_points.shape[-1])
        camera_points_homogeneous = np.hstack([camera_points, np.ones((camera_points.shape[0], 1))])
        screen_points_homogeneous = camera_points_homogeneous @ self.mapping.T
        screen_points = screen_points_homogeneous[:, :2] / screen_points_homogeneous[:, 2:3]
        return screen_points

    def map_camera_image_to_screen_image(self, camera_image, output_size):
        screen_image = cv2.warpPerspective(camera_image, self.mapping, output_size)
        return screen_image
