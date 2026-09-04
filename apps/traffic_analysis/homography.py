import cv2
import numpy as np
from typing import List, Tuple

class HomographyTransformer:
    """
    Computes and applies perspective transformation (Homography matrix H)
    to convert image pixel coordinates (u, v) into ground-plane world coordinates (x, y) in meters.
    """
    def __init__(self, pixel_pts: List[Tuple[float, float]], world_pts: List[Tuple[float, float]]):
        self.pixel_pts = np.float32(pixel_pts)
        self.world_pts = np.float32(world_pts)
        
        if len(self.pixel_pts) < 4 or len(self.world_pts) < 4:
            raise ValueError("Homography requires at least 4 point correspondences.")
            
        self.H, _ = cv2.findHomography(self.pixel_pts, self.world_pts)
        if self.H is None:
            raise RuntimeError("Failed to compute Homography matrix.")
            
        print("[Homography] Perspective transformation matrix computed successfully.")

    def transform_point(self, pixel_point: Tuple[float, float]) -> Tuple[float, float]:
        """
        Transform a single (u, v) pixel coordinate to (x, y) world coordinate in meters.
        """
        pt = np.array([[[pixel_point[0], pixel_point[1]]]], dtype=np.float32)
        world_pt = cv2.perspectiveTransform(pt, self.H)
        return float(world_pt[0, 0, 0]), float(world_pt[0, 0, 1])

    def transform_points(self, pixel_points: np.ndarray) -> np.ndarray:
        """
        Transform multiple (N, 2) pixel coordinates to (N, 2) world coordinates.
        """
        if len(pixel_points) == 0:
            return np.empty((0, 2), dtype=np.float32)
        pts = np.array([pixel_points], dtype=np.float32)
        world_pts = cv2.perspectiveTransform(pts, self.H)
        return world_pts[0]
