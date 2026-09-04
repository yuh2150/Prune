import cv2
import numpy as np
import time
from typing import List, Tuple, Optional, Any

class FPSCounter:
    """Helper class to calculate smooth FPS and latency."""
    def __init__(self, avg_frames: int = 10):
        self.avg_frames = avg_frames
        self.timestamps = []
        self.last_time = time.time()
        self.latency_ms = 0.0

    def tick(self) -> float:
        now = time.time()
        self.latency_ms = (now - self.last_time) * 1000.0
        self.last_time = now
        self.timestamps.append(now)
        if len(self.timestamps) > self.avg_frames:
            self.timestamps.pop(0)
        if len(self.timestamps) > 1:
            fps = (len(self.timestamps) - 1) / (self.timestamps[-1] - self.timestamps[0])
        else:
            fps = 0.0
        return fps


def point_in_polygon(point: Tuple[float, float], polygon: List[Tuple[int, int]]) -> bool:
    """
    Check if a 2D point (x, y) is inside a polygon using OpenCV cv2.pointPolygonTest.
    """
    pts = np.array(polygon, dtype=np.int32)
    res = cv2.pointPolygonTest(pts, (float(point[0]), float(point[1])), False)
    return res >= 0


def line_intersection(
    p1: Tuple[float, float], p2: Tuple[float, float],
    l1: Tuple[float, float], l2: Tuple[float, float]
) -> bool:
    """
    Check if line segment p1-p2 intersects segment l1-l2 using cross product.
    """
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    return ccw(p1, l1, l2) != ccw(p2, l1, l2) and ccw(p1, p2, l1) != ccw(p1, p2, l2)


def draw_polygon_overlay(
    image: np.ndarray,
    polygon: List[Tuple[int, int]],
    color: Tuple[int, int, int],
    alpha: float = 0.25,
    border_color: Optional[Tuple[int, int, int]] = None
) -> np.ndarray:
    """Draw a semi-transparent filled polygon on an image."""
    overlay = image.copy()
    pts = np.array(polygon, dtype=np.int32)
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    if border_color is not None:
        cv2.polylines(image, [pts], isClosed=True, color=border_color, thickness=2)
    return image


def draw_text_box(
    image: np.ndarray,
    text: str,
    org: Tuple[int, int],
    font_scale: float = 0.6,
    text_color: Tuple[int, int, int] = (255, 255, 255),
    bg_color: Tuple[int, int, int] = (0, 0, 0),
    thickness: int = 1,
    padding: int = 4
) -> None:
    """Draw text with a solid background box for high contrast."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (w, h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = org
    cv2.rectangle(image, (x - padding, y - h - padding), (x + w + padding, y + baseline + padding), bg_color, cv2.FILLED)
    cv2.putText(image, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)
