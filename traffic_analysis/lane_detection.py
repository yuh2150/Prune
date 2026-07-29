import cv2
import numpy as np
import supervision as sv
from typing import Dict, List, Tuple, Set, Any, Optional
from traffic_analysis.helpers import point_in_polygon, draw_polygon_overlay

class LaneManager:
    """
    Manages lane ROIs, detects lane violations (e.g. trucks in bus/car lanes),
    and computes per-lane vehicle density.
    """
    def __init__(self, config):
        self.config = config
        self.lanes = config.lanes
        self.forbidden_lanes = config.forbidden_lanes
        
        # State tracking
        self.lane_violations: Set[int] = set()

    def get_vehicle_lane(self, point: Tuple[float, float]) -> Optional[str]:
        """Find which lane polygon contains the given vehicle center point."""
        for lane_name, polygon in self.lanes.items():
            if point_in_polygon(point, polygon):
                return lane_name
        return None

    def check_violation(self, tracker_id: int, class_id: int, centroid: Tuple[float, float]) -> bool:
        """
        Check if vehicle is driving in a forbidden lane for its class.
        """
        lane = self.get_vehicle_lane(centroid)
        if lane is None:
            return False

        forbidden = self.forbidden_lanes.get(int(class_id), [])
        if lane in forbidden:
            self.lane_violations.add(tracker_id)
            return True
        else:
            self.lane_violations.discard(tracker_id)
            return False

    def compute_density(self, detections: sv.Detections) -> Dict[str, Any]:
        """
        Calculate traffic density metrics across active lanes.
        
        Returns:
            Dict containing active_vehicles, vehicles_per_lane, and density_level (Low/Moderate/Heavy/Congested).
        """
        active_count = len(detections) if detections is not None else 0
        per_lane_count = {lane_name: 0 for lane_name in self.lanes.keys()}

        if detections is not None and detections.tracker_id is not None:
            centers = detections.get_anchors_coordinates(anchor=sv.Position.CENTER)
            for center in centers:
                lane = self.get_vehicle_lane((float(center[0]), float(center[1])))
                if lane in per_lane_count:
                    per_lane_count[lane] += 1

        # Determine density level based on total active vehicles
        if active_count <= 5:
            level = "Low"
        elif active_count <= 12:
            level = "Moderate"
        elif active_count <= 20:
            level = "Heavy"
        else:
            level = "Congested"

        return {
            "active_vehicles": active_count,
            "per_lane": per_lane_count,
            "level": level
        }

    def draw_lanes(self, frame: np.ndarray) -> np.ndarray:
        """Draw lane boundaries and ROI overlays on frame."""
        colors = [(255, 100, 0), (0, 200, 255), (200, 0, 255)]
        for idx, (lane_name, polygon) in enumerate(self.lanes.items()):
            color = colors[idx % len(colors)]
            frame = draw_polygon_overlay(frame, polygon, color, alpha=0.15, border_color=color)
            
            # Label lane name
            pts = np.array(polygon, dtype=np.int32)
            M = cv2.moments(pts)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(frame, lane_name.upper(), (cx - 30, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return frame
