import numpy as np
from typing import Dict, List, Tuple, Set, Any

class WrongWayDetector:
    """
    Detects vehicles moving in the wrong direction based on trajectory vector analysis.
    """
    def __init__(self, config):
        self.config = config
        self.allowed_directions = config.allowed_directions
        self.angle_threshold = config.wrong_way_angle_threshold
        self.partition_x = config.partition_x
        
        # Active wrong-way alerts: tracker_id -> timestamp / frame_count
        self.active_alerts: Set[int] = set()

    def check(self, tracker_id: int, trajectory: List[Tuple[float, float, int]]) -> bool:
        """
        Check if a vehicle's trajectory vector violates the allowed flow direction.
        
        Args:
            tracker_id: Vehicle track ID
            trajectory: List of (cx, cy, frame_idx) tuples
            
        Returns:
            True if wrong-way violation detected, else False
        """
        if len(trajectory) < 8:
            return tracker_id in self.active_alerts

        # Trajectory vector over past N frames
        p1 = np.array([trajectory[-8][0], trajectory[-8][1]])
        p2 = np.array([trajectory[-1][0], trajectory[-1][1]])
        
        motion_vec = p2 - p1
        dist = np.linalg.norm(motion_vec)
        
        # Ignore stationary or jittering vehicles
        if dist < 15.0:
            return tracker_id in self.active_alerts

        unit_motion = motion_vec / dist

        # Select allowed direction based on lane (left vs right side of road)
        cx = trajectory[-1][0]
        if cx < self.partition_x:
            expected_dir = np.array(self.allowed_directions.get("left_side", (0.0, -1.0)))
        else:
            expected_dir = np.array(self.allowed_directions.get("right_side", (0.0, 1.0)))

        # Calculate cosine similarity and angle difference
        dot_product = np.clip(np.dot(unit_motion, expected_dir), -1.0, 1.0)
        angle_diff_deg = np.degrees(np.arccos(dot_product))

        if angle_diff_deg > self.angle_threshold:
            self.active_alerts.add(tracker_id)
            return True
        else:
            self.active_alerts.discard(tracker_id)
            return False
