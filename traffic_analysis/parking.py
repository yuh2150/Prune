import time
import numpy as np
from typing import Dict, List, Tuple, Set, Any, Optional
from traffic_analysis.helpers import point_in_polygon, draw_polygon_overlay

class IllegalParkingDetector:
    """
    Detects vehicles illegally parked (stationary beyond T seconds) inside No-Parking ROIs.
    """
    def __init__(self, config):
        self.config = config
        self.no_parking_rois = config.no_parking_rois
        self.time_threshold = config.parking_time_threshold  # in seconds
        self.dist_threshold = config.stationary_dist_threshold  # in pixels
        self.fps = config.frame_rate

        # Tracking state: tracker_id -> {'entry_frame': int, 'last_pos': (x,y), 'stationary_frames': int}
        self.parking_tracker: Dict[int, Dict[str, Any]] = {}
        self.active_parking_violations: Set[int] = set()

    def update(self, tracker_id: int, trajectory: List[Tuple[float, float, int]]) -> bool:
        """
        Update stationarity check for a vehicle.
        
        Args:
            tracker_id: Vehicle track ID
            trajectory: List of (cx, cy, frame_idx) tuples
            
        Returns:
            True if vehicle is violating illegal parking rule, else False
        """
        if len(trajectory) < 5:
            return False

        curr_pos = (trajectory[-1][0], trajectory[-1][1])
        curr_frame = trajectory[-1][2]

        # Check if vehicle center is inside any No-Parking ROI
        inside_roi = any(point_in_polygon(curr_pos, roi) for roi in self.no_parking_rois)

        if not inside_roi:
            self.parking_tracker.pop(tracker_id, None)
            self.active_parking_violations.discard(tracker_id)
            return False

        # Initialize tracking for vehicle entering ROI
        if tracker_id not in self.parking_tracker:
            self.parking_tracker[tracker_id] = {
                'start_frame': curr_frame,
                'last_pos': curr_pos,
                'stationary_frames': 0
            }
            return False

        state = self.parking_tracker[tracker_id]
        dist = np.sqrt((curr_pos[0] - state['last_pos'][0]) ** 2 + (curr_pos[1] - state['last_pos'][1]) ** 2)

        if dist < self.dist_threshold:
            state['stationary_frames'] += (curr_frame - state.get('prev_frame', curr_frame - 1))
        else:
            # Reset stationary counter if vehicle moved significantly
            state['stationary_frames'] = 0
            state['last_pos'] = curr_pos

        state['prev_frame'] = curr_frame

        # Check if stationary duration exceeds threshold
        duration_seconds = state['stationary_frames'] / float(self.fps)
        if duration_seconds >= self.time_threshold:
            self.active_parking_violations.add(tracker_id)
            return True
        else:
            self.active_parking_violations.discard(tracker_id)
            return False

    def draw_rois(self, frame: np.ndarray) -> np.ndarray:
        """Draw No-Parking ROIs on frame."""
        for roi in self.no_parking_rois:
            frame = draw_polygon_overlay(frame, roi, (0, 0, 255), alpha=0.2, border_color=(0, 0, 255))
        return frame
