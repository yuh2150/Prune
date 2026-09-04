import numpy as np
from collections import defaultdict, deque
from typing import Dict, List, Tuple
from traffic_analysis.homography import HomographyTransformer

class SpeedEstimator:
    """
    Estimates real-world speed (in km/h) for tracked vehicles using Homography coordinate transformation
    and exponential moving average (EMA) or moving average smoothing.
    """
    def __init__(self, config, homography: HomographyTransformer):
        self.config = config
        self.homography = homography
        self.fps = config.frame_rate
        self.window_size = config.speed_smoothing_window
        
        # Speed history per vehicle: tracker_id -> deque of estimated speeds
        self.speed_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=self.window_size))
        # Current smoothed speed per vehicle: tracker_id -> float (km/h)
        self.current_speeds: Dict[int, float] = {}

    def update(self, tracker_id: int, trajectory: List[Tuple[float, float, int]]) -> float:
        """
        Update speed estimation for a single vehicle based on its trajectory history.
        
        Args:
            tracker_id: Vehicle track ID
            trajectory: List of (cx, cy, frame_idx) tuples
            
        Returns:
            Smoothed speed in km/h
        """
        if len(trajectory) < 2:
            return 0.0

        # Take last point and a point N frames back for robust velocity estimate
        p2_x, p2_y, f2 = trajectory[-1]
        step = min(5, len(trajectory) - 1)
        p1_x, p1_y, f1 = trajectory[-1 - step]

        delta_frames = f2 - f1
        if delta_frames <= 0 or self.fps <= 0:
            return self.current_speeds.get(tracker_id, 0.0)

        dt = delta_frames / self.fps

        # Transform pixel coordinates to world coordinates (meters)
        w1_x, w1_y = self.homography.transform_point((p1_x, p1_y))
        w2_x, w2_y = self.homography.transform_point((p2_x, p2_y))

        # Compute Euclidean distance in meters
        distance_meters = np.sqrt((w2_x - w1_x) ** 2 + (w2_y - w1_y) ** 2)

        # Speed in m/s converted to km/h
        speed_mps = distance_meters / dt
        speed_kmh = speed_mps * 3.6

        # Cap unrealistic spikes (e.g. > 200 km/h due to tracking jitter)
        speed_kmh = min(speed_kmh, 200.0)

        # Store raw speed and compute smoothed average
        self.speed_history[tracker_id].append(speed_kmh)
        smoothed_speed = float(np.mean(self.speed_history[tracker_id]))
        self.current_speeds[tracker_id] = round(smoothed_speed, 1)

        return self.current_speeds[tracker_id]

    def get_speed(self, tracker_id: int) -> float:
        """Retrieve current smoothed speed for a vehicle."""
        return self.current_speeds.get(tracker_id, 0.0)
