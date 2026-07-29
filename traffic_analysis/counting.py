import cv2
import numpy as np
import supervision as sv
from typing import Dict, List, Set, Tuple, Any
from traffic_analysis.helpers import line_intersection

class VehicleCounter:
    """
    Handles line-crossing vehicle counting with direction classification (In/Out, Up/Down)
    and class-wise statistics.
    """
    def __init__(self, config):
        self.config = config
        line = config.counting_line
        self.line_start = line[0]
        self.line_end = line[1]
        self.partition_x = config.partition_x
        self.class_names = config.class_names

        # Cross tracking state sets
        self.crossed_ids: Set[int] = set()
        self.crossed_in_ids: Set[int] = set()
        self.crossed_out_ids: Set[int] = set()

        # Counts
        self.total_count = 0
        self.count_in = 0
        self.count_out = 0
        self.class_counts: Dict[str, int] = {name: 0 for name in self.class_names.values()}

    def update(self, detections: sv.Detections, tracker_history: Dict[int, Any]) -> Dict[str, Any]:
        """
        Check tracked vehicles against counting line and update statistics.
        
        Args:
            detections: Supervision Detections object
            tracker_history: Dict mapping tracker_id to trajectory deque
            
        Returns:
            Dict containing current count stats
        """
        if detections.tracker_id is None or len(detections.tracker_id) == 0:
            return self.get_stats()

        for track_id, class_id in zip(detections.tracker_id, detections.class_id):
            track_id = int(track_id)
            trajectory = tracker_history.get(track_id, [])
            
            if len(trajectory) < 2:
                continue

            # Take current and previous centroid positions
            p_prev = (trajectory[-2][0], trajectory[-2][1])
            p_curr = (trajectory[-1][0], trajectory[-1][1])

            # Check if trajectory segment intersects counting line
            if track_id not in self.crossed_ids:
                if line_intersection(p_prev, p_curr, self.line_start, self.line_end):
                    self.crossed_ids.add(track_id)
                    self.total_count += 1

                    # Increment class count
                    cls_name = self.class_names.get(int(class_id), "vehicle")
                    self.class_counts[cls_name] = self.class_counts.get(cls_name, 0) + 1

                    # Direction classification (In vs Out based on partition_x or Y motion)
                    cx = p_curr[0]
                    if cx < self.partition_x:
                        self.crossed_in_ids.add(track_id)
                        self.count_in += 1
                    else:
                        self.crossed_out_ids.add(track_id)
                        self.count_out += 1

        return self.get_stats()

    def get_stats(self) -> Dict[str, Any]:
        """Get current aggregated counting statistics."""
        return {
            "total": self.total_count,
            "in": self.count_in,
            "out": self.count_out,
            "per_class": self.class_counts.copy()
        }

    def draw_counting_line(self, frame: np.ndarray) -> np.ndarray:
        """Render the counting line and current counts on frame."""
        # Draw main counting line
        cv2.line(frame, self.line_start, self.line_end, (0, 0, 255), 3, cv2.LINE_AA)
        
        # Draw partition divider line if valid
        cv2.circle(frame, (self.partition_x, self.line_start[1]), 5, (255, 255, 0), -1)
        return frame
