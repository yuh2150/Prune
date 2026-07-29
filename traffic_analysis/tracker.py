import numpy as np
import supervision as sv
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Any

class VehicleTracker:
    """
    Wraps ByteTrack for multi-object tracking and manages trajectory history per vehicle ID.
    """
    def __init__(self, config):
        self.config = config
        self.tracker = sv.ByteTrack(
            frame_rate=config.frame_rate,
            track_activation_threshold=config.confidence_threshold,
            minimum_matching_threshold=0.8,
            lost_track_buffer=config.track_buffer
        )
        self.smoother = sv.DetectionsSmoother()
        
        # Trajectory history: tracker_id -> deque of (cx, cy, frame_idx)
        self.trajectories: Dict[int, deque] = defaultdict(lambda: deque(maxlen=config.trajectory_length))
        self.frame_count = 0

    def update(self, xyxy: np.ndarray, confidence: np.ndarray, class_id: np.ndarray) -> sv.Detections:
        """
        Update tracker with current frame detections.
        
        Args:
            xyxy: bounding boxes (N, 4)
            confidence: confidences (N,)
            class_id: class labels (N,)
            
        Returns:
            Supervision Detections object containing tracker_ids
        """
        self.frame_count += 1
        
        if len(xyxy) == 0:
            detections = sv.Detections(
                xyxy=np.empty((0, 4), dtype=np.float32),
                confidence=np.empty((0,), dtype=np.float32),
                class_id=np.empty((0,), dtype=int)
            )
        else:
            detections = sv.Detections(
                xyxy=xyxy,
                confidence=confidence,
                class_id=class_id
            )

        # Update ByteTrack
        detections = self.tracker.update_with_detections(detections)
        detections = self.smoother.update_with_detections(detections)

        # Record trajectory centroids
        if detections.tracker_id is not None and len(detections.tracker_id) > 0:
            centers = detections.get_anchors_coordinates(anchor=sv.Position.CENTER)
            for track_id, (cx, cy) in zip(detections.tracker_id, centers):
                self.trajectories[int(track_id)].append((float(cx), float(cy), self.frame_count))

        return detections

    def get_trajectory(self, tracker_id: int) -> List[Tuple[float, float, int]]:
        """Get position history for a given vehicle tracker ID."""
        return list(self.trajectories.get(tracker_id, []))
