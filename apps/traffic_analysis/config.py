import os
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional

@dataclass
class TrafficAnalysisConfig:
    # --- Model Configuration ---
    detector_type: str = "onnx"  # Options: 'yolov5', 'rtdetr', 'onnx', 'ultralytics'
    model_weights: str = "weights/yolov5s-pruned-finetuned.onnx"
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    device: str = "cuda"  # 'cuda' or 'cpu'
    
    # Selected vehicle class IDs (COCO mapping)
    # 2: car, 3: motorcycle, 5: bus, 7: truck
    target_classes: List[int] = field(default_factory=lambda: [2, 3, 5, 7])
    class_names: Dict[int, str] = field(default_factory=lambda: {
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck"
    })
    
    # --- Video Input / Output ---
    video_source: str = "DATA/INPUTS/cars_on_highway_2.mp4"
    output_path: str = "DATA/OUTPUTS/traffic_analysis_output.mp4"
    display: bool = True
    save_output: bool = True
    
    # --- Tracking Configuration ---
    track_buffer: int = 30
    frame_rate: int = 30
    trajectory_length: int = 60  # Number of frames to keep trajectory points
    
    # --- Counting Line Configuration ---
    # Line defined as [(x1, y1), (x2, y2)]
    counting_line: Tuple[Tuple[int, int], Tuple[int, int]] = ((0, 350), (1280, 350))
    partition_x: int = 640  # Divider for In/Out or Up/Down directional count
    
    # --- Homography & Speed Estimation ---
    # 4 Pixel points in image (quadrilateral on road)
    pixel_points: List[Tuple[float, float]] = field(default_factory=lambda: [
        (450, 250),   # Top-Left
        (830, 250),   # Top-Right
        (1100, 600),  # Bottom-Right
        (180, 600)    # Bottom-Left
    ])
    # Corresponding real-world ground plane coordinates in meters
    world_points: List[Tuple[float, float]] = field(default_factory=lambda: [
        (0.0, 0.0),    # Top-Left (0m, 0m)
        (7.5, 0.0),    # Top-Right (7.5m wide)
        (7.5, 30.0),   # Bottom-Right (30m ahead)
        (0.0, 30.0)    # Bottom-Left
    ])
    speed_smoothing_window: int = 5
    
    # --- Traffic Violation Parameters ---
    # Wrong-way: Allowed vector per direction or global (dx, dy)
    # Downward traffic vector expected: (0, 1) or Upward: (0, -1)
    allowed_directions: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "left_side": (0.0, -1.0),   # Vehicles going UP on left lane
        "right_side": (0.0, 1.0)    # Vehicles going DOWN on right lane
    })
    wrong_way_angle_threshold: float = 120.0  # Angle diff in degrees to flag wrong-way
    
    # Lane ROIs: List of Polygon vertices [(x1,y1), (x2,y2), ...]
    lanes: Dict[str, List[Tuple[int, int]]] = field(default_factory=lambda: {
        "lane_1": [(0, 350), (600, 350), (500, 720), (0, 720)],
        "lane_2": [(600, 350), (1280, 350), (1280, 720), (500, 720)]
    })
    forbidden_lanes: Dict[int, List[str]] = field(default_factory=lambda: {
        # e.g., Trucks (7) and Buses (5) forbidden in lane_1
        7: ["lane_1"],
        5: ["lane_1"]
    })
    
    # Illegal Parking: No-parking ROI polygon + stationary threshold
    no_parking_rois: List[List[Tuple[int, int]]] = field(default_factory=lambda: [
        [(50, 400), (250, 400), (250, 600), (50, 600)]
    ])
    parking_time_threshold: float = 3.0  # Seconds stationary to trigger alert
    stationary_dist_threshold: float = 5.0  # Pixels displacement threshold
    
    # --- Heatmap Generation ---
    heatmap_decay: float = 0.99
    heatmap_alpha: float = 0.4
    enable_heatmap: bool = True
    
    # --- License Plate Recognition ---
    enable_lpr: bool = False
    ocr_engine: str = "easyocr"  # 'easyocr', 'paddleocr'
    allow_synth_fallback: bool = False  # Set to True only for synthetic demo fallback plates
