import os
import sys
import argparse
import time
import cv2
import numpy as np
import supervision as sv
from pathlib import Path

# Add project root to sys.path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from traffic_analysis.config import TrafficAnalysisConfig
from traffic_analysis.detect import VehicleDetector
from traffic_analysis.tracker import VehicleTracker
from traffic_analysis.homography import HomographyTransformer
from traffic_analysis.speed_estimator import SpeedEstimator
from traffic_analysis.counting import VehicleCounter
from traffic_analysis.lane_detection import LaneManager
from traffic_analysis.wrong_way import WrongWayDetector
from traffic_analysis.parking import IllegalParkingDetector
from traffic_analysis.heatmap import HeatmapGenerator
from traffic_analysis.plate_detector import LicensePlateDetector
from traffic_analysis.plate_recognition import LicensePlateRecognizer
from traffic_analysis.database import VehicleDatabase
from traffic_analysis.dashboard import TrafficDashboard
from traffic_analysis.helpers import FPSCounter, draw_text_box

class TrafficAnalysisPipeline:
    """
    Modular Real-Time Traffic Analysis Pipeline unifies Detection, Tracking,
    Counting, Speed Estimation, Violation Detection, Heatmap, LPR, and Dashboard.
    """
    def __init__(self, config: TrafficAnalysisConfig):
        self.config = config

        print("=" * 60)
        print(" INITIALIZING TRAFFIC ANALYSIS PIPELINE")
        print("=" * 60)
        
        # 1. Detector & Tracker
        self.detector = VehicleDetector(config)
        self.tracker = VehicleTracker(config)
        
        # 2. Homography & Speed Estimation
        self.homography = HomographyTransformer(config.pixel_points, config.world_points)
        self.speed_estimator = SpeedEstimator(config, self.homography)
        
        # 3. Counting & Lane/Violation Managers
        self.counter = VehicleCounter(config)
        self.lane_manager = LaneManager(config)
        self.wrong_way_detector = WrongWayDetector(config)
        self.parking_detector = IllegalParkingDetector(config)
        
        # 4. License Plate Recognition & Database
        self.plate_detector = LicensePlateDetector()
        self.plate_recognizer = LicensePlateRecognizer(config.ocr_engine, config.allow_synth_fallback)
        self.db = VehicleDatabase()
        
        # 5. Visualizer & Dashboard
        self.dashboard = TrafficDashboard(config)
        self.fps_counter = FPSCounter()
        self.heatmap = None  # Lazy initialized on first frame

    def run(self):
        video_path = self.config.video_source
        cap = cv2.VideoCapture(int(video_path) if video_path.isdigit() else video_path)

        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {video_path}")

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_input = cap.get(cv2.CAP_PROP_FPS) or self.config.frame_rate
        self.config.frame_rate = int(fps_input)

        out = None
        if self.config.save_output:
            os.makedirs(os.path.dirname(self.config.output_path), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(self.config.output_path, fourcc, fps_input, (w, h))

        print(f"[Pipeline] Video Stream Opened ({w}x{h} @ {fps_input:.1f} FPS)")
        print(f"[Pipeline] Processing frames... Press 'q' or 'p' to exit.")

        # Initialize Heatmap Generator with exact frame resolution
        self.heatmap = HeatmapGenerator(self.config, frame_shape=(h, w))

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                fps = self.fps_counter.tick()
                latency_ms = self.fps_counter.latency_ms

                # --- Step 1: Vehicle Detection ---
                xyxy, conf, class_ids = self.detector.detect(frame)

                # --- Step 2: Multi-Object Tracking ---
                detections = self.tracker.update(xyxy, conf, class_ids)

                # --- Step 3: Analytics & Violation Processing ---
                active_centroids = []
                active_violations_frame = []
                plate_logs = []

                if detections.tracker_id is not None and len(detections.tracker_id) > 0:
                    centers = detections.get_anchors_coordinates(anchor=sv.Position.CENTER)
                    
                    for i, (track_id, bbox, cls_id) in enumerate(zip(detections.tracker_id, detections.xyxy, detections.class_id)):
                        track_id = int(track_id)
                        cls_name = self.config.class_names.get(int(cls_id), "vehicle")
                        cx, cy = float(centers[i][0]), float(centers[i][1])
                        active_centroids.append((cx, cy))

                        # Trajectory
                        traj = self.tracker.get_trajectory(track_id)

                        # Speed estimation
                        speed_kmh = self.speed_estimator.update(track_id, traj)

                        # Violation checks
                        is_wrong_way = self.wrong_way_detector.check(track_id, traj)
                        is_lane_violation = self.lane_manager.check_violation(track_id, cls_id, (cx, cy))
                        is_parking_violation = self.parking_detector.update(track_id, traj)

                        violation_str = None
                        if is_wrong_way:
                            violation_str = "Wrong Way"
                            active_violations_frame.append(f"#{track_id} Wrong Way")
                        elif is_lane_violation:
                            violation_str = "Lane Violation"
                            active_violations_frame.append(f"#{track_id} Lane Violation")
                        elif is_parking_violation:
                            violation_str = "Illegal Parking"
                            active_violations_frame.append(f"#{track_id} Illegal Parking")

                        # License Plate Recognition (on vehicle crop)
                        plate_str = "PENDING"
                        if self.config.enable_lpr and i % 5 == 0:  # Sample LPR every 5 frames
                            x1, y1, x2, y2 = map(int, bbox)
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(w, x2), min(h, y2)
                            vehicle_crop = frame[y1:y2, x1:x2]
                            
                            plate_crop = self.plate_detector.detect_and_crop(vehicle_crop)
                            plate_str = self.plate_recognizer.recognize(plate_crop, track_id)

                        # Update Vehicle Database Record
                        rec = self.db.update_vehicle(
                            tracker_id=track_id,
                            class_name=cls_name,
                            centroid=(cx, cy, self.tracker.frame_count),
                            speed=speed_kmh,
                            license_plate=plate_str,
                            violation=violation_str
                        )

                        if rec.license_plate and rec.license_plate != "PENDING":
                            plate_logs.append(f"#{track_id} {rec.class_name.upper()}: {rec.license_plate}")

                # --- Step 4: Vehicle Counting ---
                counts = self.counter.update(detections, self.tracker.trajectories)

                # --- Step 5: Traffic Density ---
                density_info = self.lane_manager.compute_density(detections)

                # --- Step 6: Heatmap Update ---
                self.heatmap.update(active_centroids)

                # --- Step 7: Visualization Rendering ---
                # A. Overlay Heatmap
                frame = self.heatmap.generate_overlay(frame)

                # B. Draw Lane & Parking ROIs, Counting Line
                frame = self.lane_manager.draw_lanes(frame)
                frame = self.parking_detector.draw_rois(frame)
                frame = self.counter.draw_counting_line(frame)

                # C. Draw Vehicle Bounding Boxes, Trajectories & Labels
                if detections.tracker_id is not None and len(detections.tracker_id) > 0:
                    centers = detections.get_anchors_coordinates(anchor=sv.Position.CENTER)
                    for i, (track_id, bbox, cls_id) in enumerate(zip(detections.tracker_id, detections.xyxy, detections.class_id)):
                        track_id = int(track_id)
                        cls_name = self.config.class_names.get(int(cls_id), "vehicle")
                        x1, y1, x2, y2 = map(int, bbox)
                        
                        rec = self.db.get_record(track_id)
                        speed = rec.current_speed if rec else 0.0
                        plate = rec.license_plate if rec else ""
                        
                        box_color = (0, 255, 0)
                        if rec and rec.violations:
                            box_color = (0, 0, 255)  # Red box for violators

                        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

                        # Label text
                        label = f"#{track_id} {cls_name} {speed:.0f}km/h"
                        if plate and plate != "PENDING":
                            label += f" [{plate}]"

                        draw_text_box(frame, label, (x1, max(15, y1 - 8)), font_scale=0.45, bg_color=box_color)

                        # Draw Trajectory Trace
                        traj = self.tracker.get_trajectory(track_id)
                        if len(traj) >= 2:
                            pts = np.array([(int(p[0]), int(p[1])) for p in traj], dtype=np.int32)
                            cv2.polylines(frame, [pts], isClosed=False, color=box_color, thickness=2)

                # D. Render HUD Dashboard
                frame = self.dashboard.draw_hud(
                    frame=frame,
                    fps=fps,
                    latency_ms=latency_ms,
                    counts=counts,
                    avg_speed=self.db.get_average_speed(),
                    density_info=density_info,
                    active_violations=active_violations_frame,
                    total_violations=self.db.get_total_violations(),
                    plate_logs=plate_logs
                )

                # --- Step 8: Display & Write Output ---
                if out:
                    out.write(frame)

                if self.config.display:
                    cv2.imshow("Real-Time Traffic Analysis System", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == ord('p'):
                        print("[Pipeline] User requested exit.")
                        break

        finally:
            cap.release()
            if out:
                out.release()
            cv2.destroyAllWindows()
            print("[Pipeline] Video stream closed successfully.")
            self._print_summary()

    def _print_summary(self):
        """Print end-of-run traffic database summary report."""
        records = self.db.get_all_records()
        print("\n" + "=" * 60)
        print(" TRAFFIC ANALYSIS PIPELINE SUMMARY REPORT")
        print("=" * 60)
        print(f"Total Tracked Vehicles: {len(records)}")
        print(f"Total Counted:          {self.counter.total_count}")
        print(f"Total Violations:       {self.db.get_total_violations()}")
        print(f"Average Vehicle Speed:  {self.db.get_average_speed():.1f} km/h")
        print("-" * 60)
        print("Sample Vehicle Database Records:")
        for t_id, r in list(records.items())[:10]:
            print(f" ID #{t_id:03d} | Class: {r.class_name:<10} | Plate: {r.license_plate:<10} | MaxSpeed: {r.max_speed:4.1f} km/h | Violations: {r.violations}")
        print("=" * 60 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Real-Time Traffic Analysis System")
    parser.add_argument("--source", type=str, default="DATA/INPUTS/cars_on_highway_2.mp4", help="Video file path or RTSP camera URL")
    parser.add_argument("--weights", type=str, default="weights/yolov5s-pruned-finetuned.onnx", help="Detector model weights (.onnx / .pt)")
    parser.add_argument("--detector", type=str, default="onnx", choices=["yolov5", "rtdetr", "onnx", "ultralytics"], help="Detector backend")
    parser.add_argument("--no-display", action="store_true", help="Disable GUI display window")
    parser.add_argument("--output", type=str, default="DATA/OUTPUTS/traffic_analysis_output.mp4", help="Output MP4 file path")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = TrafficAnalysisConfig()
    cfg.video_source = args.source
    cfg.model_weights = args.weights
    cfg.detector_type = args.detector
    cfg.display = not args.no_display
    cfg.output_path = args.output

    pipeline = TrafficAnalysisPipeline(cfg)
    pipeline.run()
