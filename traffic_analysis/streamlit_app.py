import os
import sys
import tempfile
import time
import cv2
import pandas as pd
import numpy as np
import streamlit as st
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

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Real-Time Traffic Analytics",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🚦 Real-Time Intelligent Traffic Analysis System")
st.markdown("Modular Deep Learning Traffic Analytics powered by Pruned Object Detection Models & ByteTrack.")

# --- Sidebar Options ---
st.sidebar.header("⚙️ Pipeline Configuration")

detector_type = st.sidebar.selectbox(
    "Detector Backend",
    options=["onnx", "yolov5", "rtdetr", "ultralytics"],
    index=0
)

# Scan weights folder
weights_dir = ROOT / "weights"
available_weights = [f.name for f in weights_dir.glob("*.*") if f.suffix in [".pt", ".onnx"]]
default_weight_idx = available_weights.index("yolov5s-pruned-finetuned.onnx") if "yolov5s-pruned-finetuned.onnx" in available_weights else 0

selected_weight = st.sidebar.selectbox(
    "Model Weights",
    options=available_weights,
    index=default_weight_idx
)

conf_thresh = st.sidebar.slider("Confidence Threshold", 0.1, 0.9, 0.25, 0.05)
iou_thresh = st.sidebar.slider("IoU NMS Threshold", 0.1, 0.9, 0.45, 0.05)

st.sidebar.subheader("📹 Video Source")
source_option = st.sidebar.radio("Input Source", ["Sample Video", "Upload Video File"])

video_path = None
if source_option == "Sample Video":
    sample_files = [f.name for f in (ROOT / "DATA" / "INPUTS").glob("*.mp4")]
    selected_sample = st.sidebar.selectbox("Select Sample Video", sample_files)
    video_path = str(ROOT / "DATA" / "INPUTS" / selected_sample)
else:
    uploaded_file = st.sidebar.file_uploader("Upload Video (MP4/AVI)", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        video_path = tfile.name

st.sidebar.subheader("🧩 Analytics Modules")
enable_speed = st.sidebar.checkbox("Speed Estimation (Homography)", value=True)
enable_violations = st.sidebar.checkbox("Violation Detection (Wrong-Way/Lane/Parking)", value=True)
enable_heatmap = st.sidebar.checkbox("Motion Heatmap Overlay", value=True)
enable_lpr = st.sidebar.checkbox("License Plate Recognition (OCR)", value=True)
allow_synth = st.sidebar.checkbox("Demo Synthetic Plate Fallback", value=False)

start_button = st.sidebar.button("🚀 Start Stream Processing", type="primary")
stop_button = st.sidebar.button("⏹ Stop Processing")

# --- Layout KPI Cards ---
kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)
fps_metric = kpi_col1.metric("FPS / Latency", "0.0 FPS", "0 ms")
active_metric = kpi_col2.metric("Active Vehicles", "0", "Low")
counted_metric = kpi_col3.metric("Total Counted", "0", "0 IN / 0 OUT")
speed_metric = kpi_col4.metric("Average Speed", "0.0 km/h")
violation_metric = kpi_col5.metric("Total Violations", "0 Alerts")

# --- Main Columns ---
col_video, col_charts = st.columns([2, 1])

with col_video:
    st.subheader("📹 Live Video Feed & Analytics Overlay")
    st_frame = st.empty()

with col_charts:
    st.subheader("📊 Class Distribution & Plates")
    st_chart = st.empty()
    st.subheader("🚨 Recognized License Plates")
    st_plates_table = st.empty()

st.divider()
st.subheader("📋 Vehicle Tracking Database")
st_db_table = st.empty()

# --- Execution Loop ---
if start_button and video_path:
    # Initialize Configuration
    cfg = TrafficAnalysisConfig()
    cfg.detector_type = detector_type
    cfg.model_weights = str(weights_dir / selected_weight)
    cfg.confidence_threshold = conf_thresh
    cfg.iou_threshold = iou_thresh
    cfg.video_source = video_path
    cfg.enable_speed_estimation = enable_speed
    cfg.enable_heatmap = enable_heatmap
    cfg.enable_lpr = enable_lpr
    cfg.allow_synth_fallback = allow_synth
    cfg.display = False

    # Instantiate Modules
    detector = VehicleDetector(cfg)
    tracker = VehicleTracker(cfg)
    homography = HomographyTransformer(cfg.pixel_points, cfg.world_points)
    speed_estimator = SpeedEstimator(cfg, homography)
    counter = VehicleCounter(cfg)
    lane_manager = LaneManager(cfg)
    wrong_way_detector = WrongWayDetector(cfg)
    parking_detector = IllegalParkingDetector(cfg)
    plate_detector = LicensePlateDetector()
    plate_recognizer = LicensePlateRecognizer(cfg.ocr_engine, cfg.allow_synth_fallback)
    db = VehicleDatabase()
    dashboard = TrafficDashboard(cfg)
    fps_counter = FPSCounter()

    cap = cv2.VideoCapture(video_path)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    heatmap = HeatmapGenerator(cfg, frame_shape=(h, w))

    st.success("Pipeline running! Processing stream...")

    frame_idx = 0
    while cap.isOpened():
        if stop_button:
            st.warning("Processing stopped by user.")
            break

        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        fps = fps_counter.tick()
        latency_ms = fps_counter.latency_ms

        # 1. Detect & Track
        xyxy, conf, class_ids = detector.detect(frame)
        detections = tracker.update(xyxy, conf, class_ids)

        active_centroids = []
        active_violations_frame = []
        plate_logs = []

        if detections.tracker_id is not None and len(detections.tracker_id) > 0:
            centers = detections.get_anchors_coordinates(anchor=sv.Position.CENTER)
            for i, (track_id, bbox, cls_id) in enumerate(zip(detections.tracker_id, detections.xyxy, detections.class_id)):
                track_id = int(track_id)
                cls_name = cfg.class_names.get(int(cls_id), "vehicle")
                cx, cy = float(centers[i][0]), float(centers[i][1])
                active_centroids.append((cx, cy))

                traj = tracker.get_trajectory(track_id)
                speed_kmh = speed_estimator.update(track_id, traj) if enable_speed else 0.0

                violation_str = None
                if enable_violations:
                    if wrong_way_detector.check(track_id, traj):
                        violation_str = "Wrong Way"
                        active_violations_frame.append(f"#{track_id} Wrong Way")
                    elif lane_manager.check_violation(track_id, cls_id, (cx, cy)):
                        violation_str = "Lane Violation"
                        active_violations_frame.append(f"#{track_id} Lane Violation")
                    elif parking_detector.update(track_id, traj):
                        violation_str = "Illegal Parking"
                        active_violations_frame.append(f"#{track_id} Illegal Parking")

                plate_str = "PENDING"
                if enable_lpr and frame_idx % 5 == 0:
                    x1, y1, x2, y2 = map(int, bbox)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    v_crop = frame[y1:y2, x1:x2]
                    p_crop = plate_detector.detect_and_crop(v_crop)
                    plate_str = plate_recognizer.recognize(p_crop, track_id)

                rec = db.update_vehicle(
                    tracker_id=track_id,
                    class_name=cls_name,
                    centroid=(cx, cy, tracker.frame_count),
                    speed=speed_kmh,
                    license_plate=plate_str,
                    violation=violation_str
                )

                if rec.license_plate and rec.license_plate not in ["PENDING", "N/A"]:
                    plate_logs.append({"ID": track_id, "Class": rec.class_name, "Plate": rec.license_plate})

        # Analytics Updates
        counts = counter.update(detections, tracker.trajectories)
        density_info = lane_manager.compute_density(detections)

        # Rendering
        if enable_heatmap:
            heatmap.update(active_centroids)
            frame = heatmap.generate_overlay(frame)

        frame = lane_manager.draw_lanes(frame)
        frame = parking_detector.draw_rois(frame)
        frame = counter.draw_counting_line(frame)

        if detections.tracker_id is not None and len(detections.tracker_id) > 0:
            centers = detections.get_anchors_coordinates(anchor=sv.Position.CENTER)
            for i, (track_id, bbox, cls_id) in enumerate(zip(detections.tracker_id, detections.xyxy, detections.class_id)):
                track_id = int(track_id)
                cls_name = cfg.class_names.get(int(cls_id), "vehicle")
                x1, y1, x2, y2 = map(int, bbox)
                rec = db.get_record(track_id)
                speed = rec.current_speed if rec else 0.0
                plate = rec.license_plate if rec else ""
                box_color = (0, 0, 255) if (rec and rec.violations) else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                label = f"#{track_id} {cls_name} {speed:.0f}km/h"
                draw_text_box(frame, label, (x1, max(15, y1 - 8)), font_scale=0.45, bg_color=box_color)

        frame = dashboard.draw_hud(
            frame=frame,
            fps=fps,
            latency_ms=latency_ms,
            counts=counts,
            avg_speed=db.get_average_speed(),
            density_info=density_info,
            active_violations=active_violations_frame,
            total_violations=db.get_total_violations(),
            plate_logs=[f"#{p['ID']} {p['Plate']}" for p in plate_logs]
        )

        # Streamlit Image Display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st_frame.image(frame_rgb, channels="RGB", use_container_width=True)

        # Update Metrics
        fps_metric.metric("FPS / Latency", f"{fps:.1f} FPS", f"{latency_ms:.1f} ms")
        active_metric.metric("Active Vehicles", f"{density_info['active_vehicles']}", f"Density: {density_info['level']}")
        counted_metric.metric("Total Counted", f"{counts['total']}", f"IN: {counts['in']} | OUT: {counts['out']}")
        speed_metric.metric("Average Speed", f"{db.get_average_speed():.1f} km/h")
        violation_metric.metric("Total Violations", f"{db.get_total_violations()} Alerts")

        # Class Breakdown Chart
        if counts.get("per_class"):
            df_class = pd.DataFrame(list(counts["per_class"].items()), columns=["Class", "Count"])
            st_chart.bar_chart(df_class.set_index("Class"))

        # Recognized Plates Table
        if plate_logs:
            st_plates_table.dataframe(pd.DataFrame(plate_logs).tail(5), use_container_width=True)

        # Vehicle Database Table
        all_records = db.get_all_records()
        if all_records:
            records_data = [
                {
                    "ID": r.tracker_id,
                    "Class": r.class_name,
                    "Plate": r.license_plate,
                    "Current Speed (km/h)": round(r.current_speed, 1),
                    "Max Speed (km/h)": round(r.max_speed, 1),
                    "Violations": ", ".join(r.violations) if r.violations else "None"
                }
                for r in all_records.values()
            ]
            st_db_table.dataframe(pd.DataFrame(records_data).tail(10), use_container_width=True)

    cap.release()
    st.info("Stream processing finished.")
