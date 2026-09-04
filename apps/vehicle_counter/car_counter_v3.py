import cv2 as cv
import numpy as np
import supervision as sv
import torch
import sys
import time
from pathlib import Path

# Add project root to sys.path to allow importing utils
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.general import non_max_suppression, scale_coords
from utils.augmentations import letterbox

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--weights', type=str, default='weights/yolov5s.onnx', help='model weights path')
parser.add_argument('--input', type=str, default='DATA/INPUTS/cars_on_highway_3.mp4', help='input video path')
opt = parser.parse_args()

# Initialize YOLO model and video info
model_path = opt.weights
video_path = opt.input
video_info = sv.VideoInfo.from_video_path(video_path)
w, h, fps = video_info.width, video_info.height, video_info.fps

# Setup annotators
thickness = sv.calculate_optimal_line_thickness(resolution_wh=video_info.resolution_wh)
text_scale = sv.calculate_optimal_text_scale(resolution_wh=video_info.resolution_wh)

box_annotator = sv.RoundBoxAnnotator(thickness=thickness, color_lookup=sv.ColorLookup.TRACK)
label_annotator = sv.LabelAnnotator(text_scale=text_scale, text_thickness=thickness,
                                    text_position=sv.Position.TOP_CENTER, color_lookup=sv.ColorLookup.TRACK)

# Tracker and vehicle class setup
tracker = sv.ByteTrack(
    frame_rate=video_info.fps,
    track_activation_threshold=0.35,
    lost_track_buffer=15,
    minimum_matching_threshold=0.8,
    minimum_consecutive_frames=2
)
smoother = sv.DetectionsSmoother()

# Standard COCO vehicle mapping for ONNX models
class_names = {
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck'
}
selected_classes = [2, 3, 5, 7]

# Initialize OpenVINO if installed, else fallback to ONNX Runtime
try:
    import openvino as ov
    use_openvino = True
    print("[Info] Loading model with OpenVINO...")
    core = ov.Core()
    ov_model = core.read_model(model_path)
    compiled_model = core.compile_model(ov_model, "CPU")
    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)
    print("[Info] Loaded OpenVINO model successfully.")
except ImportError:
    import onnxruntime
    use_openvino = False
    print("[Info] OpenVINO not installed. Falling back to ONNX Runtime...")
    session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    print("[Info] Loaded ONNX Runtime session successfully.")

# Initialize counters
limits = [0, 487, 1117, 487]  # Line for vehicle counting
total_counts, crossed_ids = [], set()
class_counts = {
    'car': set(),
    'motorcycle': set(),
    'bus': set(),
    'truck': set()
}


def draw_overlay(frame, pt1, pt2, alpha=0.25, color=(51, 68, 255), filled=True):
    """Draws a semi-transparent overlay rectangle."""
    overlay = frame.copy()
    rect_color = color if filled else (0, 0, 0)
    cv.rectangle(overlay, pt1, pt2, rect_color, cv.FILLED if filled else 1)
    cv.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def count_vehicles(track_id, cx, cy, limits, crossed_ids):
    """Counts vehicles crossing the line."""
    if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15 and track_id not in crossed_ids:
        crossed_ids.add(track_id)
        return True
    return False


def draw_tracks_and_count(frame, detections, total_counts, limits):
    """Annotates the frame with detected tracks and counts vehicles."""
    detections = detections[np.isin(detections.class_id, selected_classes)]  # Filter by vehicle classes

    # Annotating the Bounding Boxes and Labels
    labels = [f"#{track_id} {class_names[cls_id]}" for track_id, cls_id in
              zip(detections.tracker_id, detections.class_id)]
    box_annotator.annotate(frame, detections=detections)
    label_annotator.annotate(frame, detections=detections, labels=labels)

    # Annotate tracks and vehicles
    for track_id, center_point, cls_id in zip(detections.tracker_id,
                                              detections.get_anchors_coordinates(anchor=sv.Position.CENTER),
                                              detections.class_id):
        cx, cy = map(int, center_point)

        if count_vehicles(track_id, cx, cy, limits, crossed_ids):
            total_counts.append(track_id)
            cls_name = class_names.get(cls_id, 'car')
            class_counts[cls_name].add(track_id)
            
            sv.draw_line(frame, start=sv.Point(x=limits[0], y=limits[1]), end=sv.Point(x=limits[2], y=limits[3]),
                         color=sv.Color.ROBOFLOW, thickness=4)
            draw_overlay(frame, (0, 387), (1117, 587), alpha=0.25, color=(10, 255, 50))

    # Display the total counts on the frame
    sv.draw_text(frame, f"COUNTS: {len(total_counts)}", sv.Point(x=120, y=30), sv.Color.ROBOFLOW, 1.25,
                 2, background_color=sv.Color.WHITE)

    # Display the breakdown counts by category
    y_offset = 130
    for cls_name, track_ids in class_counts.items():
        sv.draw_text(frame, f"{cls_name.upper()}: {len(track_ids)}", sv.Point(x=120, y=y_offset), sv.Color.WHITE, 0.8,
                     2, background_color=sv.Color.BLACK)
        y_offset += 40


cap = cv.VideoCapture(video_path)
model_name = Path(model_path).stem
output_path = f"DATA/OUTPUTS/car_counter_3_{model_name}.mp4"
Path(output_path).parent.mkdir(parents=True, exist_ok=True)
out = cv.VideoWriter(output_path, cv.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

if not cap.isOpened():
    raise Exception("Error: couldn't open the video!")

# Video processing loop
prev_time = time.time()

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Calculate FPS
        curr_time = time.time()
        fps_display = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0.0
        prev_time = curr_time

        # Define region of interest (ROI)
        pts = np.array([[380, 182], [0, 412], [0, 720], [1080, 720], [861, 182]], np.int32).reshape((-1, 1, 2))
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv.fillPoly(mask, [pts], (255, 255, 255))  # Mask the polygon
        ROI = cv.bitwise_and(frame, frame, mask=mask)

        # Preprocessing
        img, ratio, (dw, dh) = letterbox(ROI, new_shape=(640, 640), auto=False)
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
        img = np.ascontiguousarray(img, dtype=np.float32) / 255.0  # normalize
        img = np.expand_dims(img, axis=0)  # batch dimension (1, 3, 640, 640)

        # Inference
        if use_openvino:
            results = compiled_model([img])[output_layer]
            pred = torch.from_numpy(results)
        else:
            ort_inputs = {input_name: img}
            ort_outs = session.run(None, ort_inputs)
            pred = torch.from_numpy(ort_outs[0])

        # NMS
        pred = non_max_suppression(pred, conf_thres=0.35, iou_thres=0.45, classes=selected_classes)

        # Rescale coords and format detections
        det = pred[0]
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], ROI.shape).round()
            xyxy = det[:, :4].cpu().numpy()
            confidence = det[:, 4].cpu().numpy()
            class_id = det[:, 5].cpu().numpy().astype(int)
            detections = sv.Detections(
                xyxy=xyxy,
                confidence=confidence,
                class_id=class_id
            )
        else:
            detections = sv.Detections(
                xyxy=np.empty((0, 4), dtype=np.float32),
                confidence=np.empty((0,), dtype=np.float32),
                class_id=np.empty((0,), dtype=int)
            )

        # Tracker & Smoother
        detections = tracker.update_with_detections(detections)
        detections = smoother.update_with_detections(detections)

        if detections.tracker_id is not None and len(detections.tracker_id) > 0:
            # Draw counting line and process vehicle tracks
            sv.draw_line(frame, start=sv.Point(x=limits[0], y=limits[1]), end=sv.Point(x=limits[2], y=limits[3]),
                         color=sv.Color.RED, thickness=4)
            draw_overlay(frame, (0, 387), (1117, 587), alpha=0.2)
            draw_tracks_and_count(frame, detections, total_counts, limits)
        else:
            # Draw counts even if tracker is empty
            sv.draw_text(frame, f"COUNTS: {len(total_counts)}", sv.Point(x=120, y=30), sv.Color.ROBOFLOW, 1.25,
                         2, background_color=sv.Color.WHITE)
            # Display the breakdown counts by category
            y_offset = 130
            for cls_name, track_ids in class_counts.items():
                sv.draw_text(frame, f"{cls_name.upper()}: {len(track_ids)}", sv.Point(x=120, y=y_offset), sv.Color.WHITE, 0.8,
                             2, background_color=sv.Color.BLACK)
                y_offset += 40

        # Draw FPS on frame
        sv.draw_text(frame, f"FPS: {fps_display:.1f}", sv.Point(x=120, y=80), sv.Color.ROBOFLOW, 1.25,
                     2, background_color=sv.Color.WHITE)

        out.write(frame)
        cv.imshow("Camera", frame)

        if cv.waitKey(1) & 0xff == ord('p'):  # Pause with 'p'
            break
finally:
    cap.release()
    out.release()
    cv.destroyAllWindows()
    print(f"[Info] Video processing finished. Output saved to: {output_path}")
