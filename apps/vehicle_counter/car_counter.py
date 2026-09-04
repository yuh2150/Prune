


import cv2 as cv
import numpy as np
import supervision as sv
import onnxruntime
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

# Initialize YOLO model and video info
onnx_model_path = "yolov5s-pruned-finetuned.onnx"
providers = ['CPUExecutionProvider']
session = onnxruntime.InferenceSession(onnx_model_path, providers=providers)

video_path = "DATA/INPUTS/cars_on_highway_2.mp4"
video_info = sv.VideoInfo.from_video_path(video_path)
w, h, fps = video_info.width, video_info.height, video_info.fps

# Setup annotators
thickness = sv.calculate_optimal_line_thickness(resolution_wh=video_info.resolution_wh)
text_scale = sv.calculate_optimal_text_scale(resolution_wh=video_info.resolution_wh)

box_annotator = sv.RoundBoxAnnotator(thickness=thickness, color_lookup=sv.ColorLookup.TRACK)
label_annotator = sv.LabelAnnotator(text_scale=text_scale, text_thickness=thickness,
                                    text_position=sv.Position.TOP_CENTER, color_lookup=sv.ColorLookup.TRACK)
trace_annotator = sv.TraceAnnotator(thickness=thickness, trace_length=fps * 2,
                                    position=sv.Position.CENTER, color_lookup=sv.ColorLookup.TRACK)

# Tracker and vehicle class setup
tracker = sv.ByteTrack(frame_rate=video_info.fps)
smoother = sv.DetectionsSmoother()
class_names = {
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck'
}
selected_classes = [2, 3, 5, 7]

# Initialize counters
limits = [0, 300, 1280, 300]  # Line for vehicle counting
partition_limit = 550
total_counts, crossed_ids = [], set()

total_counts_up, crossed_ids_up = [], set()
total_counts_down, crossed_ids_down = [], set()


def draw_overlay(frame, pt1, pt2, alpha=0.25, color=(51, 68, 255), filled=True):
    """Draws a semi-transparent overlay rectangle."""
    overlay = frame.copy()
    rect_color = color if filled else (0, 0, 0)
    cv.rectangle(overlay, pt1, pt2, rect_color, cv.FILLED if filled else 1)
    cv.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def count_vehicles(track_id, cx, cy, limits, crossed_ids):
    """Counts vehicles crossing the line."""
    if limits[0] < cx < limits[2] and limits[1] - 10 < cy < limits[1] + 10 and track_id not in crossed_ids:
        crossed_ids.add(track_id)
        return True
    return False


def count_vehicles_up(track_id, cx, cy, limits, crossed_ids_up):
    """Counts vehicles crossing the line."""
    if limits[0] < cx < partition_limit and limits[1] - 10 < cy < limits[1] + 10 and track_id not in crossed_ids_up:
        crossed_ids_up.add(track_id)
        return True
    return False


def count_vehicles_down(track_id, cx, cy, limits, crossed_ids_down):
    """Counts vehicles crossing the line."""
    if partition_limit < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15 and track_id not in crossed_ids_down:
        crossed_ids_down.add(track_id)
        return True
    return False


def draw_tracks_and_count(frame, detections, total_counts, limits):
    """Annotates the frame with detected tracks and counts vehicles."""
    detections = detections[np.isin(detections.class_id, selected_classes)]  # Filter by vehicle classes
    labels = [f"#{track_id} {class_names[cls_id]}" for track_id, cls_id in
              zip(detections.tracker_id, detections.class_id)]

    label_annotator.annotate(frame, detections=detections, labels=labels)
    box_annotator.annotate(frame, detections=detections)
    trace_annotator.annotate(frame, detections=detections)

    for track_id, center_point in zip(detections.tracker_id,
                                      detections.get_anchors_coordinates(anchor=sv.Position.CENTER)):
        cx, cy = map(int, center_point)
        cv.circle(frame, (cx, cy), 4, (0, 255, 255), cv.FILLED)  # Draw vehicle center point

        # Storing the counts
        if count_vehicles(track_id, cx, cy, limits, crossed_ids):
            total_counts.append(track_id)
            sv.draw_line(frame, start=sv.Point(x=limits[0], y=limits[1]), end=sv.Point(x=limits[2], y=limits[3]),
                         color=sv.Color.ROBOFLOW, thickness=4)
            draw_overlay(frame, (0, 200), (1287, 400), alpha=0.25, color=(10, 255, 50))

        if count_vehicles_up(track_id, cx, cy, limits, crossed_ids_up):
            total_counts_up.append(track_id)
        if count_vehicles_down(track_id, cx, cy, limits, crossed_ids_down):
            total_counts_down.append(track_id)

    # Annotating the total counts
    sv.draw_text(frame, f"COUNTS: {len(total_counts)}", sv.Point(x=120, y=30), sv.Color.ROBOFLOW, 1.25,
                 2, background_color=sv.Color.WHITE)
    sv.draw_text(frame, f"UP: {len(total_counts_up)}", sv.Point(x=560, y=280), sv.Color.WHITE, 1,
                 2)
    sv.draw_text(frame, f"DOWN: {len(total_counts_down)}", sv.Point(x=560, y=320), sv.Color.WHITE, 1,
                 2)


cap = cv.VideoCapture(video_path)
output_path = "DATA/OUTPUTS/car_counter_4.mp4"
out = cv.VideoWriter(output_path, cv.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

if not cap.isOpened():
    raise Exception("Error: couldn't open the video!")

# Video processing loop
prev_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Calculate FPS
    curr_time = time.time()
    fps_display = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0.0
    prev_time = curr_time

    crop = frame[150:, :]
    mask_b = np.zeros_like(frame, dtype=np.uint8)
    mask_w = np.ones_like(frame[150:, :], dtype=np.uint8) * 255
    mask_b[150:, :] = mask_w

    # Apply the mask to the original frame
    ROI = cv.bitwise_and(frame, mask_b)
    
    # Preprocessing
    img, ratio, (dw, dh) = letterbox(ROI, new_shape=(640, 640), auto=False)
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
    img = np.ascontiguousarray(img, dtype=np.float32) / 255.0  # normalize
    img = np.expand_dims(img, axis=0)  # batch dimension (1, 3, 640, 640)

    # Inference
    ort_inputs = {session.get_inputs()[0].name: img}
    ort_outs = session.run(None, ort_inputs)
    pred = torch.from_numpy(ort_outs[0])

    # NMS
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, classes=selected_classes)

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

    # Tracker and smoother
    detections = tracker.update_with_detections(detections)
    detections = smoother.update_with_detections(detections)

    # Draw counting line and overlay
    sv.draw_line(frame, start=sv.Point(x=limits[0], y=limits[1]), end=sv.Point(x=limits[2], y=limits[3]),
                 color=sv.Color.RED, thickness=4)
    draw_overlay(frame, (0, 200), (1287, 400), alpha=0.2)

    if detections.tracker_id is not None and len(detections.tracker_id) > 0:
        draw_tracks_and_count(frame, detections, total_counts, limits)
    else:
        # Draw counts even if tracker is empty
        sv.draw_text(frame, f"COUNTS: {len(total_counts)}", sv.Point(x=120, y=30), sv.Color.ROBOFLOW, 1.25,
                     2, background_color=sv.Color.WHITE)
        sv.draw_text(frame, f"UP: {len(total_counts_up)}", sv.Point(x=560, y=280), sv.Color.WHITE, 1,
                     2)
        sv.draw_text(frame, f"DOWN: {len(total_counts_down)}", sv.Point(x=560, y=320), sv.Color.WHITE, 1,
                     2)

    # Draw FPS on frame
    sv.draw_text(frame, f"FPS: {fps_display:.1f}", sv.Point(x=120, y=80), sv.Color.ROBOFLOW, 1.25,
                 2, background_color=sv.Color.WHITE)

    # Writing the frames to save the object
    out.write(frame)
    cv.imshow("Camera", frame)

    if cv.waitKey(1) & 0xff == ord('p'):  # Pause with 'p'
        break

# Release the resources
cap.release()
out.release()
cv.destroyAllWindows()
