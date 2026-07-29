import os
import torch
import numpy as np
import cv2
from typing import List, Tuple, Dict, Any, Union

class VehicleDetector:
    """
    Unified Vehicle Detector supporting YOLOv5 (PyTorch/ONNX), Ultralytics YOLO, and RT-DETR.
    Outputs detections in standard format: xyxy, confidence, class_id.
    """
    def __init__(self, config):
        self.config = config
        self.detector_type = config.detector_type.lower()
        self.weights = config.model_weights
        if not os.path.exists(self.weights) and os.path.exists(os.path.join("weights", self.weights)):
            self.weights = os.path.join("weights", self.weights)

        self.conf_thresh = config.confidence_threshold
        self.iou_thresh = config.iou_threshold
        self.device = torch.device(config.device if torch.cuda.is_available() and config.device == "cuda" else "cpu")
        self.target_classes = config.target_classes
        
        self.model = None
        self.session = None
        self._init_detector()

    def _init_detector(self):
        """Initialize the requested detector backend."""
        print(f"[Detector] Initializing {self.detector_type} detector with weights: {self.weights}")
        
        if self.detector_type == "onnx" or self.weights.endswith(".onnx"):
            import onnxruntime
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device.type == 'cuda' else ['CPUExecutionProvider']
            try:
                self.session = onnxruntime.InferenceSession(self.weights, providers=providers)
            except Exception:
                self.session = onnxruntime.InferenceSession(self.weights, providers=['CPUExecutionProvider'])
            self.input_name = self.session.get_inputs()[0].name
            print(f"[Detector] Loaded ONNX session with providers: {self.session.get_providers()}")

        elif self.detector_type == "ultralytics":
            from ultralytics import YOLO
            self.model = YOLO(self.weights)
            print("[Detector] Loaded Ultralytics model successfully.")

        elif self.detector_type == "rtdetr":
            if os.path.exists(self.weights) and os.path.isdir(self.weights):
                from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
                self.processor = RTDetrImageProcessor.from_pretrained(self.weights)
                self.model = RTDetrForObjectDetection.from_pretrained(self.weights).to(self.device).eval()
            else:
                from ultralytics import RTDETR
                self.model = RTDETR(self.weights)
            print("[Detector] Loaded RT-DETR model successfully.")

        elif self.detector_type == "yolov5":
            if self.weights.endswith(".onnx"):
                import onnxruntime
                self.session = onnxruntime.InferenceSession(self.weights, providers=['CPUExecutionProvider'])
                self.input_name = self.session.get_inputs()[0].name
            else:
                # Load PyTorch model checkpoint
                try:
                    from models.experimental import attempt_load
                    self.model = attempt_load(self.weights, map_location=self.device).eval()
                except Exception:
                    # Fallback to torch.hub or Ultralytics
                    self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=self.weights, trust_repo=True).eval()

    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run detection on a single frame.
        
        Args:
            frame: BGR numpy image (H, W, 3)
            
        Returns:
            xyxy: bounding boxes (N, 4)
            confidences: confidence scores (N,)
            class_ids: integer class labels (N,)
        """
        h, w, _ = frame.shape
        
        if self.session is not None:
            # ONNX Runtime Inference Pipeline
            from utils.augmentations import letterbox
            from utils.general import non_max_suppression, scale_coords
            
            img, ratio, (dw, dh) = letterbox(frame, new_shape=(640, 640), auto=False)
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
            img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
            img = np.expand_dims(img, axis=0)

            ort_inputs = {self.input_name: img}
            ort_outs = self.session.run(None, ort_inputs)
            pred = torch.from_numpy(ort_outs[0])

            # Non-Maximum Suppression
            pred = non_max_suppression(
                pred, conf_thres=self.conf_thresh, iou_thres=self.iou_thresh, classes=self.target_classes
            )

            det = pred[0]
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
                xyxy = det[:, :4].cpu().numpy()
                confidences = det[:, 4].cpu().numpy()
                class_ids = det[:, 5].cpu().numpy().astype(int)
                return xyxy, confidences, class_ids

        elif hasattr(self, 'model') and self.model is not None:
            if hasattr(self.model, 'predict'):
                # Ultralytics prediction pipeline
                results = self.model.predict(
                    source=frame,
                    conf=self.conf_thresh,
                    iou=self.iou_thresh,
                    classes=self.target_classes,
                    verbose=False
                )[0]
                boxes = results.boxes
                if len(boxes) > 0:
                    xyxy = boxes.xyxy.cpu().numpy()
                    conf = boxes.conf.cpu().numpy()
                    cls = boxes.cls.cpu().numpy().astype(int)
                    return xyxy, conf, cls
            else:
                # PyTorch YOLOv5 model pipeline
                from utils.augmentations import letterbox
                from utils.general import non_max_suppression, scale_coords
                
                img, ratio, (dw, dh) = letterbox(frame, new_shape=(640, 640), auto=False)
                img = img[:, :, ::-1].transpose(2, 0, 1)
                img = torch.from_numpy(np.ascontiguousarray(img)).to(self.device).float() / 255.0
                img = img.unsqueeze(0)

                with torch.no_grad():
                    pred = self.model(img)[0]
                    pred = non_max_suppression(pred, self.conf_thresh, self.iou_thresh, classes=self.target_classes)

                det = pred[0]
                if len(det):
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
                    xyxy = det[:, :4].cpu().numpy()
                    confidences = det[:, 4].cpu().numpy()
                    class_ids = det[:, 5].cpu().numpy().astype(int)
                    return xyxy, confidences, class_ids

        # Return empty arrays if no detections
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=int)
