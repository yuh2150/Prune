import cv2
import numpy as np
from typing import Optional, Tuple

class LicensePlateDetector:
    """
    Detects and crops license plate bounding boxes from vehicle image crops.
    Employs Morphological Top-Hat/Black-Hat transforms, Sobel edge analysis,
    and aspect-ratio bounding box isolation with padding.
    """
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.model = None
        if model_path:
            try:
                from ultralytics import YOLO
                self.model = YOLO(model_path)
            except Exception:
                pass

    def detect_and_crop(self, vehicle_crop: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract license plate crop from vehicle image.
        
        Args:
            vehicle_crop: BGR vehicle crop numpy array
            
        Returns:
            License plate BGR crop numpy array or lower vehicle region
        """
        if vehicle_crop is None or vehicle_crop.size == 0:
            return None

        h, w, _ = vehicle_crop.shape
        if h < 15 or w < 20:
            return None

        # 1. Use License Plate YOLO Model if loaded
        if self.model is not None:
            res = self.model.predict(vehicle_crop, conf=0.3, verbose=False)[0]
            if len(res.boxes) > 0:
                box = res.boxes[0].xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = box
                return vehicle_crop[y1:y2, x1:x2]

        # 2. Heuristic Contour & Morphological Plate Localization
        # Focus on lower 65% of vehicle where plates are positioned
        lower_crop = vehicle_crop[int(h * 0.35):, :]
        lh, lw, _ = lower_crop.shape
        gray = cv2.cvtColor(lower_crop, cv2.COLOR_BGR2GRAY)

        # Black-Hat Morphology to isolate dark text on light plate background (or vice versa)
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rect_kernel)

        # Sobel Horizontal Gradient
        sobel_x = cv2.Sobel(blackhat, cv2.CV_8U, 1, 0, ksize=3)
        sobel_x = cv2.convertScaleAbs(sobel_x)

        # Otsu Thresholding
        _, thresh = cv2.threshold(sobel_x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Connect horizontal characters
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, close_kernel)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            aspect_ratio = cw / float(ch)
            area = cw * ch

            # License plate aspect ratio usually 1.8 to 6.5
            if 1.8 <= aspect_ratio <= 6.5 and cw >= 22 and ch >= 8 and area >= 180:
                candidates.append((area, x, y, cw, ch))

        if candidates:
            candidates.sort(key=lambda item: item[0], reverse=True)
            _, x, y, cw, ch = candidates[0]
            
            # Add 5% padding around candidate box
            pad_x = int(cw * 0.05)
            pad_y = int(ch * 0.05)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(lw, x + cw + pad_x)
            y2 = min(lh, y + ch + pad_y)
            return lower_crop[y1:y2, x1:x2]

        # Fallback: return lower vehicle crop directly if no candidate contour isolated
        return lower_crop
