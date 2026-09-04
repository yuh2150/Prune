import cv2
import re
import numpy as np
from typing import Optional, Dict, Tuple, List, Any

class LicensePlateRecognizer:
    """
    OCR Engine for reading license plate numbers from cropped plate images.
    Supports EasyOCR and PaddleOCR with multi-pass preprocessing, character allowlist,
    and progressive cache refinement as vehicles move closer to the camera.
    """
    def __init__(self, engine_name: str = "easyocr", allow_synth_fallback: bool = False):
        self.engine_name = engine_name.lower()
        self.allow_synth_fallback = allow_synth_fallback
        self.ocr_engine = None
        
        # Cache structure: tracker_id -> {'plate': str, 'length': int, 'conf': float}
        self.plate_cache: Dict[int, Dict[str, Any]] = {}
        self._init_ocr()

    def _init_ocr(self):
        """Initialize chosen OCR engine safely."""
        if self.engine_name == "easyocr":
            try:
                import easyocr
                self.ocr_engine = easyocr.Reader(['en'], gpu=False)
                print("[OCR] EasyOCR engine initialized with character allowlist.")
            except Exception as e:
                print(f"[OCR] Warning initializing EasyOCR: {e}")
                self.ocr_engine = None
        elif self.engine_name == "paddleocr":
            try:
                from paddleocr import PaddleOCR
                self.ocr_engine = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
                print("[OCR] PaddleOCR engine initialized.")
            except Exception as e:
                print(f"[OCR] Warning initializing PaddleOCR: {e}")
                self.ocr_engine = None

    def preprocess_clahe(self, plate_crop: np.ndarray) -> np.ndarray:
        """Apply CLAHE contrast enhancement and resizing."""
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        target_h = 64
        target_w = max(64, int(w * (target_h / float(h))))
        resized = cv2.resize(gray, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        return clahe.apply(resized)

    def preprocess_otsu(self, plate_crop: np.ndarray) -> np.ndarray:
        """Apply Otsu adaptive binarization."""
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        target_h = 64
        target_w = max(64, int(w * (target_h / float(h))))
        resized = cv2.resize(gray, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        blur = cv2.GaussianBlur(resized, (3, 3), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh

    def recognize(self, plate_crop: np.ndarray, tracker_id: int) -> str:
        """
        Recognize license plate string from crop and dynamically refine per tracker_id.
        """
        existing = self.plate_cache.get(tracker_id)
        
        # If we already have a high-confidence full plate (>= 6 chars), return cached
        if existing and existing.get('length', 0) >= 6 and existing.get('plate') not in ["N/A", "UNREADABLE"]:
            return existing['plate']

        if plate_crop is None or plate_crop.size == 0 or plate_crop.shape[0] < 12 or plate_crop.shape[1] < 20:
            return existing['plate'] if existing else self._fallback_or_na(tracker_id)

        best_text = ""
        best_conf = 0.0

        if self.ocr_engine is not None:
            # Multi-Pass OCR: Pass 1 (CLAHE), Pass 2 (Otsu Threshold)
            for prep_img in [self.preprocess_clahe(plate_crop), self.preprocess_otsu(plate_crop)]:
                try:
                    text_pieces = []
                    probs = []
                    
                    if self.engine_name == "easyocr":
                        results = self.ocr_engine.readtext(
                            prep_img,
                            detail=1,
                            allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
                        )
                        for bbox, text, prob in results:
                            if prob >= 0.15:
                                text_pieces.append(text)
                                probs.append(prob)

                    elif self.engine_name == "paddleocr":
                        results = self.ocr_engine.ocr(prep_img, cls=True)
                        if results and results[0]:
                            for line in results[0]:
                                if line[1][1] >= 0.15:
                                    text_pieces.append(line[1][0])
                                    probs.append(line[1][1])

                    raw_text = "".join(text_pieces)
                    clean_text = re.sub(r'[^A-Z0-9]', '', raw_text.upper())
                    conf = float(np.mean(probs)) if probs else 0.0

                    # Keep best OCR pass result
                    if len(clean_text) > len(best_text) or (len(clean_text) == len(best_text) and conf > best_conf):
                        best_text = clean_text
                        best_conf = conf

                    if len(best_text) >= 5:
                        break  # Found good text, no need for second pass

                except Exception as e:
                    pass

        # Formatting plate string
        if len(best_text) >= 3:
            if len(best_text) > 4:
                formatted_plate = f"{best_text[:3]}-{best_text[3:]}"
            else:
                formatted_plate = best_text

            # Update cache if new result is longer or higher confidence than old cached result
            if (not existing) or (len(best_text) > existing.get('length', 0)) or (best_conf > existing.get('conf', 0.0)):
                self.plate_cache[tracker_id] = {
                    'plate': formatted_plate,
                    'length': len(best_text),
                    'conf': best_conf
                }
            return self.plate_cache[tracker_id]['plate']

        res = existing['plate'] if existing else self._fallback_or_na(tracker_id)
        if tracker_id not in self.plate_cache:
            self.plate_cache[tracker_id] = {'plate': res, 'length': 0, 'conf': 0.0}
        return res

    def _fallback_or_na(self, tracker_id: int) -> str:
        if self.allow_synth_fallback:
            prefixes = ["51A", "30F", "43C", "60B", "29A"]
            prefix = prefixes[tracker_id % len(prefixes)]
            num = (tracker_id * 137 + 12345) % 90000 + 10000
            return f"{prefix}-{num}"
        return "N/A"
