# server/vision_engine.py
import cv2
import numpy as np
import easyocr
from ultralytics import YOLO
from common.config import MODEL_PATH, ROI_MODEL_PATH
from server.corrector import reconstruct_datecode

class VisionEngine:
    def __init__(self):
        print("⏳ Loading Models...")
        self.model = YOLO(MODEL_PATH)
        self.roi_model = YOLO(ROI_MODEL_PATH)
        self.reader = easyocr.Reader(['en'], gpu=True) # Set GPU=True if server has GPU
        print("✅ Models Loaded")

    def get_datecode_roi(self, frame, box):
        x1, y1, x2, y2 = map(int, box)
        crop = frame[y1:y2, x1:x2]
        try:
            roi_results = self.roi_model(crop, verbose=False)
            boxes = roi_results[0].boxes.xyxy
            if len(boxes) == 0: return crop
            
            confs = roi_results[0].boxes.conf.cpu().numpy()
            best = confs.argmax()
            rx1, ry1, rx2, ry2 = boxes[best].cpu().numpy().astype(int)
            gx1, gy1 = x1 + rx1, y1 + ry1
            gx2, gy2 = x1 + rx2, y1 + ry2
            return frame[gy1:gy2, gx1:gx2]
        except Exception as e:
            print("ROI ERROR:", e)
            return crop

    def simple_preprocess(self, roi):
        if roi is None or roi.size == 0: return None
        h, w = roi.shape[:2]
        if w > 400 or h > 200:
            print(f"ROI too large ({w}x{h}) -> skip OCR")
            return None
        try:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
            return cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        except: return None

    def process_frame(self, frame):
        """Returns (raw_text, roi_image) or (None, None)"""
        try:
            results = self.model(frame, verbose=False)
            boxes = results[0].boxes
            if boxes is None or len(boxes) == 0: return None, None

            confs = boxes.conf.cpu().numpy()
            best_idx = confs.argmax()
            best_box = boxes.xyxy[best_idx]

            # ROI Extraction
            roi = self.get_datecode_roi(frame, best_box)
            proc = self.simple_preprocess(roi)
            
            if proc is None: return None, roi

            # OCR
            ocr_res = self.reader.readtext(proc, allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
            if not ocr_res: return None, roi
            
            best_text = max(ocr_res, key=lambda r: r[2])[1]
            return best_text, roi
        except Exception as e:
            print("Vision Process Error:", e)
            return None, None