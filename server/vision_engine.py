import cv2
import easyocr
import torch
import os
from ultralytics import YOLO
from common.config import MODEL_PATH, ROI_MODEL_PATH

class VisionEngine:
    def __init__(self):
        print("⏳ Loading Models on CPU...")
        
        # 1. Force YOLO to CPU
        self.model = YOLO(MODEL_PATH).to('cpu')
        self.roi_model = YOLO(ROI_MODEL_PATH).to('cpu')
        
        # 2. Force EasyOCR to CPU
        self.reader = easyocr.Reader(['en'], gpu=False)
        
        print("✅ Models Loaded (CPU Mode)")

    def process_frame(self, frame):
        """Main pipeline: Detect -> Crop -> OCR"""
        if frame is None: return None, None
        
        # Explicitly set device='cpu' and disable augment/half to keep it light
        results = self.model(frame, verbose=False, device='cpu')
        
        for r in results:
            for box in r.boxes:
                text, roi = self.process_ocr(frame, box)
                if text:
                    return text, roi
        return None, None

    def get_datecode_roi(self, frame, box):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = frame[y1:y2, x1:x2]
        
        # Use CPU for ROI model as well
        res_roi = self.roi_model(crop, verbose=False, device='cpu')
        for r in res_roi:
            for b in r.boxes:
                rx1, ry1, rx2, ry2 = map(int, b.xyxy[0])
                return crop[ry1:ry2, rx1:rx2]
        return None

    def simple_preprocess(self, roi):
        """Fixed: Thresholds increased to allow 900px+ images from your logs"""
        if roi is None or roi.size == 0: return None
        h, w = roi.shape[:2]
        
        # Increased to 1500 to handle your specific resolution
        if w > 1500 or h > 600: 
            print(f"⚠️ ROI too large ({w}x{h}) -> skip OCR")
            return None
            
        try:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            # scale=2 is enough for CPU; 3 might slow down the laptop too much
            scale = 2 if w > 400 else 3
            return cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        except: 
            return None

    def process_ocr(self, frame, box):
        roi = self.get_datecode_roi(frame, box)
        prepped = self.simple_preprocess(roi)
        
        if prepped is not None:
            # Running EasyOCR on CPU
            results = self.reader.readtext(
                prepped, 
                allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            )
            if results:
                full_text = "".join([res[1] for res in results]).replace(" ", "")
                return full_text, roi
        return None, roi