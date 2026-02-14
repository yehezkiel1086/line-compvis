# server/main.py
from fastapi import FastAPI, UploadFile, File
import uvicorn
import cv2
import numpy as np
import requests
import tempfile
import os
from collections import Counter
from datetime import datetime

from common.config import SERVER_PORT, PHP_UPLOAD_URL, PHP_UPLOAD_TEXT_URL
from server.vision_engine import VisionEngine
from server.plc_handler import DatabaseHandler
from server.corrector import reconstruct_datecode, majority_status, stats_digit

app = FastAPI()
vision = VisionEngine()
db = DatabaseHandler()

# --- PLC / DB ENDPOINTS ---
@app.get("/plc/input")
def get_plc_input():
    """Client calls this loop to check 0->1 transition."""
    return {"status_input": db.get_current_input()}

@app.get("/plc/pending")
def get_pending_row():
    """Client calls this to find where to write data."""
    return db.get_pending_row()

@app.get("/plc/error_code")
def get_new_error():
    return {"error_code": db.get_next_error_code()}

@app.post("/plc/write")
def write_db(data: dict):
    """Client commands Server to write to DB."""
    success = db.update_result(
        data['created_at'], 
        data['datecode'], 
        data['status'], 
        data.get('image_path'), 
        data.get('text_path')
    )
    return {"success": success}

# --- VISION & UPLOAD ENDPOINTS ---
@app.post("/inspect")
async def inspect_and_upload(files: list[UploadFile], created_at: str, error_code: str):
    """
    1. Receives images
    2. Runs Vision (YOLO/OCR)
    3. Uploads best image/text to PHP
    4. Returns Data for Client to Display
    """
    # Decode Images
    frames = []
    for file in files:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frames.append(cv2.imdecode(nparr, cv2.IMREAD_COLOR))

    # Run Vision
    raw_dates = []
    best_roi = None
    last_frame = frames[-1] if frames else None

    for frame in frames:
        text, roi = vision.process_frame(frame)
        if roi is not None: best_roi = roi
        if text:
            mapped = reconstruct_datecode([text])
            if mapped: raw_dates.append(mapped)

    # Logic
    counter = Counter(raw_dates)
    if raw_dates:
        final_dc = reconstruct_datecode(raw_dates)
        status = majority_status(counter)
        if not final_dc.strip():
            final_dc = error_code
            status = "NO VALID"
    else:
        final_dc = error_code
        status = "NO VALID"

    # Upload to PHP (Server handles this heavily lifting)
    img_path = None
    if best_roi is not None:
        # Upload the processed ROI or full frame? Original script logic favored frame if error.
        # We will upload the last full frame for context.
        img_path = upload_image_php(last_frame, created_at, final_dc)
    
    txt_path = upload_text_php(created_at, final_dc, raw_dates)

    return {
        "datecode": final_dc,
        "status": status,
        "raw_dates": raw_dates,
        "stats_digit": stats_digit(raw_dates),
        "image_path": img_path,
        "text_path": txt_path
    }

# --- HELPERS ---
def upload_image_php(frame, created_at_str, datecode):
    if frame is None: return None
    fd, tmp = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    try:
        cv2.imwrite(tmp, frame)
        with open(tmp, "rb") as f:
            r = requests.post(PHP_UPLOAD_URL, files={"image": f},
                data={"datecode": datecode, "created_at": created_at_str}, timeout=5)
            if r.status_code == 200: return r.json().get("image_path")
    except: pass
    finally: os.remove(tmp)
    return None

def upload_text_php(created_at_str, datecode, raw_dates):
    # (Same text formatting logic as before)
    content = f"DATECODE: {datecode}\nRAW: {raw_dates}"
    try:
        r = requests.post(PHP_UPLOAD_TEXT_URL, data={
            "datecode": datecode, "created_at": created_at_str, "content": content}, timeout=5)
        if r.status_code == 200: return r.json().get("text_path")
    except: pass
    return None

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT)