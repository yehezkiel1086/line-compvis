# server/main.py
from fastapi import FastAPI, UploadFile, File, Form
import uvicorn
import cv2
import numpy as np
import requests
import tempfile
import os
from collections import Counter
from common.config import SERVER_IP, SERVER_PORT, PHP_UPLOAD_URL, PHP_UPLOAD_TEXT_URL
from server.vision_engine import VisionEngine
from server.plc_handler import DatabaseHandler
from server.corrector import reconstruct_datecode, majority_status, stats_digit

app = FastAPI()
vision = VisionEngine()
db = DatabaseHandler()

@app.get("/status")
def check_status():
    """Client calls this to ask if it should capture."""
    trigger = db.check_input_trigger()
    return {"trigger": trigger}

@app.post("/inspect")
async def inspect_batch(files: list[UploadFile]):
    """Receives 5 images, processes them, uploads to PHP, updates DB."""
    
    # 1. Decode Images
    frames = []
    for file in files:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        frames.append(frame)

    # 2. Process Logic (Vision)
    raw_dates = []
    best_roi = None
    
    for frame in frames:
        text, roi = vision.process_frame(frame)
        if roi is not None: best_roi = roi
        if text:
            mapped = reconstruct_datecode([text])
            if mapped: raw_dates.append(mapped)

    # 3. Decision Logic
    counter = Counter(raw_dates)
    if raw_dates:
        final_dc = reconstruct_datecode(raw_dates)
        status = majority_status(counter)
        if not final_dc.strip():
            final_dc = db.get_next_error_code()
            status = "NO VALID"
    else:
        final_dc = db.get_next_error_code()
        status = "NO VALID"

    # 4. Database & PHP Upload Sequence
    # We need to find the created_at from the pending row
    row = db.get_pending_row()
    
    if row:
        row_id, created_at = row
        
        # A. Fast DB Update
        db.fast_update_db(created_at, final_dc, status)
        
        # B. Upload Image to PHP
        img_path = None
        if best_roi is not None: # Or use full frame if error
             # Logic from original: if error use full frame, else use best_roi?
             # Original used full frame for uploads. Let's use the last full frame.
             img_to_up = frames[-1]
             img_path = upload_image_php(img_to_up, created_at, final_dc)

        # C. Upload Text to PHP
        txt_path = upload_text_php(created_at, final_dc, raw_dates)
        
        # D. Update DB with Paths
        db.update_paths(created_at, img_path, txt_path)

    return {
        "datecode": final_dc,
        "status": status,
        "raw_dates": raw_dates,
        "stats_digit": stats_digit(raw_dates)
    }

def upload_image_php(frame, created_at, datecode):
    fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    try:
        cv2.imwrite(tmp_path, frame)
        with open(tmp_path, "rb") as f:
            r = requests.post(PHP_UPLOAD_URL, files={"image": f},
                data={"datecode": datecode, "created_at": created_at.strftime("%Y-%m-%d %H:%M:%S")}, timeout=10)
            if r.status_code == 200 and r.json().get("status") == "OK":
                return r.json().get("image_path")
    except Exception as e: print("Img Upload Err:", e)
    finally: os.remove(tmp_path)
    return None

def upload_text_php(created_at, datecode, raw_dates):
    counter = Counter(raw_dates)
    status = majority_status(counter)
    digit_stats = stats_digit(raw_dates)
    lines = [f"DATECODE: {datecode}", f"MAJORITY: {status}", ""]
    for dc, cnt in counter.items(): lines.append(f"{dc:<14} {cnt}")
    lines.append("\nMAJORITY PER DIGIT:")
    for i, cnt in enumerate(digit_stats):
        ch = cnt.most_common(1)[0][0] if cnt else "-"
        lines.append(f"Digit {i+1}: {ch}")
    
    try:
        r = requests.post(PHP_UPLOAD_TEXT_URL, data={
            "datecode": datecode,
            "created_at": created_at.strftime("%Y-%m-%d %H:%M:%S"),
            "content": "\n".join(lines)}, timeout=5)
        if r.status_code == 200 and r.json().get("status") == "OK":
            return r.json().get("text_path")
    except Exception as e: print("Txt Upload Err:", e)
    return None

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT)