# client/network.py
import requests
import cv2
from common.config import API_BASE_URL

def get_plc_input():
    try:
        r = requests.get(f"{API_BASE_URL}/plc/input", timeout=1)
        if r.status_code == 200:
            return r.json().get("status_input", 0)
    except: pass
    return 0

def get_pending_row():
    try:
        r = requests.get(f"{API_BASE_URL}/plc/pending", timeout=1)
        if r.status_code == 200:
            return r.json() # Returns dict with 'created_at' or None
    except: pass
    return None

def get_error_code():
    try:
        r = requests.get(f"{API_BASE_URL}/plc/error_code", timeout=1)
        return r.json().get("error_code", "ERROR-00000")
    except: return "ERROR-00000"

def inspect_batch(frames, created_at, error_code):
    files = []
    for i, frame in enumerate(frames):
        _, enc = cv2.imencode('.jpg', frame)
        files.append(('files', (f'img_{i}.jpg', enc.tobytes(), 'image/jpeg')))
    
    try:
        # Send everything to server for processing
        params = {"created_at": created_at, "error_code": error_code}
        r = requests.post(f"{API_BASE_URL}/inspect", files=files, params=params, timeout=20)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        print("Inspection Network Error:", e)
    return None

def write_db_result(created_at, datecode, status, img_path, txt_path):
    payload = {
        "created_at": created_at,
        "datecode": datecode,
        "status": status,
        "image_path": img_path,
        "text_path": txt_path
    }
    try:
        requests.post(f"{API_BASE_URL}/plc/write", json=payload, timeout=2)
    except Exception as e:
        print("DB Write Network Error:", e)