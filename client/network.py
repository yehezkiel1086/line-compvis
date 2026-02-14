# client/network.py
import requests
import cv2
from common.config import API_BASE_URL

def check_trigger_status():
    try:
        r = requests.get(f"{API_BASE_URL}/status", timeout=0.5)
        if r.status_code == 200:
            return r.json().get("trigger", False)
    except:
        pass
    return False

def send_batch_for_inspection(frames):
    files = []
    for i, frame in enumerate(frames):
        _, enc = cv2.imencode('.jpg', frame)
        files.append(('files', (f'img_{i}.jpg', enc.tobytes(), 'image/jpeg')))
    
    try:
        r = requests.post(f"{API_BASE_URL}/inspect", files=files, timeout=15)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        print("Network Error:", e)
    return None