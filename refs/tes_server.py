import re
import threading
import time
import tkinter as tk
from collections import Counter
from tkinter import Label
import requests, tempfile
import cv2
import easyocr
import numpy as np
from PIL import Image, ImageTk
from ultralytics import YOLO
import os

import concurrent.futures
from datetime import datetime, timedelta
import pyodbc
from queue import Queue

# try import pyav for MagnusCamera
try:
    import av
    _HAS_PYAV = True
except Exception:
    av = None
    _HAS_PYAV = False

# -------------------------
# CONFIG
# -------------------------
MODEL_PATH = "Cover_Baru.pt"
ROI_MODEL_PATH = "ROI_Baru.pt"
CAPTURE_COUNT = 5
CAPTURE_INTERVAL = 0.3

PLC_DB_SERVER = '10.19.16.21'
PLC_DB_USERNAME = 'sql_pre'
PLC_DB_PASSWORD = 'User@eng2'
PLC_DB_DATABASE = 'prod_control'
PLC_DB_TABLE = '[dbo].[z_test_vision]'
PLC_SCAN_RATE = 0.2

UPLOAD_URL = "http://10.19.23.88:8080/web/api/upload_image.php"
UPLOAD_TEXT_URL = "http://10.19.23.88:8080/web/api/upload_text.php"


# load models (may take time)
model = YOLO(MODEL_PATH)
roi_model = YOLO(ROI_MODEL_PATH)
reader = easyocr.Reader(['en'], gpu=False)

# ----------------------------------------------------------------
# helper: mapping / reconstruction (kept as in your logic)
# ----------------------------------------------------------------
MAPPING_D1 = {'0':['D','O','Q','U'],'1':['I','J','7'],'2':['S','Z','5'],'3':['B','E','8','9']}
MAPPING_D2 = {'0':['D','O','Q','U'],'1':['I','J'],'2':['Z'],'3':['E'],'4':['A','Y','H'],
              '5':['S'],'6':['G'],'7':['C'],'8':['B'],'9':['P','F']}
MAPPING_D3 = {'0':['D','O','Q','U'],'1':['I','J'],'2':['Z'],'3':['E'],'4':['Y'],
              '5':['S'],'6':['G'],'7':['C'],'8':[],'9':['P','F'],'A':['R'],'B':['H']}
MAPPING_D4 = {'0':['D','O','Q','U'],'1':['I','J'],'2':['Z'],'3':['E'],'4':['A','Y','H'],
              '5':['S'],'6':['G'],'7':['C'],'8':['B'],'9':['P','F']}
MAPPING_D5 = {'A':['R','4'],'B':['D','O','Q','8'],'C':['E','Z','2']}
MAPPING_D6 = {'1':['I','J'],'2':['Z'],'3':['E','B'],'4':['Y'],'5':['S'],'6':['G'],'7':['C']}
MAPPING_D7 = {'D':['B','H','O','Q','U','0','8']}

def map_by_position(c, pos):
    c = c.upper()
    mapping_sets = {
        0: MAPPING_D1, 1: MAPPING_D2, 2: MAPPING_D3, 3: MAPPING_D4,
        4: MAPPING_D5, 5: MAPPING_D6, 6: MAPPING_D7,
    }
    if pos in [7, 8, 9, 10]:
        mapping_sets[pos] = MAPPING_D2

    if pos in mapping_sets:
        for key, vals in mapping_sets[pos].items():
            if c == key or c in vals:
                return key
    return ""

def raw_digit_stats(list_raw):
    per_pos = [Counter() for _ in range(11)]
    for txt in list_raw:
        t = ''.join(k for k in txt.upper() if k.isalnum())[:11]
        for i, ch in enumerate(t):
            per_pos[i][ch] += 1
    return per_pos

def reconstruct_datecode(list_raw):
    if not list_raw:
        return ""

    raw_stats = raw_digit_stats(list_raw)
    per_pos = [[] for _ in range(11)]

    for txt in list_raw:
        t = ''.join(k for k in txt.upper() if k.isalnum())
        t = t[:11]
        for i in range(len(t)):
            mapped = map_by_position(t[i], i)
            if mapped:
                per_pos[i].append(mapped)

    result = ""
    for i in range(11):
        if per_pos[i]:
            result += Counter(per_pos[i]).most_common(1)[0][0]
        else:
            if i in [7, 8, 9, 10] and raw_stats[i]:
                result += raw_stats[i].most_common(1)[0][0]
            elif i == 0: result += '1'
            elif i == 6: result += 'D'
            elif i == 4: result += 'A'
            elif i == 5: result += '1'
            else:
                result += '0'
    return result

def get_today_folder(base_dir):
    today = datetime.now().strftime("%Y-%m-%d")
    day_dir = os.path.join(base_dir, today)
    os.makedirs(day_dir, exist_ok=True)
    return day_dir


# ROI logic (kept)
def get_datecode_roi(frame, box):
    x1, y1, x2, y2 = map(int, box)
    crop = frame[y1:y2, x1:x2]

    try:
        roi_results = roi_model(crop, verbose=False)
        boxes = roi_results[0].boxes.xyxy

        if len(boxes) == 0:
            return crop

        confs = roi_results[0].boxes.conf.cpu().numpy()
        best = confs.argmax()
        rx1, ry1, rx2, ry2 = boxes[best].cpu().numpy().astype(int)
        gx1 = x1 + rx1
        gy1 = y1 + ry1
        gx2 = x1 + rx2
        gy2 = y1 + ry2
        return frame[gy1:gy2, gx1:gx2]

    except Exception as e:
        print("ROI ERROR:", e)
        return crop

def simple_preprocess(roi):
    if roi is None or roi.size == 0:
        return None

    h, w = roi.shape[:2]

    # ❗ ROI KEGEDEAN → SKIP OCR
    if w > 400 or h > 200:
        print(f"ROI terlalu besar ({w}x{h}) -> skip OCR")
        return None

    try:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        return cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    except:
        return None


def process_ocr(frame, box):
    roi = get_datecode_roi(frame, box)
    proc = simple_preprocess(roi)
    if proc is None:
        return "", roi
    try:
        results = reader.readtext(proc, allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        if not results:
            return "", roi
        best = max(results, key=lambda r: r[2])
        return best[1], roi
    except Exception as e:
        print("OCR ERROR:", e)
        return "", roi

def stats_digit(list_raw):
    per_pos = [Counter() for _ in range(11)]
    for txt in list_raw:
        t = ''.join(k for k in txt.upper() if k.isalnum())[:11]
        for i, ch in enumerate(t):
            mapped = map_by_position(ch, i)
            if mapped:
                per_pos[i][mapped] += 1
    return per_pos

def resize_with_aspect_ratio_no_upscale(image, max_w, max_h):
    h, w = image.shape[:2]
    target_w = min(max_w, w)
    target_h = min(max_h, h)
    if target_w <= 0 or target_h <= 0:
        return image
    aspect = w / h
    new_w = target_w
    new_h = int(round(new_w / aspect))
    if new_h > target_h:
        new_h = target_h
        new_w = int(round(new_h * aspect))
    new_w = max(1, new_w)
    new_h = max(1, new_h)
    if new_w == w and new_h == h:
        return image
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

def majority_status(counter: Counter):
    """
    counter = Counter(raw_dates)
    """
    if not counter:
        return "NO VALID"

    values = sorted(counter.values(), reverse=True)
    total = sum(values)
    top = values[0]

    # pola distribusi
    pattern = tuple(values)

    # ===== RULE SESUAI YANG KAMU BERIKAN =====
    rules = {
        (5,): "VALID",
        (4,1): "VALID",
        (3,1,1): "VALID",

        (3,2): "WARNING",
        (2,2,1): "WARNING",
        (2,1,1,1): "WARNING",

        (1,1,1,1,1): "NO VALID",

        (4,): "VALID",
        (3,1): "VALID",

        (2,2): "WARNING",
        (2,1,1): "WARNING",

        (1,1,1,1): "NO VALID",

        (3,): "VALID",
        (2,1): "WARNING",
        (1,1,1): "NO VALID",

        (2,): "WARNING",
        (1,1): "NO VALID",

        (1,): "NO VALID",
    }

    return rules.get(pattern, "NO VALID")


# -------------------------
# MagnusCamera (PyAV wrapper) with low-latency options
# -------------------------
class MagnusCamera:
    def __init__(self, device_name="UVC Camera", width=1280, height=720, fps="30", high_fps_mode=False):
        if not _HAS_PYAV:
            raise RuntimeError("pyav not available")
        self.device_name = device_name
        self.width = width
        self.height = height
        self.fps = fps
        self.high_fps_mode = high_fps_mode
        self.latest_frame = None
        self.running = True
        rtbuf = "50M" if not high_fps_mode else "10M"
        options = {
            "video_size": f"{width}x{height}",
            "framerate": fps,
            "vcodec": "mjpeg",
            "fflags": "nobuffer",
            "rtbufsize": rtbuf,
        }
        self.container = av.open(f"video={self.device_name}", format="dshow", options=options)
        self.thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.thread.start()

    def _reader_loop(self):
        try:
            for frame in self.container.decode(video=0):
                if not self.running:
                    break
                self.latest_frame = frame.to_ndarray(format="bgr24")
        except Exception as e:
            print("MagnusCamera ERROR:", e)
            self.running = False

    def read(self):
        if self.latest_frame is None:
            return False, None
        return True, self.latest_frame.copy()

    def release(self):
        self.running = False
        try:
            self.container.close()
        except:
            pass

# -------------------------
# MAIN APP
# -------------------------
class BatteryApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Battery Inspection System")
        self.root.configure(bg="black")

        # =====================================================
        # 🔹 DATABASE OUTPUT (PERSISTENT CONNECTION)
        # =====================================================
        # Dipakai hanya oleh send_to_db()
        # Tidak buka-tutup berulang (ANTI LAMBAT)
        try:
            self.db_out_conn = pyodbc.connect(
                f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                f"SERVER={PLC_DB_SERVER};"
                f"DATABASE={PLC_DB_DATABASE};"
                f"UID={PLC_DB_USERNAME};"
                f"PWD={PLC_DB_PASSWORD}",
                autocommit=False
            )
            self.db_out_cursor = self.db_out_conn.cursor()
            print("✅ DB OUTPUT connected (persistent)")
        except Exception as e:
            print("❌ GAGAL CONNECT DB OUTPUT:", e)
            self.db_out_conn = None
            self.db_out_cursor = None

        # =============================
        # 🔹 UI LAYOUT (RESPONSIVE)
        # =============================
        main = tk.Frame(root, bg="black")
        main.grid(row=0, column=0, sticky="nsew")

        # ROOT EXPAND
        root.rowconfigure(0, weight=1)
        root.columnconfigure(0, weight=1)

        # MAIN GRID
        main.rowconfigure(0, weight=1)
        main.columnconfigure(0, weight=3)  # kiri (kamera)
        main.columnconfigure(1, weight=2)  # kanan (info)

        # LEFT (CAMERA)
        left = tk.Frame(main, bg="black")
        left.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)
        left.rowconfigure(1, weight=0)
        left.rowconfigure(2, weight=0)


        self.video_frame = tk.Frame(left, bg="black")
        self.video_frame.grid(row=0, column=0, sticky="nsew")

        self.video_frame.rowconfigure(0, weight=1)
        self.video_frame.columnconfigure(0, weight=1)


        self.video_label = tk.Label(self.video_frame, bg="black")
        self.video_label.pack(fill="both", expand=True)

        self.final_box = tk.Label(
            left,
            text="",
            font=("Consolas", 22, "bold"),
            bg="white",
            fg="black",
            bd=5,
            relief="solid",
            padx=20,
            pady=10
        )
        self.final_box.grid(row=1, column=0, pady=15, sticky="ew")

        self.info_label = tk.Label(
            left,
            text="Menunggu Battery Berhenti",
            font=("Arial", 14),
            bg="black",
            fg="white"
        )
        self.info_label.grid(row=2, column=0, sticky="ew")


        # RIGHT (INFO)
        right = tk.Frame(main, bg="black")
        self.roi_label = tk.Label(right, bg="black")
        self.roi_label.pack(pady=10)

        self.dc_box = tk.Label(
            right,
            text="",
            justify=tk.LEFT,
            font=("Consolas", 11),
            bg="white",
            fg="black",
            bd=5,
            relief="solid",
            padx=10,
            pady=10
        )
        self.dc_box.pack(pady=10)

        self.digit_box = tk.Label(
            right,
            text="",
            justify=tk.LEFT,
            font=("Consolas", 11),
            bg="white",
            fg="black",
            bd=5,
            relief="solid",
            padx=10,
            pady=10
        )
        self.digit_box.pack()

        
        right.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)


        # =====================================================
        # 🔹 CAMERA INIT
        # =====================================================
        MAGNUS_HIGH_FPS_MODE = True
        try:
            if _HAS_PYAV:
                print("Mencoba buka kamera Magnus...")
                self.cap = MagnusCamera(
                    "UVC Camera",
                    1280,
                    720,
                    fps="30",
                    high_fps_mode=MAGNUS_HIGH_FPS_MODE
                )
            else:
                raise RuntimeError("pyav not available")
        except Exception as e:
            print("Magnus init failed -> fallback OpenCV:", e)
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)




        # =====================================================
        # 🔹 THREAD & STATE CONTROL
        # =====================================================
        self.running = True
        self.current_frame = None

        self.task_queue = Queue()
        self.worker_running = True
        self.worker_busy = False
        threading.Thread(target=self.worker_loop, daemon=True).start()

        # capture state
        self.battery_present = False
        self.reset_timer_start = None
        self.can_capture = True
        self.capture_delay_time = None
        self.capture_delay = 0.5
        self.reset_delay = 1.0

        self.last_trigger_time = 0
        self.TRIGGER_COOLDOWN = 2.0

        self.last_parplt_time = None

        self.error_counter = 0
        self.load_error_counter_from_db()

        # output queue
        self.datecode_queue = Queue()
        self.OUTPUT_DELAY = 0.5
        
        # ============================
        # 🔹 DB ASYNC QUEUE & WORKER
        # ============================
        self.db_queue = Queue()
        threading.Thread(target=self.db_worker_loop, daemon=True).start()
        print("✅ DB Async Worker started")

        # =====================================================
        # 🔹 START SYSTEM
        # =====================================================
        self.update_frame()
        self.start_plc_monitor()
    

    def upload_image_to_laptop(self, frame, created_at, datecode):
        fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)  # 🔑 PENTING: lepas file handle Windows

        try:
            cv2.imwrite(tmp_path, frame)

            with open(tmp_path, "rb") as f:
                r = requests.post(
                    UPLOAD_URL,
                    files={"image": f},
                    data={
                        "datecode": datecode,
                        "created_at": created_at.strftime("%Y-%m-%d %H:%M:%S")
                    },
                    timeout=10
                )

            print("UPLOAD:", r.status_code, r.text)

            if r.status_code == 200:
                js = r.json()
                if js.get("status") == "OK":
                    return js.get("image_path")

        except Exception as e:
            print("UPLOAD ERROR:", e)

        finally:
            try:
                os.remove(tmp_path)
            except Exception as e:
                print("TMP DELETE ERROR:", e)

        return None

    def upload_text_to_laptop(self, created_at, datecode, raw_dates):
        counter = Counter(raw_dates)
        status = majority_status(counter)
        digit_stats = stats_digit(raw_dates)

        lines = []
        lines.append(f"DATECODE           : {datecode}")
        lines.append(f"MAJORITY DATECODE  : {status}")
        lines.append("")

        for dc, cnt in counter.items():
            lines.append(f"{dc:<14} {cnt}")

        lines.append("")
        lines.append("MAJORITY PER DIGIT :")
        for i, cnt in enumerate(digit_stats):
            if cnt:
                ch, c = cnt.most_common(1)[0]
                lines.append(f"Digit {i+1:<2}: {ch} ({c})")
            else:
                lines.append(f"Digit {i+1:<2}: -")

        content = "\n".join(lines)

        try:
            r = requests.post(
                UPLOAD_TEXT_URL,
                data={
                    "datecode": datecode,
                    "created_at": created_at.strftime("%Y-%m-%d %H:%M:%S"),
                    "content": content
                },
                timeout=5
            )

            print("TXT UPLOAD:", r.status_code, r.text)

            if r.status_code == 200:
                js = r.json()
                if js.get("status") == "OK":
                    return js.get("text_path")   # ✅ WAJIB RETURN

        except Exception as e:
            print("TXT UPLOAD ERROR:", e)

        return None

    # -------------------------
    # PLC DB MONITORING THREAD
    # -------------------------
    def start_plc_monitor(self):
        threading.Thread(target=self.plc_monitor_loop, daemon=True).start()

    def load_error_counter_from_db(self):
        try:
            cursor = self.db_out_conn.cursor()
            cursor.execute("""
                SELECT TOP 1 datecode
                FROM dbo.z_par_plt WITH (NOLOCK)
                WHERE datecode LIKE 'ERROR-%'
                ORDER BY created_at DESC
            """)
            row = cursor.fetchone()

            if row and row[0]:
                # ambil angka terakhir ERROR-00023 -> 23
                num = int(row[0].split("-")[-1])
                self.error_counter = num
                print(f"✅ ERROR counter lanjut dari DB: {self.error_counter}")
            else:
                self.error_counter = 0
                print("ℹ️ Belum ada ERROR-xxxx di DB")

        except Exception as e:
            print("❌ Gagal load ERROR counter:", e)
            self.error_counter = 0


    def send_to_db(self, created_at, datecode, status, frame, raw_dates, retry=3):
        if not datecode:
            print("⚠️ send_to_db: datecode kosong → SKIP DB UPDATE")
            return

        for _ in range(retry):
            try:
                # ===============================
                # 1️⃣ FAST DB UPDATE (TANPA UPLOAD)
                # ===============================
                self.db_out_cursor.execute("""
                    UPDATE dbo.z_par_plt
                    SET datecode = ?,
                        status   = ?
                    WHERE created_at BETWEEN DATEADD(ms,-500,?)
                                        AND DATEADD(ms, 500,?)
                """, datecode, status, created_at, created_at)

                self.db_out_conn.commit()
                print(f"⚡ FAST DB UPDATED: {datecode} | {status}")

                # ===============================
                # 2️⃣ BACKGROUND UPLOAD (BOLEH LAMBAT)
                # ===============================
                image_path = None
                text_path  = None

                if datecode != "NO-DETECT":

                    if datecode.startswith("ERROR"):
                        image_frame = self.current_frame
                    else:
                        image_frame = frame

                    if image_frame is not None:
                        image_path = self.upload_image_to_laptop(
                            image_frame,
                            created_at,
                            datecode
                        )

                if raw_dates and datecode != "NO-DETECT":
                    text_path = self.upload_text_to_laptop(
                        created_at,
                        datecode,
                        raw_dates
                    )

                # ===============================
                # 3️⃣ UPDATE PATH (SECOND UPDATE)
                # ===============================
                self.db_out_cursor.execute("""
                    UPDATE dbo.z_par_plt
                    SET image_path = ?,
                        text_path  = ?
                    WHERE created_at BETWEEN DATEADD(ms,-500,?)
                                        AND DATEADD(ms, 500,?)
                """, image_path, text_path, created_at, created_at)

                self.db_out_conn.commit()
                print(f"✅ DB UPDATED (PATH): {datecode}")

                return

            except Exception as e:
                self.db_out_conn.rollback()
                print("❌ DB ERROR:", e)
                time.sleep(0.2)


    def plc_monitor_loop(self):
        conn = None
        cursor = None
        last_status_input = None

        try:
            conn = pyodbc.connect(
                f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                f"SERVER={PLC_DB_SERVER};"
                f"DATABASE={PLC_DB_DATABASE};"
                f"UID={PLC_DB_USERNAME};"
                f"PWD={PLC_DB_PASSWORD}",
                autocommit=True
            )
            cursor = conn.cursor()
            print("PLC DB Monitor aktif...")

            while self.running:

                # ===============================
                # INPUT TRIGGER
                # ===============================
                cursor.execute("""
                    SELECT TOP 1 status_input
                    FROM dbo.z_test_vision WITH (NOLOCK)
                    ORDER BY created_at DESC
                """)
                row_in = cursor.fetchone()

                if row_in:
                    status_input = int(row_in[0])

                    if last_status_input is None:
                        last_status_input = status_input

                    elif status_input != last_status_input:
                        print(f"[PLC INPUT] {last_status_input} -> {status_input}")

                        if last_status_input == 0 and status_input == 1:
                            print("INPUT ON → mulai capture")
                            self.root.after(0, self.start_capture_sequence)

                        last_status_input = status_input

                # ===============================
                # OUTPUT TRIGGER
                # ===============================
                cursor.execute("""
                    SELECT TOP 1 id, created_at
                    FROM dbo.z_par_plt WITH (NOLOCK)
                    WHERE datecode IS NULL
                    ORDER BY created_at DESC
                """)
                row_out = cursor.fetchone()

                if row_out:
                    par_id = str(row_out[0]).strip()
                    created_at = row_out[1]

                    if self.last_parplt_time is None:
                        self.last_parplt_time = created_at

                    elif created_at > self.last_parplt_time:
                        print(f"[OUTPUT] row baru id={par_id}")
                        
                        if par_id == "A":

                            # =====================================
                            # 🔥 JIKA QUEUE KOSONG → SKIP (BIARKAN NULL)
                            # =====================================
                            if self.datecode_queue.empty():
                                print("⚠️ OUTPUT ON tapi datecode kosong → BIARKAN NULL (tidak kirim DB)")
                                self.last_parplt_time = created_at
                                continue

                            # =====================================
                            # 🔥 AMBIL HASIL VISION DARI QUEUE
                            # =====================================
                            datecode, status, frame, raw_dates = self.datecode_queue.get()

                            print(f"OUTPUT ON → enqueue DB: {datecode} | {status}")

                            # =====================================
                            # 🔥 MASUKKAN KE DB QUEUE (ASYNC)
                            # =====================================
                            self.db_queue.put((
                                created_at,
                                datecode,
                                status,
                                frame,
                                raw_dates
                            ))

                        # update last seen time
                        self.last_parplt_time = created_at

                time.sleep(PLC_SCAN_RATE)

        except Exception as e:
            print("PLC MONITOR ERROR:", e)

        finally:
            try:
                if cursor:
                    cursor.close()
                if conn:
                    conn.close()
            except:
                pass

    def db_worker_loop(self):
        print("🧵 DB Worker Loop aktif")

        while self.running:
            try:
                item = self.db_queue.get(timeout=1)
            except:
                continue

            if item is None:
                break

            created_at, datecode, status, frame, raw_dates = item

            try:
                # Semua yang berat dipindah ke sini
                self.send_to_db(created_at, datecode, status, frame, raw_dates)
            except Exception as e:
                print("❌ DB WORKER ERROR:", e)
            finally:
                self.db_queue.task_done()


    def is_box_in_region(self, frame, box):
        h, w, _ = frame.shape
        rx1, ry1 = int(w * 0.50), int(h * 0.40)
        rx2, ry2 = int(w * 0.67), int(h * 0.65)
        x1, y1, x2, y2 = map(int, box)
        overlap_x = not (x2 < rx1 or x1 > rx2)
        overlap_y = not (y2 < ry1 or y1 > ry2)
        return overlap_x and overlap_y

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret and frame is not None:
            self.current_frame = frame.copy()

            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            except:
                rgb = frame.copy()

            # ambil ukuran label
            w = self.video_label.winfo_width()
            h = self.video_label.winfo_height()

            if w > 10 and h > 10:
                resized = resize_with_aspect_ratio_no_upscale(rgb, w, h)
                img = ImageTk.PhotoImage(Image.fromarray(resized))
                self.video_label.configure(image=img)
                self.video_label.imgtk = img

        if self.running:
            self.root.after(20, self.update_frame)
        
    # capture triggered by space or auto-capture logic elsewhere
    def start_capture_sequence(self, event=None):
        threading.Thread(target=self.collect_frames_and_enqueue, daemon=True).start()

    def collect_frames_and_enqueue(self):
        def set_info(msg):
            self.root.after(0, lambda: self.info_label.config(text=msg))

        frames = []
        for i in range(CAPTURE_COUNT):
            set_info(f"Mengambil foto {i+1}/{CAPTURE_COUNT}...")
            if self.current_frame is not None:
                frames.append(self.current_frame.copy())
            time.sleep(CAPTURE_INTERVAL)

        # langsung enqueue - worker akan mengambil segera
        self.task_queue.put(frames)
        qsize = self.task_queue.qsize()
        set_info(f"Task masuk antrian (total: {qsize}).")

    def worker_loop(self):
        while self.worker_running:
            frames = self.task_queue.get()
            if frames is None:
                self.task_queue.task_done()
                break

            try:
                # tandai worker sibuk
                self.worker_busy = True
                # proses (blocking in worker thread)
                self.capture_sequence(frames)
            except Exception as e:
                print("ERROR worker:", e)
            finally:
                # pastikan task marked done (capture_sequence akan mematikan worker_busy)
                # but keep loop going
                self.task_queue.task_done()

    def capture_sequence(self, frames):
        raw_dates = []
        last_roi_date = None
        best_roi = None


        def set_info(msg):
            self.root.after(0, lambda: self.info_label.config(text=msg))

        total = len(frames)

        for idx, f in enumerate(frames):
            set_info(f"Memproses gambar... {idx+1}/{total}")

            try:
                # 🔹 WAJIB simpan hasil YOLO ke variabel
                results = model(f, verbose=False)
            except Exception as e:
                print("YOLO ERROR (capture_sequence):", e)
                continue

            boxes = results[0].boxes
            if boxes is None or len(boxes) == 0:
                continue

            # ==============================
            # 🔥 PILIH 1 COVER TERBAIK SAJA
            # ==============================
            confs = boxes.conf.cpu().numpy()
            best_idx = confs.argmax()
            best_box = boxes.xyxy[best_idx]

            # OCR hanya untuk 1 box
            raw, roi = process_ocr(f, best_box)
            
            if roi is not None:
                h, w = roi.shape[:2]
                if w > 400 or h > 200:
                    print("ROI invalid, skip frame")
                    continue
                
            if raw:
                mapped = reconstruct_datecode([raw])
                if mapped:
                    raw_dates.append(mapped)
                    last_roi_date = roi
                    best_roi = roi.copy()
        
        # ==========================
        # FINAL DECISION DATECODE
        # ==========================
        counter = Counter(raw_dates)
        best = ""

        try:
            if not raw_dates:
                # 🔴 OCR TIDAK JALAN / DISKIP
                self.error_counter += 1
                best = f"ERROR-{self.error_counter:05d}"
                datecode = best
                status = "NO VALID"

            else:
                best_tmp = reconstruct_datecode(raw_dates)

                if best_tmp and best_tmp.strip():
                    # 🟢 OCR JALAN & VALID
                    best = best_tmp
                    datecode = best
                    status = majority_status(counter)

                else:
                    # 🔴 OCR JALAN TAPI HASIL KACAU
                    self.error_counter += 1
                    best = f"ERROR-{self.error_counter:05d}"
                    datecode = best
                    status = "NO VALID"

        except Exception as e:
            print("❌ FINAL DATECODE ERROR:", e)
            self.error_counter += 1
            best = f"ERROR-{self.error_counter:05d}"
            datecode = best
            status = "NO VALID"


        # ⬅️ KIRIM KE QUEUE (PASTI ADA NILAI)
        self.datecode_queue.put((
            datecode,
            status,
            best_roi,
            raw_dates   # ⬅️ PENTING
        ))
        

        # ---- Release worker_busy BEFORE UI update so next task can start ----
        self.worker_busy = False

        # schedule UI update on main thread
        def update_ui():
            
            self.final_box.config(text=f"DATECODE: {best}")
            
            

            s = Counter(raw_dates)
            dc_txt = "Majority Datecode\n\n" + "\n".join(f"{k:<12} {v}" for k, v in s.items())
            self.dc_box.config(text=dc_txt)

            digit_s = stats_digit(raw_dates)
            dg = "Majority Per Digit\n\n"
            for i, cnt in enumerate(digit_s):
                if cnt:
                    b = cnt.most_common(1)[0]
                    dg += f"Digit {i+1}: {b[0]} ({b[1]})\n"
                else:
                    dg += f"Digit {i+1}: -\n"
            self.digit_box.config(text=dg)

            # show ROI greyscale preview
            self.show_roi_preview(last_roi_date)
            self.info_label.config(text="Selesai. Menunggu battery hilang untuk reset...")

        # schedule via Tk main thread
        self.root.after(0, update_ui)

    def show_roi_preview(self, roi):
        if roi is None or roi.size == 0:
            self.roi_label.config(text="(ROI Tidak Terdeteksi)", fg="white", bg="black", image="")
            return
        try:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        except:
            gray = roi
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        img = Image.fromarray(rgb).resize((200, 100))
        tk_img = ImageTk.PhotoImage(img)
        self.roi_label.configure(image=tk_img)
        self.roi_label.image = tk_img
        self.roi_label.config(text="")

    def on_close(self):
        self.running = False
        self.worker_running = False

        try:
            self.task_queue.put(None)
        except:
            pass

        try:
            if hasattr(self.cap, "release"):
                self.cap.release()
        except:
            pass

        if self.db_out_conn:
            self.db_out_conn.close()

        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()

    # ============================
    # FULLSCREEN LOGIC
    # ============================
    root.state("zoomed")          # start full layar (bisa minimize)
    root.attributes("-fullscreen", False)

    app = BatteryApp(root)

    # Toggle fullscreen (F11) & Exit fullscreen (ESC)
    is_fullscreen = False

    def toggle_fullscreen(event=None):
        global is_fullscreen
        is_fullscreen = not is_fullscreen
        root.attributes("-fullscreen", is_fullscreen)

    def exit_fullscreen(event=None):
        global is_fullscreen
        is_fullscreen = False
        root.attributes("-fullscreen", False)

    root.bind("<F11>", toggle_fullscreen)
    root.bind("<Escape>", exit_fullscreen)

    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()

