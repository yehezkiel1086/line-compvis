import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk
import threading
import time
import cv2
import client.network as net  # Requires the network.py created previously
from client.camera import MagnusCamera
from common.config import CAPTURE_COUNT, CAPTURE_INTERVAL, PLC_SCAN_RATE

class BatteryApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Battery Inspection System - Client Controller")
        self.root.configure(bg="black")

        # =====================================================
        # 🔹 UI LAYOUT (EXACTLY AS REFERENCED)
        # =====================================================
        main = tk.Frame(root, bg="black")
        main.grid(row=0, column=0, sticky="nsew")

        # Root Expand
        root.rowconfigure(0, weight=1)
        root.columnconfigure(0, weight=1)

        # Main Grid
        main.rowconfigure(0, weight=1)
        main.columnconfigure(0, weight=3)  # Left (Camera)
        main.columnconfigure(1, weight=2)  # Right (Info)

        # --- LEFT PANEL (Camera & Status) ---
        left = tk.Frame(main, bg="black")
        left.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)
        
        # Video Frame
        self.video_frame = tk.Frame(left, bg="black")
        self.video_frame.grid(row=0, column=0, sticky="nsew")
        self.video_frame.rowconfigure(0, weight=1)
        self.video_frame.columnconfigure(0, weight=1)

        self.video_label = tk.Label(self.video_frame, bg="black")
        self.video_label.pack(fill="both", expand=True)

        # Final Result Box
        self.final_box = tk.Label(
            left,
            text="READY",
            font=("Consolas", 22, "bold"),
            bg="white",
            fg="black",
            bd=5,
            relief="solid",
            padx=20,
            pady=10
        )
        self.final_box.grid(row=1, column=0, pady=15, sticky="ew")

        # Info Label
        self.info_label = tk.Label(
            left,
            text="Menunggu Battery Berhenti",
            font=("Arial", 14),
            bg="black",
            fg="white"
        )
        self.info_label.grid(row=2, column=0, sticky="ew")

        # --- RIGHT PANEL (Statistics) ---
        right = tk.Frame(main, bg="black")
        right.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

        # ROI Preview Label
        self.roi_label = tk.Label(right, bg="black")
        self.roi_label.pack(pady=10)

        # Majority Datecode Box
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
        self.dc_box.pack(pady=10, fill="x")

        # Digit Statistics Box
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
        self.digit_box.pack(fill="x")

        # =====================================================
        # 🔹 HARDWARE & STATE
        # =====================================================
        try:
            self.cap = MagnusCamera(high_fps_mode=True)
        except Exception as e:
            print("MagnusCamera failed, using CV2:", e)
            self.cap = cv2.VideoCapture(0)

        self.running = True
        self.is_processing = False
        self.current_frame = None
        self.last_input_status = None # For Rising Edge Detection

        # =====================================================
        # 🔹 THREADS
        # =====================================================
        # 1. UI Update Loop (Camera Feed)
        threading.Thread(target=self.gui_update_loop, daemon=True).start()
        
        # 2. PLC Logic Loop (The Brain)
        threading.Thread(target=self.plc_logic_loop, daemon=True).start()

    def resize_with_aspect_ratio_no_upscale(self, image, max_w, max_h):
        """Kept from original tes_server.py to maintain aspect ratio"""
        h, w = image.shape[:2]
        target_w = min(max_w, w)
        target_h = min(max_h, h)
        if target_w <= 0 or target_h <= 0: return image
        aspect = w / h
        new_w = target_w
        new_h = int(round(new_w / aspect))
        if new_h > target_h:
            new_h = target_h
            new_w = int(round(new_h * aspect))
        return cv2.resize(image, (max(1, new_w), max(1, new_h)), interpolation=cv2.INTER_AREA)

    def gui_update_loop(self):
        """Standard loop to keep the UI responsive and showing video"""
        while self.running:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                self.current_frame = frame.copy()
                
                # Update UI Image
                try:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    w = self.video_label.winfo_width()
                    h = self.video_label.winfo_height()
                    
                    if w > 10 and h > 10:
                        resized = self.resize_with_aspect_ratio_no_upscale(rgb, w, h)
                        img = ImageTk.PhotoImage(Image.fromarray(resized))
                        
                        # Use lambda to update safely on main thread
                        self.root.after(0, lambda i=img: self.update_video_label(i))
                except Exception:
                    pass
            time.sleep(0.03) # ~30 FPS UI update

    def update_video_label(self, img):
        self.video_label.configure(image=img)
        self.video_label.imgtk = img

    def plc_logic_loop(self):
        """
        🔥 CORE LOGIC ENGINE
        Polls the Server for PLC Input status and triggers logic on Rising Edge (0->1).
        """
        print("✅ PLC Logic Engine Started")
        
        while self.running:
            if not self.is_processing:
                # 1. Poll PLC Status from Server
                current_status = net.get_plc_input()

                # Initialize if first run
                if self.last_input_status is None:
                    self.last_input_status = current_status
                    time.sleep(PLC_SCAN_RATE)
                    continue

                # 2. Detect Rising Edge (0 -> 1)
                if self.last_input_status == 0 and current_status == 1:
                    print("⚡ Trigger Detected (0->1) -> Starting Inspection")
                    self.execute_inspection_cycle()
                
                self.last_input_status = current_status
            
            time.sleep(PLC_SCAN_RATE)

    def execute_inspection_cycle(self):
        """Orchestrates the entire Capture -> Inspect -> Write DB flow"""
        self.is_processing = True
        self.update_info("Mencari Row Pending di DB...")

        # A. Get Target Row and Error Code
        pending = net.get_pending_row()
        if not pending:
            print("⚠️ No pending row found in DB.")
            self.update_info("DB: Tidak ada row pending.")
            self.is_processing = False
            return

        created_at = pending['created_at']
        error_code = net.get_error_code()

        # B. Capture Images
        frames = []
        for i in range(CAPTURE_COUNT):
            self.update_info(f"Mengambil foto {i+1}/{CAPTURE_COUNT}...")
            if self.current_frame is not None:
                frames.append(self.current_frame.copy())
            time.sleep(CAPTURE_INTERVAL)

        # C. Send to Server for Vision Processing
        self.update_info("Memproses OCR ke Server...")
        result = net.inspect_batch(frames, created_at, error_code)

        if result:
            # D. Parse Results
            dc = result.get("datecode", "NO-DETECT")
            status = result.get("status", "NO VALID")
            raw_dates = result.get("raw_dates", [])
            stats_digit = result.get("stats_digit", [])
            img_path = result.get("image_path")
            txt_path = result.get("text_path")

            # E. Update UI (Main Result)
            self.root.after(0, lambda: self.final_box.config(text=f"{dc}"))
            
            # F. Update UI (Majority Box)
            self.update_stats_ui(raw_dates, stats_digit)

            # G. Write Final Result to DB (via Server)
            self.update_info("Menulis hasil ke DB...")
            net.write_db_result(created_at, dc, status, img_path, txt_path)
            
            self.update_info("Selesai.")
        else:
            self.update_info("Server Error / Timeout")

        time.sleep(1.0) # Short debounce/cooldown
        self.is_processing = False
        self.update_info("Menunggu Battery Berhenti")

    def update_stats_ui(self, raw_dates, stats_digit):
        """Updates the detailed statistics boxes on the right panel"""
        from collections import Counter
        
        # 1. Majority Datecode
        s = Counter(raw_dates)
        dc_txt = "Majority Datecode\n\n" + "\n".join(f"{k:<12} {v}" for k, v in s.items())
        
        # 2. Majority Per Digit
        dg_txt = "Majority Per Digit\n\n"
        # stats_digit comes from server as list of Counters or dicts
        # Adjusting strictly to how tes_server.py displayed it:
        for i, item in enumerate(stats_digit):
             # If item is a list/tuple from Counter.most_common
             if item: 
                 # Handle if server sends raw dict or list
                 # Assuming server logic sends a simplified list for display
                 dg_txt += f"Digit {i+1}: {item}\n"
             else:
                 dg_txt += f"Digit {i+1}: -\n"

        self.root.after(0, lambda: self.dc_box.config(text=dc_txt))
        self.root.after(0, lambda: self.digit_box.config(text=dg_txt))

    def update_info(self, text):
        self.root.after(0, lambda: self.info_label.config(text=text))

    def on_close(self):
        self.running = False
        if hasattr(self, 'cap'):
            self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    root.state('zoomed')
    # Prevent fullscreen lock if needed, matching original
    root.attributes("-fullscreen", False) 
    app = BatteryApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()