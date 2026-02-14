# client/main.py
import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk
import threading
import time
import cv2
from client.camera import MagnusCamera
from client.network import check_trigger_status, send_batch_for_inspection
from common.config import CAPTURE_COUNT, CAPTURE_INTERVAL, PLC_SCAN_RATE

class BatteryApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Client Inspection")
        self.root.configure(bg="black")
        
        # UI Setup
        self.setup_ui()
        
        # Camera
        self.cap = MagnusCamera(high_fps_mode=True)
        
        # State
        self.running = True
        self.is_capturing = False
        
        # Threads
        threading.Thread(target=self.gui_update_loop, daemon=True).start()
        threading.Thread(target=self.trigger_monitor_loop, daemon=True).start()

    def setup_ui(self):
        # (Simplified layout from your original code)
        main = tk.Frame(self.root, bg="black")
        main.pack(fill="both", expand=True)
        
        self.video_label = Label(main, bg="black")
        self.video_label.pack(side="left", fill="both", expand=True)
        
        right_panel = tk.Frame(main, bg="white", width=300)
        right_panel.pack(side="right", fill="y")
        
        self.final_box = Label(right_panel, text="WAITING...", font=("Consolas", 20))
        self.final_box.pack(pady=20)
        
        self.info_label = Label(right_panel, text="-", font=("Arial", 10))
        self.info_label.pack()
        
        self.digit_box = Label(right_panel, text="", font=("Consolas", 10), justify="left")
        self.digit_box.pack(pady=10)

    def gui_update_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                self.current_frame = frame
                # Resize and show
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (640, 360)) # Fixed size for preview
                tk_img = ImageTk.PhotoImage(Image.fromarray(img))
                
                # Update on Main Thread
                self.root.after(0, lambda i=tk_img: self.video_label.configure(image=i))
                self.video_label.image = tk_img # Keep ref
            time.sleep(0.03)

    def trigger_monitor_loop(self):
        while self.running:
            if not self.is_capturing:
                if check_trigger_status():
                    print("🚀 Trigger Detected! Starting Capture...")
                    self.start_capture()
            time.sleep(PLC_SCAN_RATE)

    def start_capture(self):
        self.is_capturing = True
        threading.Thread(target=self.capture_sequence, daemon=True).start()

    def capture_sequence(self):
        frames = []
        for i in range(CAPTURE_COUNT):
            self.root.after(0, lambda x=i: self.info_label.config(text=f"Capturing {x+1}/{CAPTURE_COUNT}"))
            if hasattr(self, 'current_frame'):
                frames.append(self.current_frame.copy())
            time.sleep(CAPTURE_INTERVAL)
            
        self.root.after(0, lambda: self.info_label.config(text="Sending to Server..."))
        
        # Send to Server
        result = send_batch_for_inspection(frames)
        
        if result:
            dc = result.get("datecode", "ERR")
            st = result.get("status", "ERR")
            stats = result.get("stats_digit", [])
            
            # Format Digit Stats
            dg_txt = "Digit Stats:\n"
            for i, cnt in enumerate(stats):
                 # stats is list of Counters, we need to handle format
                 # Simpler: just show raw text
                 pass 

            self.root.after(0, lambda: self.final_box.config(text=f"{dc}\n{st}"))
            self.root.after(0, lambda: self.info_label.config(text="Done."))
        else:
            self.root.after(0, lambda: self.info_label.config(text="Server Error"))

        time.sleep(1.0) # Debounce
        self.is_capturing = False

    def on_close(self):
        self.running = False
        self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    root.state('zoomed')
    app = BatteryApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()