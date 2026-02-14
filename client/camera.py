# client/camera.py
import cv2
import threading
import time

try:
    import av
    _HAS_PYAV = True
except ImportError:
    av = None
    _HAS_PYAV = False

class MagnusCamera:
    def __init__(self, device_name="UVC Camera", width=1280, height=720, fps="30", high_fps_mode=False):
        if not _HAS_PYAV:
            print("❌ PyAV not installed. Using OpenCV fallback.")
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            self.cap.set(3, width)
            self.cap.set(4, height)
            self.use_pyav = False
            return

        self.use_pyav = True
        self.device_name = device_name
        self.running = True
        self.latest_frame = None
        
        rtbuf = "50M" if not high_fps_mode else "10M"
        options = {
            "video_size": f"{width}x{height}",
            "framerate": str(fps),
            "vcodec": "mjpeg",
            "fflags": "nobuffer",
            "rtbufsize": rtbuf,
        }
        
        try:
            self.container = av.open(f"video={device_name}", format="dshow", options=options)
            self.thread = threading.Thread(target=self._reader_loop, daemon=True)
            self.thread.start()
        except Exception as e:
            print(f"MagnusCamera Init Failed: {e}. Fallback to OpenCV.")
            self.use_pyav = False
            self.cap = cv2.VideoCapture(0)

    def _reader_loop(self):
        try:
            for frame in self.container.decode(video=0):
                if not self.running: break
                self.latest_frame = frame.to_ndarray(format="bgr24")
        except: self.running = False

    def read(self):
        if not self.use_pyav:
            return self.cap.read()
        if self.latest_frame is None:
            return False, None
        return True, self.latest_frame.copy()

    def release(self):
        self.running = False
        if self.use_pyav:
            try: self.container.close()
            except: pass
        else:
            self.cap.release()