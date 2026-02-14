# client/plc_handler.py
import requests
from common.config import SERVER_IP, SERVER_PORT

class PLCAPIHandler:
    def __init__(self):
        self.base_url = f"http://{SERVER_IP}:{SERVER_PORT}"
        self.last_status_input = None
        print("✅ Client PLC-API Handler Initialized (No DB Connection)")

    def check_input_trigger(self):
        """Polls Server to check z_test_vision. Returns True on rising edge (0->1)."""
        try:
            response = requests.get(f"{self.base_url}/plc/input", timeout=1)
            if response.status_code == 200:
                status = int(response.json().get("status_input", 0))
                
                if self.last_status_input is None:
                    self.last_status_input = status
                    return False
                
                # Rising Edge Detection: 0 to 1
                is_triggered = (self.last_status_input == 0 and status == 1)
                self.last_status_input = status
                return is_triggered
        except Exception as e:
            print(f"Server Communication Error (Input): {e}")
        return False

    def get_pending_row(self):
        """Asks server for the row ID in z_par_plt waiting for result."""
        try:
            response = requests.get(f"{self.base_url}/plc/pending", timeout=1)
            if response.status_code == 200:
                return response.json() # Returns {'id': x, 'created_at': '...'}
        except Exception as e:
            print(f"Server Communication Error (Pending): {e}")
        return None

    def get_next_error_code(self):
        """Asks server for the next incremented ERROR-XXXXX string."""
        try:
            response = requests.get(f"{self.base_url}/plc/error_code", timeout=1)
            if response.status_code == 200:
                return response.json().get("error_code")
        except:
            return "ERROR-99999" # Fallback

    def write_final_result(self, created_at, datecode, status, image_path=None, text_path=None):
        """Tells server to write results to the Database."""
        payload = {
            "created_at": created_at,
            "datecode": datecode,
            "status": status,
            "image_path": image_path,
            "text_path": text_path
        }
        try:
            response = requests.post(f"{self.base_url}/plc/write", json=payload, timeout=2)
            return response.status_code == 200
        except Exception as e:
            print(f"Server Communication Error (Write): {e}")
            return False