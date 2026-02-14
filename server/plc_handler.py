# server/plc_handler.py
import pyodbc
import time
from common.config import PLC_DB_SERVER, PLC_DB_DATABASE, PLC_DB_USERNAME, PLC_DB_PASSWORD

class DatabaseHandler:
    def __init__(self):
        self.conn = self._connect()
        self.cursor = self.conn.cursor()
        self.last_status_input = None
        self.last_parplt_time = None
        self.error_counter = 0
        self.load_error_counter()
        print("✅ PLC Database Handler Initialized")

    def _connect(self):
        return pyodbc.connect(
            f"DRIVER={{ODBC Driver 17 for SQL Server}};"
            f"SERVER={PLC_DB_SERVER};DATABASE={PLC_DB_DATABASE};"
            f"UID={PLC_DB_USERNAME};PWD={PLC_DB_PASSWORD}",
            autocommit=True
        )

    def load_error_counter(self):
        try:
            self.cursor.execute("SELECT TOP 1 datecode FROM dbo.z_par_plt WITH (NOLOCK) WHERE datecode LIKE 'ERROR-%' ORDER BY created_at DESC")
            row = self.cursor.fetchone()
            if row and row[0]:
                self.error_counter = int(row[0].split("-")[-1])
        except: self.error_counter = 0

    def get_next_error_code(self):
        self.error_counter += 1
        return f"ERROR-{self.error_counter:05d}"

    def check_input_trigger(self):
        """Polls z_test_vision. Returns True on rising edge (0->1)."""
        try:
            self.cursor.execute("SELECT TOP 1 status_input FROM dbo.z_test_vision WITH (NOLOCK) ORDER BY created_at DESC")
            row = self.cursor.fetchone()
            if row:
                status = int(row[0])
                if self.last_status_input is None:
                    self.last_status_input = status
                    return False
                if self.last_status_input == 0 and status == 1:
                    self.last_status_input = status
                    return True
                self.last_status_input = status
        except Exception as e:
            print(f"PLC Read Error: {e}")
            try: self.conn = self._connect(); self.cursor = self.conn.cursor()
            except: pass
        return False

    def get_pending_row(self):
        """Gets the row ID in z_par_plt waiting for result."""
        try:
            self.cursor.execute("""
                SELECT TOP 1 id, created_at FROM dbo.z_par_plt WITH (NOLOCK)
                WHERE datecode IS NULL ORDER BY created_at DESC
            """)
            return self.cursor.fetchone()
        except: return None

    def fast_update_db(self, created_at, datecode, status):
        try:
            self.cursor.execute("""
                UPDATE dbo.z_par_plt SET datecode = ?, status = ?
                WHERE created_at BETWEEN DATEADD(ms,-500,?) AND DATEADD(ms, 500,?)
            """, datecode, status, created_at, created_at)
        except Exception as e:
            print("Fast Update Error:", e)

    def update_paths(self, created_at, image_path, text_path):
        try:
            self.cursor.execute("""
                UPDATE dbo.z_par_plt SET image_path = ?, text_path = ?
                WHERE created_at BETWEEN DATEADD(ms,-500,?) AND DATEADD(ms, 500,?)
            """, image_path, text_path, created_at, created_at)
        except Exception as e:
            print("Path Update Error:", e)