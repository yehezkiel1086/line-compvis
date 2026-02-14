# server/plc_handler.py
import pyodbc
import time
from common.config import PLC_DB_SERVER, PLC_DB_DATABASE, PLC_DB_USERNAME, PLC_DB_PASSWORD

class DatabaseHandler:
    def __init__(self):
        self.conn = self._connect()
        self.cursor = self.conn.cursor()
        print("✅ Server DB Handler Initialized")

    def _connect(self):
        return pyodbc.connect(
            f"DRIVER={{ODBC Driver 17 for SQL Server}};"
            f"SERVER={PLC_DB_SERVER};DATABASE={PLC_DB_DATABASE};"
            f"UID={PLC_DB_USERNAME};PWD={PLC_DB_PASSWORD}",
            autocommit=True
        )

    def get_current_input(self):
        """Just returns the current value (0 or 1). Logic is on Client."""
        try:
            self.cursor.execute("SELECT TOP 1 status_input FROM dbo.z_test_vision WITH (NOLOCK) ORDER BY created_at DESC")
            row = self.cursor.fetchone()
            return int(row[0]) if row else 0
        except:
            self._reconnect()
            return 0

    def get_pending_row(self):
        """Returns row waiting for result."""
        try:
            self.cursor.execute("""
                SELECT TOP 1 id, created_at FROM dbo.z_par_plt WITH (NOLOCK)
                WHERE datecode IS NULL ORDER BY created_at DESC
            """)
            row = self.cursor.fetchone()
            if row:
                # Return created_at as ISO string for JSON serialization
                return {"id": row[0], "created_at": row[1].strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]}
        except: pass
        return None

    def update_result(self, created_at, datecode, status, image_path, text_path):
        """Writes the final result and paths to DB."""
        try:
            # Flexible query: Matches row roughly created at the same time
            self.cursor.execute("""
                UPDATE dbo.z_par_plt 
                SET datecode = ?, status = ?, image_path = ?, text_path = ?
                WHERE created_at BETWEEN DATEADD(ms,-900,?) AND DATEADD(ms, 900,?)
            """, datecode, status, image_path, text_path, created_at, created_at)
            return True
        except Exception as e:
            print(f"DB Write Error: {e}")
            return False

    def get_next_error_code(self):
        try:
            self.cursor.execute("SELECT TOP 1 datecode FROM dbo.z_par_plt WITH (NOLOCK) WHERE datecode LIKE 'ERROR-%' ORDER BY created_at DESC")
            row = self.cursor.fetchone()
            num = int(row[0].split("-")[-1]) if row else 0
            return f"ERROR-{num + 1:05d}"
        except:
            return "ERROR-00001"

    def _reconnect(self):
        try:
            self.conn = self._connect()
            self.cursor = self.conn.cursor()
        except: pass