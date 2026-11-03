from pathlib import Path
from datetime import datetime
import os
import sqlite3

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = Path(os.getenv("SENTINELAI_DB_PATH", str(BASE_DIR / "predictions.db")))

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    machine_type TEXT NOT NULL,
    air_temperature REAL NOT NULL,
    process_temperature REAL NOT NULL,
    rotational_speed REAL NOT NULL,
    torque REAL NOT NULL,
    tool_wear REAL NOT NULL,
    risk_probability REAL NOT NULL,
    threshold REAL NOT NULL,
    machine_failure INTEGER NOT NULL,
    status TEXT NOT NULL
)
"""

def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)

def init_db():
    with get_connection() as conn:
        conn.execute(CREATE_TABLE_SQL)

def save_prediction(payload, result):
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO predictions (
                timestamp, machine_type, air_temperature,
                process_temperature, rotational_speed, torque,
                tool_wear, risk_probability, threshold,
                machine_failure, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now().astimezone().isoformat(timespec="seconds"),
                payload["machine_type"],
                payload["air_temperature"],
                payload["process_temperature"],
                payload["rotational_speed"],
                payload["torque"],
                payload["tool_wear"],
                result["risk_probability"],
                result["threshold"],
                int(result["machine_failure"]),
                result["status"],
            ),
        )
