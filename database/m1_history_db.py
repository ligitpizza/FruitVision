"""
Lightweight SQLite logging for prediction results.
Shared table so /predict and /analyse (and later members) all write here.
"""
import os
import sqlite3
from datetime import datetime

DB_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(DB_DIR, "fruitvision.db")


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    os.makedirs(DB_DIR, exist_ok=True)
    conn = _connect()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            member TEXT NOT NULL,
            filename TEXT,
            fruit TEXT NOT NULL,
            label TEXT NOT NULL,
            confidence REAL NOT NULL,
            annotated_path TEXT,
            source TEXT,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def log_result(member, fruit, label, confidence, filename=None, annotated_path=None, source="predict"):
    """Insert one prediction result. Call this right after predict_ripeness() returns."""
    conn = _connect()
    conn.execute(
        """INSERT INTO results (member, filename, fruit, label, confidence, annotated_path, source, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (member, filename, fruit, label, confidence, annotated_path, source, datetime.now().isoformat(timespec="seconds")),
    )
    conn.commit()
    conn.close()


def get_recent(member=None, limit=50):
    """Fetch most recent results, optionally filtered by member (e.g. 'member_1_ab')."""
    conn = _connect()
    if member:
        rows = conn.execute(
            "SELECT * FROM results WHERE member = ? ORDER BY id DESC LIMIT ?",
            (member, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM results ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


init_db()