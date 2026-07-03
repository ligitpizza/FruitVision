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

def get_paginated(member=None, fruit=None, page=1, per_page=20):
    """
    Fetch a page of results, optionally filtered by member and/or fruit.
    Returns (rows, total) where rows is a list of dicts for the requested
    page and total is the count of all rows matching the filters.
    """
    page = max(page, 1)
    per_page = max(per_page, 1)
    offset = (page - 1) * per_page

    conn = _connect()

    where_clauses = []
    params = []
    if member:
        where_clauses.append("member = ?")
        params.append(member)
    if fruit:
        where_clauses.append("fruit = ?")
        params.append(fruit)
    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

    total = conn.execute(
        f"SELECT COUNT(*) FROM results {where_sql}", params
    ).fetchone()[0]

    rows = conn.execute(
        f"SELECT * FROM results {where_sql} ORDER BY id DESC LIMIT ? OFFSET ?",
        (*params, per_page, offset),
    ).fetchall()

    conn.close()
    return [dict(r) for r in rows], total

def get_by_id(result_id):
    """Fetch a single result by its id. Returns a dict, or None if not found."""
    conn = _connect()
    row = conn.execute(
        "SELECT * FROM results WHERE id = ?", (result_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def update_result(result_id, **fields):
    """
    Update one or more columns on a result row.
    Usage: update_result(5, label="ripe", confidence=92.3)
    Only whitelisted columns can be updated. Returns True if a row was updated.
    """
    allowed = {"member", "filename", "fruit", "label", "confidence", "annotated_path", "source"}
    updates = {k: v for k, v in fields.items() if k in allowed}
    if not updates:
        return False

    set_sql = ", ".join(f"{col} = ?" for col in updates)
    params = list(updates.values()) + [result_id]

    conn = _connect()
    cur = conn.execute(f"UPDATE results SET {set_sql} WHERE id = ?", params)
    conn.commit()
    updated = cur.rowcount > 0
    conn.close()
    return updated


def delete_result(result_id):
    """Delete a result by id. Returns True if a row was deleted."""
    conn = _connect()
    cur = conn.execute("DELETE FROM results WHERE id = ?", (result_id,))
    conn.commit()
    deleted = cur.rowcount > 0
    conn.close()
    return deleted


def get_stats(member=None):
    """
    Summary stats for the dashboard, optionally filtered by member.
    Returns a dict: total count, counts per label, counts per fruit,
    overall average confidence, and average confidence per fruit.
    """
    conn = _connect()

    where_sql = "WHERE member = ?" if member else ""
    params = (member,) if member else ()

    total = conn.execute(
        f"SELECT COUNT(*) FROM results {where_sql}", params
    ).fetchone()[0]

    avg_confidence_row = conn.execute(
        f"SELECT AVG(confidence) FROM results {where_sql}", params
    ).fetchone()
    avg_confidence = round(avg_confidence_row[0], 2) if avg_confidence_row[0] is not None else 0

    label_rows = conn.execute(
        f"SELECT label, COUNT(*) as cnt FROM results {where_sql} GROUP BY label", params
    ).fetchall()
    by_label = {r["label"]: r["cnt"] for r in label_rows}

    fruit_rows = conn.execute(
        f"SELECT fruit, COUNT(*) as cnt FROM results {where_sql} GROUP BY fruit", params
    ).fetchall()
    by_fruit = {r["fruit"]: r["cnt"] for r in fruit_rows}

    avg_by_fruit_rows = conn.execute(
        f"SELECT fruit, AVG(confidence) as avg_conf FROM results {where_sql} GROUP BY fruit", params
    ).fetchall()
    avg_confidence_by_fruit = {r["fruit"]: round(r["avg_conf"], 2) for r in avg_by_fruit_rows}

    conn.close()
    return {
        "total": total,
        "avg_confidence": avg_confidence,
        "by_label": by_label,
        "by_fruit": by_fruit,
        "avg_confidence_by_fruit": avg_confidence_by_fruit,
    }

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