"""
Lightweight SQLite logging for prediction results.

Renamed from m1_history_db.py -> history_db.py. The logic is unchanged: it
was already member-agnostic (every function takes `member` as an optional
filter), it just lived under a filename that made it look like it belonged
only to member 1. Every route in the global app.py, and every member's
dashboard, reads/writes through this single module + table.
"""
import os
import sqlite3
from datetime import datetime, timedelta

DB_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(DB_DIR, "fruitvision.db")


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def _add_date_filters(where_clauses, params, date_from=None, date_to=None):
    """Filters on created_at by calendar date (YYYY-MM-DD), inclusive on both
    ends. ISO timestamp strings sort/compare correctly as plain strings, so
    date_from (a date-only prefix) is naturally <= any timestamp that day,
    and date_to gets end-of-day appended so the whole end date is included."""
    if date_from:
        where_clauses.append("created_at >= ?")
        params.append(date_from)
    if date_to:
        where_clauses.append("created_at <= ?")
        params.append(f"{date_to}T23:59:59")


def get_paginated(member=None, fruit=None, user_id=None, date_from=None, date_to=None, page=1, per_page=20):
    """
    Fetch a page of results, optionally filtered by member, fruit, owning
    user, and/or a created_at date range. Returns (rows, total) where rows
    is a list of dicts for the requested page and total is the count of all
    rows matching the filters.
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
    if user_id is not None:
        where_clauses.append("user_id = ?")
        params.append(user_id)
    _add_date_filters(where_clauses, params, date_from, date_to)
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
    allowed = {
        "member", "filename", "fruit", "label", "confidence", "annotated_path",
        "fruit_area_px", "blemish_area_px", "blemish_percentage",
        "quality_grade", "surface_path", "source", "marketability_status",
        "dispatch_priority", "marketability_min_days", "marketability_max_days",
        "marketability_action", "marketability_reliability",
        "marketability_storage_assumption",
    }
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


def get_stats(member=None, fruit=None, user_id=None, since_hours=None, date_from=None, date_to=None):
    """
    Summary stats for a dashboard, optionally filtered by member, fruit,
    user, a rolling time window (since_hours), and/or a created_at date
    range. Returns a dict: total count, counts per label, counts per fruit,
    overall average confidence, average confidence per fruit, and average
    inference latency.
    """
    conn = _connect()

    where_clauses = []
    params = []
    if member:
        where_clauses.append("member = ?")
        params.append(member)
    if fruit:
        where_clauses.append("fruit = ?")
        params.append(fruit)
    if user_id is not None:
        where_clauses.append("user_id = ?")
        params.append(user_id)
    if since_hours is not None:
        cutoff = (datetime.now() - timedelta(hours=since_hours)).isoformat(timespec="seconds")
        where_clauses.append("created_at >= ?")
        params.append(cutoff)
    _add_date_filters(where_clauses, params, date_from, date_to)
    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
    params = tuple(params)

    total = conn.execute(
        f"SELECT COUNT(*) FROM results {where_sql}", params
    ).fetchone()[0]

    avg_confidence_row = conn.execute(
        f"SELECT AVG(confidence) FROM results {where_sql}", params
    ).fetchone()
    avg_confidence = round(avg_confidence_row[0], 2) if avg_confidence_row[0] is not None else 0

    avg_latency_row = conn.execute(
        f"SELECT AVG(latency_ms) FROM results {where_sql}", params
    ).fetchone()
    avg_latency_ms = round(avg_latency_row[0], 1) if avg_latency_row[0] is not None else None

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

    avg_blemish_row = conn.execute(
        f"SELECT AVG(blemish_percentage) FROM results {where_sql}", params
    ).fetchone()
    avg_blemish_percentage = (
        round(avg_blemish_row[0], 2) if avg_blemish_row[0] is not None else None
    )

    grade_rows = conn.execute(
        f"SELECT quality_grade, COUNT(*) as cnt FROM results {where_sql} "
        "AND quality_grade IS NOT NULL GROUP BY quality_grade"
        if where_sql
        else "SELECT quality_grade, COUNT(*) as cnt FROM results "
             "WHERE quality_grade IS NOT NULL GROUP BY quality_grade",
        params,
    ).fetchall()
    by_quality_grade = {r["quality_grade"]: r["cnt"] for r in grade_rows}

    conn.close()
    return {
        "total": total,
        "avg_confidence": avg_confidence,
        "avg_latency_ms": avg_latency_ms,
        "by_label": by_label,
        "by_fruit": by_fruit,
        "avg_confidence_by_fruit": avg_confidence_by_fruit,
        "avg_blemish_percentage": avg_blemish_percentage,
        "by_quality_grade": by_quality_grade,
    }


def get_stats_since(hours=24, member=None, user_id=None):
    """Convenience wrapper: stats for the rolling window, e.g. 'last 24h'."""
    return get_stats(member=member, user_id=user_id, since_hours=hours)

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
            fruit_area_px INTEGER,
            blemish_area_px INTEGER,
            blemish_percentage REAL,
            quality_grade TEXT,
            surface_path TEXT,
            source TEXT,
            created_at TEXT NOT NULL,
            user_id INTEGER,
            latency_ms REAL,
            flagged INTEGER DEFAULT 0,
            marketability_status TEXT,
            dispatch_priority TEXT,
            marketability_min_days INTEGER,
            marketability_max_days INTEGER,
            marketability_action TEXT,
            marketability_reliability TEXT,
            marketability_storage_assumption TEXT
        )
    """)
    # Compatibility migration for databases created by earlier versions.
    # SQLite's ADD COLUMN preserves every existing row and yields NULL for the
    # new surface fields, which correctly means "not analysed" rather than 0%.
    existing_columns = {
        row["name"] for row in conn.execute("PRAGMA table_info(results)").fetchall()
    }
    migrations = {
        "fruit_area_px": "INTEGER",
        "blemish_area_px": "INTEGER",
        "blemish_percentage": "REAL",
        "quality_grade": "TEXT",
        "surface_path": "TEXT",
        "user_id": "INTEGER",
        "latency_ms": "REAL",
        "flagged": "INTEGER DEFAULT 0",
        "detection_breakdown": "TEXT",
        "marketability_status": "TEXT",
        "dispatch_priority": "TEXT",
        "marketability_min_days": "INTEGER",
        "marketability_max_days": "INTEGER",
        "marketability_action": "TEXT",
        "marketability_reliability": "TEXT",
        "marketability_storage_assumption": "TEXT",
    }
    for column, data_type in migrations.items():
        if column not in existing_columns:
            conn.execute(f"ALTER TABLE results ADD COLUMN {column} {data_type}")
    conn.commit()
    conn.close()


def log_result(
    member,
    fruit,
    label,
    confidence,
    filename=None,
    annotated_path=None,
    source="predict",
    fruit_area_px=None,
    blemish_area_px=None,
    blemish_percentage=None,
    quality_grade=None,
    surface_path=None,
    user_id=None,
    latency_ms=None,
    flagged=0,
    detection_breakdown=None,
    marketability_status=None,
    dispatch_priority=None,
    marketability_min_days=None,
    marketability_max_days=None,
    marketability_action=None,
    marketability_reliability=None,
    marketability_storage_assumption=None,
):
    """Insert one prediction result. Call this right after predict_ripeness() returns.

    detection_breakdown: optional JSON string like '{"ripe": 2, "unripe": 2,
    "rotten": 1}', set only by the multi-fruit-per-photo batch path (see
    app.py's /analyse) when one photo's majority label is being logged as a
    single row -- None for every ordinary single-fruit prediction.
    """
    conn = _connect()
    conn.execute(
        """INSERT INTO results (
               member, filename, fruit, label, confidence, annotated_path,
               fruit_area_px, blemish_area_px, blemish_percentage,
               quality_grade, surface_path, source, created_at,
               user_id, latency_ms, flagged, detection_breakdown,
               marketability_status, dispatch_priority,
               marketability_min_days, marketability_max_days,
               marketability_action, marketability_reliability,
               marketability_storage_assumption
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            member, filename, fruit, label, confidence, annotated_path,
            fruit_area_px, blemish_area_px, blemish_percentage,
            quality_grade, surface_path, source,
            datetime.now().isoformat(timespec="seconds"),
            user_id, latency_ms, int(bool(flagged)), detection_breakdown,
            marketability_status, dispatch_priority,
            marketability_min_days, marketability_max_days,
            marketability_action, marketability_reliability,
            marketability_storage_assumption,
        ),
    )
    conn.commit()
    conn.close()


def get_all(member=None, fruit=None, user_id=None, date_from=None, date_to=None):
    """Unpaginated fetch, for CSV export. Same filters as get_paginated."""
    conn = _connect()
    where_clauses = []
    params = []
    if member:
        where_clauses.append("member = ?")
        params.append(member)
    if fruit:
        where_clauses.append("fruit = ?")
        params.append(fruit)
    if user_id is not None:
        where_clauses.append("user_id = ?")
        params.append(user_id)
    _add_date_filters(where_clauses, params, date_from, date_to)
    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
    rows = conn.execute(
        f"SELECT * FROM results {where_sql} ORDER BY id DESC", params
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_recent(member=None, user_id=None, limit=50):
    """Fetch most recent results, optionally filtered by member (e.g. 'ensemble_ab') and/or owning user."""
    conn = _connect()
    where_clauses = []
    params = []
    if member:
        where_clauses.append("member = ?")
        params.append(member)
    if user_id is not None:
        where_clauses.append("user_id = ?")
        params.append(user_id)
    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
    rows = conn.execute(
        f"SELECT * FROM results {where_sql} ORDER BY id DESC LIMIT ?", (*params, limit)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


init_db()
