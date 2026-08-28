"""
Fruit stock ledger -- SQLite logging for inventory movements.

Mirrors history_db.py's shape: every stock change is one row (an event), not
a running total column, so totals are always computed on read via SUM/GROUP
BY and can never drift out of sync with the ledger. Quantity is signed so a
manual entry can also record stock leaving (sold/discarded/correction), not
just stock arriving.
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


def _add_date_filters(where_clauses, params, date_from=None, date_to=None):
    if date_from:
        where_clauses.append("created_at >= ?")
        params.append(date_from)
    if date_to:
        where_clauses.append("created_at <= ?")
        params.append(f"{date_to}T23:59:59")


def init_db():
    os.makedirs(DB_DIR, exist_ok=True)
    conn = _connect()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS stock_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fruit TEXT NOT NULL,
            label TEXT NOT NULL,
            quantity INTEGER NOT NULL,
            source TEXT NOT NULL,
            note TEXT,
            track_tag TEXT,
            created_at TEXT NOT NULL,
            user_id INTEGER
        )
    """)
    conn.commit()
    conn.close()


def log_stock_event(fruit, label, quantity, source, note=None, track_tag=None, user_id=None):
    """Insert one stock movement. quantity may be negative (stock leaving)."""
    conn = _connect()
    conn.execute(
        """INSERT INTO stock_events (
               fruit, label, quantity, source, note, track_tag, created_at, user_id
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            fruit, label, quantity, source, note, track_tag,
            datetime.now().isoformat(timespec="seconds"), user_id,
        ),
    )
    conn.commit()
    conn.close()


def get_paginated(fruit=None, label=None, source=None, user_id=None, date_from=None, date_to=None, page=1, per_page=20):
    page = max(page, 1)
    per_page = max(per_page, 1)
    offset = (page - 1) * per_page

    conn = _connect()
    where_clauses = []
    params = []
    if fruit:
        where_clauses.append("fruit = ?")
        params.append(fruit)
    if label:
        where_clauses.append("label = ?")
        params.append(label)
    if source:
        where_clauses.append("source = ?")
        params.append(source)
    if user_id is not None:
        where_clauses.append("user_id = ?")
        params.append(user_id)
    _add_date_filters(where_clauses, params, date_from, date_to)
    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

    total = conn.execute(
        f"SELECT COUNT(*) FROM stock_events {where_sql}", params
    ).fetchone()[0]

    rows = conn.execute(
        f"SELECT * FROM stock_events {where_sql} ORDER BY id DESC LIMIT ? OFFSET ?",
        (*params, per_page, offset),
    ).fetchall()

    conn.close()
    return [dict(r) for r in rows], total


def get_all(fruit=None, label=None, source=None, user_id=None, date_from=None, date_to=None):
    """Unpaginated fetch, for CSV/PDF export. Same filters as get_paginated."""
    conn = _connect()
    where_clauses = []
    params = []
    if fruit:
        where_clauses.append("fruit = ?")
        params.append(fruit)
    if label:
        where_clauses.append("label = ?")
        params.append(label)
    if source:
        where_clauses.append("source = ?")
        params.append(source)
    if user_id is not None:
        where_clauses.append("user_id = ?")
        params.append(user_id)
    _add_date_filters(where_clauses, params, date_from, date_to)
    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
    rows = conn.execute(
        f"SELECT * FROM stock_events {where_sql} ORDER BY id DESC", params
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_by_id(event_id):
    conn = _connect()
    row = conn.execute("SELECT * FROM stock_events WHERE id = ?", (event_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def update_stock_event(event_id, **fields):
    """Update one or more columns on a stock event. Only whitelisted columns
    can be updated. Returns True if a row was updated."""
    allowed = {"fruit", "label", "quantity", "note"}
    updates = {k: v for k, v in fields.items() if k in allowed}
    if not updates:
        return False

    set_sql = ", ".join(f"{col} = ?" for col in updates)
    params = list(updates.values()) + [event_id]

    conn = _connect()
    cur = conn.execute(f"UPDATE stock_events SET {set_sql} WHERE id = ?", params)
    conn.commit()
    updated = cur.rowcount > 0
    conn.close()
    return updated


def delete_stock_event(event_id):
    conn = _connect()
    cur = conn.execute("DELETE FROM stock_events WHERE id = ?", (event_id,))
    conn.commit()
    deleted = cur.rowcount > 0
    conn.close()
    return deleted


def get_summary(fruit=None, user_id=None, date_from=None, date_to=None):
    """On-hand totals grouped by fruit and ripeness label, summed from the
    ledger (never a stored running total). Returns:
      {
        "grand_total": int,
        "by_fruit": {fruit: total_qty},
        "by_label": {label: total_qty},
        "matrix": {fruit: {label: qty}},
      }
    """
    conn = _connect()
    where_clauses = []
    params = []
    if fruit:
        where_clauses.append("fruit = ?")
        params.append(fruit)
    if user_id is not None:
        where_clauses.append("user_id = ?")
        params.append(user_id)
    _add_date_filters(where_clauses, params, date_from, date_to)
    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

    rows = conn.execute(
        f"""SELECT fruit, label, SUM(quantity) as qty FROM stock_events
            {where_sql} GROUP BY fruit, label""",
        params,
    ).fetchall()
    conn.close()

    matrix = {}
    by_fruit = {}
    by_label = {}
    grand_total = 0
    for r in rows:
        qty = r["qty"] or 0
        matrix.setdefault(r["fruit"], {})[r["label"]] = qty
        by_fruit[r["fruit"]] = by_fruit.get(r["fruit"], 0) + qty
        by_label[r["label"]] = by_label.get(r["label"], 0) + qty
        grand_total += qty

    return {
        "grand_total": grand_total,
        "by_fruit": by_fruit,
        "by_label": by_label,
        "matrix": matrix,
    }


init_db()
