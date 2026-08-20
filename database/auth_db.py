"""
Auth + admin/settings persistence: users, activity log, and a small
key/value settings store. Same connect-per-call sqlite pattern as
history_db.py, same DB file.
"""
import os
import sqlite3
import secrets
import string
from datetime import datetime, timedelta

from werkzeug.security import generate_password_hash, check_password_hash

DB_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(DB_DIR, "fruitvision.db")

ROLES = ("admin", "farmer")


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _column_exists(conn, table, column):
    cols = [r["name"] for r in conn.execute(f"PRAGMA table_info({table})")]
    return column in cols


def init_db():
    os.makedirs(DB_DIR, exist_ok=True)
    conn = _connect()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'farmer',
            created_at TEXT NOT NULL,
            last_active TEXT
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS activity_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            action TEXT NOT NULL,
            detail TEXT,
            created_at TEXT NOT NULL
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)

    # Migrate the pre-existing `results` table (from history_db.py) with the
    # new nullable columns needed for per-user stats, latency, and flagging.
    if conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='results'"
    ).fetchone()[0]:
        if not _column_exists(conn, "results", "user_id"):
            conn.execute("ALTER TABLE results ADD COLUMN user_id INTEGER")
        if not _column_exists(conn, "results", "latency_ms"):
            conn.execute("ALTER TABLE results ADD COLUMN latency_ms REAL")
        if not _column_exists(conn, "results", "flagged"):
            conn.execute("ALTER TABLE results ADD COLUMN flagged INTEGER DEFAULT 0")

    conn.commit()

    seeded_accounts = []
    if conn.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0:
        for name, email, role, password in (
            ("Admin", "admin@fruitvision.local", "admin", "admin123"),
            ("Demo Farmer", "farmer@fruitvision.local", "farmer", "farmer123"),
        ):
            conn.execute(
                """INSERT INTO users (name, email, password_hash, role, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (name, email, generate_password_hash(password), role,
                 datetime.now().isoformat(timespec="seconds")),
            )
            seeded_accounts.append((email, password))
        conn.commit()

    conn.close()

    if seeded_accounts:
        lines = "\n".join(f"  {email} / {password}" for email, password in seeded_accounts)
        print(
            "\n[FruitVision] Seeded default accounts:\n"
            f"{lines}\n"
            "  (change these from Settings after logging in)\n"
        )


def _row_to_dict(row):
    return dict(row) if row else None


def create_user(name, email, password, role="farmer"):
    """Raises sqlite3.IntegrityError if the email is already taken."""
    conn = _connect()
    cur = conn.execute(
        """INSERT INTO users (name, email, password_hash, role, created_at)
           VALUES (?, ?, ?, ?, ?)""",
        (
            name,
            email,
            generate_password_hash(password),
            role if role in ROLES else "farmer",
            datetime.now().isoformat(timespec="seconds"),
        ),
    )
    conn.commit()
    user_id = cur.lastrowid
    conn.close()
    return user_id


def generate_temp_password(length=10):
    alphabet = string.ascii_letters + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


def get_user_by_id(user_id):
    conn = _connect()
    row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    return _row_to_dict(row)


def get_user_by_email(email):
    conn = _connect()
    row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    conn.close()
    return _row_to_dict(row)


def verify_login(email, password):
    """Returns the user dict on success, else None."""
    user = get_user_by_email(email)
    if user and check_password_hash(user["password_hash"], password):
        return user
    return None


def set_password(user_id, new_password):
    conn = _connect()
    conn.execute(
        "UPDATE users SET password_hash = ? WHERE id = ?",
        (generate_password_hash(new_password), user_id),
    )
    conn.commit()
    conn.close()


def update_user_name(user_id, name):
    conn = _connect()
    conn.execute("UPDATE users SET name = ? WHERE id = ?", (name, user_id))
    conn.commit()
    conn.close()


def list_users():
    conn = _connect()
    rows = conn.execute("SELECT * FROM users ORDER BY id ASC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def update_user_role(user_id, role):
    if role not in ROLES:
        return False
    conn = _connect()
    cur = conn.execute("UPDATE users SET role = ? WHERE id = ?", (role, user_id))
    conn.commit()
    updated = cur.rowcount > 0
    conn.close()
    return updated


def delete_user(user_id):
    conn = _connect()
    cur = conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
    conn.commit()
    deleted = cur.rowcount > 0
    conn.close()
    return deleted


def touch_last_active(user_id):
    conn = _connect()
    conn.execute(
        "UPDATE users SET last_active = ? WHERE id = ?",
        (datetime.now().isoformat(timespec="seconds"), user_id),
    )
    conn.commit()
    conn.close()


def admin_count():
    conn = _connect()
    count = conn.execute(
        "SELECT COUNT(*) FROM users WHERE role = 'admin'"
    ).fetchone()[0]
    conn.close()
    return count


def log_activity(user_id, action, detail=None):
    conn = _connect()
    conn.execute(
        """INSERT INTO activity_log (user_id, action, detail, created_at)
           VALUES (?, ?, ?, ?)""",
        (user_id, action, detail, datetime.now().isoformat(timespec="seconds")),
    )
    conn.commit()
    conn.close()


def get_recent_activity(limit=15, user_id=None):
    conn = _connect()
    if user_id is not None:
        rows = conn.execute(
            """SELECT activity_log.*, users.name AS user_name FROM activity_log
               LEFT JOIN users ON users.id = activity_log.user_id
               WHERE activity_log.user_id = ?
               ORDER BY activity_log.id DESC LIMIT ?""",
            (user_id, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            """SELECT activity_log.*, users.name AS user_name FROM activity_log
               LEFT JOIN users ON users.id = activity_log.user_id
               ORDER BY activity_log.id DESC LIMIT ?""",
            (limit,),
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


DEFAULT_SETTINGS = {
    "default_model": "ab",
    "confidence_threshold": "0",
}


def get_setting(key, default=None):
    conn = _connect()
    row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    conn.close()
    if row is not None:
        return row["value"]
    return DEFAULT_SETTINGS.get(key, default)


def get_all_settings():
    return {key: get_setting(key) for key in DEFAULT_SETTINGS}


def set_setting(key, value):
    conn = _connect()
    conn.execute(
        "INSERT INTO settings (key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, str(value)),
    )
    conn.commit()
    conn.close()


init_db()
