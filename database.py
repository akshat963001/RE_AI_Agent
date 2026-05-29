import sqlite3
from config import DB_PATH


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            phone     TEXT NOT NULL,
            role      TEXT NOT NULL,      -- 'user' (agent/owner) or 'assistant' (AI reply)
            content   TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS leads (
            phone             TEXT PRIMARY KEY,
            first_contact     DATETIME DEFAULT CURRENT_TIMESTAMP,
            last_contact      DATETIME DEFAULT CURRENT_TIMESTAMP,
            status            TEXT DEFAULT 'active',
            -- active | escalated | human_takeover | closed | discarded | blacklisted
            lead_score        INTEGER DEFAULT 5,
            escalation_count  INTEGER DEFAULT 0,
            building_name     TEXT DEFAULT '',
            br_type           TEXT DEFAULT '',   -- '1br' or '2br'
            asking_psf        REAL,
            area_sqft         REAL,
            notes             TEXT DEFAULT ''
        )
    """)

    conn.commit()
    conn.close()
    print("[DB] Initialized")


# ── Messages ──────────────────────────────────────────────────────────────────

def save_message(phone: str, role: str, content: str):
    conn = get_conn()
    c = conn.cursor()
    c.execute(
        "INSERT INTO messages (phone, role, content) VALUES (?, ?, ?)",
        (phone, role, content)
    )
    c.execute("INSERT OR IGNORE INTO leads (phone) VALUES (?)", (phone,))
    c.execute(
        "UPDATE leads SET last_contact = CURRENT_TIMESTAMP WHERE phone = ?",
        (phone,)
    )
    conn.commit()
    conn.close()


def get_conversation(phone: str, limit: int = 20) -> list[dict]:
    """Return up to `limit` most-recent messages in chronological order."""
    conn = get_conn()
    c = conn.cursor()
    c.execute(
        """
        SELECT role, content FROM messages
        WHERE phone = ?
        ORDER BY timestamp DESC
        LIMIT ?
        """,
        (phone, limit)
    )
    rows = c.fetchall()
    conn.close()
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]


# ── Leads ─────────────────────────────────────────────────────────────────────

def get_lead(phone: str) -> dict | None:
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT * FROM leads WHERE phone = ?", (phone,))
    row = c.fetchone()
    conn.close()
    return dict(row) if row else None


def update_lead(phone: str, **kwargs):
    if not kwargs:
        return
    conn = get_conn()
    c = conn.cursor()
    c.execute("INSERT OR IGNORE INTO leads (phone) VALUES (?)", (phone,))
    sets = ", ".join(f"{k} = ?" for k in kwargs)
    values = list(kwargs.values()) + [phone]
    c.execute(f"UPDATE leads SET {sets} WHERE phone = ?", values)
    conn.commit()
    conn.close()


def increment_escalation_count(phone: str):
    conn = get_conn()
    c = conn.cursor()
    c.execute("INSERT OR IGNORE INTO leads (phone) VALUES (?)", (phone,))
    c.execute(
        "UPDATE leads SET escalation_count = escalation_count + 1 WHERE phone = ?",
        (phone,)
    )
    conn.commit()
    conn.close()


def get_all_leads() -> list[dict]:
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT * FROM leads ORDER BY lead_score DESC, last_contact DESC")
    rows = c.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def is_blocked(phone: str) -> bool:
    """Returns True if the lead is blacklisted or discarded — no reply should be sent."""
    lead = get_lead(phone)
    if not lead:
        return False
    return lead.get("status") in ("blacklisted", "discarded")
