# database.py
"""SQLite data layer. All other modules talk to the database only through this file."""
import hashlib
import json
import secrets
import sqlite3
from contextlib import contextmanager
from datetime import datetime

from config import (
    DATABASE_PATH,
    DEFAULT_APPLICATION_LIMIT,
    DEFAULT_APPLICATIONS_ENABLED,
    SAMPLE_OPPORTUNITIES_FILE,
    APPLICATION_PENDING,
)


# ============================================================
# GENERAL HELPERS
# ============================================================
def now() -> str:
    """Current date/time as an ISO string."""
    return datetime.now().isoformat(timespec="seconds")


@contextmanager
def get_connection():
    """
    Open a SQLite connection that commits on success, rolls back on error
    and ALWAYS closes (the plain sqlite3 context manager does not close).
    """
    connection = sqlite3.connect(str(DATABASE_PATH), timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


# ============================================================
# PASSWORD HASHING
# ============================================================
_PBKDF2_ROUNDS = 120_000


def hash_password(password: str) -> str:
    """PBKDF2-SHA256 hash, stored as 'salt$hash'."""
    salt = secrets.token_hex(16)
    password_hash = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt.encode("utf-8"), _PBKDF2_ROUNDS
    ).hex()
    return f"{salt}${password_hash}"


def verify_password(password: str, stored_password: str) -> bool:
    """Check a password against a stored 'salt$hash' value."""
    try:
        salt, expected_hash = stored_password.split("$", 1)
    except (ValueError, AttributeError):
        return False
    actual_hash = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt.encode("utf-8"), _PBKDF2_ROUNDS
    ).hex()
    return secrets.compare_digest(actual_hash, expected_hash)


# ============================================================
# SCHEMA
# ============================================================
SCHEMA = """
-- USERS
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('ADMIN', 'USER')),
    name TEXT NOT NULL,
    created_at TEXT NOT NULL
);

-- USER PROFILES
CREATE TABLE IF NOT EXISTS user_profiles (
    user_id INTEGER PRIMARY KEY,
    gender TEXT,
    family_income REAL,
    state TEXT,
    district TEXT,
    category TEXT,
    course TEXT,
    year INTEGER,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- USER DOCUMENTS
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    document_type TEXT NOT NULL,
    file_path TEXT NOT NULL,
    uploaded_at TEXT NOT NULL,
    UNIQUE (user_id, document_type),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- SCHOLARSHIPS / SCHEMES
CREATE TABLE IF NOT EXISTS opportunities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    organization TEXT,
    description TEXT,
    application_limit INTEGER,
    registration_start TEXT,
    registration_end TEXT,
    applications_enabled INTEGER NOT NULL DEFAULT 1,
    official_url TEXT,
    natural_language_rules TEXT,
    rules_verified INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- ELIGIBILITY RULES
CREATE TABLE IF NOT EXISTS eligibility_rules (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    opportunity_id INTEGER NOT NULL,
    field TEXT NOT NULL,
    operator TEXT NOT NULL,
    value_json TEXT NOT NULL,
    human_readable TEXT NOT NULL,
    FOREIGN KEY (opportunity_id) REFERENCES opportunities(id) ON DELETE CASCADE
);

-- REQUIRED DOCUMENTS
CREATE TABLE IF NOT EXISTS required_documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    opportunity_id INTEGER NOT NULL,
    document_type TEXT NOT NULL,
    FOREIGN KEY (opportunity_id) REFERENCES opportunities(id) ON DELETE CASCADE
);

-- APPLICATIONS
CREATE TABLE IF NOT EXISTS applications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    opportunity_id INTEGER NOT NULL,
    status TEXT NOT NULL,
    reason TEXT,
    applied_at TEXT NOT NULL,
    UNIQUE (user_id, opportunity_id),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (opportunity_id) REFERENCES opportunities(id) ON DELETE CASCADE
);

-- RAG SOURCE DOCUMENTS
CREATE TABLE IF NOT EXISTS opportunity_sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    opportunity_id INTEGER NOT NULL,
    file_name TEXT NOT NULL,
    file_path TEXT NOT NULL,
    uploaded_at TEXT NOT NULL,
    FOREIGN KEY (opportunity_id) REFERENCES opportunities(id) ON DELETE CASCADE
);
"""


def init_db():
    """Create all tables. Safe to call on every start."""
    with get_connection() as connection:
        connection.executescript(SCHEMA)


# ============================================================
# DEMO DATA
# ============================================================
DEMO_USERS = [
    ("admin", "admin123", "ADMIN", "System Admin"),
    ("student01", "student123", "USER", "Aarav Sharma"),
    ("student02", "student123", "USER", "Ananya Patil"),
    ("student03", "student123", "USER", "Rohan Shah"),
    ("student04", "student123", "USER", "Sneha Kulkarni"),
    ("student05", "student123", "USER", "Vikram Joshi"),
    ("student06", "student123", "USER", "Meera Desai"),
    ("student07", "student123", "USER", "Kabir Mehta"),
]

DEMO_PROFILES = {
    "student01": ("Male", 250000, "Maharashtra", "Mumbai", "SC", "Engineering", 3),
    "student02": ("Female", 180000, "Maharashtra", "Pune", "OBC", "Engineering", 2),
    "student03": ("Male", 500000, "Maharashtra", "Mumbai", "General", "Engineering", 4),
    "student04": ("Female", 220000, "Gujarat", "Ahmedabad", "SC", "Engineering", 3),
    "student05": ("Male", 150000, "Maharashtra", "Nashik", "ST", "Arts", 2),
    "student06": ("Female", 300000, "Maharashtra", "Mumbai", "General", "Engineering", 1),
    "student07": ("Female", 120000, "Maharashtra", "Thane", "OBC", "Engineering", 4),
}


def seed_demo_users():
    """Create demo accounts (and profiles for students) if they don't exist."""
    with get_connection() as connection:
        for username, password, role, name in DEMO_USERS:
            existing = connection.execute(
                "SELECT id FROM users WHERE username = ?", (username,)
            ).fetchone()
            if existing:
                continue

            cursor = connection.execute(
                """
                INSERT INTO users (username, password_hash, role, name, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (username, hash_password(password), role, name, now()),
            )
            user_id = cursor.lastrowid

            if role == "USER" and username in DEMO_PROFILES:
                gender, income, state, district, category, course, year = DEMO_PROFILES[username]
                connection.execute(
                    """
                    INSERT INTO user_profiles
                        (user_id, gender, family_income, state, district, category, course, year)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (user_id, gender, income, state, district, category, course, year),
                )


def seed_sample_opportunities():
    """
    Load data/sample_opportunities.json when the opportunities table is empty.
    Returns the number of opportunities inserted.
    """
    if not SAMPLE_OPPORTUNITIES_FILE.exists():
        return 0
    if get_all_opportunities():
        return 0

    with open(SAMPLE_OPPORTUNITIES_FILE, "r", encoding="utf-8") as file:
        samples = json.load(file)

    inserted = 0
    for sample in samples:
        opportunity_id = create_opportunity(sample)
        save_rules(opportunity_id, sample.get("rules", []))
        save_required_documents(opportunity_id, sample.get("required_documents", []))
        inserted += 1
    return inserted


# ============================================================
# LOGIN
# ============================================================
def authenticate_user(username: str, password: str):
    """
    Returns a user dict on success, None on failure.
    The dict contains BOTH 'id' and 'user_id' (same value) because the
    pages use 'id'.
    """
    username = (username or "").strip()
    with get_connection() as connection:
        user = connection.execute(
            "SELECT id, username, password_hash, role, name FROM users WHERE username = ?",
            (username,),
        ).fetchone()

    if user is None or not verify_password(password or "", user["password_hash"]):
        return None

    return {
        "id": user["id"],
        "user_id": user["id"],
        "username": user["username"],
        "role": user["role"],
        "name": user["name"],
    }


# ============================================================
# USER PROFILE
# ============================================================
def get_user_profile(user_id: int):
    """Full student profile (profile columns are None if not filled in yet)."""
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT u.id AS user_id, u.username, u.name,
                   p.gender, p.family_income, p.state, p.district,
                   p.category, p.course, p.year
            FROM users u
            LEFT JOIN user_profiles p ON u.id = p.user_id
            WHERE u.id = ?
            """,
            (user_id,),
        ).fetchone()
    return dict(row) if row else None


def save_user_profile(user_id: int, profile: dict = None, **fields):
    """
    Save or update a student profile.
    Accepts either a dict:  save_user_profile(1, {"gender": "Male", ...})
    or keyword arguments:   save_user_profile(1, gender="Male", ...)
    """
    data = dict(profile or {})
    data.update(fields)

    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO user_profiles
                (user_id, gender, family_income, state, district, category, course, year)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                gender = excluded.gender,
                family_income = excluded.family_income,
                state = excluded.state,
                district = excluded.district,
                category = excluded.category,
                course = excluded.course,
                year = excluded.year
            """,
            (
                user_id,
                data.get("gender"),
                data.get("family_income"),
                data.get("state"),
                data.get("district"),
                data.get("category"),
                data.get("course"),
                data.get("year"),
            ),
        )


# ============================================================
# OPPORTUNITIES
# ============================================================
def create_opportunity(opportunity_data: dict = None, **fields) -> int:
    """
    Create a scholarship/scheme and return its id.
    Accepts a dict or keyword arguments.
    """
    data = dict(opportunity_data or {})
    data.update(fields)

    if not str(data.get("name", "")).strip():
        raise ValueError("Opportunity name is required.")

    limit = data.get("application_limit")
    if limit is None:
        limit = DEFAULT_APPLICATION_LIMIT
    enabled = data.get("applications_enabled")
    if enabled is None:
        enabled = DEFAULT_APPLICATIONS_ENABLED

    timestamp = now()
    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO opportunities
                (name, organization, description, application_limit,
                 registration_start, registration_end, applications_enabled,
                 official_url, natural_language_rules, rules_verified,
                 created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
            """,
            (
                data["name"].strip(),
                data.get("organization", ""),
                data.get("description", ""),
                int(limit),
                data.get("registration_start"),
                data.get("registration_end"),
                int(bool(enabled)),
                data.get("official_url", ""),
                data.get("natural_language_rules", ""),
                timestamp,
                timestamp,
            ),
        )
        return cursor.lastrowid


def get_opportunity(opportunity_id: int):
    with get_connection() as connection:
        row = connection.execute(
            "SELECT * FROM opportunities WHERE id = ?", (opportunity_id,)
        ).fetchone()
    return dict(row) if row else None


def get_all_opportunities():
    with get_connection() as connection:
        rows = connection.execute("SELECT * FROM opportunities ORDER BY id DESC").fetchall()
    return [dict(row) for row in rows]


_UPDATABLE_OPPORTUNITY_FIELDS = {
    "name",
    "organization",
    "description",
    "application_limit",
    "registration_start",
    "registration_end",
    "applications_enabled",
    "official_url",
    "natural_language_rules",
    "rules_verified",
}


def update_opportunity(opportunity_id: int, **fields) -> bool:
    """Update known columns of an opportunity. Returns True if a row changed."""
    clean = {k: v for k, v in fields.items() if k in _UPDATABLE_OPPORTUNITY_FIELDS}
    if not clean:
        return False

    for key in ("applications_enabled", "rules_verified"):
        if key in clean:
            clean[key] = int(bool(clean[key]))

    clean["updated_at"] = now()
    assignments = ", ".join(f"{key} = ?" for key in clean)  # keys come from the whitelist
    values = list(clean.values()) + [opportunity_id]

    with get_connection() as connection:
        cursor = connection.execute(
            f"UPDATE opportunities SET {assignments} WHERE id = ?", values
        )
        return cursor.rowcount > 0


# ============================================================
# ELIGIBILITY RULES
# ============================================================
def save_rules(opportunity_id: int, rules: list):
    """
    Replace the eligibility rules of an opportunity and mark them verified.
    Only call this after the admin has verified the AI-generated rules.
    """
    with get_connection() as connection:
        connection.execute(
            "DELETE FROM eligibility_rules WHERE opportunity_id = ?", (opportunity_id,)
        )
        for rule in rules:
            field = rule["field"]
            operator = rule["operator"]
            human_readable = rule.get("human_readable") or f"{field} {operator} {rule['value']}"
            connection.execute(
                """
                INSERT INTO eligibility_rules
                    (opportunity_id, field, operator, value_json, human_readable)
                VALUES (?, ?, ?, ?, ?)
                """,
                (opportunity_id, field, operator, json.dumps(rule["value"]), human_readable),
            )
        connection.execute(
            "UPDATE opportunities SET rules_verified = ?, updated_at = ? WHERE id = ?",
            (1 if rules else 0, now(), opportunity_id),
        )


def get_rules(opportunity_id: int) -> list:
    """Structured rules; each rule has a decoded 'value'."""
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT id, opportunity_id, field, operator, value_json, human_readable
            FROM eligibility_rules
            WHERE opportunity_id = ?
            ORDER BY id
            """,
            (opportunity_id,),
        ).fetchall()

    rules = []
    for row in rows:
        rule = dict(row)
        rule["value"] = json.loads(rule.pop("value_json"))
        rules.append(rule)
    return rules


# ============================================================
# REQUIRED DOCUMENTS
# ============================================================
def save_required_documents(opportunity_id: int, document_types: list):
    """Replace the required documents of an opportunity."""
    with get_connection() as connection:
        connection.execute(
            "DELETE FROM required_documents WHERE opportunity_id = ?", (opportunity_id,)
        )
        seen = set()
        for document_type in document_types or []:
            document_type = str(document_type).strip()
            if not document_type or document_type.lower() in seen:
                continue
            seen.add(document_type.lower())
            connection.execute(
                "INSERT INTO required_documents (opportunity_id, document_type) VALUES (?, ?)",
                (opportunity_id, document_type),
            )


def get_required_documents(opportunity_id: int) -> list:
    """List of document-type strings."""
    with get_connection() as connection:
        rows = connection.execute(
            "SELECT document_type FROM required_documents WHERE opportunity_id = ? ORDER BY id",
            (opportunity_id,),
        ).fetchall()
    return [row["document_type"] for row in rows]


# ============================================================
# STUDENT DOCUMENTS
# ============================================================
def get_document_record(user_id: int, document_type: str):
    with get_connection() as connection:
        row = connection.execute(
            "SELECT * FROM documents WHERE user_id = ? AND document_type = ?",
            (user_id, document_type),
        ).fetchone()
    return dict(row) if row else None


def save_document_metadata(user_id: int, document_type: str, file_path: str):
    """Insert or replace the uploaded-document record (one per user + type)."""
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO documents (user_id, document_type, file_path, uploaded_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id, document_type) DO UPDATE SET
                file_path = excluded.file_path,
                uploaded_at = excluded.uploaded_at
            """,
            (user_id, document_type, str(file_path), now()),
        )


def get_user_documents(user_id: int) -> list:
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT id, user_id, document_type, file_path, uploaded_at
            FROM documents
            WHERE user_id = ?
            ORDER BY document_type
            """,
            (user_id,),
        ).fetchall()
    return [dict(row) for row in rows]


# ============================================================
# APPLICATIONS
# ============================================================
def create_application(
    user_id: int,
    opportunity_id: int,
    status: str = APPLICATION_PENDING,
    reason: str = "",
) -> int:
    """
    Create an application. Raises ValueError if the student already applied
    or the application limit has been reached (checked atomically).
    """
    with get_connection() as connection:
        connection.execute("BEGIN IMMEDIATE")

        opportunity = connection.execute(
            "SELECT application_limit FROM opportunities WHERE id = ?", (opportunity_id,)
        ).fetchone()
        if opportunity is None:
            raise ValueError("Opportunity not found.")

        already = connection.execute(
            "SELECT 1 FROM applications WHERE user_id = ? AND opportunity_id = ?",
            (user_id, opportunity_id),
        ).fetchone()
        if already:
            raise ValueError("You have already applied to this opportunity.")

        limit = opportunity["application_limit"]
        if limit:
            count = connection.execute(
                "SELECT COUNT(*) AS c FROM applications WHERE opportunity_id = ?",
                (opportunity_id,),
            ).fetchone()["c"]
            if count >= limit:
                raise ValueError("The application limit has been reached.")

        cursor = connection.execute(
            """
            INSERT INTO applications (user_id, opportunity_id, status, reason, applied_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (user_id, opportunity_id, status, reason, now()),
        )
        return cursor.lastrowid


def has_applied(user_id: int, opportunity_id: int) -> bool:
    with get_connection() as connection:
        row = connection.execute(
            "SELECT id FROM applications WHERE user_id = ? AND opportunity_id = ?",
            (user_id, opportunity_id),
        ).fetchone()
    return row is not None


def get_application_count(opportunity_id: int) -> int:
    with get_connection() as connection:
        row = connection.execute(
            "SELECT COUNT(*) AS count FROM applications WHERE opportunity_id = ?",
            (opportunity_id,),
        ).fetchone()
    return row["count"]


def get_user_applications(user_id: int) -> list:
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT a.id, a.user_id, a.opportunity_id, a.status, a.reason,
                   a.applied_at, o.name AS opportunity_name
            FROM applications a
            JOIN opportunities o ON o.id = a.opportunity_id
            WHERE a.user_id = ?
            ORDER BY a.applied_at DESC, a.id DESC
            """,
            (user_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_opportunity_applications(opportunity_id: int) -> list:
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT a.id, a.user_id, a.opportunity_id, a.status, a.reason, a.applied_at,
                   u.username, u.name,
                   p.gender, p.family_income, p.state, p.district,
                   p.category, p.course, p.year
            FROM applications a
            JOIN users u ON u.id = a.user_id
            LEFT JOIN user_profiles p ON p.user_id = a.user_id
            WHERE a.opportunity_id = ?
            ORDER BY a.applied_at DESC, a.id DESC
            """,
            (opportunity_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def update_application_status(application_id: int, status: str, reason: str = "") -> bool:
    with get_connection() as connection:
        cursor = connection.execute(
            "UPDATE applications SET status = ?, reason = ? WHERE id = ?",
            (status, reason, application_id),
        )
        return cursor.rowcount > 0


# ============================================================
# RAG SOURCE DOCUMENTS
# ============================================================
def save_opportunity_source(opportunity_id: int, file_name: str, file_path: str) -> int:
    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO opportunity_sources (opportunity_id, file_name, file_path, uploaded_at)
            VALUES (?, ?, ?, ?)
            """,
            (opportunity_id, file_name, str(file_path), now()),
        )
        return cursor.lastrowid


def get_opportunity_sources(opportunity_id: int) -> list:
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT * FROM opportunity_sources
            WHERE opportunity_id = ?
            ORDER BY uploaded_at DESC, id DESC
            """,
            (opportunity_id,),
        ).fetchall()
    return [dict(row) for row in rows]


# ============================================================
# COMMAND LINE: python database.py
# ============================================================
if __name__ == "__main__":
    print("Initializing Scholarship AI database...")
    init_db()
    seed_demo_users()
    count = seed_sample_opportunities()
    print(f"Database ready: {DATABASE_PATH} ({count} sample opportunities added)")
