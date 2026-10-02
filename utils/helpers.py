# utils/helpers.py
"""Small reusable helpers used across the project."""
from datetime import date, datetime
from pathlib import Path


# ============================================================
# Number / currency formatting
# ============================================================
def _format_indian_integer(number: int) -> str:
    digits = str(number)
    if len(digits) <= 3:
        return digits
    last_three, remaining = digits[-3:], digits[:-3]
    groups = []
    while len(remaining) > 2:
        groups.insert(0, remaining[-2:])
        remaining = remaining[:-2]
    if remaining:
        groups.insert(0, remaining)
    return ",".join(groups + [last_three])


def _format_indian_number(number) -> str:
    is_negative = number < 0
    number = abs(number)
    if isinstance(number, float) and not number.is_integer():
        integer_part, decimal_part = f"{number:.2f}".split(".")
        result = f"{_format_indian_integer(int(integer_part))}.{decimal_part}"
    else:
        result = _format_indian_integer(int(number))
    return f"-{result}" if is_negative else result


def format_currency(amount) -> str:
    """300000 -> ₹3,00,000"""
    if amount is None or amount == "":
        return "—"
    try:
        amount = float(amount)
    except (TypeError, ValueError):
        return str(amount)
    if amount.is_integer():
        amount = int(amount)
    return f"₹{_format_indian_number(amount)}"


# ============================================================
# Dates
# ============================================================
_DATE_FORMATS = ["%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d"]


def parse_date(value):
    """Convert many date formats into a date object (None if impossible)."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def format_date(value) -> str:
    """2026-10-20 -> 20 Oct 2026"""
    if value is None or value == "":
        return "—"
    parsed = parse_date(value)
    return parsed.strftime("%d %b %Y") if parsed else str(value)


def registration_status(start, end, enabled=True, today=None) -> str:
    """
    Returns one of: 'OPEN', 'NOT_STARTED', 'ENDED', 'CLOSED'
    ('CLOSED' = switched off by the admin).
    """
    today = today or date.today()
    start_date, end_date = parse_date(start), parse_date(end)

    if start_date and today < start_date:
        return "NOT_STARTED"
    if end_date and today > end_date:
        return "ENDED"
    if not enabled:
        return "CLOSED"
    return "OPEN"


# ============================================================
# Text
# ============================================================
def clean_text(value) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def title_case(value) -> str:
    return clean_text(value).title()


def truncate_text(value, max_length: int = 100) -> str:
    value = clean_text(value)
    if len(value) <= max_length:
        return value
    return value[: max_length - 3].rstrip() + "..."


# ============================================================
# Status labels
# ============================================================
_STATUS_LABELS = {
    "ELIGIBLE": "Eligible",
    "NOT_ELIGIBLE": "Not Eligible",
    "NEEDS_MORE_INFORMATION": "Needs More Information",
    "PENDING": "Pending",
    "APPROVED": "Approved",
    "REJECTED": "Rejected",
    "OPEN": "Open",
    "CLOSED": "Closed",
    "STOPPED": "Stopped",
    "ACTIVE": "Active",
    "INACTIVE": "Inactive",
}

_STATUS_ICONS = {
    "ELIGIBLE": "✅",
    "NOT_ELIGIBLE": "❌",
    "NEEDS_MORE_INFORMATION": "⚠️",
    "PENDING": "⏳",
    "APPROVED": "✅",
    "REJECTED": "❌",
    "OPEN": "🟢",
    "CLOSED": "⚪",
    "STOPPED": "⏹️",
    "ACTIVE": "🟢",
    "INACTIVE": "⚪",
}


def get_status_label(status) -> str:
    if not status:
        return "Unknown"
    return _STATUS_LABELS.get(str(status).upper(), str(status).replace("_", " ").title())


def get_status_emoji(status) -> str:
    if not status:
        return "•"
    return _STATUS_ICONS.get(str(status).upper(), "•")


# ============================================================
# Profile
# ============================================================
_PROFILE_FIELDS = {
    "name": "Name",
    "gender": "Gender",
    "family_income": "Family Income",
    "state": "State",
    "district": "District",
    "category": "Category",
    "course": "Course",
    "year": "Year",
}

# Fields that the eligibility engine needs (district is optional).
REQUIRED_PROFILE_FIELDS = ["gender", "family_income", "state", "category", "course", "year"]


def _is_filled(value) -> bool:
    return value is not None and str(value).strip() != ""


def profile_completion(profile: dict) -> int:
    """Approximate profile completion percentage."""
    if not profile:
        return 0
    done = sum(1 for field in _PROFILE_FIELDS if _is_filled(profile.get(field)))
    return round(done / len(_PROFILE_FIELDS) * 100)


def get_missing_profile_fields(profile: dict, required_only: bool = False) -> list:
    """Labels of profile fields that are still empty."""
    profile = profile or {}
    fields = (
        {f: _PROFILE_FIELDS[f] for f in REQUIRED_PROFILE_FIELDS}
        if required_only
        else _PROFILE_FIELDS
    )
    return [label for field, label in fields.items() if not _is_filled(profile.get(field))]


# ============================================================
# Files
# ============================================================
ALLOWED_DOCUMENT_EXTENSIONS = {"pdf", "png", "jpg", "jpeg", "webp"}


def get_file_extension(filename: str) -> str:
    if not filename:
        return ""
    return Path(filename).suffix.lower().lstrip(".")


def is_allowed_document(filename: str) -> bool:
    return get_file_extension(filename) in ALLOWED_DOCUMENT_EXTENSIONS


def format_file_size(size_bytes) -> str:
    if size_bytes is None:
        return "0 B"
    try:
        size = float(size_bytes)
    except (TypeError, ValueError):
        return "Unknown"
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{int(size)} B" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return "Unknown"


# ============================================================
# General
# ============================================================
def safe_int(value, default=0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_float(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def yes_no(value) -> str:
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if value is None:
        return "No"
    return "Yes" if str(value).strip().lower() in {"true", "1", "yes", "y"} else "No"
