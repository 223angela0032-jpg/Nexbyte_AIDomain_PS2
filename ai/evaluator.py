# ai/evaluator.py
"""
Deterministic eligibility engine.

The LLM is NOT used to make the eligibility decision. A student's profile is
compared with structured rules using plain Python logic.
"""
import json
from typing import Any

from config import ELIGIBLE, NOT_ELIGIBLE, NEEDS_MORE_INFORMATION

# ---------------------------------------------------------------- operators
# Canonical operator names + accepted aliases (so old data / hand-typed rules still work).
OPERATOR_ALIASES = {
    "equals": "equals", "equal": "equals", "==": "equals", "=": "equals", "eq": "equals",
    "not_equals": "not_equals", "not_equal": "not_equals", "!=": "not_equals", "ne": "not_equals",
    "in": "in",
    "not_in": "not_in",
    "contains": "contains", "includes": "contains",
    "less_than": "less_than", "<": "less_than", "lt": "less_than",
    "less_than_or_equal": "less_than_or_equal", "less_equal": "less_than_or_equal",
    "<=": "less_than_or_equal", "max": "less_than_or_equal", "lte": "less_than_or_equal",
    "greater_than": "greater_than", ">": "greater_than", "gt": "greater_than",
    "greater_than_or_equal": "greater_than_or_equal", "greater_equal": "greater_than_or_equal",
    ">=": "greater_than_or_equal", "min": "greater_than_or_equal", "gte": "greater_than_or_equal",
}

NUMERIC_OPERATORS = {"less_than", "less_than_or_equal", "greater_than", "greater_than_or_equal"}


def normalize_operator(operator: Any) -> str:
    """Return the canonical operator name, or '' if unknown."""
    return OPERATOR_ALIASES.get(str(operator or "").strip().lower(), "")


# ---------------------------------------------------------------- course groups
# A rule like  course contains "STEM"  or  "undergraduate"  describes a *group* of
# courses, while the student's profile holds a concrete course such as "B.Tech".
_STEM_WORDS = (
    "stem", "engineering", "b.tech", "btech", "b.e", "m.tech", "mtech", "technology",
    "science", "b.sc", "bsc", "m.sc", "msc", "mathematics", "maths", "math",
    "computer", "bca", "mca", "information technology", "statistics",
)
_UNDERGRAD_WORDS = (
    "undergraduate", "b.tech", "btech", "b.e", "b.sc", "bsc", "b.a", "b.com", "bcom",
    "bba", "bca", "b.arch", "engineering", "arts", "commerce", "science", "bachelor",
    "diploma", "mbbs", "llb",
)

COURSE_GROUPS = {
    "stem": _STEM_WORDS,
    "undergraduate": _UNDERGRAD_WORDS,
    "ug": _UNDERGRAD_WORDS,
    # Any enrolled course counts as "higher education".
    "higher education": None,
}


def course_matches(actual: str, expected: str) -> bool:
    """True if the student's course satisfies the (possibly group-style) expected course."""
    actual = str(actual).strip().lower()
    expected = str(expected).strip().lower()
    if not actual or not expected:
        return False
    if expected in actual or actual in expected:
        return True
    if expected in COURSE_GROUPS:
        words = COURSE_GROUPS[expected]
        if words is None:
            return True
        return any(word in actual for word in words)
    return False


# ---------------------------------------------------------------- value helpers
def _to_number(value: Any):
    """Return a float if the value is numeric (or numeric text), else None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip().replace(",", ""))
        except ValueError:
            return None
    return None


def _normalize(value: Any):
    """Lower-case/strip strings so 'Maharashtra' == 'maharashtra'."""
    if isinstance(value, str):
        return value.strip().lower()
    return value


def _as_list(value: Any) -> list:
    """Turn an expected value into a list (JSON text and comma-separated text accepted)."""
    if isinstance(value, (list, tuple, set)):
        return list(value)
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("["):
            try:
                loaded = json.loads(text)
                if isinstance(loaded, list):
                    return loaded
            except json.JSONDecodeError:
                pass
        return [part.strip() for part in text.split(",") if part.strip()]
    return [value]


def _same(a: Any, b: Any) -> bool:
    """Equality that treats 3, 3.0 and '3' as the same value and ignores text case."""
    number_a, number_b = _to_number(a), _to_number(b)
    if number_a is not None and number_b is not None:
        return number_a == number_b
    return _normalize(a) == _normalize(b)


# ---------------------------------------------------------------- single rule
def evaluate_rule(actual_value: Any, operator: str, expected_value: Any, field: str = "") -> bool:
    """Evaluate ONE rule. Unknown operators evaluate to False."""
    operator = normalize_operator(operator)

    if operator == "equals":
        if field == "course":
            return course_matches(actual_value, expected_value)
        return _same(actual_value, expected_value)

    if operator == "not_equals":
        if field == "course":
            return not course_matches(actual_value, expected_value)
        return not _same(actual_value, expected_value)

    if operator == "in":
        if field == "course":
            return any(course_matches(actual_value, item) for item in _as_list(expected_value))
        return any(_same(actual_value, item) for item in _as_list(expected_value))

    if operator == "not_in":
        if field == "course":
            return not any(course_matches(actual_value, item) for item in _as_list(expected_value))
        return not any(_same(actual_value, item) for item in _as_list(expected_value))

    if operator == "contains":
        if field == "course":
            return course_matches(actual_value, expected_value)
        return str(_normalize(expected_value)) in str(_normalize(actual_value))

    if operator in NUMERIC_OPERATORS:
        actual_number = _to_number(actual_value)
        expected_number = _to_number(expected_value)
        if actual_number is None or expected_number is None:
            return False
        if operator == "less_than":
            return actual_number < expected_number
        if operator == "less_than_or_equal":
            return actual_number <= expected_number
        if operator == "greater_than":
            return actual_number > expected_number
        return actual_number >= expected_number

    return False


# ---------------------------------------------------------------- documents
def _document_type(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("document_type") or "").strip()
    return str(item or "").strip()


def check_missing_documents(user_documents: list, required_documents: list) -> list:
    """Required document types the student has not uploaded (original spelling kept)."""
    uploaded = {_document_type(d).lower() for d in (user_documents or []) if _document_type(d)}
    missing = []
    for required in required_documents or []:
        name = _document_type(required)
        if name and name.lower() not in uploaded:
            missing.append(name)
    return missing


# ---------------------------------------------------------------- full opportunity
def _rule_summary(rule: dict, actual: Any = None) -> dict:
    field = rule.get("field")
    operator = rule.get("operator")
    expected = rule.get("value")
    return {
        "field": field,
        "operator": operator,
        "actual_value": actual,
        "expected_value": expected,
        "human_readable": rule.get("human_readable") or f"{field} {operator} {expected}",
    }


def evaluate_opportunity(
    user_profile: dict,
    rules: list,
    user_documents: list = None,
    required_documents: list = None,
) -> dict:
    """
    Compare a student profile with an opportunity's rules.

    Status logic:
        any rule definitely failed        -> NOT_ELIGIBLE
        else any profile data missing     -> NEEDS_MORE_INFORMATION
        else                              -> ELIGIBLE
    Missing documents are reported separately (they do not change the status).
    """
    user_profile = user_profile or {}
    passed_rules, failed_rules, missing_information = [], [], []

    for rule in rules or []:
        if not isinstance(rule, dict):
            continue
        field = rule.get("field")
        if not field:
            continue

        actual = user_profile.get(field)
        if actual is None or str(actual).strip() == "":
            missing_information.append(
                {
                    "field": field,
                    "message": f"Please provide your {str(field).replace('_', ' ')}.",
                    "rule": rule,
                }
            )
            continue

        if not normalize_operator(rule.get("operator")):
            # An unreadable rule must never silently pass or fail a student.
            missing_information.append(
                {
                    "field": field,
                    "message": f"Rule could not be checked automatically: "
                    f"{rule.get('human_readable') or field}",
                    "rule": rule,
                }
            )
            continue

        if evaluate_rule(actual, rule.get("operator"), rule.get("value"), field=field):
            passed_rules.append(_rule_summary(rule, actual))
        else:
            failed_rules.append(_rule_summary(rule, actual))

    missing_documents = check_missing_documents(user_documents or [], required_documents or [])

    if not rules:
        status = NEEDS_MORE_INFORMATION
    elif failed_rules:
        status = NOT_ELIGIBLE
    elif missing_information:
        status = NEEDS_MORE_INFORMATION
    else:
        status = ELIGIBLE

    return {
        "status": status,
        "passed_rules": passed_rules,
        "failed_rules": failed_rules,
        "missing_information": missing_information,
        "missing_documents": missing_documents,
        "has_rules": bool(rules),
    }
