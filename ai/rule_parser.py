# ai/rule_parser.py
"""Turns an administrator's natural-language rules into validated structured rules."""
import re

from ai.evaluator import normalize_operator
from ai.llm import ask_llm_json
from ai.prompts import RULE_GENERATION_PROMPT, render_prompt
from utils.helpers import format_currency

ALLOWED_FIELDS = {"gender", "category", "family_income", "state", "district", "course", "year"}

FIELD_LABELS = {
    "gender": "Gender",
    "category": "Category",
    "family_income": "Family income",
    "state": "State",
    "district": "District",
    "course": "Course",
    "year": "Year",
}

NUMERIC_FIELDS = {"family_income", "year"}


def generate_rules(admin_text: str) -> list:
    """Natural language -> validated list of rule dicts (calls the LLM)."""
    if not admin_text or not admin_text.strip():
        return []

    prompt = render_prompt(RULE_GENERATION_PROMPT, admin_text=admin_text.strip())
    result = ask_llm_json(prompt)

    # Accept {"rules": [...]} as well as a bare list.
    rules = result.get("rules", []) if isinstance(result, dict) else result
    if not isinstance(rules, list):
        raise ValueError("LLM returned an invalid rules format.")
    return validate_rules(rules)


# Name used by the admin page.
parse_rules = generate_rules


def _coerce_value(field: str, operator: str, value):
    """Make LLM values consistent: numbers for income/year, lower-case gender."""
    if isinstance(value, list):
        return [_coerce_value(field, "equals", item) for item in value]

    if field in NUMERIC_FIELDS:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return int(value) if float(value).is_integer() else value
        if isinstance(value, str):
            cleaned = re.sub(r"[₹,\s]", "", value)
            try:
                number = float(cleaned)
                return int(number) if number.is_integer() else number
            except ValueError:
                return value
        return value

    if field == "gender" and isinstance(value, str):
        mapping = {
            "woman": "female", "women": "female", "girl": "female", "female": "female",
            "man": "male", "men": "male", "boy": "male", "male": "male",
        }
        return mapping.get(value.strip().lower(), value.strip())

    return value.strip() if isinstance(value, str) else value


def validate_rules(rules: list) -> list:
    """Drop unusable rules and normalize the rest."""
    validated = []
    for rule in rules:
        if not isinstance(rule, dict):
            continue

        field = str(rule.get("field") or "").strip()
        operator = normalize_operator(rule.get("operator"))
        value = rule.get("value")

        if field not in ALLOWED_FIELDS or not operator:
            continue
        if value is None or value == "" or value == []:
            continue

        value = _coerce_value(field, operator, value)
        human_readable = (rule.get("human_readable") or "").strip() or build_human_readable(
            field, operator, value
        )
        validated.append(
            {"field": field, "operator": operator, "value": value, "human_readable": human_readable}
        )
    return validated


def build_human_readable(field, operator, value) -> str:
    """Readable sentence for a rule."""
    field_name = FIELD_LABELS.get(field, str(field))

    if isinstance(value, list):
        value_display = ", ".join(str(v) for v in value)
    elif field == "family_income" and isinstance(value, (int, float)):
        value_display = format_currency(value)
    else:
        value_display = str(value)

    texts = {
        "equals": f"{field_name} must be {value_display}",
        "not_equals": f"{field_name} must not be {value_display}",
        "in": f"{field_name} must be one of: {value_display}",
        "not_in": f"{field_name} must not be one of: {value_display}",
        "contains": f"{field_name} must include {value_display}",
        "less_than": f"{field_name} must be below {value_display}",
        "less_than_or_equal": f"{field_name} must be at most {value_display}",
        "greater_than": f"{field_name} must be above {value_display}",
        "greater_than_or_equal": f"{field_name} must be at least {value_display}",
    }
    return texts.get(normalize_operator(operator), f"{field_name}: {operator} {value_display}")


def format_rules_for_admin(rules: list) -> list:
    """Readable strings for the admin UI."""
    return [r.get("human_readable", "") for r in rules if r.get("human_readable")]
