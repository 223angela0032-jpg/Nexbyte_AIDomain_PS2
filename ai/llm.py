# ai/llm.py
"""Thin wrapper around the Gemini API. Every LLM call in the project goes through here."""
import json

from config import GEMINI_API_KEY, GEMINI_MODEL
from ai.prompts import (
    ELIGIBILITY_EXPLANATION_PROMPT,
    RAG_EXPLANATION_PROMPT,
    render_prompt,
)

_client = None


def get_client():
    """Create the Gemini client on first use."""
    global _client
    if _client is None:
        if not GEMINI_API_KEY:
            raise RuntimeError(
                "GEMINI_API_KEY is not configured. Add GEMINI_API_KEY to your .env file."
            )
        from google import genai  # imported lazily so the app starts without the SDK

        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


def is_llm_configured() -> bool:
    return bool(GEMINI_API_KEY)


def ask_llm(prompt: str) -> str:
    """Send a prompt to Gemini and return the text answer."""
    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty.")

    response = get_client().models.generate_content(model=GEMINI_MODEL, contents=prompt)
    if not response.text:
        raise RuntimeError("Gemini returned an empty response.")
    return response.text.strip()


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    return text.strip()


def ask_llm_json(prompt: str):
    """Send a prompt to Gemini and return the parsed JSON answer."""
    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty.")

    response = get_client().models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config={"response_mime_type": "application/json"},
    )
    if not response.text:
        raise RuntimeError("Gemini returned an empty JSON response.")

    response_text = _strip_code_fences(response.text)
    try:
        return json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Gemini returned invalid JSON.\n\nResponse:\n{response_text}") from exc


def _compact(data) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, default=str)


def generate_eligibility_explanation(
    profile: dict,
    opportunity: dict,
    evaluation: dict,
    rag_context: str = "",
) -> str:
    """Student-friendly explanation of a deterministic eligibility result."""
    profile_view = {
        key: profile.get(key)
        for key in ("gender", "family_income", "state", "district", "category", "course", "year")
    }
    opportunity_view = {
        key: opportunity.get(key)
        for key in ("name", "organization", "description", "registration_start", "registration_end")
    }
    prompt = render_prompt(
        ELIGIBILITY_EXPLANATION_PROMPT,
        user_profile=_compact(profile_view),
        opportunity=_compact(opportunity_view),
        evaluation=_compact(evaluation),
        evidence=rag_context or "No official document evidence is available.",
    )
    return ask_llm(prompt)


def answer_question_with_rag(question: str, rag_context: str) -> str:
    """Answer a free-text question using only retrieved official clauses."""
    prompt = render_prompt(
        RAG_EXPLANATION_PROMPT,
        evidence=rag_context or "No official document evidence is available.",
        question=question,
    )
    return ask_llm(prompt)
