# utils/ui.py
"""Tiny Streamlit helpers shared by the pages."""
import streamlit as st

_FLASH_KEY = "_flash_messages"


def flash(kind: str, message: str):
    """
    Queue a message that survives st.rerun().
    kind: 'success' | 'error' | 'warning' | 'info'
    """
    st.session_state.setdefault(_FLASH_KEY, []).append((kind, message))


def show_flash():
    """Display and clear queued messages (call near the top of a page)."""
    messages = st.session_state.pop(_FLASH_KEY, [])
    for kind, message in messages:
        getattr(st, kind if kind in {"success", "error", "warning", "info"} else "info")(message)
