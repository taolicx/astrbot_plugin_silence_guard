from __future__ import annotations

import importlib
from typing import Any, Iterable


FOCUS_SESSION_MODULE_CANDIDATES = (
    "astrbot_plugin_focus_session.main",
    "astrbot_plugin_focus_session",
    "focus_session.main",
    "focus_session",
)


def build_focus_session_key(event: Any) -> str:
    origin = str(getattr(event, "unified_msg_origin", "") or "").strip()
    sender_getter = getattr(event, "get_sender_id", None)
    if not callable(sender_getter):
        return ""
    sender_id = str(sender_getter() or "").strip()
    if not origin or not sender_id:
        return ""
    return f"{origin}::{sender_id}"


def clear_focus_session(
    session_key: str,
    module_names: Iterable[str] = FOCUS_SESSION_MODULE_CANDIDATES,
) -> bool:
    if not session_key:
        return False

    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue

        sessions = getattr(module, "_FOCUS_SESSIONS", None)
        if isinstance(sessions, dict) and session_key in sessions:
            sessions.pop(session_key, None)
            return True

    return False


def has_focus_session(
    session_key: str,
    module_names: Iterable[str] = FOCUS_SESSION_MODULE_CANDIDATES,
) -> bool:
    if not session_key:
        return False

    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue

        sessions = getattr(module, "_FOCUS_SESSIONS", None)
        if isinstance(sessions, dict) and session_key in sessions:
            return True

    return False


def mark_focus_session_expired(
    session_key: str,
    module_names: Iterable[str] = FOCUS_SESSION_MODULE_CANDIDATES,
) -> bool:
    if not session_key:
        return False

    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue

        sessions = getattr(module, "_FOCUS_SESSIONS", None)
        if isinstance(sessions, dict) and session_key in sessions:
            session = sessions.get(session_key)
            if session is not None:
                try:
                    session.expires_at = 0
                except Exception:
                    pass
                try:
                    session.last_notice_at = 0
                except Exception:
                    pass
            return True

    return False
