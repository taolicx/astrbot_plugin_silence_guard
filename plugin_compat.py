from __future__ import annotations

import importlib
import sys
from typing import Any, Iterable


FOCUS_SESSION_MODULE_CANDIDATES = (
    "data.plugins.astrbot_plugin_focus_session.main",
    "data.plugins.astrbot_plugin_focus_session",
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

    cleared = False
    for module in _iter_focus_session_modules(module_names):
        sessions = getattr(module, "_FOCUS_SESSIONS", None)
        if isinstance(sessions, dict) and session_key in sessions:
            sessions.pop(session_key, None)
            cleared = True

    return cleared


def clear_all_focus_sessions(
    module_names: Iterable[str] = FOCUS_SESSION_MODULE_CANDIDATES,
) -> int:
    cleared = 0
    seen_sessions: set[int] = set()

    for module in _iter_focus_session_modules(module_names):
        sessions = getattr(module, "_FOCUS_SESSIONS", None)
        if not isinstance(sessions, dict):
            continue
        sessions_id = id(sessions)
        if sessions_id in seen_sessions:
            continue
        seen_sessions.add(sessions_id)
        cleared += len(sessions)
        sessions.clear()

    return cleared


def has_focus_session(
    session_key: str,
    module_names: Iterable[str] = FOCUS_SESSION_MODULE_CANDIDATES,
) -> bool:
    if not session_key:
        return False

    for module in _iter_focus_session_modules(module_names):
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

    marked = False
    for module in _iter_focus_session_modules(module_names):
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
            marked = True

    return marked


def _iter_focus_session_modules(module_names: Iterable[str]) -> list[Any]:
    modules: list[Any] = []
    seen: set[int] = set()

    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            module = sys.modules.get(module_name)
        if module is not None and id(module) not in seen:
            modules.append(module)
            seen.add(id(module))

    for module_name, module in tuple(sys.modules.items()):
        if module is None or id(module) in seen:
            continue
        if not (
            module_name.endswith("astrbot_plugin_focus_session")
            or module_name.endswith("astrbot_plugin_focus_session.main")
        ):
            continue
        if isinstance(getattr(module, "_FOCUS_SESSIONS", None), dict):
            modules.append(module)
            seen.add(id(module))

    return modules
