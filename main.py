from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

try:
    from astrbot.api import AstrBotConfig, logger
    from astrbot.api.event import AstrMessageEvent, filter
    from astrbot.api.event.filter import EventMessageType
    from astrbot.api.star import Context, Star, register
    from astrbot.core.message.components import At
except ImportError:  # pragma: no cover - local tests without AstrBot installed
    class _LocalLogger:
        def info(self, *args: Any, **kwargs: Any) -> None:
            pass

        def warning(self, *args: Any, **kwargs: Any) -> None:
            pass

    class _LocalFilter:
        class PermissionType:
            ADMIN = object()

        class EventMessageType:
            ALL = object()

        @staticmethod
        def event_message_type(*args: Any, **kwargs: Any):
            return lambda func: func

        @staticmethod
        def permission_type(*args: Any, **kwargs: Any):
            return lambda func: func

        @staticmethod
        def command(*args: Any, **kwargs: Any):
            return lambda func: func

        @staticmethod
        def after_message_sent(*args: Any, **kwargs: Any):
            return lambda func: func

    class _LocalStar:
        def __init__(self, context: Any) -> None:
            self.context = context

    class _LocalAt:
        pass

    def register(*args: Any, **kwargs: Any):
        return lambda cls: cls

    AstrBotConfig = dict
    AstrMessageEvent = Any
    Context = Any
    EventMessageType = _LocalFilter.EventMessageType
    Star = _LocalStar
    At = _LocalAt
    filter = _LocalFilter()
    logger = _LocalLogger()

try:
    from .deepseek_judge import DeepSeekJudge
    from .plugin_compat import (
        build_focus_session_key,
        clear_all_focus_sessions,
        clear_focus_session,
        has_focus_session,
        mark_focus_session_expired,
        parse_command_list,
        strip_wake_prefix,
    )
    from .silence_logic import (
        MUTE,
        NO_REPLY,
        REPLY,
        UNCERTAIN,
        WAKE,
        ConversationState,
        Decision,
        DEFAULT_ACKS,
        DEFAULT_FAREWELL_KEYWORDS,
        DEFAULT_HARD_CLOSERS,
        DEFAULT_SILENCE_KEYWORDS,
        DEFAULT_WAKE_KEYWORDS,
        analyze_rules,
        apply_decision_to_state,
        is_active_context,
    )
except ImportError:
    from deepseek_judge import DeepSeekJudge
    from plugin_compat import (
        build_focus_session_key,
        clear_all_focus_sessions,
        clear_focus_session,
        has_focus_session,
        mark_focus_session_expired,
        parse_command_list,
        strip_wake_prefix,
    )
    from silence_logic import (
        MUTE,
        NO_REPLY,
        REPLY,
        UNCERTAIN,
        WAKE,
        ConversationState,
        Decision,
        DEFAULT_ACKS,
        DEFAULT_FAREWELL_KEYWORDS,
        DEFAULT_HARD_CLOSERS,
        DEFAULT_SILENCE_KEYWORDS,
        DEFAULT_WAKE_KEYWORDS,
        analyze_rules,
        apply_decision_to_state,
        is_active_context,
    )


PLUGIN_NAME = "astrbot_plugin_silence_guard"
DEFAULT_CLEAR_ALL_COMMANDS = (
    "结束所有对话",
    "清空连续对话",
    "停止所有监听",
    "结束所有监听",
    "关闭所有连续对话",
)


@dataclass
class ListenSession:
    expires_at: float
    last_notice_at: float = 0.0


DEFAULT_CONFIG: dict[str, Any] = {
    "enabled": True,
    "enable_private": True,
    "enable_group": True,
    "listen_mode_enabled": True,
    "listen_seconds": 300,
    "listen_refresh_on_each_message": True,
    "listen_enable_private": True,
    "listen_enable_group": True,
    "listen_ignore_self_messages": True,
    "listen_ignore_wake_prefix_commands": True,
    "listen_sender_whitelist": [],
    "listen_sender_blacklist": [],
    "listen_group_whitelist": [],
    "listen_group_blacklist": [],
    "listen_send_activation_notice": False,
    "listen_activation_notice": "已进入连续对话模式，{minutes} 分钟内不用再 @ 我。",
    "listen_send_expire_notice": False,
    "listen_expire_notice": "连续对话已超时，需要再 @ 或使用唤醒词。",
    "smart_mode": True,
    "judge_mode": "ambiguous_only",
    "judge_provider_id": "",
    "judge_timeout_seconds": 3,
    "judge_max_tokens": 96,
    "judge_context_turns": 6,
    "judge_min_confidence": 0.72,
    "judge_cache_seconds": 120,
    "fallback_when_uncertain": "reply",
    "default_mute_seconds": 600,
    "farewell_cooldown_seconds": 1800,
    "active_chat_seconds": 300,
    "reply_once_to_farewell": True,
    "debug_log": False,
    "admin_clear_all_commands": list(DEFAULT_CLEAR_ALL_COMMANDS),
    "silence_keywords": DEFAULT_SILENCE_KEYWORDS,
    "farewell_keywords": DEFAULT_FAREWELL_KEYWORDS,
    "wake_keywords": DEFAULT_WAKE_KEYWORDS,
    "hard_closers": DEFAULT_HARD_CLOSERS,
    "ack_keywords": DEFAULT_ACKS,
}


@register(
    PLUGIN_NAME,
    "taolicx",
    "整合连续监听和不回复守门：被唤醒后可免 @ 连续对话，也能按语义收住回复。",
    "2.0.0",
)
class SilenceGuardPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig | None = None) -> None:
        super().__init__(context)
        self.raw_config = config or {}
        self.states: dict[str, ConversationState] = {}
        self.listen_sessions: dict[str, ListenSession] = {}
        self.judge = DeepSeekJudge(self._config(), self.context, logger=logger)

    async def initialize(self) -> None:
        cfg = self._config()
        self.judge = DeepSeekJudge(cfg, self.context, logger=logger)
        logger.info(
            "[SilenceGuard] 已启动，内置连续监听窗口 %s 秒",
            int(cfg.get("listen_seconds", 300)),
        )

    @filter.event_message_type(EventMessageType.ALL, priority=100)
    async def silence_guard(self, event: AstrMessageEvent) -> None:
        cfg = self._config()
        if not cfg.get("enabled", True):
            return

        is_private = bool(event.is_private_chat())
        is_group = not is_private
        if is_private and not cfg.get("enable_private", True):
            return
        if is_group and not cfg.get("enable_group", True):
            return

        text = event.message_str or event.get_message_str()
        if self._is_clear_all_command(event, text):
            cleared = self._clear_all_sessions()
            yield event.plain_result(f"已结束所有正在监听的连续对话：{cleared} 个").stop_event()
            return

        session_key = event.unified_msg_origin
        state = self._state(session_key)
        now = time.time()
        listen_session_key = build_focus_session_key(event)
        in_listen_session = await self._prepare_listen_session(
            event,
            cfg,
            now,
            listen_session_key,
        )
        in_external_focus_session = has_focus_session(listen_session_key)
        is_directed = bool(
            event.is_at_or_wake_command
            or is_private
            or in_listen_session
            or in_external_focus_session
        )

        decision = analyze_rules(
            text,
            state,
            cfg,
            is_directed=is_directed,
            is_group=is_group,
            now=now,
        )
        if decision.action == UNCERTAIN and self._should_use_judge(cfg):
            judge_decision = await self._judge(
                text,
                state,
                cfg,
                is_group,
                is_directed,
                now,
                umo=event.unified_msg_origin,
            )
            decision = self._merge_uncertain(cfg, judge_decision)
        elif decision.action == UNCERTAIN:
            decision = self._fallback_decision(cfg, decision)

        apply_decision_to_state(decision, state, cfg, now=now)
        state.remember_user(text, now)
        event.set_extra("silence_guard_decision", decision.to_dict())

        if decision.action in {NO_REPLY, MUTE}:
            self._sync_focus_session(event)
            event.should_call_llm(True)
            event.stop_event()
            return

        if decision.action == WAKE:
            event.continue_event()
            return

    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.command(
        "结束所有对话",
        alias={"清空连续对话", "停止所有监听", "结束所有监听", "关闭所有连续对话"},
    )
    async def clear_all_focus_sessions_cmd(self, event: AstrMessageEvent):
        cleared = self._clear_all_sessions()
        yield event.plain_result(f"已结束所有正在监听的连续对话：{cleared} 个").stop_event()

    @filter.after_message_sent(priority=-100)
    async def record_bot_reply(self, event: AstrMessageEvent) -> None:
        cfg = self._config()
        if not cfg.get("enabled", True):
            return
        text = self._extract_result_text(event)
        if not text:
            return
        state = self._state(event.unified_msg_origin)
        state.remember_bot(text, time.time())

    async def terminate(self) -> None:
        self.states.clear()
        self.listen_sessions.clear()

    def _state(self, session_key: str) -> ConversationState:
        if session_key not in self.states:
            self.states[session_key] = ConversationState()
        return self.states[session_key]

    def _config(self) -> dict[str, Any]:
        cfg = dict(DEFAULT_CONFIG)
        try:
            cfg.update(dict(self.raw_config))
        except Exception:  # noqa: BLE001 - AstrBotConfig should be dict-like
            pass
        return cfg

    def _should_use_judge(self, cfg: dict[str, Any]) -> bool:
        if not cfg.get("smart_mode", True):
            return False
        if cfg.get("judge_mode", "ambiguous_only") != "ambiguous_only":
            return False
        return self.judge.available()

    async def _judge(
        self,
        text: str,
        state: ConversationState,
        cfg: dict[str, Any],
        is_group: bool,
        is_directed: bool,
        now: float,
        *,
        umo: str | None = None,
    ) -> Decision:
        if self.judge.config != cfg:
            self.judge = DeepSeekJudge(cfg, self.context, logger=logger)
        context_turns = max(1, int(cfg.get("judge_context_turns", 6)))
        recent_context = list(state.history)[-context_turns:]
        state_summary = {
            "last_bot_message": state.last_bot_text[-500:],
            "last_user_message": state.last_user_text[-500:],
            "is_group": is_group,
            "is_directed": is_directed,
            "active_context": is_active_context(
                state,
                int(cfg.get("active_chat_seconds", 300)),
                now,
            ),
            "muted": state.is_muted(now),
            "sleeping": state.is_sleeping(now),
        }
        return await self.judge.classify(
            current_user_message=text,
            recent_context=recent_context,
            state_summary=state_summary,
            umo=umo,
        )

    def _merge_uncertain(self, cfg: dict[str, Any], decision: Decision) -> Decision:
        min_confidence = float(cfg.get("judge_min_confidence", 0.72))
        if decision.action != UNCERTAIN and decision.confidence >= min_confidence:
            return decision
        return self._fallback_decision(cfg, decision)

    def _fallback_decision(self, cfg: dict[str, Any], decision: Decision) -> Decision:
        fallback = str(cfg.get("fallback_when_uncertain", "reply")).lower()
        if fallback == "no_reply":
            return Decision(
                NO_REPLY,
                f"uncertain_fallback_no_reply:{decision.reason}",
                confidence=decision.confidence,
                source=decision.source,
            )
        return Decision(
            REPLY,
            f"uncertain_fallback_reply:{decision.reason}",
            confidence=decision.confidence,
            source=decision.source,
        )

    def _extract_result_text(self, event: AstrMessageEvent) -> str:
        result = event.get_result()
        if not result or not getattr(result, "chain", None):
            return ""
        parts: list[str] = []
        for comp in result.chain:
            text = getattr(comp, "text", None)
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
        return "\n".join(parts).strip()

    def _sync_focus_session(self, event: AstrMessageEvent) -> None:
        session_key = build_focus_session_key(event)
        if not session_key:
            return
        self.listen_sessions.pop(session_key, None)
        mark_focus_session_expired(session_key)
        clear_focus_session(session_key)

    async def _prepare_listen_session(
        self,
        event: AstrMessageEvent,
        cfg: dict[str, Any],
        now: float,
        session_key: str,
    ) -> bool:
        if not cfg.get("listen_mode_enabled", True):
            return False
        if not session_key or self._should_ignore_listen_event(event, cfg):
            return False

        self._cleanup_expired_listen_sessions(now, keep_key=session_key)

        if self._is_explicit_listen_wake(event):
            self._open_listen_session(session_key, cfg, now)
            self._schedule_activation_notice(event, cfg)
            return True

        if cfg.get("listen_ignore_wake_prefix_commands", True) and self._looks_like_command(event):
            return False

        active = self.listen_sessions.get(session_key)
        if active is None:
            return False

        if active.expires_at <= now:
            self.listen_sessions.pop(session_key, None)
            await self._maybe_send_expire_notice(event, active, cfg, now)
            return False

        event.is_wake = True
        event.is_at_or_wake_command = True
        event.should_call_llm(False)
        if cfg.get("listen_refresh_on_each_message", True):
            active.expires_at = now + max(1, int(cfg.get("listen_seconds", 300)))

        if cfg.get("debug_log", False):
            logger.info(
                "[SilenceGuard] 连续监听放行 user=%s origin=%s expires_in=%ss",
                event.get_sender_id(),
                event.unified_msg_origin,
                int(active.expires_at - now),
            )

        return True

    def _open_listen_session(
        self,
        session_key: str,
        cfg: dict[str, Any],
        now: float,
    ) -> None:
        self.listen_sessions[session_key] = ListenSession(
            expires_at=now + max(1, int(cfg.get("listen_seconds", 300))),
        )
        if cfg.get("debug_log", False):
            logger.info("[SilenceGuard] 已开启连续监听 %s", session_key)

    def _is_explicit_listen_wake(self, event: AstrMessageEvent) -> bool:
        if bool(getattr(event, "is_at_or_wake_command", False)):
            return True
        return self._message_mentions_self(event) or self._message_uses_wake_prefix(event)

    def _message_mentions_self(self, event: AstrMessageEvent) -> bool:
        self_id = str(event.get_self_id() or "").strip()
        if not self_id:
            return False
        for component in event.get_messages():
            if not isinstance(component, At):
                continue
            qq = str(getattr(component, "qq", "") or "").strip()
            if qq == self_id:
                return True
        return False

    def _message_uses_wake_prefix(self, event: AstrMessageEvent) -> bool:
        text = str(event.get_message_str() or "").strip()
        return bool(text) and any(prefix and text.startswith(prefix) for prefix in self._wake_prefixes())

    def _looks_like_command(self, event: AstrMessageEvent) -> bool:
        text = str(event.get_message_str() or "").strip()
        return bool(text) and any(prefix and text.startswith(prefix) for prefix in self._wake_prefixes())

    def _should_ignore_listen_event(self, event: AstrMessageEvent, cfg: dict[str, Any]) -> bool:
        sender_id = str(event.get_sender_id() or "").strip()
        if cfg.get("listen_ignore_self_messages", True) and sender_id == str(event.get_self_id() or "").strip():
            return True

        sender_blacklist = self._parse_id_set(cfg.get("listen_sender_blacklist"))
        if sender_blacklist and sender_id in sender_blacklist:
            return True
        sender_whitelist = self._parse_id_set(cfg.get("listen_sender_whitelist"))
        if sender_whitelist and sender_id not in sender_whitelist:
            return True

        is_private = bool(event.is_private_chat())
        if is_private and not cfg.get("listen_enable_private", True):
            return True
        if not is_private and not cfg.get("listen_enable_group", True):
            return True

        group_id = str(event.get_group_id() or "").strip()
        if group_id:
            group_blacklist = self._parse_id_set(cfg.get("listen_group_blacklist"))
            if group_blacklist and group_id in group_blacklist:
                return True
            group_whitelist = self._parse_id_set(cfg.get("listen_group_whitelist"))
            if group_whitelist and group_id not in group_whitelist:
                return True

        return False

    def _schedule_activation_notice(self, event: AstrMessageEvent, cfg: dict[str, Any]) -> None:
        if not cfg.get("listen_send_activation_notice", False):
            return
        seconds = max(1, int(cfg.get("listen_seconds", 300)))
        minutes = max(1, round(seconds / 60))
        text = str(cfg.get("listen_activation_notice", "")).format(
            seconds=seconds,
            minutes=minutes,
        )
        asyncio.create_task(self._send_notice_later(event, text))

    async def _send_notice_later(self, event: AstrMessageEvent, text: str) -> None:
        if not text:
            return
        await asyncio.sleep(0.2)
        await event.send(event.plain_result(text))

    async def _maybe_send_expire_notice(
        self,
        event: AstrMessageEvent,
        session: ListenSession,
        cfg: dict[str, Any],
        now: float,
    ) -> None:
        if not cfg.get("listen_send_expire_notice", False):
            return
        if now - session.last_notice_at < 5:
            return
        session.last_notice_at = now
        text = str(cfg.get("listen_expire_notice", "")).strip()
        if text:
            await event.send(event.plain_result(text))

    def _cleanup_expired_listen_sessions(self, now: float, keep_key: str = "") -> None:
        expired_keys = [
            key
            for key, session in self.listen_sessions.items()
            if key != keep_key and session.expires_at <= now
        ]
        for key in expired_keys:
            self.listen_sessions.pop(key, None)

    def _parse_id_set(self, value: Any) -> set[str] | None:
        parsed = set(parse_command_list(value))
        return parsed or None

    def _clear_all_sessions(self) -> int:
        cleared = len(self.listen_sessions)
        self.listen_sessions.clear()
        cleared += clear_all_focus_sessions()
        self.states.clear()
        return cleared

    def _is_clear_all_command(self, event: AstrMessageEvent, text: str) -> bool:
        is_admin = getattr(event, "is_admin", None)
        if not callable(is_admin):
            return False
        try:
            if not is_admin():
                return False
        except Exception:
            return False
        normalized = strip_wake_prefix(text, self._wake_prefixes())
        if not normalized:
            return False

        return normalized in self._clear_all_commands()

    def _clear_all_commands(self) -> set[str]:
        cfg = self._config()
        commands = set(DEFAULT_CLEAR_ALL_COMMANDS)
        commands.update(parse_command_list(cfg.get("admin_clear_all_commands")))
        return commands

    def _wake_prefixes(self) -> tuple[str, ...]:
        try:
            global_config = self.context.get_config()
            raw = global_config.get("wake_prefix", ["/"])
        except Exception:
            raw = ["/"]

        if isinstance(raw, str):
            values = [raw]
        elif isinstance(raw, (list, tuple, set)):
            values = [str(item) for item in raw]
        else:
            values = ["/"]

        prefixes = tuple(prefix.strip() for prefix in values if str(prefix).strip())
        return prefixes or ("/",)
