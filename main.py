from __future__ import annotations

import time
from typing import Any

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.event.filter import EventMessageType
from astrbot.api.star import Context, Star, register

try:
    from .deepseek_judge import DeepSeekJudge
    from .plugin_compat import (
        build_focus_session_key,
        clear_all_focus_sessions,
        clear_focus_session,
        has_focus_session,
        mark_focus_session_expired,
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
CLEAR_ALL_COMMANDS = {
    "结束所有对话",
    "清空连续对话",
    "停止所有监听",
    "结束所有监听",
    "关闭所有连续对话",
}


DEFAULT_CONFIG: dict[str, Any] = {
    "enabled": True,
    "enable_private": True,
    "enable_group": True,
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
    "silence_keywords": DEFAULT_SILENCE_KEYWORDS,
    "farewell_keywords": DEFAULT_FAREWELL_KEYWORDS,
    "wake_keywords": DEFAULT_WAKE_KEYWORDS,
    "hard_closers": DEFAULT_HARD_CLOSERS,
    "ack_keywords": DEFAULT_ACKS,
}


@register(
    PLUGIN_NAME,
    "taolicx",
    "根据上下文判断 AstrBot 什么时候应该安静，不该回复时直接拦截事件。",
    "1.1.0",
)
class SilenceGuardPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig | None = None) -> None:
        super().__init__(context)
        self.raw_config = config or {}
        self.states: dict[str, ConversationState] = {}
        self.judge = DeepSeekJudge(self._config(), self.context, logger=logger)

    async def initialize(self) -> None:
        cfg = self._config()
        self.judge = DeepSeekJudge(cfg, self.context, logger=logger)

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
        focus_session_key = build_focus_session_key(event)
        in_focus_session = has_focus_session(focus_session_key)
        is_directed = bool(event.is_at_or_wake_command or is_private or in_focus_session)

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
        mark_focus_session_expired(session_key)
        clear_focus_session(session_key)

    def _clear_all_sessions(self) -> int:
        cleared = clear_all_focus_sessions()
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
        normalized = " ".join(str(text or "").strip().split())
        if not normalized:
            return False

        wake_prefixes = self._wake_prefixes()
        for prefix in wake_prefixes:
            if prefix and normalized.startswith(prefix):
                normalized = normalized[len(prefix) :].strip()
                break

        return normalized in CLEAR_ALL_COMMANDS

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
