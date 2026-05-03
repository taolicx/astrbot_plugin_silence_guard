from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any

try:
    from astrbot.api.star import Context
    from astrbot.core.provider import Provider
except ImportError:  # pragma: no cover - allow local unit tests without AstrBot installed
    from typing import Any as Context
    from typing import Any as Provider

try:
    from .silence_logic import Decision, MUTE, NO_REPLY, REPLY, UNCERTAIN, WAKE
except ImportError:
    from silence_logic import Decision, MUTE, NO_REPLY, REPLY, UNCERTAIN, WAKE


SYSTEM_PROMPT = """你是 AstrBot 插件里的“不回复判断器”。你的任务不是聊天，而是判断当前用户消息是否应该让机器人继续回复。

只输出 json 对象，不要输出 markdown，不要解释。

可选 action:
- REPLY: 用户正在提问、继续请求、表达需要机器人处理，应该交给机器人回复。
- NO_REPLY: 用户在礼貌收尾、附和确认、明确表示不用回复、要求当前这条不要回复。
- MUTE: 用户要求机器人闭嘴、安静、停止说话一段时间。
- WAKE: 用户要求恢复回复、继续说话、解除静默。
- UNCERTAIN: 无法可靠判断。

判断原则:
- 明确问题、任务、求助，优先 REPLY。
- “闭嘴、别回、不要回复、不用管、我想静静”一类真实指令，判 NO_REPLY 或 MUTE。
- 如果用户只是讨论“闭嘴”这个词、假设“如果我说闭嘴会怎样”，不要当成静默指令。
- 最近机器人刚说“晚安/拜拜/再见”，用户也回“晚安/拜拜/再见”，判 NO_REPLY。
- “嗯嗯、好吧、行、收到、算了、不用了”要结合上下文；如果只是收尾确认，判 NO_REPLY。

json 输出例子:
{"action":"NO_REPLY","confidence":0.91,"mute_seconds":0,"reason":"用户在结束对话"}
"""


@dataclass
class JudgeResult:
    decision: Decision
    raw: dict[str, Any] | None = None


class DeepSeekJudge:
    def __init__(self, config: dict[str, Any], context: Context, logger: Any | None = None) -> None:
        self.config = config
        self.context = context
        self.logger = logger
        self._cache: dict[str, tuple[float, Decision]] = {}

    def available(self) -> bool:
        provider_id = self._provider_id()
        if provider_id is not None:
            try:
                provider = self.context.get_provider_by_id(provider_id)
            except Exception:
                provider = None
            return self._is_provider(provider)
        return self._get_default_provider() is not None

    async def classify(
        self,
        *,
        current_user_message: str,
        recent_context: list[dict[str, str]],
        state_summary: dict[str, Any],
        umo: str | None = None,
    ) -> Decision:
        provider = self._resolve_provider(umo)
        if provider is None:
            return Decision(UNCERTAIN, "judge_provider_not_found", source="judge")

        cache_key = self._cache_key(current_user_message, recent_context, state_summary, provider)
        cached = self._get_cache(cache_key)
        if cached:
            return cached

        try:
            decision = await asyncio.wait_for(
                self._request(
                    provider,
                    current_user_message=current_user_message,
                    recent_context=recent_context,
                    state_summary=state_summary,
                ),
                timeout=float(self.config.get("judge_timeout_seconds", 3)),
            )
        except asyncio.TimeoutError:
            self._log_debug("SilenceGuard judge timeout; falling back to rules.")
            decision = Decision(UNCERTAIN, "judge_timeout", source="judge")
        except Exception as exc:  # noqa: BLE001 - plugin should fail open
            self._log_warning(
                "SilenceGuard judge failed: %s (%s)",
                exc.__class__.__name__,
                exc,
            )
            decision = Decision(UNCERTAIN, "judge_request_failed", source="judge")

        self._set_cache(cache_key, decision)
        return decision

    async def _request(
        self,
        provider: Provider,
        *,
        current_user_message: str,
        recent_context: list[dict[str, str]],
        state_summary: dict[str, Any],
    ) -> Decision:
        payload = json.dumps(
            {
                "recent_context": recent_context,
                "current_user_message": current_user_message,
                "state": state_summary,
            },
            ensure_ascii=False,
        )
        response = await provider.text_chat(
            prompt=payload,
            system_prompt=SYSTEM_PROMPT,
            model=provider.get_model() or None,
            temperature=0,
            max_tokens=int(self.config.get("judge_max_tokens", 96)),
        )

        content = getattr(response, "completion_text", "") or ""
        if not content.strip():
            return Decision(UNCERTAIN, "judge_empty_content", source="judge")

        parsed = self._extract_json(content)
        if parsed is None:
            return Decision(UNCERTAIN, "judge_invalid_json", source="judge")

        action = str(parsed.get("action", UNCERTAIN)).upper()
        if action not in {REPLY, NO_REPLY, MUTE, WAKE, UNCERTAIN}:
            action = UNCERTAIN
        confidence = float(parsed.get("confidence", 0.0) or 0.0)
        mute_seconds = int(parsed.get("mute_seconds", 0) or 0)
        reason = str(parsed.get("reason", "judge_result"))[:120]
        return Decision(
            action=action,
            reason=reason,
            confidence=max(0.0, min(1.0, confidence)),
            mute_seconds=max(0, mute_seconds),
            source="judge",
        )

    def _resolve_provider(self, umo: str | None) -> Provider | None:
        provider_id = self._provider_id()
        if provider_id:
            provider = self.context.get_provider_by_id(provider_id)
            if self._is_provider(provider):
                return provider
            if self.logger:
                self.logger.warning("SilenceGuard judge provider `%s` not found.", provider_id)
            return None

        provider = self._get_default_provider(umo)
        if self._is_provider(provider):
            return provider
        return None

    def _get_default_provider(self, umo: str | None = None) -> Provider | None:
        try:
            provider = self.context.get_using_provider(umo=umo)
        except Exception:
            provider = None
        if self._is_provider(provider):
            return provider

        try:
            providers = self.context.get_all_providers()
        except Exception:
            providers = []
        if providers:
            first_provider = providers[0]
            if self._is_provider(first_provider):
                return first_provider
        return None

    def _is_provider(self, value: object) -> bool:
        provider_type = Provider if isinstance(Provider, type) else None
        if provider_type is not None:
            return isinstance(value, provider_type)
        return all(
            hasattr(value, attr)
            for attr in ("text_chat", "get_model", "meta")
        )

    def _provider_id(self) -> str | None:
        provider_id = self.config.get("judge_provider_id")
        if isinstance(provider_id, str) and provider_id.strip():
            return provider_id.strip()
        return None

    def _log_debug(self, message: str, *args: Any) -> None:
        if not self.logger or not self.config.get("debug_log", False):
            return
        debug = getattr(self.logger, "debug", None)
        if callable(debug):
            debug(message, *args)

    def _log_warning(self, message: str, *args: Any) -> None:
        if not self.logger or not self.config.get("debug_log", False):
            return
        warning = getattr(self.logger, "warning", None)
        if callable(warning):
            warning(message, *args)

    def _extract_json(self, content: str) -> dict[str, Any] | None:
        text = content.strip()
        if not text:
            return None
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:].strip()
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            pass

        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(text[start : end + 1])
                return parsed if isinstance(parsed, dict) else None
            except Exception:
                return None
        return None

    def _cache_key(
        self,
        current_user_message: str,
        recent_context: list[dict[str, str]],
        state_summary: dict[str, Any],
        provider: Provider,
    ) -> str:
        last_bot = str(state_summary.get("last_bot_message", ""))[-80:]
        provider_id = ""
        try:
            provider_id = provider.meta().id
        except Exception:
            provider_id = provider.provider_config.get("id", "") if hasattr(provider, "provider_config") else ""
        return json.dumps(
            {
                "p": provider_id,
                "m": current_user_message.strip().lower(),
                "b": last_bot.strip().lower(),
                "c": recent_context[-2:],
            },
            ensure_ascii=False,
            sort_keys=True,
        )

    def _get_cache(self, key: str) -> Decision | None:
        ttl = int(self.config.get("judge_cache_seconds", 120))
        if ttl <= 0:
            return None
        item = self._cache.get(key)
        if not item:
            return None
        expires_at, decision = item
        if expires_at <= time.time():
            self._cache.pop(key, None)
            return None
        return decision

    def _set_cache(self, key: str, decision: Decision) -> None:
        ttl = int(self.config.get("judge_cache_seconds", 120))
        if ttl <= 0:
            return
        self._cache[key] = (time.time() + ttl, decision)


ProviderJudge = DeepSeekJudge
