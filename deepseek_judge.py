from __future__ import annotations

import asyncio
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from .silence_logic import Decision, MUTE, NO_REPLY, REPLY, UNCERTAIN, WAKE


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
    def __init__(self, config: dict[str, Any], logger: Any | None = None) -> None:
        self.config = config
        self.logger = logger
        self._cache: dict[str, tuple[float, Decision]] = {}

    def available(self) -> bool:
        return bool(self.config.get("judge_api_key")) and bool(self.config.get("judge_base_url"))

    async def classify(
        self,
        *,
        current_user_message: str,
        recent_context: list[dict[str, str]],
        state_summary: dict[str, Any],
    ) -> Decision:
        if not self.available():
            return Decision(UNCERTAIN, "judge_not_configured", source="judge")

        cache_key = self._cache_key(current_user_message, recent_context, state_summary)
        cached = self._get_cache(cache_key)
        if cached:
            return cached

        try:
            decision = await asyncio.to_thread(
                self._request,
                current_user_message,
                recent_context,
                state_summary,
            )
        except Exception as exc:  # noqa: BLE001 - plugin should fail open
            if self.logger:
                self.logger.warning("SilenceGuard judge failed: %s", exc)
            decision = Decision(UNCERTAIN, "judge_request_failed", source="judge")

        self._set_cache(cache_key, decision)
        return decision

    def _request(
        self,
        current_user_message: str,
        recent_context: list[dict[str, str]],
        state_summary: dict[str, Any],
    ) -> Decision:
        base_url = str(self.config.get("judge_base_url", "https://api.deepseek.com")).rstrip("/")
        endpoint = f"{base_url}/chat/completions"
        body = {
            "model": self.config.get("judge_model", "deepseek-v4-flash"),
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "recent_context": recent_context,
                            "current_user_message": current_user_message,
                            "state": state_summary,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            "response_format": {"type": "json_object"},
            "thinking": {"type": "disabled"},
            "temperature": 0,
            "max_tokens": int(self.config.get("judge_max_tokens", 96)),
        }
        headers = {
            "Authorization": f"Bearer {self.config.get('judge_api_key')}",
            "Content-Type": "application/json",
        }
        req = urllib.request.Request(
            endpoint,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        timeout = float(self.config.get("judge_timeout_seconds", 3))
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"DeepSeek HTTP {exc.code}: {error_body}") from exc

        content = (
            payload.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        if not content:
            return Decision(UNCERTAIN, "judge_empty_content", source="judge")

        parsed = json.loads(content)
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

    def _cache_key(
        self,
        current_user_message: str,
        recent_context: list[dict[str, str]],
        state_summary: dict[str, Any],
    ) -> str:
        last_bot = str(state_summary.get("last_bot_message", ""))[-80:]
        return json.dumps(
            {
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
