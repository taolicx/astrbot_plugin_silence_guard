from __future__ import annotations

import re
import time
import unicodedata
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterable


REPLY = "REPLY"
NO_REPLY = "NO_REPLY"
MUTE = "MUTE"
WAKE = "WAKE"
UNCERTAIN = "UNCERTAIN"


DEFAULT_SILENCE_KEYWORDS = [
    "闭嘴",
    "别回",
    "不要回",
    "不用回",
    "无需回复",
    "不必回复",
    "别说了",
    "不要说话",
    "安静",
    "住口",
    "shut up",
    "be quiet",
    "stop talking",
]

DEFAULT_FAREWELL_KEYWORDS = [
    "晚安",
    "睡了",
    "我睡了",
    "拜拜",
    "拜",
    "再见",
    "886",
    "88",
    "下了",
    "回头聊",
]

DEFAULT_WAKE_KEYWORDS = [
    "继续",
    "出来",
    "可以说话了",
    "恢复回复",
    "解除静默",
    "别闭嘴了",
    "说话",
    "回答我",
    "在吗",
    "醒醒",
]

DEFAULT_HARD_CLOSERS = [
    "先这样",
    "就这样",
    "到此为止",
    "不聊了",
    "别聊了",
    "这个不聊了",
    "不用管了",
    "你不用管",
    "不用分析",
    "不用解释",
    "不用回复",
    "别回复",
    "不要回复",
    "别理我",
    "不用理我",
    "我想静静",
    "让我静静",
]

DEFAULT_ACKS = [
    "嗯",
    "嗯嗯",
    "恩",
    "恩恩",
    "好",
    "好的",
    "好吧",
    "行",
    "行吧",
    "可以",
    "ok",
    "okay",
    "收到",
    "明白",
    "了解",
    "知道了",
    "懂了",
    "是的",
    "对",
    "对的",
    "没事",
    "没事了",
    "不用了",
    "算了",
]


QUESTION_MARKERS = ["?", "？", "吗", "么", "呢", "什么", "怎么", "怎样", "怎么样", "为什么", "如何", "能不能", "可不可以"]
HYPOTHETICAL_MARKERS = ["如果", "假如", "比如", "例如", "所谓", "假设", "测试", "会怎样", "什么意思"]


@dataclass
class Decision:
    action: str
    reason: str
    confidence: float = 1.0
    mute_seconds: int = 0
    source: str = "rules"

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "reason": self.reason,
            "confidence": self.confidence,
            "mute_seconds": self.mute_seconds,
            "source": self.source,
        }


@dataclass
class ConversationState:
    mute_until: float = 0.0
    sleep_until: float = 0.0
    last_user_text: str = ""
    last_user_at: float = 0.0
    last_bot_text: str = ""
    last_bot_at: float = 0.0
    history: deque[dict[str, str]] = field(default_factory=lambda: deque(maxlen=24))

    def is_muted(self, now: float | None = None) -> bool:
        return self.mute_until > (now if now is not None else time.time())

    def is_sleeping(self, now: float | None = None) -> bool:
        return self.sleep_until > (now if now is not None else time.time())

    def remember_user(self, text: str, now: float | None = None) -> None:
        self.last_user_text = text
        self.last_user_at = now if now is not None else time.time()
        if text.strip():
            self.history.append({"role": "user", "content": text.strip()})

    def remember_bot(self, text: str, now: float | None = None) -> None:
        self.last_bot_text = text
        self.last_bot_at = now if now is not None else time.time()
        if text.strip():
            self.history.append({"role": "assistant", "content": text.strip()})


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text or "")
    normalized = normalized.replace("\u200b", "").replace("\ufeff", "")
    return re.sub(r"\s+", " ", normalized).strip()


def compact_text(text: str) -> str:
    text = normalize_text(text).lower()
    return re.sub(r"[\s,，.。!！?？~～、;；:：\"'“”‘’()（）\[\]【】<>《》]+", "", text)


def normalize_keyword_list(value: Any, defaults: Iterable[str]) -> list[str]:
    if value is None:
        return list(defaults)
    if isinstance(value, str):
        items = re.split(r"[\n,，]+", value)
    elif isinstance(value, Iterable):
        items = [str(item) for item in value]
    else:
        return list(defaults)
    cleaned = [normalize_text(item) for item in items if normalize_text(item)]
    return cleaned or list(defaults)


def is_question_like(text: str) -> bool:
    compact = compact_text(text)
    return any(marker in compact for marker in QUESTION_MARKERS)


def is_hypothetical_or_meta(text: str) -> bool:
    compact = compact_text(text)
    return any(marker in compact for marker in HYPOTHETICAL_MARKERS)


def contains_any(text: str, keywords: Iterable[str]) -> bool:
    compact = compact_text(text)
    for keyword in keywords:
        key = compact_text(keyword)
        if key and key in compact:
            return True
    return False


def exactish_any(text: str, keywords: Iterable[str], max_extra: int = 2) -> bool:
    compact = compact_text(text)
    if not compact:
        return False
    for keyword in keywords:
        key = compact_text(keyword)
        if not key:
            continue
        if compact == key:
            return True
        if compact.startswith(key) and len(compact) <= len(key) + max_extra:
            return True
        if compact.endswith(key) and len(compact) <= len(key) + max_extra:
            return True
    return False


CN_NUMBERS = {
    "零": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}


def parse_small_cn_number(text: str) -> int | None:
    text = compact_text(text)
    if not text:
        return None
    if text.isdigit():
        return int(text)
    if text == "十":
        return 10
    if "十" in text:
        left, _, right = text.partition("十")
        tens = CN_NUMBERS.get(left, 1 if left == "" else None)
        ones = CN_NUMBERS.get(right, 0 if right == "" else None)
        if tens is None or ones is None:
            return None
        return tens * 10 + ones
    return CN_NUMBERS.get(text)


def parse_duration_seconds(text: str, default_seconds: int) -> int:
    normalized = normalize_text(text)
    compact = compact_text(normalized)
    if "半小时" in compact or "半个小时" in compact:
        return 30 * 60

    pattern = re.compile(r"(\d+|[一二两三四五六七八九十]{1,3})\s*(秒|分钟|分|小时|时|天)")
    match = pattern.search(normalized)
    if not match:
        return max(0, int(default_seconds))
    value = parse_small_cn_number(match.group(1))
    if value is None:
        return max(0, int(default_seconds))
    unit = match.group(2)
    if unit == "秒":
        return value
    if unit in {"分钟", "分"}:
        return value * 60
    if unit in {"小时", "时"}:
        return value * 3600
    if unit == "天":
        return value * 86400
    return max(0, int(default_seconds))


def is_active_context(state: ConversationState, active_seconds: int, now: float) -> bool:
    return bool(state.last_bot_text and now - state.last_bot_at <= max(0, active_seconds))


def is_low_signal_reply(text: str, farewell_keywords: Iterable[str], ack_keywords: Iterable[str]) -> bool:
    compact = compact_text(text)
    if not compact:
        return True
    if exactish_any(text, farewell_keywords, max_extra=3):
        return True
    if exactish_any(text, ack_keywords, max_extra=2):
        return True
    return len(compact) <= 2 and not is_question_like(text)


def analyze_rules(
    text: str,
    state: ConversationState,
    config: dict[str, Any],
    *,
    is_directed: bool,
    is_group: bool,
    now: float | None = None,
) -> Decision:
    now = now if now is not None else time.time()
    text = normalize_text(text)
    silence_keywords = normalize_keyword_list(
        config.get("silence_keywords"),
        DEFAULT_SILENCE_KEYWORDS,
    )
    farewell_keywords = normalize_keyword_list(
        config.get("farewell_keywords"),
        DEFAULT_FAREWELL_KEYWORDS,
    )
    wake_keywords = normalize_keyword_list(config.get("wake_keywords"), DEFAULT_WAKE_KEYWORDS)
    hard_closers = normalize_keyword_list(config.get("hard_closers"), DEFAULT_HARD_CLOSERS)
    ack_keywords = normalize_keyword_list(config.get("ack_keywords"), DEFAULT_ACKS)
    default_mute_seconds = int(config.get("default_mute_seconds", 600))
    farewell_cooldown_seconds = int(config.get("farewell_cooldown_seconds", 1800))
    active_seconds = int(config.get("active_chat_seconds", 300))
    active = is_active_context(state, active_seconds, now)
    compact = compact_text(text)

    if not compact:
        return Decision(REPLY, "empty_message")

    if is_hypothetical_or_meta(text) and is_question_like(text):
        return Decision(REPLY, "hypothetical_question")

    if exactish_any(text, wake_keywords, max_extra=4):
        return Decision(WAKE, "wake_keyword")

    if state.is_muted(now):
        return Decision(NO_REPLY, "conversation_is_muted")

    if state.is_sleeping(now) and is_low_signal_reply(text, farewell_keywords, ack_keywords):
        return Decision(NO_REPLY, "farewell_cooldown")

    # In groups, only judge unaddressed messages when they are part of a recent bot thread.
    if is_group and not is_directed and not active:
        return Decision(REPLY, "group_message_not_directed")

    mentions_silence = contains_any(text, silence_keywords)
    if mentions_silence:
        duration = parse_duration_seconds(text, default_mute_seconds)
        short_command = len(compact) <= 16
        explicit_pattern = re.search(
            r"(闭嘴|住口|安静|别说了|不要说话|别回|不用回|不要回|别回复|不用回复|不要回复)",
            compact,
        )
        if explicit_pattern and not (is_hypothetical_or_meta(text) and is_question_like(text)):
            if short_command or re.search(r"(一会|一下|分钟|小时|秒|天|以后|现在|先)", compact):
                return Decision(MUTE, "silence_command", mute_seconds=duration)
            return Decision(NO_REPLY, "no_reply_command")

    if contains_any(text, hard_closers) and not is_question_like(text):
        return Decision(NO_REPLY, "hard_conversation_closer")

    last_bot_is_farewell = exactish_any(state.last_bot_text, farewell_keywords, max_extra=6)
    current_is_farewell = exactish_any(text, farewell_keywords, max_extra=4)
    if last_bot_is_farewell and current_is_farewell:
        if farewell_cooldown_seconds > 0:
            state.sleep_until = max(state.sleep_until, now + farewell_cooldown_seconds)
        return Decision(NO_REPLY, "mirrored_farewell")

    if active and current_is_farewell and not bool(config.get("reply_once_to_farewell", True)):
        if farewell_cooldown_seconds > 0:
            state.sleep_until = max(state.sleep_until, now + farewell_cooldown_seconds)
        return Decision(NO_REPLY, "farewell_end")

    if active and is_low_signal_reply(text, farewell_keywords, ack_keywords):
        return Decision(UNCERTAIN, "short_ack_or_farewell")

    ambiguous_markers = ["算了", "不用了", "没事了", "不管了", "随便", "先不", "回头再说"]
    if active and any(marker in compact for marker in ambiguous_markers) and not is_question_like(text):
        return Decision(UNCERTAIN, "ambiguous_closer")

    return Decision(REPLY, "no_silence_rule_matched")


def apply_decision_to_state(
    decision: Decision,
    state: ConversationState,
    config: dict[str, Any],
    *,
    now: float | None = None,
) -> None:
    now = now if now is not None else time.time()
    if decision.action == MUTE:
        seconds = decision.mute_seconds or int(config.get("default_mute_seconds", 600))
        if seconds > 0:
            state.mute_until = max(state.mute_until, now + seconds)
    elif decision.action == WAKE:
        state.mute_until = 0
        state.sleep_until = 0
    elif decision.action == NO_REPLY and decision.reason in {"mirrored_farewell", "farewell_end"}:
        seconds = int(config.get("farewell_cooldown_seconds", 1800))
        if seconds > 0:
            state.sleep_until = max(state.sleep_until, now + seconds)
