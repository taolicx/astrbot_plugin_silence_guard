import sys
import time
import unittest
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from silence_logic import (
    MUTE,
    NO_REPLY,
    REPLY,
    UNCERTAIN,
    WAKE,
    ConversationState,
    analyze_rules,
    apply_decision_to_state,
)

from deepseek_judge import DeepSeekJudge
from plugin_compat import (
    build_focus_session_key,
    clear_focus_session,
    has_focus_session,
    mark_focus_session_expired,
)


BASE_CONFIG = {
    "default_mute_seconds": 600,
    "farewell_cooldown_seconds": 1800,
    "active_chat_seconds": 300,
    "reply_once_to_farewell": True,
}


class DummyProvider:
    def __init__(self, model: str = "dummy-model", text: str = '{"action":"NO_REPLY","confidence":0.95,"mute_seconds":0,"reason":"done"}') -> None:
        self._model = model
        self._text = text

    def get_model(self) -> str:
        return self._model

    def meta(self):
        return type("Meta", (), {"id": "dummy-provider"})()

    async def text_chat(self, **kwargs):
        return type("Resp", (), {"completion_text": self._text})()


class DummyContext:
    def __init__(self, provider=None) -> None:
        self.provider = provider or DummyProvider()

    def get_provider_by_id(self, provider_id: str):
        if provider_id == "dummy-provider":
            return self.provider
        return None

    def get_using_provider(self, umo=None):
        return self.provider

    def get_all_providers(self):
        return [self.provider]


class DummyEvent:
    def __init__(self, origin: str = "default(aiocqhttp)", sender_id: str = "1556592332") -> None:
        self.unified_msg_origin = origin
        self._sender_id = sender_id

    def get_sender_id(self):
        return self._sender_id


class SilenceLogicTest(unittest.TestCase):
    def test_silence_command_mutes(self):
        state = ConversationState()
        decision = analyze_rules(
            "闭嘴 10 分钟",
            state,
            BASE_CONFIG,
            is_directed=True,
            is_group=False,
            now=time.time(),
        )
        self.assertEqual(decision.action, MUTE)
        self.assertEqual(decision.mute_seconds, 600)

    def test_plain_silence_command_only_stops_current_reply(self):
        state = ConversationState()
        decision = analyze_rules(
            "闭嘴",
            state,
            BASE_CONFIG,
            is_directed=True,
            is_group=False,
            now=time.time(),
        )
        self.assertEqual(decision.action, NO_REPLY)
        self.assertEqual(decision.mute_seconds, 0)

    def test_hush_only_stops_focus_session(self):
        state = ConversationState()
        decision = analyze_rules(
            "嘘",
            state,
            BASE_CONFIG,
            is_directed=True,
            is_group=False,
            now=time.time(),
        )
        self.assertEqual(decision.action, NO_REPLY)
        self.assertEqual(decision.reason, "hush_command")

    def test_hypothetical_silence_is_not_muted(self):
        state = ConversationState()
        decision = analyze_rules(
            "如果我说闭嘴你会怎样？",
            state,
            BASE_CONFIG,
            is_directed=True,
            is_group=False,
            now=time.time(),
        )
        self.assertEqual(decision.action, REPLY)

    def test_mirrored_farewell_no_reply(self):
        state = ConversationState()
        state.remember_bot("晚安，好梦", time.time())
        decision = analyze_rules(
            "晚安",
            state,
            BASE_CONFIG,
            is_directed=True,
            is_group=False,
            now=time.time(),
        )
        self.assertEqual(decision.action, NO_REPLY)

    def test_wake_keyword(self):
        state = ConversationState(mute_until=time.time() + 100)
        decision = analyze_rules(
            "可以说话了",
            state,
            BASE_CONFIG,
            is_directed=True,
            is_group=False,
            now=time.time(),
        )
        self.assertEqual(decision.action, WAKE)
        apply_decision_to_state(decision, state, BASE_CONFIG)
        self.assertFalse(state.is_muted())

    def test_more_builtin_silence_words_mute(self):
        state = ConversationState()
        for text in ("别回我", "不回我", "不回复", "别打扰", "暂停回复", "先别说话", "shut up"):
            decision = analyze_rules(
                text,
                state,
                BASE_CONFIG,
                is_directed=True,
                is_group=False,
                now=time.time(),
            )
            self.assertEqual(decision.action, NO_REPLY, text)

    def test_duration_hint_silence_words_mute(self):
        state = ConversationState()
        for text in ("别回我 10 分钟", "暂停回复一会", "先别说话半小时"):
            decision = analyze_rules(
                text,
                state,
                BASE_CONFIG,
                is_directed=True,
                is_group=False,
                now=time.time(),
            )
            self.assertEqual(decision.action, MUTE, text)

    def test_user_keywords_extend_builtin_keywords(self):
        state = ConversationState()
        config = {**BASE_CONFIG, "silence_keywords": ["收声"]}
        custom = analyze_rules(
            "收声",
            state,
            config,
            is_directed=True,
            is_group=False,
            now=time.time(),
        )
        builtin = analyze_rules(
            "闭嘴",
            state,
            config,
            is_directed=True,
            is_group=False,
            now=time.time(),
        )
        self.assertEqual(custom.action, NO_REPLY)
        self.assertEqual(builtin.action, NO_REPLY)

    def test_more_builtin_wake_words(self):
        state = ConversationState(mute_until=time.time() + 100)
        decision = analyze_rules(
            "继续说",
            state,
            BASE_CONFIG,
            is_directed=True,
            is_group=False,
            now=time.time(),
        )
        self.assertEqual(decision.action, WAKE)

    def test_more_hard_closers_no_reply(self):
        state = ConversationState()
        decision = analyze_rules(
            "打住",
            state,
            BASE_CONFIG,
            is_directed=True,
            is_group=False,
            now=time.time(),
        )
        self.assertEqual(decision.action, NO_REPLY)

    def test_focus_session_key_builder(self):
        key = build_focus_session_key(DummyEvent())
        self.assertEqual(key, "default(aiocqhttp)::1556592332")

    def test_focus_session_clear_helper(self):
        module = types.SimpleNamespace(_FOCUS_SESSIONS={"default(aiocqhttp)::1556592332": object()})
        sys.modules["astrbot_plugin_focus_session"] = module
        try:
            self.assertTrue(has_focus_session("default(aiocqhttp)::1556592332", ("astrbot_plugin_focus_session",)))
            cleared = clear_focus_session("default(aiocqhttp)::1556592332", ("astrbot_plugin_focus_session",))
            self.assertTrue(cleared)
            self.assertNotIn("default(aiocqhttp)::1556592332", module._FOCUS_SESSIONS)
        finally:
            sys.modules.pop("astrbot_plugin_focus_session", None)

    def test_focus_session_clear_helper_finds_astrbot_loaded_module(self):
        key = "default(aiocqhttp)::1556592332"
        module = types.SimpleNamespace(_FOCUS_SESSIONS={key: object()})
        sys.modules["data.plugins.astrbot_plugin_focus_session.main"] = module
        try:
            self.assertTrue(has_focus_session(key, ()))
            self.assertTrue(clear_focus_session(key, ()))
            self.assertNotIn(key, module._FOCUS_SESSIONS)
        finally:
            sys.modules.pop("data.plugins.astrbot_plugin_focus_session.main", None)

    def test_focus_session_mark_expired_helper(self):
        session = type("Session", (), {"expires_at": 123, "last_notice_at": 456})()
        module = types.SimpleNamespace(_FOCUS_SESSIONS={"default(aiocqhttp)::1556592332": session})
        sys.modules["astrbot_plugin_focus_session"] = module
        try:
            marked = mark_focus_session_expired(
                "default(aiocqhttp)::1556592332",
                ("astrbot_plugin_focus_session",),
            )
            self.assertTrue(marked)
            self.assertEqual(session.expires_at, 0)
            self.assertEqual(session.last_notice_at, 0)
        finally:
            sys.modules.pop("astrbot_plugin_focus_session", None)

    def test_short_ack_is_uncertain_when_active(self):
        state = ConversationState()
        state.remember_bot("这个问题可以这样处理。", time.time())
        decision = analyze_rules(
            "嗯嗯",
            state,
            BASE_CONFIG,
            is_directed=True,
            is_group=False,
            now=time.time(),
        )
        self.assertEqual(decision.action, UNCERTAIN)

    def test_provider_judge_uses_default_provider_when_not_selected(self):
        judge = DeepSeekJudge(
            {
                "judge_provider_id": "",
                "judge_max_tokens": 32,
                "judge_cache_seconds": 0,
            },
            DummyContext(),
        )
        decision = self._run_async(
            judge.classify(
                current_user_message="嗯嗯",
                recent_context=[{"role": "assistant", "content": "这个问题可以这样处理。"}],
                state_summary={"last_bot_message": "这个问题可以这样处理。"},
            )
        )
        self.assertEqual(decision.action, NO_REPLY)
        self.assertEqual(decision.source, "judge")

    def test_provider_judge_uses_selected_provider(self):
        selected = DummyProvider(model="selected-model")
        context = DummyContext(provider=selected)
        judge = DeepSeekJudge(
            {
                "judge_provider_id": "dummy-provider",
                "judge_max_tokens": 32,
                "judge_cache_seconds": 0,
            },
            context,
        )
        decision = self._run_async(
            judge.classify(
                current_user_message="晚安",
                recent_context=[{"role": "assistant", "content": "晚安，好梦。"}],
                state_summary={"last_bot_message": "晚安，好梦。"},
            )
        )
        self.assertEqual(decision.action, NO_REPLY)

    def _run_async(self, coro):
        import asyncio

        return asyncio.run(coro)


if __name__ == "__main__":
    unittest.main()
