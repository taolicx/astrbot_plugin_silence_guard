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
from main import ListenSession, SilenceGuardPlugin
from plugin_compat import (
    build_focus_session_key,
    clear_all_focus_sessions,
    clear_focus_session,
    has_focus_session,
    mark_focus_session_expired,
    parse_command_list,
    strip_wake_prefix,
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


class TimeoutProvider(DummyProvider):
    async def text_chat(self, **kwargs):
        import asyncio

        await asyncio.sleep(0.05)
        return type("Resp", (), {"completion_text": "{}"})()


class DummyLogger:
    def __init__(self) -> None:
        self.warnings: list[tuple] = []
        self.debugs: list[tuple] = []

    def warning(self, *args):
        self.warnings.append(args)

    def debug(self, *args):
        self.debugs.append(args)


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

    def get_config(self):
        return {"wake_prefix": ["/", "!"]}


class DummyEvent:
    def __init__(
        self,
        origin: str = "default(aiocqhttp)",
        sender_id: str = "1556592332",
        text: str = "",
        private: bool = False,
        group_id: str = "10001",
        self_id: str = "123456",
        admin: bool = False,
    ) -> None:
        self.unified_msg_origin = origin
        self._sender_id = sender_id
        self.message_str = text
        self._private = private
        self._group_id = group_id
        self._self_id = self_id
        self.role = "admin" if admin else "member"
        self.is_wake = False
        self.is_at_or_wake_command = False
        self.call_llm = False
        self.sent = []

    def get_sender_id(self):
        return self._sender_id

    def get_self_id(self):
        return self._self_id

    def get_group_id(self):
        return "" if self._private else self._group_id

    def is_private_chat(self):
        return self._private

    def get_message_str(self):
        return self.message_str

    def get_messages(self):
        return []

    def is_admin(self):
        return self.role == "admin"

    def should_call_llm(self, call_llm: bool) -> None:
        self.call_llm = call_llm

    async def send(self, result):
        self.sent.append(result)

    def plain_result(self, text: str):
        return type("DummyResult", (), {"text": text, "stop_event": lambda self: self})()


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

    def test_focus_session_clear_all_helper(self):
        module = types.SimpleNamespace(
            _FOCUS_SESSIONS={
                "default(aiocqhttp)::1": object(),
                "default(aiocqhttp)::2": object(),
            }
        )
        sys.modules["data.plugins.astrbot_plugin_focus_session.main"] = module
        try:
            cleared = clear_all_focus_sessions(())
            self.assertEqual(cleared, 2)
            self.assertEqual(module._FOCUS_SESSIONS, {})
        finally:
            sys.modules.pop("data.plugins.astrbot_plugin_focus_session.main", None)

    def test_builtin_listen_session_marks_followup_as_wake(self):
        plugin = SilenceGuardPlugin(DummyContext(), {})
        key = build_focus_session_key(DummyEvent())
        plugin.listen_sessions[key] = ListenSession(expires_at=time.time() + 60)
        event = DummyEvent(text="继续讲")
        cfg = plugin._config()
        active = self._run_async(
            plugin._prepare_listen_session(
                event,
                cfg,
                time.time(),
                build_focus_session_key(event),
            )
        )
        self.assertTrue(active)
        self.assertTrue(event.is_wake)
        self.assertTrue(event.is_at_or_wake_command)
        self.assertFalse(event.call_llm)

    def test_builtin_listen_session_clear_current(self):
        plugin = SilenceGuardPlugin(DummyContext(), {})
        event = DummyEvent()
        key = build_focus_session_key(event)
        plugin.listen_sessions[key] = ListenSession(expires_at=time.time() + 60)
        plugin._sync_focus_session(event)
        self.assertNotIn(key, plugin.listen_sessions)

    def test_builtin_listen_session_clear_all(self):
        plugin = SilenceGuardPlugin(DummyContext(), {})
        plugin.listen_sessions["a"] = ListenSession(expires_at=time.time() + 60)
        plugin.listen_sessions["b"] = ListenSession(expires_at=time.time() + 60)
        cleared = plugin._clear_all_sessions()
        self.assertEqual(cleared, 2)
        self.assertEqual(plugin.listen_sessions, {})

    def test_custom_admin_command_list_parser(self):
        commands = parse_command_list("清场，全部停止\n停止监听, 清场")
        self.assertEqual(commands, ("清场", "全部停止", "停止监听"))
        object_commands = parse_command_list([{"value": " 全部闭嘴 "}, {"text": "收工"}])
        self.assertEqual(object_commands, ("全部闭嘴", "收工"))

    def test_strip_wake_prefix_for_custom_admin_command(self):
        self.assertEqual(strip_wake_prefix("/清场", ("/", "!",)), "清场")
        self.assertEqual(strip_wake_prefix("! 全部停止", ("/", "!")), "全部停止")
        self.assertEqual(strip_wake_prefix("停止监听", ("/",)), "停止监听")

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

    def test_provider_judge_timeout_is_quiet_fallback(self):
        logger = DummyLogger()
        judge = DeepSeekJudge(
            {
                "judge_provider_id": "",
                "judge_timeout_seconds": 0.001,
                "judge_cache_seconds": 0,
                "debug_log": False,
            },
            DummyContext(provider=TimeoutProvider()),
            logger=logger,
        )
        decision = self._run_async(
            judge.classify(
                current_user_message="嗯",
                recent_context=[],
                state_summary={},
            )
        )
        self.assertEqual(decision.action, UNCERTAIN)
        self.assertEqual(decision.reason, "judge_timeout")
        self.assertEqual(logger.warnings, [])
        self.assertEqual(logger.debugs, [])

    def _run_async(self, coro):
        import asyncio

        return asyncio.run(coro)


if __name__ == "__main__":
    unittest.main()
