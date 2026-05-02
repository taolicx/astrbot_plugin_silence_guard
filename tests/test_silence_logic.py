import sys
import time
import unittest
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


BASE_CONFIG = {
    "default_mute_seconds": 600,
    "farewell_cooldown_seconds": 1800,
    "active_chat_seconds": 300,
    "reply_once_to_farewell": True,
}


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


if __name__ == "__main__":
    unittest.main()
