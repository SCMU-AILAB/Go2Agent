from __future__ import annotations

import unittest

from agent.decision import DecisionAgentError
from agent.social_vision import SocialVisionAgent
from robot import RobotState
from skills import build_g1_autonomy_skills, build_go2_autonomy_skills
from skills.motions import HandshakeSkill, WaveSkill
from tests.test_vision_policy import FakeVisionInvoker, camera_frame


class SocialVisionTests(unittest.IsolatedAsyncioTestCase):
    async def test_speaking_none_requires_evidence_and_remains_silent(self):
        payload = {
            "gesture": "none",
            "hand_visible": False,
            "directed_at_robot": False,
            "present_in_latest": True,
            "speech": None,
        }
        invoker = FakeVisionInvoker([payload])
        agent = SocialVisionAgent(
            invoker=invoker, prompt_profile="egocentric", generate_speech=True
        )
        with self.assertRaises(DecisionAgentError):
            await agent.decide(
                [camera_frame(1), camera_frame(2)],
                RobotState(hardware=False, connected=True),
                [HandshakeSkill()],
            )
        payload["evidence"] = "none"
        result = await agent.decide(
            [camera_frame(1), camera_frame(2)],
            RobotState(hardware=False, connected=True),
            [HandshakeSkill()],
        )
        self.assertEqual(result.action, "ignore")
        self.assertIsNone(result.speech)
        self.assertIn('"evidence":"none","speech":null', invoker.calls[-1][1])
        self.assertIn("arm hanging down beside the thigh", invoker.calls[-1][1])

    async def test_speaking_action_without_evidence_is_rejected(self):
        agent = SocialVisionAgent(
            generate_speech=True,
            invoker=FakeVisionInvoker(
                [
                    {
                        "gesture": "handshake",
                        "hand_visible": True,
                        "directed_at_robot": True,
                        "present_in_latest": True,
                        "speech": "你好！",
                    }
                ]
            ),
        )
        with self.assertRaises(DecisionAgentError):
            await agent.decide(
                [camera_frame(1), camera_frame(2)],
                RobotState(hardware=False, connected=True),
                [HandshakeSkill()],
            )

    async def test_generated_speech_is_preserved_only_for_confirmed_action(self):
        for directed in (True, False):
            invoker = FakeVisionInvoker(
                [
                    {
                        "gesture": "handshake",
                        "evidence": "offered_hand",
                        "hand_visible": True,
                        "present_in_latest": True,
                        "directed_at_robot": directed,
                        "speech": "你好，见到你很开心！",
                    }
                ]
            )
            agent = SocialVisionAgent(
                invoker=invoker, prompt_profile="egocentric", generate_speech=True
            )
            decision = await agent.decide(
                [camera_frame(1), camera_frame(2)],
                RobotState(hardware=False, connected=True),
                [HandshakeSkill()],
            )
            self.assertEqual(
                decision.action, "execute_and_speak" if directed else "ignore"
            )
            self.assertEqual(
                decision.speech, "你好，见到你很开心！" if directed else None
            )
            self.assertIn('"speech"', invoker.calls[0][1])

    async def test_active_gesture_does_not_speak_again(self):
        agent = SocialVisionAgent(
            generate_speech=True,
            invoker=FakeVisionInvoker(
                [
                    {
                        "gesture": "wave",
                        "evidence": "side_to_side",
                        "hand_visible": True,
                        "directed_at_robot": True,
                        "present_in_latest": True,
                        "speech": "你好呀！",
                    }
                ]
            ),
        )
        decision = await agent.decide(
            [camera_frame(1), camera_frame(2)],
            RobotState(hardware=False, connected=True),
            [WaveSkill()],
            policy_context={"active_skill": "wave"},
        )
        self.assertEqual(decision.action, "continue")
        self.assertIsNone(decision.speech)

    async def test_wave_recipient_check_remains_required(self):
        for directed, action in ((False, "ignore"), (True, "execute_skill")):
            invoker = FakeVisionInvoker(
                [
                    {
                        "gesture": "wave",
                        "evidence": "side_to_side",
                        "hand_visible": True,
                        "present_in_latest": True,
                        "directed_at_robot": directed,
                    }
                ]
            )
            agent = SocialVisionAgent(invoker=invoker, prompt_profile="egocentric")
            decision = await agent.decide(
                [camera_frame(1), camera_frame(2)],
                RobotState(hardware=False, connected=True),
                [WaveSkill()],
            )
            self.assertEqual(decision.action, action)
            self.assertIn("NOT whether", invoker.calls[0][1])
            if not directed:
                self.assertIn("recipient_unconfirmed", decision.reason)

    async def test_egocentric_prompt_keeps_all_execution_checks(self):
        for field in (None, "hand_visible", "directed_at_robot", "present_in_latest"):
            payload = {
                "gesture": "handshake",
                "hand_visible": True,
                "directed_at_robot": True,
                "present_in_latest": True,
                "evidence": "offered_hand",
            }
            if field is not None:
                payload[field] = False
            invoker = FakeVisionInvoker([payload])
            agent = SocialVisionAgent(invoker=invoker, prompt_profile="egocentric")
            decision = await agent.decide(
                [camera_frame(1), camera_frame(2)],
                RobotState(hardware=False, connected=True),
                [HandshakeSkill()],
            )
            self.assertEqual(
                decision.action, "execute_skill" if field is None else "ignore"
            )
            self.assertIn("The camera is the recipient", invoker.calls[0][1])
            self.assertEqual(agent.last_metrics["prompt_profile"], "egocentric")

    def test_unknown_profile_is_rejected(self):
        with self.assertRaises(ValueError):
            SocialVisionAgent(invoker=FakeVisionInvoker([]), prompt_profile="unknown")

    async def decide(self, **changes):
        payload = {
            "gesture": "handshake",
            "hand_visible": True,
            "directed_at_robot": True,
            "present_in_latest": True,
            "evidence": "offered_hand",
        }
        payload.update(changes)
        invoker = FakeVisionInvoker([payload])
        decision = await SocialVisionAgent(invoker=invoker).decide(
            [camera_frame(1), camera_frame(1.4), camera_frame(1.8)],
            RobotState(hardware=False, connected=True),
            [HandshakeSkill(), WaveSkill()],
        )
        return decision, invoker

    async def test_handshake_with_visible_evidence(self):
        decision, invoker = await self.decide()
        self.assertEqual(decision.skill, "handshake")
        self.assertEqual(decision.arguments, {})
        self.assertIn("[-0.8, -0.4, 0.0]", invoker.calls[0][1])
        self.assertNotIn("argument_defaults", invoker.calls[0][1])

    async def test_wave_requires_motion_evidence(self):
        decision, _ = await self.decide(gesture="wave", evidence="side_to_side")
        self.assertEqual(decision.skill, "wave")

    async def test_ambiguous_or_absent_evidence_does_not_execute(self):
        for changes in (
            {"hand_visible": False},
            {"directed_at_robot": False},
            {"present_in_latest": False},
            {"evidence": "raised_palm"},
            {"gesture": "uncertain"},
            {"gesture": "none"},
            {"gesture": "high_five", "evidence": "raised_palm"},
        ):
            with self.subTest(changes=changes):
                decision, _ = await self.decide(**changes)
                self.assertEqual(decision.action, "ignore")

    async def test_strict_schema_rejects_schema_copy_and_extra_fields(self):
        for changes in (
            {"hand_visible": "true"},
            {"arguments": {}},
            {"gesture": {"type": "string"}},
            {"evidence": "offered_hand for handshake"},
            {"evidence": "hand is visible and reaching towards the robot"},
        ):
            with self.subTest(changes=changes), self.assertRaises(DecisionAgentError):
                await self.decide(**changes)

    async def test_truncated_json_does_not_execute(self):
        agent = SocialVisionAgent(invoker=FakeVisionInvoker(['{"gesture":"handshake"']))
        with self.assertRaises(DecisionAgentError):
            await agent.decide(
                [camera_frame(1), camera_frame(2)],
                RobotState(hardware=False, connected=True),
                [HandshakeSkill()],
            )

    async def test_single_markdown_json_fence_is_safely_unwrapped(self):
        output = """```json
{"gesture":"wave","hand_visible":true,"directed_at_robot":true,
"present_in_latest":true,"evidence":"side_to_side"}
```"""
        agent = SocialVisionAgent(invoker=FakeVisionInvoker([output]))
        decision = await agent.decide(
            [camera_frame(1), camera_frame(2)],
            RobotState(hardware=False, connected=True),
            build_g1_autonomy_skills(),
        )

        self.assertEqual(decision.action, "execute_skill")
        self.assertEqual(decision.skill, "wave")

    async def test_single_frame_waits_without_inference(self):
        invoker = FakeVisionInvoker([])
        decision = await SocialVisionAgent(invoker=invoker).decide(
            [camera_frame(1)],
            RobotState(hardware=False, connected=True),
            [],
        )
        self.assertEqual(decision.action, "ignore")
        self.assertFalse(invoker.calls)

    async def test_high_five_and_active_handshake(self):
        for gesture, evidence, context, expected in (
            ("high_five", "raised_palm", {}, "execute_skill"),
            ("handshake", "offered_hand", {"active_skill": "handshake"}, "continue"),
        ):
            agent = SocialVisionAgent(
                invoker=FakeVisionInvoker(
                    [
                        {
                            "gesture": gesture,
                            "evidence": evidence,
                            "hand_visible": True,
                            "directed_at_robot": True,
                            "present_in_latest": True,
                        }
                    ]
                )
            )
            decision = await agent.decide(
                [camera_frame(1), camera_frame(2)],
                RobotState(hardware=False, connected=True),
                build_g1_autonomy_skills(),
                policy_context=context,
            )
            self.assertEqual(decision.action, expected)
            self.assertEqual(
                agent.last_metrics["gesture_observation"]["gesture"], gesture
            )

    async def test_go2_heart_gesture_maps_to_registered_skill(self):
        agent = SocialVisionAgent(
            invoker=FakeVisionInvoker(
                [
                    {
                        "gesture": "heart",
                        "evidence": "heart_shape",
                        "hand_visible": True,
                        "directed_at_robot": True,
                        "present_in_latest": True,
                    }
                ]
            )
        )
        decision = await agent.decide(
            [camera_frame(1), camera_frame(2)],
            RobotState(hardware=False, connected=True),
            build_go2_autonomy_skills(),
        )

        self.assertEqual(decision.action, "execute_skill")
        self.assertEqual(decision.skill, "heart")

    async def test_duplicate_frame_timestamps_are_rejected(self):
        invoker = FakeVisionInvoker([])
        decision = await SocialVisionAgent(invoker=invoker).decide(
            [camera_frame(1), camera_frame(1)],
            RobotState(hardware=False, connected=True),
            [HandshakeSkill()],
        )
        self.assertEqual(decision.action, "ignore")
        self.assertFalse(invoker.calls)
