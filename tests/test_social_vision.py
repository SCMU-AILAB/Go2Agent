"""SocialVisionAgent: the VLM decides directly from frames and the task."""

from __future__ import annotations

import unittest

from agent.social_vision import SocialVisionAgent
from robot import RobotState
from skills import build_go2_all_skills
from tests.test_vision_policy import FakeVisionInvoker, camera_frame


def _agent(payloads, **kwargs) -> tuple[SocialVisionAgent, FakeVisionInvoker]:
    invoker = FakeVisionInvoker(payloads)
    agent = SocialVisionAgent(
        invoker=invoker,
        task_context=kwargs.pop("task_context", "按画面和任务选择 Go2 动作"),
        **kwargs,
    )
    return agent, invoker


async def _decide(agent, *, at=(1.0, 2.0), policy_context=None, state=None):
    return await agent.decide(
        [camera_frame(t) for t in at],
        state or RobotState(hardware=False, connected=True),
        build_go2_all_skills(),
        policy_context=policy_context,
    )


class SocialVisionTests(unittest.IsolatedAsyncioTestCase):
    async def test_model_decision_executes_without_extra_gating(self):
        agent, _ = _agent([{"action": "execute_skill", "skill": "sit", "reason": "asked"}])

        decision = await _decide(agent)

        self.assertEqual(decision.action, "execute_skill")
        self.assertEqual(decision.skill, "sit")

    async def test_frames_and_task_go_to_the_vision_model(self):
        agent, invoker = _agent([{"action": "ignore"}], task_context="看到有人挥手就打招呼")

        await _decide(agent, at=(1.0, 1.4, 1.8))

        frames, prompt = invoker.calls[0]
        self.assertEqual(len(frames), 3)
        self.assertIn("看到有人挥手就打招呼", prompt)
        self.assertIn('"name": "hello"', prompt)
        self.assertIn("frame_offsets_s: [-0.8, -0.4, 0.0]", prompt)

    async def test_prompt_has_no_skill_or_gesture_special_cases(self):
        agent, invoker = _agent([{"action": "ignore"}])

        await _decide(
            agent,
            state=RobotState(
                hardware=True,
                connected=True,
                details={"mode": 1, "supported_loco_actions": ["hello", "heart"]},
            ),
        )

        prompt = invoker.calls[0][1]
        template = prompt.split("Operator task:")[0]
        for patched in ("wave", "heart", "follow_person", "peace", "thumbs", "center"):
            self.assertNotIn(patched, template)
        self.assertNotIn("supported_loco_actions", prompt)
        self.assertIn('"mode": 1', prompt)

    async def test_speech_follows_generate_speech(self):
        payload = {"action": "execute_and_speak", "skill": "heart", "speech": "给你比心"}
        agent, _ = _agent([payload], generate_speech=True)
        decision = await _decide(agent)
        self.assertEqual(decision.action, "execute_and_speak")
        self.assertEqual(decision.speech, "给你比心")

        muted, _ = _agent([payload], generate_speech=False)
        decision = await _decide(muted)
        self.assertEqual(decision.action, "execute_skill")
        self.assertIsNone(decision.speech)

        speak_only, _ = _agent([{"action": "speak", "speech": "你好"}])
        self.assertEqual((await _decide(speak_only)).action, "ignore")

    async def test_unregistered_skill_is_ignored(self):
        agent, _ = _agent([{"action": "execute_skill", "skill": "handshake"}])

        decision = await _decide(agent)

        self.assertEqual(decision.action, "ignore")
        self.assertIn("handshake", decision.reason)

    async def test_invalid_output_is_ignored(self):
        for output in ("bad json", '[{"point_2d":[436,573]}]', {"action": "fly"}):
            with self.subTest(output=output):
                agent, _ = _agent([output])
                self.assertEqual((await _decide(agent)).action, "ignore")

    async def test_tool_call_name_nested_in_arguments_is_accepted(self):
        agent, _ = _agent(
            [{"action": "execute_skill", "arguments": {"name": "move_forward", "distance_m": 0.2}}]
        )

        decision = await _decide(agent)

        self.assertEqual(decision.skill, "move_forward")
        self.assertEqual(decision.arguments, {"distance_m": 0.2})

    async def test_continue_and_interrupt_depend_on_active_skill(self):
        agent, _ = _agent([{"action": "continue"}])
        self.assertEqual((await _decide(agent)).action, "ignore")
        self.assertEqual(
            (await _decide(agent, policy_context={"active_skill": "follow_person"})).action,
            "continue",
        )

        agent, _ = _agent([{"action": "interrupt", "reason": "goal changed"}])
        self.assertEqual((await _decide(agent)).action, "interrupt")

    async def test_reselecting_the_active_skill_becomes_continue(self):
        agent, _ = _agent([{"action": "execute_skill", "skill": "follow_person"}])

        decision = await _decide(agent, policy_context={"active_skill": "follow_person"})

        self.assertEqual(decision.action, "continue")

    async def test_non_chronological_frames_are_ignored(self):
        agent, invoker = _agent([{"action": "execute_skill", "skill": "sit"}])

        decision = await _decide(agent, at=(2.0, 2.0))

        self.assertEqual(decision.action, "ignore")
        self.assertEqual(invoker.calls, [])

    async def test_optional_hold_uses_camera_time(self):
        payload = {"action": "execute_skill", "skill": "hello"}
        agent, _ = _agent([payload], confirm_hold_s=0.5)

        first = await _decide(agent, at=(0.8, 1.0))
        short = await _decide(agent, at=(1.1, 1.2))
        held = await _decide(agent, at=(1.4, 1.6))

        self.assertEqual(first.action, "ignore")
        self.assertEqual(short.action, "ignore")
        self.assertEqual(held.skill, "hello")

    async def test_hold_is_disabled_by_default(self):
        agent, _ = _agent([{"action": "execute_skill", "skill": "hello"}])

        self.assertEqual(agent.confirm_hold_s, 0.0)
        self.assertEqual((await _decide(agent)).skill, "hello")


if __name__ == "__main__":
    unittest.main()
