"""Task-driven SocialVisionAgent tests (no hardcoded gesture enum)."""

from __future__ import annotations

import unittest

from agent.social_vision import SocialVisionAgent
from robot import RobotState
from skills import build_go2_autonomy_skills
from tests.test_vision_policy import FakeVisionInvoker, camera_frame


class SocialVisionTests(unittest.IsolatedAsyncioTestCase):
    def _decide(self, payloads, skills=None, **kwargs):
        policy_context = kwargs.pop("policy_context", None)
        agent = SocialVisionAgent(
            invoker=FakeVisionInvoker(payloads),
            task_context=kwargs.pop(
                "task_context", "有人比耶或比心就比心，打招呼就打招呼"
            ),
            generate_speech=kwargs.pop("generate_speech", False),
            response_format=kwargs.pop("response_format", "json"),
            **kwargs,
        )
        return agent.decide(
            [camera_frame(1), camera_frame(2)],
            RobotState(hardware=False, connected=True),
            skills if skills is not None else build_go2_autonomy_skills(),
            policy_context=policy_context,
        )

    async def test_ignore_when_no_social_response(self):
        decision = await self._decide(
            [
                {
                    "action": "ignore",
                    "skill": None,
                    "observation": "person walking by",
                    "hand_visible": False,
                    "directed_at_robot": False,
                    "present_in_latest": False,
                }
            ]
        )
        self.assertEqual(decision.action, "ignore")

    async def test_execute_registered_skill_from_catalog(self):
        decision = await self._decide(
            [
                {
                    "action": "execute_skill",
                    "skill": "heart",
                    "observation": "V-sign near face",
                    "hand_visible": True,
                    "directed_at_robot": True,
                    "present_in_latest": True,
                }
            ]
        )
        self.assertEqual(decision.action, "execute_skill")
        self.assertEqual(decision.skill, "heart")

    async def test_corrects_peace_sign_misclassified_as_wave(self):
        agent = SocialVisionAgent(
            invoker=FakeVisionInvoker(
                [
                    {
                        "action": "execute_skill",
                        "skill": "wave",
                        "observation": "stationary peace sign near face",
                        "hand_visible": True,
                        "directed_at_robot": True,
                        "present_in_latest": True,
                    }
                ]
            ),
            task_context="用户比耶就比心，挥手就打招呼",
        )
        decision = await agent.decide(
            [camera_frame(1), camera_frame(2)],
            RobotState(hardware=False, connected=True),
            build_go2_autonomy_skills(),
        )
        self.assertEqual(decision.skill, "heart")
        self.assertEqual(
            agent.last_metrics["gesture_observation"]["resolved_skill"], "heart"
        )

    async def test_real_wave_is_not_corrected_to_heart(self):
        decision = await self._decide(
            [
                {
                    "action": "execute_skill",
                    "skill": "wave",
                    "observation": "open hand waving side-to-side",
                    "hand_visible": True,
                    "directed_at_robot": True,
                    "present_in_latest": True,
                }
            ]
        )
        self.assertEqual(decision.skill, "wave")

    async def test_speech_only_when_generate_speech_enabled(self):
        payload = {
            "action": "execute_skill",
            "skill": "wave",
            "observation": "waving",
            "hand_visible": True,
            "directed_at_robot": True,
            "present_in_latest": True,
            "speech": "你好",
        }
        silent = await self._decide([payload], generate_speech=False)
        self.assertEqual(silent.action, "execute_skill")
        self.assertIsNone(silent.speech)
        speaking = await self._decide([payload], generate_speech=True)
        self.assertEqual(speaking.action, "execute_and_speak")
        self.assertEqual(speaking.speech, "你好")

    async def test_requires_directed_and_present(self):
        decision = await self._decide(
            [
                {
                    "action": "execute_skill",
                    "skill": "heart",
                    "observation": "heart shape",
                    "hand_visible": True,
                    "directed_at_robot": False,
                    "present_in_latest": True,
                }
            ]
        )
        self.assertEqual(decision.action, "ignore")

    async def test_unregistered_skill_rejected(self):
        decision = await self._decide(
            [
                {
                    "action": "execute_skill",
                    "skill": "launch_missile",
                    "observation": "whatever",
                    "hand_visible": True,
                    "directed_at_robot": True,
                    "present_in_latest": True,
                }
            ]
        )
        self.assertEqual(decision.action, "ignore")

    async def test_duplicate_timestamps_rejected(self):
        agent = SocialVisionAgent(invoker=FakeVisionInvoker([]))
        frame = camera_frame(1)
        decision = await agent.decide(
            [frame, frame],
            RobotState(hardware=False, connected=True),
            build_go2_autonomy_skills(),
        )
        self.assertEqual(decision.action, "ignore")

    async def test_prompt_includes_task_and_catalog(self):
        invoker = FakeVisionInvoker(
            [
                {
                    "action": "ignore",
                    "skill": None,
                    "observation": "none",
                    "hand_visible": False,
                    "directed_at_robot": False,
                    "present_in_latest": False,
                }
            ]
        )
        agent = SocialVisionAgent(
            invoker=invoker,
            task_context="用户比耶就比心",
        )
        await agent.decide(
            [camera_frame(1), camera_frame(2)],
            RobotState(hardware=False, connected=True),
            build_go2_autonomy_skills(),
        )
        prompt = invoker.calls[-1][1]
        self.assertIn("用户比耶就比心", prompt)
        self.assertIn("heart", prompt)
        self.assertIn("wave", prompt)
        self.assertIn("is NOT waving", prompt)
        self.assertIn("choose heart", prompt)
        self.assertNotIn("damp", prompt)

    async def test_gesture_label_maps_peace_sign_to_requested_heart(self):
        invoker = FakeVisionInvoker(["peace_sign"])
        agent = SocialVisionAgent(
            invoker=invoker,
            task_context="用户打招呼就打招呼，比耶或比心就比心",
            response_format="gesture_label",
        )
        decision = await agent.decide(
            [camera_frame(1), camera_frame(2)],
            RobotState(hardware=False, connected=True),
            build_go2_autonomy_skills(),
        )
        self.assertEqual(decision.action, "execute_skill")
        self.assertEqual(decision.skill, "heart")
        prompt = invoker.calls[-1][1]
        self.assertIn("Reply with exactly one label", prompt)
        self.assertNotIn("JSON object", prompt)

    async def test_gesture_label_none_is_ignored(self):
        decision = await self._decide(
            ["none"],
            response_format="gesture_label",
        )
        self.assertEqual(decision.action, "ignore")

    async def test_gesture_label_not_requested_is_ignored(self):
        decision = await self._decide(
            ["high_five"],
            response_format="gesture_label",
        )
        self.assertEqual(decision.action, "ignore")

    async def test_gesture_label_rejects_coordinate_output(self):
        with self.assertRaisesRegex(Exception, "unsupported gesture label"):
            await self._decide(
                ['[{"point": [236, 738]}]'],
                response_format="gesture_label",
            )

    async def test_gesture_label_thumbs_up_random_dance_when_task_asks(self):
        invoker = FakeVisionInvoker(["thumbs_up"])
        agent = SocialVisionAgent(
            invoker=invoker,
            task_context="当用户给你竖起来大拇指的时候就随机选一个预设的舞蹈跳跃",
            response_format="gesture_label",
        )
        decision = await agent.decide(
            [camera_frame(1), camera_frame(2)],
            RobotState(hardware=False, connected=True),
            build_go2_autonomy_skills(),
        )
        self.assertEqual(decision.action, "execute_skill")
        self.assertEqual(decision.skill, "random_dance")
        prompt = invoker.calls[-1][1]
        self.assertIn("thumbs_up", prompt)
        self.assertIn("Reply with exactly one label", prompt)

    async def test_gesture_label_thumbs_up_unrelated_task_is_ignored(self):
        decision = await self._decide(
            ["thumbs_up"],
            response_format="gesture_label",
            task_context="用户打招呼时打招呼",
        )
        self.assertEqual(decision.action, "ignore")

    async def test_gesture_label_thumbs_up_without_dance_task_is_ignored(self):
        decision = await self._decide(
            ["thumbs_up"],
            response_format="gesture_label",
            task_context="用户点赞时比心",
        )
        self.assertEqual(decision.action, "ignore")

    async def test_gesture_label_rejects_point_2d(self):
        for label in ("point", "point_2d", 'point_2d'):
            with self.assertRaisesRegex(Exception, "unsupported gesture label"):
                await self._decide(
                    [label],
                    response_format="gesture_label",
                    task_context="用户点赞时随机跳舞",
                )

    async def test_json_path_can_select_random_dance(self):
        decision = await self._decide(
            [
                {
                    "action": "execute_skill",
                    "skill": "random_dance",
                    "observation": "thumbs up toward camera",
                    "hand_visible": True,
                    "directed_at_robot": True,
                    "present_in_latest": True,
                }
            ],
            task_context="用户竖起大拇指时随机跳舞",
        )
        self.assertEqual(decision.action, "execute_skill")
        self.assertEqual(decision.skill, "random_dance")

    async def test_json_active_random_dance_returns_continue(self):
        decision = await self._decide(
            [
                {
                    "action": "execute_skill",
                    "skill": "random_dance",
                    "observation": "thumbs up still held",
                    "hand_visible": True,
                    "directed_at_robot": True,
                    "present_in_latest": True,
                }
            ],
            task_context="用户竖起大拇指时随机跳舞",
            policy_context={"active_skill": "random_dance"},
        )
        self.assertEqual(decision.action, "continue")


if __name__ == "__main__":
    unittest.main()
