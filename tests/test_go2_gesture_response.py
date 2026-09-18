import unittest

from agent.social_vision import SocialVisionAgent
from robot import RobotState
from skills.go2_catalog import build_go2_autonomy_skills
from tests.test_vision_policy import FakeVisionInvoker, camera_frame


class TaskDrivenVisionTests(unittest.IsolatedAsyncioTestCase):
    async def test_model_selected_heart_is_executed(self):
        agent = SocialVisionAgent(
            task_context="用户打招呼就打招呼，比耶或比心就比心",
            generate_speech=True,
            invoker=FakeVisionInvoker(
                [
                    {
                        "action": "execute_skill",
                        "skill": "heart",
                        "observation": "peace sign near face",
                        "hand_visible": True,
                        "directed_at_robot": True,
                        "present_in_latest": True,
                        "speech": "给你比心",
                    }
                ]
            ),
        )
        decision = await agent.decide(
            [camera_frame(1), camera_frame(2)],
            RobotState(hardware=False, connected=True),
            build_go2_autonomy_skills(),
        )
        self.assertEqual(decision.action, "execute_and_speak")
        self.assertEqual(decision.skill, "heart")
        self.assertEqual(decision.speech, "给你比心")

    async def test_model_selected_wave_is_executed(self):
        agent = SocialVisionAgent(
            task_context="有人打招呼就打招呼",
            invoker=FakeVisionInvoker(
                [
                    {
                        "action": "execute_skill",
                        "skill": "wave",
                        "observation": "person waving hand",
                        "hand_visible": True,
                        "directed_at_robot": True,
                        "present_in_latest": True,
                    }
                ]
            ),
        )
        decision = await agent.decide(
            [camera_frame(1), camera_frame(2)],
            RobotState(hardware=False, connected=True),
            build_go2_autonomy_skills(),
        )
        self.assertEqual(decision.action, "execute_skill")
        self.assertEqual(decision.skill, "wave")

    async def test_peace_sign_returned_as_wave_is_resolved_to_heart(self):
        agent = SocialVisionAgent(
            task_context="用户打招呼就打招呼，比耶或比心就比心",
            invoker=FakeVisionInvoker(
                [
                    {
                        "action": "execute_skill",
                        "skill": "wave",
                        "observation": "V-sign held near face",
                        "hand_visible": True,
                        "directed_at_robot": True,
                        "present_in_latest": True,
                    }
                ]
            ),
        )
        decision = await agent.decide(
            [camera_frame(1), camera_frame(2)],
            RobotState(hardware=False, connected=True),
            build_go2_autonomy_skills(),
        )
        self.assertEqual(decision.skill, "heart")

    async def test_unregistered_or_operator_skill_is_ignored(self):
        for skill in ("damp", "front_flip", "not_a_skill"):
            agent = SocialVisionAgent(
                invoker=FakeVisionInvoker(
                    [
                        {
                            "action": "execute_skill",
                            "skill": skill,
                            "observation": "something",
                            "hand_visible": True,
                            "directed_at_robot": True,
                            "present_in_latest": True,
                        }
                    ]
                ),
            )
            decision = await agent.decide(
                [camera_frame(1), camera_frame(2)],
                RobotState(hardware=False, connected=True),
                build_go2_autonomy_skills(),
            )
            self.assertEqual(decision.action, "ignore")

    async def test_unconfirmed_directed_at_robot_is_ignored(self):
        agent = SocialVisionAgent(
            invoker=FakeVisionInvoker(
                [
                    {
                        "action": "execute_skill",
                        "skill": "heart",
                        "observation": "two-hand heart",
                        "hand_visible": True,
                        "directed_at_robot": False,
                        "present_in_latest": True,
                    }
                ]
            ),
        )
        decision = await agent.decide(
            [camera_frame(1), camera_frame(2)],
            RobotState(hardware=False, connected=True),
            build_go2_autonomy_skills(),
        )
        self.assertEqual(decision.action, "ignore")

    async def test_ignore_with_no_skill_stays_silent(self):
        agent = SocialVisionAgent(
            invoker=FakeVisionInvoker(
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
            ),
        )
        decision = await agent.decide(
            [camera_frame(1), camera_frame(2)],
            RobotState(hardware=False, connected=True),
            build_go2_autonomy_skills(),
        )
        self.assertEqual(decision.action, "ignore")
        self.assertIsNone(decision.skill)

    async def test_active_skill_returns_continue(self):
        agent = SocialVisionAgent(
            invoker=FakeVisionInvoker(
                [
                    {
                        "action": "execute_skill",
                        "skill": "heart",
                        "observation": "heart still held",
                        "hand_visible": True,
                        "directed_at_robot": True,
                        "present_in_latest": True,
                    }
                ]
            ),
        )
        decision = await agent.decide(
            [camera_frame(1), camera_frame(2)],
            RobotState(hardware=False, connected=True),
            build_go2_autonomy_skills(),
            policy_context={"active_skill": "heart"},
        )
        self.assertEqual(decision.action, "continue")


if __name__ == "__main__":
    unittest.main()
