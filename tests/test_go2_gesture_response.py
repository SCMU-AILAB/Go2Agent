"""Vision decisions reach Go2 only through the shared SkillRuntime."""

import unittest

from agent.social_vision import SocialVisionAgent
from core.runtime import SkillRuntime
from robot import RobotState, SimulatedRobotAdapter
from skills import register_go2_skills
from tests.test_vision_policy import FakeVisionInvoker, camera_frame


class Go2VisionExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def _decide_and_execute(self, payload):
        robot = SimulatedRobotAdapter()
        runtime = SkillRuntime(robot)
        register_go2_skills(runtime)
        agent = SocialVisionAgent(
            invoker=FakeVisionInvoker([payload]),
            task_context="按画面和任务选择 Go2 动作",
            generate_speech=True,
        )
        decision = await agent.decide(
            [camera_frame(1), camera_frame(2)],
            RobotState(hardware=False, connected=True),
            runtime.registry.list(),
        )
        result = None
        if decision.skill is not None:
            result = await runtime.execute(decision.skill, **decision.arguments)
        return decision, result, robot

    async def test_model_chosen_native_action_runs_on_go2(self):
        for skill in ("hello", "heart", "sit"):
            with self.subTest(skill=skill):
                _decision, result, robot = await self._decide_and_execute(
                    {"action": "execute_skill", "skill": skill}
                )
                self.assertTrue(result.success)
                self.assertEqual(robot.events, [("loco_action", (skill, {}))])

    async def test_speech_rides_along_with_the_skill(self):
        decision, _result, robot = await self._decide_and_execute(
            {"action": "execute_and_speak", "skill": "heart", "speech": "给你比心"}
        )
        self.assertEqual(decision.speech, "给你比心")
        self.assertEqual(robot.events, [("loco_action", ("heart", {}))])

    async def test_invalid_arguments_are_rejected_by_runtime(self):
        _decision, result, robot = await self._decide_and_execute(
            {"action": "execute_skill", "skill": "move_forward", "arguments": {"distance_m": 5}}
        )
        self.assertFalse(result.success)
        self.assertEqual(robot.events, [])

    async def test_ignore_never_touches_the_robot(self):
        decision, result, robot = await self._decide_and_execute({"action": "ignore"})
        self.assertEqual(decision.action, "ignore")
        self.assertIsNone(result)
        self.assertEqual(robot.events, [])


if __name__ == "__main__":
    unittest.main()
