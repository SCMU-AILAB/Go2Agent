from __future__ import annotations

import unittest

from agent import LocalVoiceCommandAgent
from core.runtime import SkillRuntime
from robot import SimulatedRobotAdapter
from skills import register_go2_skills


class LocalVoiceCommandAgentTests(unittest.IsolatedAsyncioTestCase):
    def build_agent(
        self,
    ) -> tuple[LocalVoiceCommandAgent, SimulatedRobotAdapter]:
        robot = SimulatedRobotAdapter()
        runtime = SkillRuntime(robot)
        register_go2_skills(runtime)
        return LocalVoiceCommandAgent(runtime), robot

    async def test_heart_executes_registered_skill(self) -> None:
        agent, robot = self.build_agent()

        reply = await agent.chat("请给我比个心")

        self.assertEqual(reply, "好的，比心指令已发送。")
        self.assertIn(("loco_action", ("heart", {})), robot.events)

    async def test_stop_has_priority_over_other_action_words(self) -> None:
        agent, robot = self.build_agent()

        reply = await agent.chat("别跳舞了，马上停止")

        self.assertEqual(reply, "好的，已经停止。")
        self.assertIn(("stop", None), robot.events)
        self.assertNotIn(("loco_action", ("dance1", {})), robot.events)

    async def test_unknown_text_does_not_guess_a_skill(self) -> None:
        agent, robot = self.build_agent()

        reply = await agent.chat("今天天气怎么样")

        self.assertIn("没有识别到明确动作", reply)
        self.assertEqual(robot.events, [])

    def test_lateral_move_and_turn_are_distinct(self) -> None:
        move = LocalVoiceCommandAgent._match("向左走一点")
        turn = LocalVoiceCommandAgent._match("向左转一点")

        self.assertIsNotNone(move)
        self.assertIsNotNone(turn)
        self.assertEqual(move.skill, "move_left")
        self.assertEqual(turn.skill, "turn_left")


if __name__ == "__main__":
    unittest.main()
