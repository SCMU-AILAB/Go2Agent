from __future__ import annotations

import json
import unittest
from dataclasses import replace

from agent.social_vision import SocialVisionAgent, _compact_skill_catalog
from agent.vision_policy import VisionDecisionAgent, _skill_catalog_payload
from core.runtime import SkillRuntime
from perception import CameraFrame, PerceptionResult
from robot import RobotState, SimulatedRobotAdapter
from skills import build_go2_all_skills, register_go2_skills


class FakeVisionInvoker:
    def __init__(self, response: object) -> None:
        self.response = response
        self.prompt = ""

    async def ainvoke(self, frames: object, prompt: str) -> object:
        del frames
        self.prompt = prompt
        return self.response


def frames() -> list[CameraFrame]:
    return [
        CameraFrame(
            observed_at_s=1.0,
            rgb=b"frame-1",
            depth=None,
            observation=PerceptionResult(observed_at_s=1.0),
        ),
        CameraFrame(
            observed_at_s=2.0,
            rgb=b"frame-2",
            depth=None,
            observation=PerceptionResult(observed_at_s=2.0),
        ),
    ]


class Go2VisionCatalogTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.skills = build_go2_all_skills()
        self.names = {skill.metadata.name for skill in self.skills}
        self.state = RobotState(hardware=False, connected=True)

    async def test_visual_agent_receives_complete_catalog(self) -> None:
        invoker = FakeVisionInvoker(
            {"action": "execute_skill", "skill": "damp", "arguments": {}}
        )
        agent = SocialVisionAgent(
            invoker=invoker,
            task_context="明确切换 Go2 阻尼模式",
        )

        decision = await agent.decide(frames(), self.state, self.skills)

        self.assertEqual(decision.skill, "damp")
        self.assertTrue(agent.allow_operator_skills)
        self.assertEqual(agent.response_format, "decision")
        catalog = json.loads(invoker.prompt.split("Registered skills: ")[1].split("\nRobot state:")[0])
        self.assertEqual({entry["name"] for entry in catalog}, self.names)

    def test_visual_payload_contains_same_skills_and_schemas(self) -> None:
        payload = _skill_catalog_payload(self.skills)
        payload_names = {entry["name"] for entry in payload}

        self.assertEqual(payload_names, self.names)
        pose = next(entry for entry in payload if entry["name"] == "pose")
        self.assertIn("flag", pose["arguments_schema"]["properties"])
        damp = next(entry for entry in payload if entry["name"] == "damp")
        self.assertIn("operator_only", damp["tags"])

    def test_compaction_preserves_argument_contracts_and_safety_tags(self) -> None:
        original = _skill_catalog_payload(self.skills)
        compact = _compact_skill_catalog(self.skills)
        self.assertLess(len(json.dumps(compact)), len(json.dumps(original)))
        for before, after in zip(original, compact, strict=True):
            self.assertEqual(before["name"], after["name"])
            self.assertEqual(before["tags"], after["tags"])
            schema = after["arguments_schema"]
            self.assertEqual(schema["required"], before["arguments_schema"]["required"])
            for name, field in before["arguments_schema"]["properties"].items():
                self.assertEqual(schema["properties"][name], {k: v for k, v in field.items() if k != "title"})

    def test_recovery_allowlist_does_not_filter_operator_skills(self) -> None:
        recovered = VisionDecisionAgent._recoverable_skill_names(self.skills)

        self.assertEqual(recovered, self.names)

    async def test_every_registered_skill_can_be_selected_with_its_arguments(
        self,
    ) -> None:
        observed_frames = [
            replace(
                frame,
                observation=PerceptionResult(
                    observed_at_s=frame.observed_at_s,
                    person_count=1,
                    nearest_person_distance_m=2.0,
                    person_center_x=0.5,
                ),
                nearest_obstacle_distance_m=2.0,
            )
            for frame in frames()
        ]
        for skill in self.skills:
            with self.subTest(skill=skill.metadata.name):
                arguments = (
                    {"flag": False} if "flag" in skill.args_model.model_fields else {}
                )
                agent = SocialVisionAgent(
                    invoker=FakeVisionInvoker(
                        {
                            "action": "execute_skill",
                            "skill": skill.metadata.name,
                            "arguments": arguments,
                        }
                    ),
                    task_context="按画面选择 Go2 动作",
                )
                decision = await agent.decide(observed_frames, self.state, self.skills)
                self.assertEqual(decision.action, "execute_skill")
                self.assertEqual(decision.skill, skill.metadata.name)
                self.assertEqual(decision.arguments, arguments)

    async def test_operator_flag_decision_executes_through_shared_runtime(self) -> None:
        robot = SimulatedRobotAdapter()
        runtime = SkillRuntime(robot)
        register_go2_skills(runtime)
        agent = SocialVisionAgent(
            invoker=FakeVisionInvoker(
                {
                    "action": "execute_skill",
                    "skill": "auto_recover_set",
                    "arguments": {"flag": False},
                }
            ),
        )
        decision = await agent.decide(frames(), self.state, runtime.registry.list())
        result = await runtime.execute(decision.skill, **decision.arguments)

        self.assertTrue(result.success)
        self.assertEqual(
            robot.events, [("loco_action", ("auto_recover_set", {"flag": False}))]
        )


if __name__ == "__main__":
    unittest.main()
