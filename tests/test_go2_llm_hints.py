"""P0: tool descriptions and system prompt must teach the cloud model selection."""

from __future__ import annotations

import unittest

from adapters.langchain import build_langchain_tools
from agent.service import GO2_SYSTEM_PROMPT, system_prompt_for
from core.runtime import SkillRuntime
from robot import SimulatedRobotAdapter
from skills import register_go2_skills


class Go2LlmHintTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runtime = SkillRuntime(SimulatedRobotAdapter())
        register_go2_skills(self.runtime)
        self.tools = {t.name: t for t in build_langchain_tools(self.runtime)}

    def test_key_tools_carry_chinese_triggers(self) -> None:
        expectations = {
            "heart": ("比心",),
            "hello": ("打招呼",),
            "stretch": ("伸懒腰",),
            "sit": ("坐下",),
            "stand_up": ("站起来",),
            "dance1": ("跳舞",),
            "move_forward": ("向前",),
            "turn_left": ("左转",),
        }
        for name, needles in expectations.items():
            description = self.tools[name].description or ""
            for needle in needles:
                with self.subTest(tool=name, needle=needle):
                    self.assertIn(needle, description)

    def test_hello_description_forbids_heart_substitution(self) -> None:
        description = self.tools["hello"].description or ""
        self.assertIn("Never substitute", description)
        self.assertIn("heart", description)

    def test_move_descriptions_state_bounds_and_stand_up(self) -> None:
        description = self.tools["move_forward"].description or ""
        self.assertIn("0.3", description)
        self.assertIn("stand_up", description)
        self.assertIn("navigation", description.lower())

    def test_operator_only_tools_marked_for_llm(self) -> None:
        runtime = SkillRuntime(SimulatedRobotAdapter())
        register_go2_skills(runtime, include_operator_only=True)
        tools = {t.name: t for t in build_langchain_tools(runtime)}
        for name in ("front_flip", "damp", "trot_run", "hand_stand", "switch_joystick"):
            description = tools[name].description or ""
            with self.subTest(tool=name):
                self.assertIn("OPERATOR-ONLY", description)

    def test_go2_system_prompt_mapping_and_anti_hello_rule(self) -> None:
        self.assertEqual(system_prompt_for("go2"), GO2_SYSTEM_PROMPT)
        for phrase in (
            "比心 / 送心 / heart -> heart",
            "站起来 / 起身 / stand up -> stand_up",
            "Never substitute hello/wave",
            "stand_up first",
            "Operator-only tools",
        ):
            self.assertIn(phrase, GO2_SYSTEM_PROMPT)

    def test_pose_flag_description_documents_boolean(self) -> None:
        description = self.tools["pose"].description or ""
        self.assertIn("flag=true", description)
        self.assertIn("flag=false", description)


if __name__ == "__main__":
    unittest.main()
