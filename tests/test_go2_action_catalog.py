"""Verify the new Go2 catalog through the tool/runtime/native adapter boundary."""

import json
import unittest
from unittest.mock import Mock

from adapters.langchain import build_langchain_tools
from core.runtime import SkillRuntime
from robot import Go2Bindings, UnitreeGo2Adapter
from skills import register_go2_skills

NO_ARG_ACTIONS = {
    "stretch",
    "content",
    "heart",
    "scrape",
    "dance1",
    "dance2",
    "front_flip",
    "front_jump",
    "front_pounce",
    "left_flip",
    "back_flip",
    "free_walk",
    "static_walk",
    "trot_run",
    "economic_gait",
    "switch_avoid_mode",
}
FLAG_ACTIONS = {
    "pose",
    "hand_stand",
    "free_bound",
    "free_jump",
    "free_avoid",
    "classic_walk",
    "walk_upright",
    "cross_step",
    "switch_joystick",
    "auto_recover_set",
}
DEFAULT_ACTIONS = NO_ARG_ACTIONS | FLAG_ACTIONS


class Go2ActionCatalogTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.sport = Mock()
        for name in NO_ARG_ACTIONS | FLAG_ACTIONS:
            getattr(self.sport, name).return_value = 0
        self.robot = UnitreeGo2Adapter(bindings=Go2Bindings(Mock(), lambda: self.sport))
        await self.robot.connect()
        self.runtime = SkillRuntime(self.robot)
        register_go2_skills(self.runtime, include_operator_only=True)
        self.tools = {tool.name: tool for tool in build_langchain_tools(self.runtime)}

    async def asyncTearDown(self):
        await self.robot.close()

    async def test_every_added_tool_reaches_exact_native_method(self):
        for name in NO_ARG_ACTIONS:
            with self.subTest(name=name):
                result = json.loads(await self.tools[name].ainvoke({}))
                self.assertTrue(result["success"])
                self.assertIn("command accepted", result["message"])
                getattr(self.sport, name).assert_called_once_with()
        for name in FLAG_ACTIONS:
            for flag in (True, False):
                with self.subTest(name=name, flag=flag):
                    result = json.loads(await self.tools[name].ainvoke({"flag": flag}))
                    self.assertTrue(result["success"])
                    getattr(self.sport, name).assert_called_with(flag)

    async def test_flag_validation_rejects_coercions_missing_and_extra_values(self):
        for name in FLAG_ACTIONS:
            for arguments in (
                {},
                {"flag": 1},
                {"flag": "false"},
                {"flag": None},
                {"flag": True, "unexpected": 1},
            ):
                with self.subTest(name=name, arguments=arguments):
                    result = await self.runtime.execute(name, **arguments)
                    self.assertFalse(result.success)
                    with self.assertRaisesRegex(RuntimeError, "boolean flag"):
                        await self.robot.execute_loco_action(name, arguments)
            getattr(self.sport, name).assert_not_called()

    async def test_default_catalog_and_schema(self):
        runtime = SkillRuntime(self.robot)
        register_go2_skills(runtime)
        names = {tool.name for tool in build_langchain_tools(runtime)}
        self.assertTrue(DEFAULT_ACTIONS <= names)
        self.assertTrue(DEFAULT_ACTIONS <= names)
        for name in FLAG_ACTIONS:
            schema = self.tools[name].args_schema.model_json_schema()
            self.assertEqual(schema["properties"]["flag"]["type"], "boolean")
            self.assertIn("flag", schema["required"])
        state = await self.robot.get_state()
        self.assertTrue(
            (NO_ARG_ACTIONS | FLAG_ACTIONS)
            <= set(state.details["supported_loco_actions"])
        )

    async def test_sdk_error_and_missing_method_are_failures(self):
        self.sport.heart.return_value = 1234
        result = await self.runtime.execute("heart")
        self.assertFalse(result.success)
        self.assertIn("1234", result.message)
        self.sport.dance1 = None
        result = await self.runtime.execute("dance1")
        self.assertFalse(result.success)
        self.assertIn("do not provide dance1", result.message)

    async def test_no_argument_actions_reject_extra_parameters(self):
        for name in NO_ARG_ACTIONS:
            result = await self.runtime.execute(name, flag=True)
            self.assertFalse(result.success)
            with self.assertRaisesRegex(RuntimeError, "does not accept arguments"):
                await self.robot.execute_loco_action(name, {"flag": True})
            getattr(self.sport, name).assert_not_called()
