from __future__ import annotations

import asyncio
import threading
import unittest
from unittest.mock import Mock, patch

from robot import Go2Bindings, RobotCommandError, UnitreeGo2Adapter, UnitreeGo2Config


class Go2AdapterTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.channel = Mock()
        self.sport = Mock()
        self.sport.move.return_value = 0
        self.sport.stop_move.return_value = 0
        self.factory = Mock(return_value=self.sport)
        self.robot = UnitreeGo2Adapter(
            UnitreeGo2Config(network_interface="eth0", domain_id=3),
            bindings=Go2Bindings(self.channel, self.factory),
        )

    async def asyncTearDown(self) -> None:
        await self.robot.close()

    async def test_connect_close_are_idempotent_and_do_not_move(self) -> None:
        self.assertFalse((await self.robot.get_state()).connected)
        await self.robot.connect()
        await self.robot.connect()
        self.channel.initialize.assert_called_once_with(3, "eth0")
        self.sport.set_timeout.assert_called_once_with(2.0)
        self.sport.init.assert_called_once_with()
        state = await self.robot.get_state()
        self.assertTrue(state.hardware and state.connected)
        self.assertFalse(state.details["telemetry_available"])
        self.sport.move.assert_not_called()
        self.sport.stand_up.assert_not_called()
        await self.robot.close()
        await self.robot.close()
        self.channel.release.assert_called_once_with()
        self.sport.stop_move.assert_not_called()
        self.assertFalse(self.robot.connected)

    async def test_connect_failure_releases_dds_and_allows_retry(self) -> None:
        self.sport.init.side_effect = RuntimeError("init failed")
        with self.assertRaisesRegex(RobotCommandError, "init failed"):
            await self.robot.connect()
        self.assertFalse(self.robot.connected)
        self.channel.release.assert_called_once_with()
        self.sport.init.side_effect = None
        await self.robot.connect()
        self.assertTrue(self.robot.connected)

    async def test_channel_init_failure_does_not_release_unowned_channel(self) -> None:
        self.channel.initialize.side_effect = RuntimeError("DDS failed")
        with self.assertRaisesRegex(RobotCommandError, "DDS failed"):
            await self.robot.connect()
        self.channel.release.assert_not_called()
        self.factory.assert_not_called()

    async def test_move_uses_go2_three_argument_signature_and_stop(self) -> None:
        await self.robot.connect()
        await self.robot.move_velocity(0.2, -0.1, 0.4)
        self.sport.move.assert_called_once_with(0.2, -0.1, 0.4)
        await self.robot.stop()
        self.sport.stop_move.assert_called_once_with()

    async def test_disconnected_commands_fail(self) -> None:
        for command in (
            self.robot.stop(),
            self.robot.move_velocity(0.1, 0, 0),
            self.robot.execute_loco_action("stand_up"),
        ):
            with self.assertRaisesRegex(RobotCommandError, "not connected"):
                await command
        self.sport.move.assert_not_called()

    async def test_invalid_velocities_never_reach_sdk(self) -> None:
        await self.robot.connect()
        for values in (
            (float("nan"), 0, 0),
            (0, float("inf"), 0),
            (0, 0, float("-inf")),
            (0.31, 0, 0),
            (0, -0.31, 0),
            (0, 0, 0.61),
            (True, 0, 0),
        ):
            with self.subTest(values=values):
                with self.assertRaises(RobotCommandError):
                    await self.robot.move_velocity(*values)
        self.sport.move.assert_not_called()

    async def test_sdk_errors_and_invalid_status_are_not_success(self) -> None:
        await self.robot.connect()
        for status in (7301, -1, None, False, "0"):
            self.sport.move.return_value = status
            with self.subTest(status=status):
                with self.assertRaises(RobotCommandError):
                    await self.robot.move_velocity(0.1, 0, 0)
        self.sport.stop_move.side_effect = OSError("transport lost")
        with self.assertRaisesRegex(RobotCommandError, "transport lost"):
            await self.robot.stop()

    async def test_go2_postures_use_exact_methods(self) -> None:
        await self.robot.connect()
        for action in (
            "stand_up",
            "stand_down",
            "balance_stand",
            "sit",
            "rise_sit",
            "stop_move",
            "damp",
            "recovery_stand",
            "hello",
            "stretch",
        ):
            method = getattr(self.sport, action)
            method.return_value = 0
            await self.robot.execute_loco_action(action)
            method.assert_called_once_with()

    async def test_g1_actions_unknown_arguments_and_missing_methods_rejected(
        self,
    ) -> None:
        await self.robot.connect()
        for action in ("start", "zero_torque", "squat", "set_fsm_id", "__init__"):
            with self.assertRaisesRegex(RobotCommandError, "unsupported Go2"):
                await self.robot.execute_loco_action(action)
        with self.assertRaisesRegex(RobotCommandError, "does not accept"):
            await self.robot.execute_loco_action("stand_up", {"unexpected": 1})
        self.sport.stand_up.assert_not_called()
        self.sport.hello = None
        with self.assertRaisesRegex(RobotCommandError, "do not provide hello"):
            await self.robot.execute_loco_action("hello")

    async def test_arm_operations_fail_and_release_is_a_noop(self) -> None:
        await self.robot.connect()
        self.sport.reset_mock()
        for command in (
            self.robot.wave("right"),
            self.robot.wait_for_wave_completion("right", 1),
            self.robot.execute_arm_action(27, "handshake"),
            self.robot.execute_custom_arm_action("custom"),
            self.robot.stop_custom_arm_action(),
            self.robot.wait_for_arm_action_completion(25, "wave", 1),
        ):
            with self.assertRaisesRegex(RobotCommandError, "does not support G1 arm"):
                await command
        await self.robot.release_arm()
        self.assertEqual(self.sport.mock_calls, [])

    async def test_cancelled_native_move_finishes_before_stop(self) -> None:
        await self.robot.connect()
        entered = threading.Event()
        release = threading.Event()
        order: list[str] = []

        def move(*args: object) -> int:
            entered.set()
            if not release.wait(2):
                raise TimeoutError("test move was not released")
            order.append("move_finished")
            return 0

        def stop() -> int:
            order.append("stop")
            return 0

        self.sport.move.side_effect = move
        self.sport.stop_move.side_effect = stop
        moving = asyncio.create_task(self.robot.move_velocity(0.1, 0, 0))
        stopping = None
        try:
            self.assertTrue(await asyncio.to_thread(entered.wait, 1))
            moving.cancel()
            stopping = asyncio.create_task(self.robot.stop())
            await asyncio.sleep(0.02)
            self.assertEqual(order, [])
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await moving
            await stopping
            self.assertEqual(order, ["move_finished", "stop"])
        finally:
            release.set()
            await asyncio.gather(
                moving, *([stopping] if stopping else []), return_exceptions=True
            )

    async def test_bindings_are_loaded_lazily_and_missing_sdk_is_explained(
        self,
    ) -> None:
        with patch("robot.go2_adapter.importlib.import_module") as load:
            robot = UnitreeGo2Adapter()
            load.assert_not_called()
            load.side_effect = ImportError("not installed")
            with self.assertRaisesRegex(RobotCommandError, "robot.go2.SportClient"):
                await robot.connect()

    def test_config_rejects_invalid_limits(self) -> None:
        for kwargs in (
            {"domain_id": -1},
            {"domain_id": True},
            {"timeout_s": 0},
            {"timeout_s": float("nan")},
            {"max_forward_m_s": float("inf")},
            {"max_lateral_m_s": -1},
            {"max_yaw_rad_s": True},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    UnitreeGo2Config(**kwargs)


if __name__ == "__main__":
    unittest.main()
