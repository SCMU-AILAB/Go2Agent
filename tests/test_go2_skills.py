from __future__ import annotations

import time
import unittest

from core.runtime import SkillRuntime
from robot import (
    UnitreeGo2Adapter,
    create_hardware_robot,
    create_simulated_robot,
)
from skills import (
    build_go2_autonomy_skills,
    register_g1_skills,
    register_go2_skills,
)
from skills.motions.go2_motion import GO2_VELOCITY_REFRESH_S


class RecordingGo2Adapter:
    def __init__(self) -> None:
        self.events: list[tuple[str, object]] = []
        self._connected = True

    @property
    def connected(self) -> bool:
        return self._connected

    async def get_state(self):
        from robot.base import RobotState

        return RobotState(hardware=True, connected=self._connected, details={})

    async def stop(self) -> None:
        self.events.append(("stop", None))

    async def wave(self, arm: str) -> None:
        raise RuntimeError("unsupported")

    async def wait_for_wave_completion(self, arm: str, timeout_s: float):
        raise RuntimeError("unsupported")

    async def execute_arm_action(self, action_id: int, action_name: str) -> None:
        raise RuntimeError("unsupported")

    async def execute_custom_arm_action(self, action_name: str) -> None:
        raise RuntimeError("unsupported")

    async def stop_custom_arm_action(self) -> None:
        raise RuntimeError("unsupported")

    async def wait_for_arm_action_completion(
        self, action_id: int, action_name: str, timeout_s: float
    ):
        raise RuntimeError("unsupported")

    async def release_arm(self) -> None:
        self.events.append(("release_arm", None))

    async def execute_loco_action(
        self, action: str, arguments: dict[str, object] | None = None
    ) -> None:
        self.events.append(("loco", (action, dict(arguments or {}))))

    async def move_velocity(
        self, forward_m_s: float, lateral_m_s: float, yaw_rad_s: float
    ) -> None:
        self.events.append(("move", (forward_m_s, lateral_m_s, yaw_rad_s)))


class Go2FactoryTests(unittest.TestCase):
    def test_factory_builds_go2_and_g1_hardware(self) -> None:
        go2 = create_hardware_robot("go2", network_interface="eth0", domain_id=1)
        self.assertIsInstance(go2, UnitreeGo2Adapter)
        self.assertEqual(go2.config.network_interface, "eth0")
        self.assertEqual(go2.config.domain_id, 1)

        g1 = create_hardware_robot("g1", network_interface="eth0")
        self.assertEqual(type(g1).__name__, "UnitreeG1Adapter")

        sim = create_simulated_robot("go2")
        self.assertFalse(sim.events)

    def test_factory_rejects_unknown_model(self) -> None:
        with self.assertRaises(ValueError):
            create_hardware_robot("a1")  # type: ignore[arg-type]


class Go2CatalogTests(unittest.TestCase):
    def test_autonomy_catalog_excludes_g1_only_skills(self) -> None:
        names = {skill.metadata.name for skill in build_go2_autonomy_skills()}
        self.assertIn("stand_up", names)
        self.assertIn("hello", names)
        self.assertIn("wave", names)
        self.assertIn("move", names)
        self.assertIn("stop", names)
        self.assertNotIn("handshake", names)
        self.assertNotIn("high_five", names)
        self.assertNotIn("squat", names)
        self.assertNotIn("start", names)
        self.assertNotIn("damp", names)
        self.assertNotIn("recovery_stand", names)

    def test_operator_catalog_adds_damp_and_recovery(self) -> None:
        robot = RecordingGo2Adapter()
        runtime = SkillRuntime(robot)  # type: ignore[arg-type]
        register_go2_skills(runtime, include_operator_only=True)
        names = {skill.metadata.name for skill in runtime.registry.list()}
        self.assertIn("damp", names)
        self.assertIn("recovery_stand", names)
        damp = runtime.registry.get("damp")
        self.assertIn("operator_only", damp.metadata.tags)
        self.assertIn("dangerous", damp.metadata.tags)

    def test_g1_catalog_does_not_register_go2_hello(self) -> None:
        robot = RecordingGo2Adapter()
        runtime = SkillRuntime(robot)  # type: ignore[arg-type]
        register_g1_skills(runtime)
        names = {skill.metadata.name for skill in runtime.registry.list()}
        self.assertNotIn("hello", names)
        self.assertIn("wave", names)


class Go2SkillExecutionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.robot = RecordingGo2Adapter()
        self.runtime = SkillRuntime(self.robot)  # type: ignore[arg-type]
        register_go2_skills(self.runtime)

    async def test_wave_skill_uses_native_hello(self) -> None:
        result = await self.runtime.execute("wave")
        self.assertTrue(result.success)
        self.assertEqual(self.robot.events, [("loco", ("hello", {}))])

    async def test_stand_up_maps_to_sport_client(self) -> None:
        result = await self.runtime.execute("stand_up")
        self.assertTrue(result.success)
        self.assertEqual(self.robot.events, [("loco", ("stand_up", {}))])

    async def test_stop_stops_and_releases_arm(self) -> None:
        result = await self.runtime.execute("stop")
        self.assertTrue(result.success)
        self.assertEqual(self.robot.events, [("stop", None), ("release_arm", None)])

    async def test_move_refreshes_velocity_periodically_and_stops(self) -> None:
        started = time.monotonic()
        result = await self.runtime.execute(
            "move",
            forward_m_s=0.1,
            lateral_m_s=0.0,
            yaw_rad_s=0.0,
            duration_s=0.12,
        )
        elapsed = time.monotonic() - started
        self.assertTrue(result.success)
        move_events = [e for e in self.robot.events if e[0] == "move"]
        stop_events = [e for e in self.robot.events if e[0] == "stop"]
        self.assertGreaterEqual(len(move_events), 2)
        self.assertGreaterEqual(
            result.data.get("velocity_refresh_s"), GO2_VELOCITY_REFRESH_S - 1e-9
        )
        self.assertEqual(len(stop_events), 1)
        self.assertEqual(self.robot.events[-1], ("stop", None))
        self.assertGreaterEqual(elapsed, 0.1)

    async def test_move_cleanup_stops_after_failure(self) -> None:
        async def boom(*args: object, **kwargs: object) -> None:
            raise RuntimeError("adapter exploded")

        self.robot.move_velocity = boom  # type: ignore[method-assign]
        result = await self.runtime.execute(
            "move",
            forward_m_s=0.1,
            duration_s=1.0,
        )
        self.assertFalse(result.success)
        self.assertEqual(self.robot.events[-1], ("stop", None))

    async def test_directional_move_uses_refresh_loop(self) -> None:
        result = await self.runtime.execute("move_forward", distance_m=0.1)
        self.assertTrue(result.success)
        moves = [e for e in self.robot.events if e[0] == "move"]
        self.assertGreaterEqual(len(moves), 2)
        self.assertEqual(self.robot.events[-1], ("stop", None))


if __name__ == "__main__":
    unittest.main()
