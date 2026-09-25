from __future__ import annotations

import asyncio
import time
import unittest

from core.runtime import SkillRuntime
from perception import CameraFrame, PerceptionResult
from robot import RobotState, SimulatedRobotAdapter
from skills import register_go2_skills
from skills.motions.go2_follow import FollowPersonSkill


def frame(
    *,
    count: int = 1,
    distance: float | None = 2.2,
    center: float | None = 0.5,
    obstacle: float | None = 2.0,
    age_s: float = 0.0,
) -> CameraFrame:
    now = time.monotonic() - age_s
    return CameraFrame(
        observed_at_s=now,
        rgb=None,
        depth=None,
        observation=PerceptionResult(
            observed_at_s=now,
            person_count=count,
            nearest_person_distance_m=distance,
            person_center_x=center,
        ),
        nearest_obstacle_distance_m=obstacle,
    )


class Go2FollowTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.robot = SimulatedRobotAdapter()
        self.runtime = SkillRuntime(self.robot)
        register_go2_skills(self.runtime)
        skill = self.runtime.registry.get("follow_person")
        self.assertIsInstance(skill, FollowPersonSkill)
        self.skill = skill

    async def test_requires_fresh_unambiguous_depth_target(self) -> None:
        result = await self.runtime.execute("follow_person")
        self.assertFalse(result.success)
        for unsafe in (
            frame(count=2),
            frame(distance=None),
            frame(center=None),
            frame(obstacle=0.5),
            frame(obstacle=float("nan")),
            frame(age_s=2.0),
        ):
            self.skill.observe_frame(unsafe)
            result = await self.runtime.execute("follow_person")
            self.assertFalse(result.success)
        self.assertFalse(any(name == "move_velocity" for name, _ in self.robot.events))

    async def test_follows_then_stops_at_target_distance(self) -> None:
        self.skill.observe_frame(frame())
        task = asyncio.create_task(self.runtime.execute("follow_person"))
        try:
            await asyncio.sleep(0.08)
            self.assertTrue(
                any(name == "move_velocity" and value[0] == 0.15 for name, value in self.robot.events)
            )
            self.skill.observe_frame(frame(distance=1.5))
            await asyncio.sleep(0.06)
            self.assertIn(("stop", None), self.robot.events)
            self.skill.observe_frame(frame(count=0, distance=None, center=None))
            result = await asyncio.wait_for(task, 1.0)
            self.assertFalse(result.success)
            self.assertIn("person lost", result.message)
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    async def test_tolerates_one_hog_miss_but_not_a_stale_target(self) -> None:
        self.skill.observe_frame(frame())
        task = asyncio.create_task(self.runtime.execute("follow_person"))
        try:
            await asyncio.sleep(0.05)
            self.skill.observe_frame(frame(count=0, distance=None, center=None))
            await asyncio.sleep(0.05)
            self.assertFalse(task.done())
            self.skill.observe_frame(
                frame(count=0, distance=None, center=None, age_s=0.4)
            )
            result = await asyncio.wait_for(task, 1.0)
            self.assertFalse(result.success)
            self.assertIn("person lost", result.message)
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    async def test_lost_or_ambiguous_target_and_obstacle_stop_motion(self) -> None:
        for unsafe in (
            frame(count=2),
            frame(obstacle=0.8),
        ):
            with self.subTest(unsafe=unsafe):
                self.skill.observe_frame(frame())
                task = asyncio.create_task(self.runtime.execute("follow_person"))
                await asyncio.sleep(0.05)
                self.skill.observe_frame(unsafe)
                result = await asyncio.wait_for(task, 1.0)
                self.assertFalse(result.success)
                self.assertEqual(self.robot.events[-1], ("stop", None))

    async def test_cancellation_stops_robot(self) -> None:
        self.skill.observe_frame(frame())
        task = asyncio.create_task(self.runtime.execute("follow_person"))
        await asyncio.sleep(0.05)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        self.assertEqual(self.robot.events[-1], ("stop", None))

    async def test_hardware_requires_fresh_telemetry(self) -> None:
        class NoTelemetryRobot(SimulatedRobotAdapter):
            async def get_state(self) -> RobotState:
                return RobotState(
                    hardware=True, connected=True,
                    details={"telemetry_available": False},
                )

        robot = NoTelemetryRobot()
        runtime = SkillRuntime(robot)
        register_go2_skills(runtime)
        skill = runtime.registry.get("follow_person")
        self.assertIsInstance(skill, FollowPersonSkill)
        skill.observe_frame(frame())
        result = await runtime.execute("follow_person")
        self.assertFalse(result.success)
        self.assertIn("telemetry", result.message)
        self.assertFalse(any(name == "move_velocity" for name, _ in robot.events))

    async def test_turns_toward_target_then_stops_on_large_track_jump(self) -> None:
        self.skill.observe_frame(frame(center=0.8))
        task = asyncio.create_task(self.runtime.execute("follow_person"))
        await asyncio.sleep(0.05)
        self.assertTrue(any(
            name == "move_velocity" and value[0] == 0.0 and value[2] < 0
            for name, value in self.robot.events
        ))
        self.skill.observe_frame(frame(center=0.1))
        result = await asyncio.wait_for(task, 1.0)
        self.assertFalse(result.success)
        self.assertIn("target changed abruptly", result.message)
        self.assertEqual(self.robot.events[-1], ("stop", None))


if __name__ == "__main__":
    unittest.main()
