from __future__ import annotations

import asyncio
import unittest

from core.context import SkillContext
from core.models import SkillArgs, SkillMetadata, SkillResult
from core.registry import SkillRegistry
from core.resources import ResourceManager
from core.runtime import SkillRuntime
from core.skill import RobotSkill
from core.types import FailureCode, SkillStatus
from robot import SimulatedRobotAdapter
from robot.base import RobotCommandError, RobotState
from skills import register_go2_skills
from tests.go2_helpers import go2_skill


class FakeRobotAdapter:
    def __init__(
        self,
        *,
        hardware: bool = False,
        connected: bool = False,
        wave_error: bool = False,
        move_error: bool = False,
    ) -> None:
        self.hardware = hardware
        self.connected = connected
        self.wave_error = wave_error
        self.move_error = move_error
        self.waves: list[str] = []
        self.velocity_commands: list[tuple[float, float, float]] = []
        self.stop_count = 0

    async def get_state(self) -> RobotState:
        return RobotState(hardware=self.hardware, connected=self.connected)

    async def stop(self) -> None:
        self.stop_count += 1

    async def execute_loco_action(self, action: str, arguments=None) -> None:
        if self.wave_error:
            raise RobotCommandError("hello command rejected")
        self.waves.append(action)

    async def move_velocity(
        self,
        forward_m_s: float,
        lateral_m_s: float,
        yaw_rad_s: float,
    ) -> None:
        self.velocity_commands.append((forward_m_s, lateral_m_s, yaw_rad_s))
        if self.move_error:
            raise RobotCommandError("move command rejected")


class EmptyArgs(SkillArgs):
    pass


class SlowSkill(RobotSkill[EmptyArgs]):
    metadata = SkillMetadata(
        name="slow",
        description="A skill used to verify timeouts.",
        timeout_s=0.01,
    )
    args_model = EmptyArgs

    async def execute(self, ctx: SkillContext, args: EmptyArgs) -> SkillResult:
        await asyncio.sleep(1)
        return SkillResult.ok()


class ResourceSkill(RobotSkill[EmptyArgs]):
    metadata = SkillMetadata(
        name="resource_skill",
        description="A skill used to verify resource locking.",
        required_resources=("shared_resource",),
        timeout_s=1.0,
    )
    args_model = EmptyArgs

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.execution_count = 0

    async def execute(self, ctx: SkillContext, args: EmptyArgs) -> SkillResult:
        self.execution_count += 1
        self.started.set()
        await self.release.wait()
        return SkillResult.ok()


class VerificationSkill(RobotSkill[EmptyArgs]):
    metadata = SkillMetadata(
        name="verification_skill",
        description="A skill used to verify lifecycle ordering.",
        timeout_s=0.05,
    )
    args_model = EmptyArgs

    def __init__(self, *, block_verification: bool = False) -> None:
        self.block_verification = block_verification
        self.phases: list[str] = []

    async def execute(self, ctx: SkillContext, args: EmptyArgs) -> SkillResult:
        self.phases.append("execute")
        return SkillResult.ok("command accepted")

    async def verify(
        self,
        ctx: SkillContext,
        args: EmptyArgs,
        result: SkillResult,
    ) -> SkillResult:
        self.phases.append("verify")
        if self.block_verification:
            await asyncio.sleep(1)
        result.verification = {"completed": True}
        return result

    async def cleanup(self, ctx: SkillContext, args: EmptyArgs) -> None:
        self.phases.append("cleanup")


class SkillRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancel_waiting_for_second_resource_releases_first(self) -> None:
        resources = ResourceManager()
        async with resources.acquire(("b",)):
            async def wait_for_b() -> None:
                async with resources.acquire(("a", "b")):
                    pass

            waiter = asyncio.create_task(wait_for_b())
            await asyncio.sleep(0)
            self.assertTrue(resources._locks["a"].locked())
            waiter.cancel()
            await asyncio.gather(waiter, return_exceptions=True)
        async with asyncio.timeout(0.1):
            async with resources.acquire(("a",)):
                pass

    async def test_stop_preempts_active_motion(self) -> None:
        robot = SimulatedRobotAdapter()
        runtime = SkillRuntime(robot)
        register_go2_skills(runtime)
        moving = asyncio.create_task(runtime.execute("move", duration_s=1.0))
        await asyncio.sleep(0.04)

        stopped = await asyncio.wait_for(runtime.execute("stop"), 0.3)
        self.assertTrue(stopped.success)
        self.assertTrue(moving.done())
        after_stop = len(robot.events)
        await asyncio.sleep(0.05)
        self.assertFalse(any(name == "move_velocity" for name, _ in robot.events[after_stop:]))

    async def test_wave_runs_through_runtime(self) -> None:
        robot = FakeRobotAdapter()
        runtime = SkillRuntime(robot)
        runtime.register(go2_skill("wave"))

        result = await runtime.execute("wave")

        self.assertTrue(result.success)
        self.assertEqual(result.status, SkillStatus.SUCCEEDED)
        self.assertEqual(result.data["sdk_action"], "hello")
        self.assertEqual(result.verification, {})
        self.assertEqual(robot.waves, ["hello"])
        self.assertIsNotNone(result.duration_s)

    async def test_wave_rejects_unsupported_arm(self) -> None:
        robot = FakeRobotAdapter()
        runtime = SkillRuntime(robot)
        runtime.register(go2_skill("wave"))

        result = await runtime.execute("wave", arm="left")

        self.assertFalse(result.success)
        self.assertEqual(result.failure_code, FailureCode.INVALID_ARGUMENTS)
        self.assertEqual(robot.waves, [])

    async def test_wave_requires_connected_hardware(self) -> None:
        robot = FakeRobotAdapter(hardware=True, connected=False)
        runtime = SkillRuntime(robot)
        runtime.register(go2_skill("wave"))

        result = await runtime.execute("wave")

        self.assertEqual(result.status, SkillStatus.PRECONDITION_FAILED)
        self.assertEqual(result.failure_code, FailureCode.PRECONDITION_NOT_MET)
        self.assertEqual(robot.waves, [])

    async def test_robot_error_is_structured_without_implicit_stop(self) -> None:
        robot = FakeRobotAdapter(wave_error=True)
        runtime = SkillRuntime(robot)
        runtime.register(go2_skill("wave"))

        result = await runtime.execute("wave")

        self.assertEqual(result.failure_code, FailureCode.ROBOT_ERROR)
        self.assertEqual(robot.stop_count, 0)

    async def test_verify_runs_before_cleanup(self) -> None:
        runtime = SkillRuntime(FakeRobotAdapter())
        skill = VerificationSkill()
        runtime.register(skill)

        result = await runtime.execute("verification_skill")

        self.assertTrue(result.success)
        self.assertEqual(result.verification, {"completed": True})
        self.assertEqual(skill.phases, ["execute", "verify", "cleanup"])

    async def test_verification_uses_skill_execution_timeout(self) -> None:
        runtime = SkillRuntime(FakeRobotAdapter())
        skill = VerificationSkill(block_verification=True)
        runtime.register(skill)

        result = await runtime.execute("verification_skill")

        self.assertEqual(result.status, SkillStatus.TIMEOUT)
        self.assertEqual(skill.phases, ["execute", "verify", "cleanup"])

    async def test_unknown_skill_returns_structured_failure(self) -> None:
        runtime = SkillRuntime(FakeRobotAdapter())

        result = await runtime.execute("missing")

        self.assertFalse(result.success)
        self.assertEqual(result.failure_code, FailureCode.INVALID_ARGUMENTS)

    async def test_timeout_does_not_assume_a_stop_policy(self) -> None:
        robot = FakeRobotAdapter()
        runtime = SkillRuntime(robot)
        runtime.register(SlowSkill())

        result = await runtime.execute("slow")

        self.assertEqual(result.status, SkillStatus.TIMEOUT)
        self.assertEqual(robot.stop_count, 0)

    async def test_move_backward_is_bounded_and_always_stops(self) -> None:
        robot = FakeRobotAdapter()
        runtime = SkillRuntime(robot)
        runtime.register(go2_skill("move_backward"))

        result = await runtime.execute("move_backward", distance_m=0.05)

        self.assertTrue(result.success)
        self.assertGreater(len(robot.velocity_commands), 1)
        self.assertTrue(
            all(command == (-0.1, 0.0, 0.0) for command in robot.velocity_commands)
        )
        self.assertEqual(robot.stop_count, 1)

    async def test_move_backward_stops_after_command_error(self) -> None:
        robot = FakeRobotAdapter(move_error=True)
        runtime = SkillRuntime(robot)
        runtime.register(go2_skill("move_backward"))

        result = await runtime.execute("move_backward")

        self.assertEqual(result.failure_code, FailureCode.ROBOT_ERROR)
        self.assertEqual(robot.stop_count, 1)

    async def test_move_backward_rejects_unsafe_distance(self) -> None:
        robot = FakeRobotAdapter()
        runtime = SkillRuntime(robot)
        runtime.register(go2_skill("move_backward"))

        result = await runtime.execute("move_backward", distance_m=2.0)

        self.assertEqual(result.failure_code, FailureCode.INVALID_ARGUMENTS)
        self.assertEqual(robot.velocity_commands, [])
        self.assertEqual(robot.stop_count, 0)

    async def test_required_resources_serialize_concurrent_invocations(self) -> None:
        runtime = SkillRuntime(FakeRobotAdapter())
        skill = ResourceSkill()
        runtime.register(skill)

        first = asyncio.create_task(runtime.execute("resource_skill"))
        await asyncio.wait_for(skill.started.wait(), timeout=0.2)
        second = asyncio.create_task(runtime.execute("resource_skill"))
        await asyncio.sleep(0)
        self.assertEqual(skill.execution_count, 1)

        skill.release.set()
        first_result, second_result = await asyncio.gather(first, second)
        self.assertTrue(first_result.success)
        self.assertTrue(second_result.success)
        self.assertEqual(skill.execution_count, 2)

    def test_registry_rejects_duplicate_names(self) -> None:
        registry = SkillRegistry()
        registry.register(go2_skill("wave"))

        with self.assertRaisesRegex(ValueError, "duplicate skill"):
            registry.register(go2_skill("wave"))
