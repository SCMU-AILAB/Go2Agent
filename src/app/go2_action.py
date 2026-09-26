"""Run one Go2 skill directly, without an Agent or language model."""

from __future__ import annotations

import argparse
import asyncio
import json

from core.models import SkillResult
from core.runtime import SkillRuntime
from core.types import FailureCode, SkillStatus
from robot import (
    RobotAdapter,
    RobotCommandError,
    create_hardware_robot,
    create_simulated_robot,
)
from skills import register_go2_skills


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("skill", nargs="?", default="hello")
    parser.add_argument(
        "--arguments",
        default="{}",
        help="JSON object passed to the skill (for example: '{\"flag\":true}')",
    )
    parser.add_argument("--hardware", action="store_true")
    parser.add_argument("--network", default="")
    parser.add_argument("--domain-id", type=int, default=0)
    return parser.parse_args()


async def run_action(
    skill: str = "hello",
    *,
    arguments: dict[str, object] | None = None,
    hardware: bool = False,
    network_interface: str = "",
    domain_id: int = 0,
) -> SkillResult:
    hardware_robot = (
        create_hardware_robot(
            "go2", network_interface=network_interface, domain_id=domain_id
        )
        if hardware
        else None
    )
    robot: RobotAdapter = hardware_robot or create_simulated_robot("go2")
    runtime = SkillRuntime(robot)
    register_go2_skills(runtime)
    try:
        if hardware_robot is not None:
            await hardware_robot.connect()
        return await runtime.execute(skill, **(arguments or {}))
    finally:
        if hardware_robot is not None:
            await hardware_robot.close()


async def _run(args: argparse.Namespace) -> int:
    try:
        try:
            arguments = json.loads(args.arguments)
        except json.JSONDecodeError as exc:
            raise ValueError(f"--arguments must be a JSON object: {exc}") from exc
        if not isinstance(arguments, dict):
            raise TypeError("--arguments must be a JSON object")
        result = await run_action(
            args.skill,
            arguments=arguments,
            hardware=args.hardware,
            network_interface=args.network,
            domain_id=args.domain_id,
        )
    except (RobotCommandError, RuntimeError, TypeError, ValueError) as exc:
        result = SkillResult.fail(
            SkillStatus.FAILED,
            str(exc),
            failure_code=FailureCode.ROBOT_ERROR,
        )
    print(json.dumps(result.to_dict(), ensure_ascii=False, default=str))
    return 0 if result.success else 1


def main() -> int:
    return asyncio.run(_run(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
