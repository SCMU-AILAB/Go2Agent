"""Public facade for registering and invoking robot skills."""

import asyncio

from robot.base import RobotAdapter

from .executor import SkillExecutor
from .models import SkillArgs, SkillInvocation, SkillResult
from .registry import SkillRegistry
from .resources import ResourceManager
from .skill import RobotSkill


class SkillRuntime:
    def __init__(self, robot: RobotAdapter) -> None:
        self.robot = robot
        self.registry = SkillRegistry()
        self.resources = ResourceManager()
        self.executor = SkillExecutor(self.registry, robot, self.resources)
        self._active_tasks: dict[asyncio.Task[SkillResult], str] = {}

    def register[ArgsT: SkillArgs](self, skill: RobotSkill[ArgsT]) -> None:
        self.registry.register(skill)

    async def execute(
        self,
        skill_name: str,
        /,
        **arguments: object,
    ) -> SkillResult:
        if skill_name in {"stop", "stop_move"}:
            await self.cancel_active()
        invocation = SkillInvocation(skill_name=skill_name, arguments=arguments)
        task = asyncio.current_task()
        if task is not None:
            self._active_tasks[task] = skill_name
        try:
            return await self.executor.execute(invocation)
        finally:
            if task is not None:
                self._active_tasks.pop(task, None)

    async def cancel_active(self) -> None:
        current = asyncio.current_task()
        tasks = [
            task
            for task, skill_name in self._active_tasks.items()
            if task is not current
            and skill_name not in {"stop", "stop_move"}
            and not task.done()
        ]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
