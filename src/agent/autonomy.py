"""Execute event decisions through SkillRuntime and optional speech output."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from adapters.host_audio import SpeechOutput
from core.models import SkillResult
from core.runtime import SkillRuntime
from perception import EventDetector, PerceptionResult, WorldEvent, WorldState

from .decision import AgentDecision, DecisionAgent


@dataclass(frozen=True, slots=True)
class DecisionOutcome:
    event: WorldEvent
    decision: AgentDecision
    skill_result: SkillResult | None = None
    speech_spoken: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "event": self.event.to_dict(),
            "decision": self.decision.to_dict(),
            "skill_result": (
                self.skill_result.to_dict() if self.skill_result is not None else None
            ),
            "speech_spoken": self.speech_spoken,
        }


@dataclass(frozen=True, slots=True)
class _DecisionRequest:
    event: WorldEvent
    decision: AgentDecision


class AutonomousDecisionLoop:
    def __init__(
        self,
        runtime: SkillRuntime,
        decision_agent: DecisionAgent,
        *,
        world_state: WorldState | None = None,
        event_detector: EventDetector | None = None,
        speech: SpeechOutput | None = None,
        queue_size: int = 32,
    ) -> None:
        if queue_size <= 0:
            raise ValueError("queue_size must be greater than zero")
        self.runtime = runtime
        self.decision_agent = decision_agent
        self.world_state = world_state or WorldState()
        self.event_detector = event_detector or EventDetector()
        self.speech = speech
        self._lock = asyncio.Lock()
        self._event_queue: asyncio.Queue[WorldEvent] = asyncio.Queue(maxsize=queue_size)
        self._decision_queue: asyncio.Queue[_DecisionRequest] = asyncio.Queue(
            maxsize=queue_size
        )
        self._outcome_queue: asyncio.Queue[DecisionOutcome] = asyncio.Queue(
            maxsize=queue_size
        )
        self._error_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=queue_size)
        self._worker_tasks: tuple[asyncio.Task[None], ...] = ()

    @property
    def running(self) -> bool:
        return bool(self._worker_tasks)

    async def start_workers(self) -> None:
        if self.running:
            return
        self._worker_tasks = (
            asyncio.create_task(self._decision_worker(), name="decision-worker"),
            asyncio.create_task(self._execution_worker(), name="execution-worker"),
        )

    async def stop_workers(self) -> None:
        tasks = self._worker_tasks
        self._worker_tasks = ()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def observe(self, observation: PerceptionResult) -> tuple[WorldEvent, ...]:
        """Update WorldState and enqueue sparse events without waiting for the Agent."""
        async with self._lock:
            events = self.event_detector.update(observation, self.world_state)
        for event in events:
            self._put_latest(self._event_queue, event)
        return events

    async def next_outcome(self, timeout: float | None = None) -> DecisionOutcome:
        if timeout is None:
            return await self._outcome_queue.get()
        async with asyncio.timeout(timeout):
            return await self._outcome_queue.get()

    def drain_outcomes(self) -> tuple[DecisionOutcome, ...]:
        outcomes: list[DecisionOutcome] = []
        while True:
            try:
                outcomes.append(self._outcome_queue.get_nowait())
            except asyncio.QueueEmpty:
                return tuple(outcomes)

    def drain_errors(self) -> tuple[str, ...]:
        errors: list[str] = []
        while True:
            try:
                errors.append(self._error_queue.get_nowait())
            except asyncio.QueueEmpty:
                return tuple(errors)

    async def process(
        self,
        observation: PerceptionResult,
    ) -> tuple[DecisionOutcome, ...]:
        async with self._lock:
            events = self.event_detector.update(observation, self.world_state)
        outcomes: list[DecisionOutcome] = []
        for event in events:
            async with self._lock:
                world_state = self.world_state.to_dict()
            decision = await self.decision_agent.decide(event, world_state)
            outcomes.append(await self._execute(event, decision))
        return tuple(outcomes)

    async def _decision_worker(self) -> None:
        while True:
            event = await self._event_queue.get()
            try:
                async with self._lock:
                    world_state = self.world_state.to_dict()
                decision = await self.decision_agent.decide(event, world_state)
                self._put_latest(
                    self._decision_queue,
                    _DecisionRequest(event=event, decision=decision),
                )
            except Exception as exc:  # noqa: BLE001 - worker must remain alive
                self._put_latest(self._error_queue, str(exc))

    async def _execution_worker(self) -> None:
        while True:
            item = await self._decision_queue.get()
            try:
                outcome = await self._execute(item.event, item.decision)
                self._put_latest(self._outcome_queue, outcome)
            except Exception as exc:  # noqa: BLE001 - worker must remain alive
                self._put_latest(self._error_queue, str(exc))

    @staticmethod
    def _put_latest[T](queue: asyncio.Queue[T], item: T) -> None:
        try:
            queue.put_nowait(item)
        except asyncio.QueueFull:
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            queue.put_nowait(item)

    async def _execute(
        self,
        event: WorldEvent,
        decision: AgentDecision,
    ) -> DecisionOutcome:
        async with self._lock:
            stale = (
                event.type.value == "person_entered"
                and (
                    not self.world_state.person_visible
                    or self.world_state.person_greeted
                )
            ) or (
                event.type.value == "person_too_close"
                and (
                    not self.world_state.person_visible
                    or not self.world_state.person_too_close
                )
            )
        if stale:
            return DecisionOutcome(
                event=event,
                decision=AgentDecision(
                    action="ignore",
                    reason="event is no longer current",
                ),
            )

        skill_result: SkillResult | None = None
        speech_spoken = False
        if decision.action == "execute_and_speak":
            if decision.skill is None:
                raise RuntimeError("validated decision is missing a skill")
            if decision.speech is None:
                raise RuntimeError("validated decision is missing speech")
            if self.speech is None:
                skill_result = await self.runtime.execute(
                    decision.skill,
                    **decision.arguments,
                )
            else:
                skill_task = asyncio.create_task(
                    self.runtime.execute(
                        decision.skill,
                        **decision.arguments,
                    ),
                    name="autonomy-skill-execution",
                )
                speech_task = asyncio.create_task(
                    self.speech.speak(decision.speech),
                    name="autonomy-speech-output",
                )
                skill_outcome, speech_outcome = await asyncio.gather(
                    skill_task,
                    speech_task,
                    return_exceptions=True,
                )
                if isinstance(skill_outcome, BaseException):
                    raise skill_outcome
                if isinstance(speech_outcome, BaseException):
                    raise speech_outcome
                skill_result = skill_outcome
                speech_spoken = True
        elif decision.action == "execute_skill":
            if decision.skill is None:
                raise RuntimeError("validated decision is missing a skill")
            skill_result = await self.runtime.execute(
                decision.skill,
                **decision.arguments,
            )

        if (
            skill_result is not None
            and skill_result.success
            and decision.skill == "wave"
        ):
            async with self._lock:
                if self.world_state.person_visible:
                    self.world_state.mark_greeted()

        if decision.action == "speak":
            if decision.speech is None:
                raise RuntimeError("validated decision is missing speech")
            if self.speech is not None:
                await self.speech.speak(decision.speech)
                speech_spoken = True

        return DecisionOutcome(
            event=event,
            decision=decision,
            skill_result=skill_result,
            speech_spoken=speech_spoken,
        )
