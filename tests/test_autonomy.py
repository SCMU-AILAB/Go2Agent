from __future__ import annotations

import asyncio
import json
import unittest
from collections.abc import Mapping
from typing import cast
from unittest.mock import AsyncMock, patch

from langchain_core.messages import HumanMessage
from pydantic import ValidationError

from agent import (
    AgentDecision,
    AutonomousDecisionLoop,
    DecisionAgentError,
    EventDecisionAgent,
    build_decision_system_prompt,
)
from core.context import SkillContext
from core.models import SkillArgs, SkillMetadata, SkillResult
from core.runtime import SkillRuntime
from core.skill import RobotSkill
from perception import PerceptionResult, WorldEvent, WorldEventType, WorldState
from robot import SimulatedRobotAdapter
from skills.motions import MoveBackwardSkill, WaveSkill


class FakeDecisionInvoker:
    def __init__(self, structured_response: object) -> None:
        self.structured_response = structured_response
        self.inputs: list[dict[str, object]] = []

    async def ainvoke(self, input_state: dict[str, object]) -> object:
        self.inputs.append(input_state)
        return {"structured_response": self.structured_response}


class SlowDecisionInvoker:
    async def ainvoke(self, input_state: dict[str, object]) -> object:
        await asyncio.sleep(1.0)
        return {"structured_response": {"action": "ignore"}}


class ScriptedDecisionAgent:
    def __init__(self) -> None:
        self.calls: list[tuple[WorldEvent, Mapping[str, object]]] = []

    async def decide(
        self,
        event: WorldEvent,
        world_state: Mapping[str, object],
    ) -> AgentDecision:
        self.calls.append((event, world_state))
        if event.type == WorldEventType.PERSON_ENTERED:
            return AgentDecision(
                action="execute_and_speak",
                skill="wave",
                arguments={"arm": "right"},
                speech="你好！",
                reason="new person",
            )
        if event.type == WorldEventType.PERSON_TOO_CLOSE:
            return AgentDecision(
                action="execute_and_speak",
                skill="move_backward",
                arguments={"distance_m": 0.2},
                speech="请稍微保持一点距离。",
                reason="person is too close",
            )
        return AgentDecision(action="ignore", reason="no action needed")


class BlockingDecisionAgent(ScriptedDecisionAgent):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def decide(
        self,
        event: WorldEvent,
        world_state: Mapping[str, object],
    ) -> AgentDecision:
        self.calls.append((event, world_state))
        self.started.set()
        await self.release.wait()
        return AgentDecision(action="ignore")


class FakeSpeechOutput:
    def __init__(self) -> None:
        self.messages: list[str] = []

    async def speak(self, text: str) -> None:
        self.messages.append(text)


def observation(at_s: float, distance_m: float) -> PerceptionResult:
    return PerceptionResult(
        observed_at_s=at_s,
        person_count=1,
        nearest_person_distance_m=distance_m,
    )


class AgentDecisionTests(unittest.IsolatedAsyncioTestCase):
    def test_normalizes_multimodal_tool_name_inside_arguments(self) -> None:
        decision = AgentDecision.model_validate(
            {
                "action": "execute_skill",
                "arguments": {"name": "follow_person"},
            }
        )
        self.assertEqual(decision.skill, "follow_person")
        self.assertEqual(decision.arguments, {})

    def test_decision_prompt_is_built_from_registry_catalog(self) -> None:
        runtime = SkillRuntime(SimulatedRobotAdapter())
        runtime.register(WaveSkill())
        runtime.register(MoveBackwardSkill())

        prompt = build_decision_system_prompt(runtime.registry.list())

        self.assertIn("wave:", prompt)
        self.assertIn("move_backward:", prompt)

    def test_decision_contract_rejects_missing_skill(self) -> None:
        with self.assertRaises(ValidationError):
            AgentDecision(action="execute_skill")

    def test_decision_normalizes_empty_optional_strings(self) -> None:
        decision = AgentDecision.model_validate(
            {"action": "ignore", "arguments": {}, "speech": "", "skill": ""}
        )

        self.assertIsNone(decision.speech)
        self.assertIsNone(decision.skill)

    def test_decision_normalizes_compact_skill_action(self) -> None:
        decision = AgentDecision.model_validate(
            {
                "action": "wave",
                "skill": "wave",
                "arguments": {"arm": "right"},
                "speech": "您好！",
            }
        )

        self.assertEqual(decision.action, "execute_and_speak")
        self.assertEqual(decision.skill, "wave")

    def test_decision_infers_skill_from_compact_action(self) -> None:
        decision = AgentDecision.model_validate(
            {"action": "wave", "arguments": {"arm": "right"}}
        )

        self.assertEqual(decision.action, "execute_skill")
        self.assertEqual(decision.skill, "wave")

    def test_decision_normalizes_noop_alias(self) -> None:
        decision = AgentDecision.model_validate(
            {"action": "none", "arguments": {}, "speech": ""}
        )

        self.assertEqual(decision.action, "ignore")
        self.assertIsNone(decision.speech)

    def test_decision_upgrades_execute_skill_with_speech(self) -> None:
        decision = AgentDecision.model_validate(
            {
                "action": "execute_skill",
                "skill": "wave",
                "arguments": {"arm": "right"},
                "speech": "您好，欢迎来到我们的机器人！",
            }
        )

        self.assertEqual(decision.action, "execute_and_speak")
        self.assertEqual(decision.skill, "wave")

    async def test_event_agent_accepts_compact_skill_action(self) -> None:
        agent = EventDecisionAgent(
            invoker=FakeDecisionInvoker(
                {
                    "action": "wave",
                    "skill": "wave",
                    "arguments": {"arm": "right"},
                    "speech": "您好！",
                }
            ),
            skill_catalog=cast(
                tuple[RobotSkill[SkillArgs], ...],
                (WaveSkill(),),
            ),
        )
        event = WorldEvent(type=WorldEventType.PERSON_ENTERED, timestamp_s=1.0)

        decision = await agent.decide(event, {})

        self.assertEqual(decision.action, "execute_and_speak")
        self.assertEqual(decision.skill, "wave")

    async def test_event_agent_returns_validated_structured_decision(self) -> None:
        invoker = FakeDecisionInvoker(
            {
                "action": "execute_and_speak",
                "skill": "wave",
                "arguments": {"arm": "right"},
                "speech": "你好！",
            }
        )
        agent = EventDecisionAgent(invoker=invoker)
        event = WorldEvent(
            type=WorldEventType.PERSON_ENTERED,
            timestamp_s=1.0,
            entity_id="nearest_person",
        )

        decision = await agent.decide(event, {"person_greeted": False})

        self.assertEqual(decision.skill, "wave")
        messages = invoker.inputs[0].get("messages")
        if not isinstance(messages, list) or not isinstance(messages[0], HumanMessage):
            self.fail("Decision Agent input must contain a HumanMessage")
        payload = json.loads(str(messages[0].content))
        self.assertEqual(payload["event"]["type"], "person_entered")
        self.assertFalse(payload["world_state"]["person_greeted"])

    async def test_event_agent_rejects_invalid_structured_response(self) -> None:
        agent = EventDecisionAgent(
            invoker=FakeDecisionInvoker(
                {"action": "execute_skill", "skill": "unknown"}
            ),
            skill_catalog=cast(
                tuple[RobotSkill[SkillArgs], ...],
                (WaveSkill(),),
            ),
        )
        event = WorldEvent(
            type=WorldEventType.PERSON_ENTERED,
            timestamp_s=1.0,
        )

        with self.assertRaises(DecisionAgentError):
            await agent.decide(event, {})

    async def test_event_agent_times_out_slow_invocation(self) -> None:
        agent = EventDecisionAgent(
            invoker=SlowDecisionInvoker(),
            timeout_s=0.01,
        )
        event = WorldEvent(
            type=WorldEventType.PERSON_ENTERED,
            timestamp_s=1.0,
        )

        with self.assertRaisesRegex(DecisionAgentError, "timed out after 0.01"):
            await agent.decide(event, {})

    async def test_person_left_bypasses_model(self) -> None:
        invoker = FakeDecisionInvoker({"action": "execute_skill", "skill": "wave"})
        agent = EventDecisionAgent(invoker=invoker)
        event = WorldEvent(
            type=WorldEventType.PERSON_LEFT,
            timestamp_s=1.0,
        )

        decision = await agent.decide(event, {})

        self.assertEqual(decision.action, "ignore")
        self.assertEqual(invoker.inputs, [])


class AutonomousDecisionLoopTests(unittest.IsolatedAsyncioTestCase):
    def build_loop(
        self,
    ) -> tuple[
        AutonomousDecisionLoop,
        SimulatedRobotAdapter,
        ScriptedDecisionAgent,
        FakeSpeechOutput,
    ]:
        robot = SimulatedRobotAdapter()
        runtime = SkillRuntime(robot)
        runtime.register(WaveSkill())
        runtime.register(MoveBackwardSkill())
        agent = ScriptedDecisionAgent()
        speech = FakeSpeechOutput()
        return (
            AutonomousDecisionLoop(runtime, agent, speech=speech),
            robot,
            agent,
            speech,
        )

    async def test_new_person_is_greeted_and_marked_once(self) -> None:
        loop, robot, agent, speech = self.build_loop()

        first = await loop.process(observation(1.0, 2.0))
        repeated = await loop.process(observation(1.1, 2.0))

        self.assertEqual(len(first), 1)
        self.assertEqual(first[0].decision.skill, "wave")
        self.assertTrue(first[0].skill_result and first[0].skill_result.success)
        self.assertTrue(first[0].speech_spoken)
        self.assertEqual(repeated, ())
        self.assertTrue(loop.world_state.person_greeted)
        self.assertEqual(robot.events, [("wave", "right")])
        self.assertEqual(speech.messages, ["你好！"])
        self.assertEqual(len(agent.calls), 1)
        self.assertFalse(agent.calls[0][1]["person_greeted"])

    async def test_execute_and_speak_runs_action_and_speech_concurrently(self) -> None:
        action_started = asyncio.Event()
        speech_started = asyncio.Event()

        class CoordinatedArgs(SkillArgs):
            pass

        class CoordinatedSkill(RobotSkill[CoordinatedArgs]):
            metadata = SkillMetadata(
                name="coordinated_action",
                description="Test concurrent action and speech.",
            )
            args_model = CoordinatedArgs

            async def execute(
                self,
                ctx: SkillContext,
                args: CoordinatedArgs,
            ) -> SkillResult:
                action_started.set()
                await speech_started.wait()
                return SkillResult.ok()

        class CoordinatedSpeech:
            async def speak(self, text: str) -> None:
                speech_started.set()
                await action_started.wait()

        runtime = SkillRuntime(SimulatedRobotAdapter())
        runtime.register(CoordinatedSkill())
        loop = AutonomousDecisionLoop(
            runtime,
            ScriptedDecisionAgent(),
            speech=CoordinatedSpeech(),
        )

        outcome = await asyncio.wait_for(
            loop._execute(
                WorldEvent(type=WorldEventType.PERSON_LEFT, timestamp_s=1.0),
                AgentDecision(
                    action="execute_and_speak",
                    skill="coordinated_action",
                    speech="你好",
                ),
            ),
            timeout=0.2,
        )

        self.assertTrue(outcome.skill_result and outcome.skill_result.success)
        self.assertTrue(outcome.speech_spoken)
        self.assertTrue(action_started.is_set())
        self.assertTrue(speech_started.is_set())

    async def test_too_close_person_selects_bounded_backward_skill(self) -> None:
        loop, robot, agent, speech = self.build_loop()
        await loop.process(observation(1.0, 2.0))

        with patch(
            "skills.motions.move_backward.asyncio.sleep",
            new=AsyncMock(),
        ):
            outcomes = await loop.process(observation(2.0, 0.6))

        self.assertEqual(len(outcomes), 1)
        self.assertEqual(outcomes[0].decision.skill, "move_backward")
        self.assertTrue(
            outcomes[0].skill_result and outcomes[0].skill_result.success
        )
        self.assertEqual(
            robot.events,
            [
                ("wave", "right"),
                ("move_velocity", (-0.2, 0.0, 0.0)),
                ("stop", None),
            ],
        )
        self.assertEqual(
            speech.messages,
            ["你好！", "请稍微保持一点距离。"],
        )
        self.assertEqual(len(agent.calls), 2)

    async def test_workers_keep_observing_while_decision_is_blocked(self) -> None:
        robot = SimulatedRobotAdapter()
        runtime = SkillRuntime(robot)
        agent = BlockingDecisionAgent()
        loop = AutonomousDecisionLoop(
            runtime,
            agent,
            world_state=WorldState(absence_reset_s=0.5),
        )
        await loop.start_workers()
        try:
            first_events = await loop.observe(observation(1.0, 2.0))
            await asyncio.wait_for(agent.started.wait(), timeout=0.2)

            second_events = await asyncio.wait_for(
                loop.observe(
                    PerceptionResult(
                        observed_at_s=2.0,
                        person_count=0,
                    )
                ),
                timeout=0.2,
            )

            self.assertEqual(len(first_events), 1)
            self.assertEqual(len(second_events), 1)
            self.assertEqual(second_events[0].type, WorldEventType.PERSON_LEFT)
            agent.release.set()
            first_outcome = await loop.next_outcome(timeout=0.5)
            second_outcome = await loop.next_outcome(timeout=0.5)
            self.assertEqual(first_outcome.event.type, WorldEventType.PERSON_ENTERED)
            self.assertEqual(second_outcome.event.type, WorldEventType.PERSON_LEFT)
        finally:
            await loop.stop_workers()
