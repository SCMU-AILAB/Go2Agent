from __future__ import annotations

import asyncio
import json
import time
import unittest
from collections.abc import Sequence

from agent import (
    AgentDecision,
    VisionDecisionAgent,
    VisionPolicyDecision,
    VisionPolicyWorker,
)
from agent.vision_policy import _skill_catalog_payload, _VisionDecisionRequest
from core.context import SkillContext
from core.models import SkillArgs, SkillMetadata, SkillResult
from core.runtime import SkillRuntime
from core.skill import RobotSkill
from perception import CameraFrame, PerceptionResult, VideoBuffer
from robot import RobotState, SimulatedRobotAdapter
from skills.motions import HandshakeSkill, MoveBackwardSkill, WaveHandSkill, WaveSkill


def camera_frame(at_s: float, *, obstacle_distance_m: float = 2.0) -> CameraFrame:
    observation = PerceptionResult(observed_at_s=at_s, source="camera:test")
    return CameraFrame(
        observed_at_s=at_s,
        rgb=b"jpeg",
        depth=None,
        observation=observation,
        nearest_obstacle_distance_m=obstacle_distance_m,
    )


class FakeVisionInvoker:
    def __init__(self, responses: Sequence[object]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[Sequence[object], str]] = []

    async def ainvoke(self, frames: Sequence[object], prompt: str) -> object:
        self.calls.append((frames, prompt))
        if len(self.responses) > 1:
            return self.responses.pop(0)
        return self.responses[0]


class VideoBufferTests(unittest.TestCase):
    def test_window_evicts_old_frames_and_samples_endpoints(self) -> None:
        buffer = VideoBuffer(window_s=2.0, max_frames=60)
        for index in range(6):
            buffer.push(camera_frame(float(index)))

        sampled = buffer.sample(3)

        self.assertEqual([frame.observed_at_s for frame in sampled], [3.0, 4.0, 5.0])

    def test_uniform_sample_includes_first_and_latest_frame(self) -> None:
        buffer = VideoBuffer(window_s=10.0, max_frames=60)
        for index in range(10):
            buffer.push(camera_frame(float(index)))

        sampled = buffer.sample(4)

        self.assertEqual([frame.observed_at_s for frame in sampled], [0.0, 3.0, 6.0, 9.0])


class VisionDecisionAgentTests(unittest.IsolatedAsyncioTestCase):
    def test_different_generated_speech_cannot_bypass_action_cooldown(self):
        first = AgentDecision(action="execute_and_speak", skill="wave", speech="你好")
        second = AgentDecision(action="execute_and_speak", skill="wave", speech="很高兴见到你")
        silent = AgentDecision(action="execute_skill", skill="wave")
        self.assertEqual(VisionPolicyWorker._decision_signature(first), VisionPolicyWorker._decision_signature(second))
        self.assertEqual(VisionPolicyWorker._decision_signature(first), VisionPolicyWorker._decision_signature(silent))

    async def test_video_frames_and_runtime_context_produce_agent_decision(self) -> None:
        invoker = FakeVisionInvoker(
            [
                json.dumps(
                    {
                        "action": "execute_skill",
                        "skill": "wave",
                        "arguments": {"arm": "right"},
                        "speech": None,
                        "reason": "person approached",
                    }
                )
            ]
        )
        runtime = SkillRuntime(SimulatedRobotAdapter())
        runtime.register(WaveSkill())
        runtime.register(WaveHandSkill())
        agent = VisionDecisionAgent(invoker=invoker)

        decision = await agent.decide(
            [camera_frame(1.0), camera_frame(2.0)],
            RobotState(hardware=False, connected=True),
            runtime.registry.list(),
            policy_context={
                "last_selected_skill": "wave",
                "seconds_since_last_selection": 3.0,
            },
        )

        self.assertEqual(decision.skill, "wave")
        self.assertEqual(len(invoker.calls[0][0]), 2)
        self.assertIn('"frame_count": 2', invoker.calls[0][1])
        self.assertIn('"name": "wave"', invoker.calls[0][1])
        self.assertNotIn('"name": "wave_hand"', invoker.calls[0][1])
        self.assertIn('"last_selected_skill": "wave"', invoker.calls[0][1])

    async def test_continue_and_interrupt_are_valid_noop_decisions(self) -> None:
        for action in ("continue", "interrupt"):
            decision = AgentDecision(action=action)
            self.assertEqual(decision.action, action)

    async def test_policy_decision_serializes_window_state_and_metrics(self) -> None:
        payload = VisionPolicyDecision(
            decided_at_s=3.0,
            frame_count=2,
            window_start_s=1.0,
            window_end_s=2.0,
            decision=AgentDecision(action="ignore"),
            robot_state=RobotState(hardware=False, connected=True),
            policy_context={"last_selected_skill": "wave"},
            model_metrics={"inference_s": 0.25},
        ).to_dict()

        self.assertEqual(payload["frame_count"], 2)
        self.assertEqual(payload["window_start_s"], 1.0)
        self.assertEqual(
            payload["decision"],
            AgentDecision(action="ignore").to_dict(),
        )
        self.assertEqual(payload["model_metrics"], {"inference_s": 0.25})
        self.assertEqual(
            payload["policy_context"],
            {"last_selected_skill": "wave"},
        )
        self.assertEqual(payload["decision_age_s"], 1.0)

    def test_compatibility_aliases_share_action_signature(self) -> None:
        wave = AgentDecision(
            action="execute_skill",
            skill="wave",
            arguments={"arm": "right"},
        )
        wave_hand = AgentDecision(
            action="execute_skill",
            skill="wave_hand",
            arguments={"arm": "right"},
        )

        self.assertEqual(
            VisionPolicyWorker._decision_signature(wave),
            VisionPolicyWorker._decision_signature(wave_hand),
        )

    async def test_malformed_ignore_is_recovered_as_safe_noop(self) -> None:
        decision = VisionDecisionAgent._parse_output(
            '{"action":"ignore","skill":null,"arguments":{"bad":'
        )

        self.assertEqual(decision.action, "ignore")
        self.assertEqual(decision.arguments, {})

    async def test_malformed_execute_decision_is_rejected(self) -> None:
        with self.assertRaisesRegex(Exception, "JSON object"):
            VisionDecisionAgent._parse_output(
                '{"action":"execute_skill","skill":"wave"'
            )

    async def test_truncated_handshake_is_rejected_even_with_allowlist(
        self,
    ) -> None:
        with self.assertRaisesRegex(Exception, "JSON object"):
            VisionDecisionAgent._parse_output(
                '{"action":"execute_skill","skill":"handshake","arguments":',
                recoverable_skills={"handshake"},
            )

    async def test_truncated_skill_is_rejected_without_recovery_allowlist(
        self,
    ) -> None:
        with self.assertRaisesRegex(Exception, "JSON object"):
            VisionDecisionAgent._parse_output(
                '{"action":"execute_skill","skill":"handshake","arguments":'
            )

    async def test_skill_catalog_uses_values_instead_of_json_schema(self) -> None:
        payload = _skill_catalog_payload([HandshakeSkill()])

        self.assertEqual(
            payload,
            [
                {
                    "name": "handshake",
                    "description": HandshakeSkill.metadata.description,
                    "argument_defaults": {"duration_s": 4.0},
                    "required_arguments": [],
                    "interruptible": True,
                }
            ],
        )

    async def test_noisy_json_noop_discards_hallucinated_arguments(self) -> None:
        decision = VisionDecisionAgent._parse_output(
            json.dumps(
                {
                    "action": "ignore",
                    "skill": None,
                    "arguments": {"skill_catalog": {"wave": "definition"}},
                    "speech": None,
                    "reason": "nothing actionable",
                }
            )
        )

        self.assertEqual(decision.action, "ignore")
        self.assertEqual(decision.arguments, {})

    async def test_noisy_mapping_continue_discards_hallucinated_arguments(self) -> None:
        decision = VisionDecisionAgent._parse_output(
            {
                "structured_response": {
                    "action": " continue ",
                    "skill": " ",
                    "arguments": {"distance_m": 0.2},
                    "speech": "",
                }
            }
        )

        self.assertEqual(decision.action, "continue")
        self.assertIsNone(decision.skill)
        self.assertEqual(decision.arguments, {})
        self.assertIsNone(decision.speech)

    async def test_noisy_interrupt_remains_strictly_rejected(self) -> None:
        with self.assertRaisesRegex(Exception, "requires a skill"):
            VisionDecisionAgent._parse_output(
                {
                    "action": "interrupt",
                    "skill": None,
                    "arguments": {"distance_m": 0.2},
                    "speech": None,
                }
            )


class VisionPolicyWorkerTests(unittest.IsolatedAsyncioTestCase):
    async def test_model_metrics_include_request_and_latency_fields(self) -> None:
        invoker = FakeVisionInvoker([{"action": "ignore"}])
        invoker.last_metrics = {
            "round_trip_s": 0.475,
            "client_encode_s": 0.005,
            "http_round_trip_s": 0.47,
            "transport_residual_s": 0.044,
            "network_rtt_s": 0.08,
            "inference_s": 0.426,
        }
        worker = VisionPolicyWorker(
            SkillRuntime(SimulatedRobotAdapter()),
            VisionDecisionAgent(invoker=invoker),
            VideoBuffer(window_s=2.0, max_frames=60),
        )

        metrics = worker._build_model_metrics(
            request_id="vision-000001",
            captured_at_s=10.0,
            sent_at_s=10.05,
            finished_at_s=10.525,
            frame_count=8,
        )

        self.assertEqual(metrics["request_id"], "vision-000001")
        self.assertEqual(metrics["frame_count"], 8)
        self.assertEqual(metrics["frame_age_ms"], 525.0)
        self.assertEqual(metrics["end_to_end_latency_ms"], 525.0)
        self.assertEqual(metrics["client_encode_ms"], 5.0)
        self.assertEqual(metrics["http_round_trip_ms"], 470.0)
        self.assertEqual(metrics["transport_residual_ms"], 44.0)
        self.assertEqual(metrics["network_rtt_ms"], 80.0)
        self.assertEqual(metrics["inference_latency_ms"], 426.0)

    async def test_policy_context_reports_active_skill_and_last_decision(self) -> None:
        worker = VisionPolicyWorker(
            SkillRuntime(SimulatedRobotAdapter()),
            VisionDecisionAgent(invoker=FakeVisionInvoker([{"action": "ignore"}])),
            VideoBuffer(window_s=2.0, max_frames=60),
        )
        worker._active_skill = "wave"
        worker._active_started_at_s = time.monotonic() - 1.0
        worker._active_task = asyncio.create_task(asyncio.sleep(1.0))
        worker._last_decision = AgentDecision(
            action="execute_skill",
            skill="wave",
            arguments={"arm": "right"},
        )

        try:
            context = worker._build_policy_context()
        finally:
            worker._active_task.cancel()
            await asyncio.gather(worker._active_task, return_exceptions=True)

        self.assertEqual(context["active_skill"], "wave")
        self.assertEqual(context["robot_motion"], "executing")
        self.assertGreaterEqual(float(context["active_skill_elapsed_s"]), 1.0)
        self.assertEqual(
            context["last_decision"],
            {
                "action": "execute_skill",
                "skill": "wave",
                "arguments": {"arm": "right"},
                "speech": None,
                "reason": None,
            },
        )

    async def test_execution_drops_action_that_ages_in_queue(self) -> None:
        robot = SimulatedRobotAdapter()
        runtime = SkillRuntime(robot)
        runtime.register(WaveSkill())
        frame = camera_frame(time.monotonic() - 1.0)
        decision = AgentDecision(
            action="execute_skill",
            skill="wave",
            arguments={"arm": "right"},
        )
        worker = VisionPolicyWorker(
            runtime,
            VisionDecisionAgent(invoker=FakeVisionInvoker([{"action": "ignore"}])),
            VideoBuffer(window_s=2.0, max_frames=60),
            max_decision_age_s=0.1,
        )
        await worker._decision_queue.put(
            _VisionDecisionRequest(
                decision=decision,
                frames=(frame,),
                robot_state=RobotState(hardware=False, connected=True),
                request_id="vision-queue-old",
                model_metrics={"request_id": "vision-queue-old"},
            )
        )
        task = asyncio.create_task(worker._execution_loop())
        try:
            deadline = time.monotonic() + 0.2
            while not worker._outcome_queue.qsize() and time.monotonic() < deadline:
                await asyncio.sleep(0.005)
            outcomes = worker.drain_outcomes()
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

        self.assertNotIn(("wave", "right"), robot.events)
        self.assertEqual(outcomes[0].request_id, "vision-queue-old")
        self.assertIn("before execution", outcomes[0].suppressed_reason or "")

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
        worker = VisionPolicyWorker(
            runtime,
            VisionDecisionAgent(invoker=FakeVisionInvoker([{"action": "ignore"}])),
            VideoBuffer(window_s=2.0, max_frames=60),
            speech=CoordinatedSpeech(),
        )

        skill_result, speech_spoken = await asyncio.wait_for(
            worker._execute(
                AgentDecision(
                    action="execute_and_speak",
                    skill="coordinated_action",
                    speech="你好",
                )
            ),
            timeout=0.2,
        )

        self.assertTrue(skill_result and skill_result.success)
        self.assertTrue(speech_spoken)
        self.assertTrue(action_started.is_set())
        self.assertTrue(speech_started.is_set())

    async def test_recovered_handshake_requires_recent_close_depth(self) -> None:
        robot = SimulatedRobotAdapter()
        runtime = SkillRuntime(robot)
        runtime.register(HandshakeSkill())
        buffer = VideoBuffer(window_s=2.0, max_frames=60)
        buffer.push(camera_frame(time.monotonic()))
        invoker = FakeVisionInvoker(
            ['{"action":"execute_skill","skill":"handshake","arguments":']
        )
        worker = VisionPolicyWorker(
            runtime,
            VisionDecisionAgent(invoker=invoker),
            buffer,
            interval_s=0.01,
        )

        await worker.start()
        try:
            await asyncio.sleep(0.04)
        finally:
            await worker.stop()

        self.assertNotIn("arm_action", [event[0] for event in robot.events])
        self.assertTrue(worker.drain_errors())

    async def test_recent_close_depth_cannot_authorize_malformed_handshake(self) -> None:
        robot = SimulatedRobotAdapter()
        runtime = SkillRuntime(robot)
        runtime.register(HandshakeSkill())
        buffer = VideoBuffer(window_s=2.0, max_frames=60)
        frame = camera_frame(
            time.monotonic(),
            obstacle_distance_m=0.35,
        )
        buffer.push(frame)
        invoker = FakeVisionInvoker(
            [
                '{"action":"execute_skill","skill":"handshake","arguments":',
                '{"action":"continue"}',
            ]
        )
        worker = VisionPolicyWorker(
            runtime,
            VisionDecisionAgent(invoker=invoker),
            buffer,
            interval_s=0.01,
        )
        worker.observe_frame(frame)

        await worker.start()
        try:
            await asyncio.sleep(0.04)
        finally:
            await worker.stop()

        self.assertNotIn("arm_action", [event[0] for event in robot.events])
        self.assertTrue(worker.drain_errors())

    async def test_stale_visual_action_is_not_executed(self) -> None:
        robot = SimulatedRobotAdapter()
        runtime = SkillRuntime(robot)
        runtime.register(WaveSkill())
        buffer = VideoBuffer(window_s=2.0, max_frames=60)
        buffer.push(camera_frame(time.monotonic() - 10.0))
        invoker = FakeVisionInvoker(
            [
                {
                    "action": "execute_skill",
                    "skill": "wave",
                    "arguments": {"arm": "right"},
                }
            ]
        )
        worker = VisionPolicyWorker(
            runtime,
            VisionDecisionAgent(invoker=invoker),
            buffer,
            interval_s=0.01,
            max_decision_age_s=5.0,
        )

        await worker.start()
        try:
            await asyncio.sleep(0.04)
        finally:
            await worker.stop()

        self.assertNotIn(("wave", "right"), robot.events)
        self.assertTrue(
            any(
                outcome.suppressed_reason is not None
                and "stale visual decision" in outcome.suppressed_reason
                for outcome in worker.drain_outcomes()
            )
        )

    async def test_stale_handshake_is_rejected_despite_fresh_close_depth(
        self,
    ) -> None:
        robot = SimulatedRobotAdapter()
        runtime = SkillRuntime(robot)
        runtime.register(HandshakeSkill())
        buffer = VideoBuffer(window_s=2.0, max_frames=60)
        buffer.push(camera_frame(time.monotonic() - 10.0))
        invoker = FakeVisionInvoker(
            [
                {
                    "action": "execute_skill",
                    "skill": "handshake",
                    "arguments": {},
                },
                {"action": "continue"},
            ]
        )
        worker = VisionPolicyWorker(
            runtime,
            VisionDecisionAgent(invoker=invoker),
            buffer,
            interval_s=0.01,
            max_decision_age_s=5.0,
        )
        worker.observe_frame(
            camera_frame(
                time.monotonic(),
                obstacle_distance_m=0.35,
            )
        )

        await worker.start()
        try:
            await asyncio.sleep(0.04)
        finally:
            await worker.stop()

        self.assertNotIn("arm_action", [event[0] for event in robot.events])
        self.assertTrue(any(
            "stale visual decision" in (outcome.suppressed_reason or "")
            for outcome in worker.drain_outcomes()
        ))

    async def test_depth_safety_latch_blocks_mobile_base_skill(self) -> None:
        robot = SimulatedRobotAdapter()
        runtime = SkillRuntime(robot)
        runtime.register(MoveBackwardSkill())
        buffer = VideoBuffer(window_s=2.0, max_frames=60)
        buffer.push(camera_frame(1.0))
        invoker = FakeVisionInvoker(
            [
                {
                    "action": "execute_skill",
                    "skill": "move_backward",
                    "arguments": {"distance_m": 0.2},
                }
            ]
        )
        worker = VisionPolicyWorker(
            runtime,
            VisionDecisionAgent(invoker=invoker),
            buffer,
            interval_s=0.01,
        )
        worker.set_safety_latched(True)

        await worker.start()
        try:
            await asyncio.sleep(0.04)
        finally:
            await worker.stop()

        self.assertNotIn("move_velocity", [event[0] for event in robot.events])
        self.assertTrue(
            any(
                outcome.suppressed_reason
                == "depth safety latch blocks mobile-base skills"
                for outcome in worker.drain_outcomes()
            )
        )

    async def test_depth_safety_latch_allows_upper_body_skill(self) -> None:
        robot = SimulatedRobotAdapter()
        runtime = SkillRuntime(robot)
        runtime.register(WaveSkill())
        buffer = VideoBuffer(window_s=2.0, max_frames=60)
        buffer.push(camera_frame(1.0))
        invoker = FakeVisionInvoker(
            [
                {
                    "action": "execute_skill",
                    "skill": "wave",
                    "arguments": {"arm": "right"},
                }
            ]
        )
        worker = VisionPolicyWorker(
            runtime,
            VisionDecisionAgent(invoker=invoker),
            buffer,
            interval_s=0.01,
        )
        worker.set_safety_latched(True)

        await worker.start()
        try:
            await asyncio.sleep(0.04)
        finally:
            await worker.stop()

        self.assertIn(("wave", "right"), robot.events)

    async def test_depth_safety_stop_preserves_active_handshake(self) -> None:
        robot = SimulatedRobotAdapter()
        runtime = SkillRuntime(robot)
        runtime.register(HandshakeSkill())
        buffer = VideoBuffer(window_s=2.0, max_frames=60)
        buffer.push(camera_frame(1.0))
        invoker = FakeVisionInvoker(
            [
                {
                    "action": "execute_skill",
                    "skill": "handshake",
                    "arguments": {"duration_s": 1.0},
                },
                {"action": "continue"},
            ]
        )
        worker = VisionPolicyWorker(
            runtime,
            VisionDecisionAgent(invoker=invoker),
            buffer,
            interval_s=0.01,
        )

        await worker.start()
        try:
            await asyncio.sleep(0.04)
            interrupted = await worker.stop_locomotion_for_safety()
            self.assertFalse(interrupted)
            self.assertIsNotNone(worker.active_behavior)
            self.assertNotIn(("release_arm", None), robot.events)
        finally:
            await worker.stop()

        self.assertIn(("stop", None), robot.events)
        self.assertIn(("release_arm", None), robot.events)

    async def test_depth_safety_stop_cancels_active_locomotion(self) -> None:
        robot = SimulatedRobotAdapter()
        runtime = SkillRuntime(robot)
        runtime.register(MoveBackwardSkill())
        buffer = VideoBuffer(window_s=2.0, max_frames=60)
        buffer.push(camera_frame(1.0))
        invoker = FakeVisionInvoker(
            [
                {
                    "action": "execute_skill",
                    "skill": "move_backward",
                    "arguments": {"distance_m": 0.3},
                },
                {"action": "continue"},
            ]
        )
        worker = VisionPolicyWorker(
            runtime,
            VisionDecisionAgent(invoker=invoker),
            buffer,
            interval_s=0.01,
        )

        await worker.start()
        try:
            await asyncio.sleep(0.04)
            interrupted = await worker.stop_locomotion_for_safety()
            self.assertTrue(interrupted)
            self.assertIsNone(worker.active_behavior)
        finally:
            await worker.stop()

        self.assertIn("move_velocity", [event[0] for event in robot.events])
        self.assertIn(("stop", None), robot.events)

    async def test_identical_skill_is_suppressed_during_cooldown(self) -> None:
        robot = SimulatedRobotAdapter()
        runtime = SkillRuntime(robot)
        runtime.register(WaveSkill())
        buffer = VideoBuffer(window_s=2.0, max_frames=60)
        buffer.push(camera_frame(1.0))
        invoker = FakeVisionInvoker(
            [
                {
                    "action": "execute_skill",
                    "skill": "wave",
                    "arguments": {"arm": "right"},
                }
            ]
        )
        worker = VisionPolicyWorker(
            runtime,
            VisionDecisionAgent(invoker=invoker),
            buffer,
            interval_s=0.01,
            action_cooldown_s=10.0,
        )

        await worker.start()
        try:
            await asyncio.sleep(0.08)
        finally:
            await worker.stop()

        self.assertEqual(robot.events.count(("wave", "right")), 1)
        self.assertTrue(
            any(
                outcome.suppressed_reason == "identical behavior is in cooldown"
                for outcome in worker.drain_outcomes()
            )
        )

    async def test_interrupt_cancels_active_skill_and_stops_robot(self) -> None:
        robot = SimulatedRobotAdapter()
        runtime = SkillRuntime(robot)
        runtime.register(MoveBackwardSkill())
        buffer = VideoBuffer(window_s=2.0, max_frames=60)
        buffer.push(camera_frame(1.0))
        invoker = FakeVisionInvoker(
            [
                {
                    "action": "execute_skill",
                    "skill": "move_backward",
                    "arguments": {"distance_m": 0.3},
                },
                {"action": "interrupt", "reason": "path blocked"},
                {"action": "continue"},
            ]
        )
        worker = VisionPolicyWorker(
            runtime,
            VisionDecisionAgent(invoker=invoker),
            buffer,
            interval_s=0.01,
        )

        await worker.start()
        try:
            await asyncio.sleep(0.08)
        finally:
            await worker.stop()

        self.assertIn("move_velocity", [event[0] for event in robot.events])
        self.assertIn(("stop", None), robot.events)


if __name__ == "__main__":
    unittest.main()
