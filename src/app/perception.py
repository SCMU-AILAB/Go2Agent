"""Run the D435i event-to-Agent-to-skill autonomous decision loop."""

from __future__ import annotations

import argparse
import asyncio
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from adapters import AudioOutputError, UnitreeAudioOutput
from agent import (
    DEFAULT_LLAMA_CPP_MODEL,
    DEFAULT_LLAMA_CPP_URL,
    DEFAULT_UNIFOLM_MODEL,
    DEFAULT_UNIFOLM_URL,
    DEFAULT_VISION_GOAL,
    DEFAULT_VISION_MODEL,
    AutonomousDecisionLoop,
    CudaVisionInvoker,
    DecisionAgentError,
    DecisionOutcome,
    EventDecisionAgent,
    LlamaCppVisionInvoker,
    OllamaVisionInvoker,
    UnifolmVisionInvoker,
    VisionDecisionAgent,
    VisionPolicyOutcome,
    VisionPolicyWorker,
)
from agent.social_vision import (
    SocialVisionAgent,
    TaskDrivenObservation,
)
from agent.vision_capture import VisionCapture
from core.runtime import SkillRuntime
from perception import (
    EventDetector,
    PerceptionError,
    PerceptionResult,
    RealSensePersonDetector,
    VideoBuffer,
    WorldEvent,
    WorldState,
)
from robot import (
    ROBOT_MODELS,
    HardwareRobot,
    RobotAdapter,
    RobotCommandError,
    RobotModel,
    create_hardware_robot,
    create_simulated_robot,
)
from skills import register_g1_skills, register_go2_skills

from .structured_log import configure_structured_logging, emit_log

_LOG_OWNERS = (
    "robot.audio",
    "agent.vision_capture",
    "app.perception",
    "perception.camera",
    "perception.realsense",
    "perception.detector",
    "perception.safety",
    "agent.event_policy",
    "agent.vision_policy",
    "agent.cuda_worker",
    "runtime.skill",
    "robot.adapter",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hardware",
        action="store_true",
        help="connect to a physical G1 instead of the simulated adapter",
    )
    parser.add_argument(
        "--network",
        default="",
        help="Unitree DDS interface, e.g. eth0",
    )
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument(
        "--robot",
        choices=ROBOT_MODELS,
        default=os.getenv("G1_ROBOT_MODEL", "g1"),
        help="robot model to assemble; go2 uses SportClient and a reduced skill catalog",
    )
    parser.add_argument(
        "--vision-generate-speech",
        action="store_true",
        help="generate a short contextual utterance with each confirmed social gesture",
    )
    parser.add_argument(
        "--vision-rotation-deg",
        type=int,
        choices=(0, 90, 180, 270),
        default=0,
        help="clockwise RGB rotation for VLM only; depth safety remains native",
    )
    parser.add_argument(
        "--vision-capture-dir",
        type=Path,
        help="opt-in local model input capture (Ollama only)",
    )
    parser.add_argument(
        "--vision-capture-limit",
        type=int,
        default=20,
        help="maximum captured windows per run; stops saving at limit",
    )
    parser.add_argument(
        "--vision-json-mode",
        choices=("schema", "json", "prompt"),
        default="schema",
        help="Ollama schema or generic JSON constraints (prompt is a legacy alias for json)",
    )
    parser.add_argument(
        "--vision-task",
        choices=("general", "social"),
        default="general",
        help="social classifies gestures only; general exposes the skill catalog",
    )
    parser.add_argument(
        "--vision-social-profile",
        choices=("legacy", "egocentric"),
        default="legacy",
        help="social prompt profile; egocentric makes camera-relative gesture intent explicit",
    )
    parser.add_argument(
        "--vision-disable-thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="disable Ollama thinking output for supported models such as Qwen3.5",
    )
    parser.add_argument(
        "--include-operator-only-skills",
        action="store_true",
        help=(
            "expose low-level and dangerous SDK controls to the policy; "
            "disabled by default"
        ),
    )
    parser.add_argument(
        "--policy",
        choices=("vision", "event"),
        default="vision",
        help="rolling video policy (default) or legacy sparse-event policy",
    )
    parser.add_argument(
        "--model",
        help="policy model override; defaults depend on --policy",
    )
    parser.add_argument("--ollama-url", default=os.getenv("OLLAMA_HOST"))
    parser.add_argument(
        "--vision-url",
        default=os.getenv("VISION_URL") or os.getenv("UNIFOLM_VISION_URL"),
        help="vision HTTP endpoint for UnifoLM or llama.cpp",
    )
    parser.add_argument(
        "--decision-timeout-s",
        type=float,
        default=8.0,
        help="maximum time to wait for one Ollama event decision",
    )
    parser.add_argument(
        "--vision-timeout-s",
        type=float,
        default=120.0,
        help="maximum time to wait for one video-window decision",
    )
    parser.add_argument(
        "--vision-backend",
        choices=("cuda", "transformers", "ollama", "unifolm", "llamacpp"),
        default="cuda",
        help=(
            "Jetson CUDA worker (default), in-process Transformers, Ollama, "
            "remote Unitree UnifoLM, or local llama.cpp INT4"
        ),
    )
    parser.add_argument(
        "--vision-python",
        help="Python executable for the CUDA worker (default: /usr/bin/python3)",
    )
    parser.add_argument(
        "--vision-packages",
        help="isolated CUDA package directory (default: .cuda-packages)",
    )
    parser.add_argument(
        "--vision-max-new-tokens",
        type=int,
        default=40,
        help="maximum decision tokens generated by the VLM",
    )
    parser.add_argument(
        "--vision-interval-s",
        type=float,
        default=0.5,
        help="target interval between video policy decisions (default: 0.5 s)",
    )
    parser.add_argument(
        "--video-window-s",
        type=float,
        default=2.0,
        help="duration retained in the sliding video window",
    )
    parser.add_argument(
        "--vision-frame-count",
        type=int,
        default=1,
        help="uniformly sampled frames supplied to each VLM invocation",
    )
    parser.add_argument(
        "--max-decision-age-s",
        type=float,
        default=5.0,
        help=(
            "suppress visual speech/actions based on frames older than this "
            "many seconds"
        ),
    )
    parser.add_argument(
        "--vision-goal",
        default=DEFAULT_VISION_GOAL,
    )
    parser.add_argument(
        "--action-cooldown-s",
        type=float,
        default=5.0,
        help="minimum interval before an identical action may execute again",
    )
    parser.add_argument("--no-audio", action="store_true")
    parser.add_argument("--speaker-id", type=int, default=0)
    parser.add_argument("--camera-serial", help="D435i serial number")
    parser.add_argument(
        "--camera-python",
        help="Python with working pyrealsense2 bindings (default: /usr/bin/python3)",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--detection-fps",
        type=float,
        default=5.0,
        help="person detection rate; RGB capture continues at --fps",
    )
    parser.add_argument("--frame-timeout-ms", type=int, default=5000)
    parser.add_argument("--min-score", type=float, default=0.5)
    parser.add_argument(
        "--max-distance-m",
        type=float,
        default=4.0,
        help="ignore people with a valid depth beyond this distance",
    )
    parser.add_argument("--absence-reset-s", type=float, default=2.0)
    parser.add_argument("--too-close-m", type=float, default=0.8)
    parser.add_argument("--too-close-release-m", type=float, default=1.0)
    parser.add_argument(
        "--safety-stop-m",
        type=float,
        default=0.4,
        help="immediate software stop distance from central depth samples",
    )
    parser.add_argument(
        "--safety-release-m",
        type=float,
        default=0.55,
        help="distance required to clear the depth safety latch",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="exit after the first observation/event or vision decision",
    )
    parser.add_argument(
        "--observation-interval-s",
        type=float,
        default=5.0,
        help="maximum frequency for unchanged observation logs (default: 5 s)",
    )
    parser.add_argument(
        "--verbose-observations",
        action="store_true",
        help="print every camera observation instead of rate-limiting logs",
    )
    return parser.parse_args()


@dataclass(slots=True)
class _ObservationLogGate:
    """Keep high-rate camera frames from hiding sparse event logs."""

    interval_s: float = 5.0
    verbose: bool = False
    immediate_state_changes: bool = True
    _last_observation: PerceptionResult | None = None
    _last_printed_at_s: float | None = None

    def __post_init__(self) -> None:
        if self.interval_s <= 0:
            raise ValueError("observation log interval must be greater than zero")

    def should_print(self, observation: PerceptionResult) -> bool:
        previous = self._last_observation
        self._last_observation = observation
        if self.verbose or previous is None:
            self._last_printed_at_s = observation.observed_at_s
            return True

        state_changed = (
            previous.person_detected != observation.person_detected
            or previous.person_count != observation.person_count
            or (previous.nearest_person_distance_m is None)
            != (observation.nearest_person_distance_m is None)
        )
        interval_elapsed = (
            self._last_printed_at_s is None
            or observation.observed_at_s - self._last_printed_at_s >= self.interval_s
        )
        if not interval_elapsed and not (
            self.immediate_state_changes and state_changed
        ):
            return False
        self._last_printed_at_s = observation.observed_at_s
        return True


@dataclass(slots=True)
class _RepeatedErrorLogGate:
    """Emit repeated policy errors at a bounded rate with a suppression count."""

    interval_s: float = 5.0
    _last_key: tuple[str, str] | None = None
    _last_printed_at_s: float | None = None
    _suppressed_count: int = 0

    def __post_init__(self) -> None:
        if self.interval_s <= 0:
            raise ValueError("error log interval must be greater than zero")

    def admit(
        self,
        stage: str,
        message: str,
        *,
        now_s: float | None = None,
    ) -> int | None:
        observed_at_s = time.monotonic() if now_s is None else now_s
        key = (stage, message)
        if key != self._last_key:
            self._last_key = key
            self._last_printed_at_s = observed_at_s
            self._suppressed_count = 0
            return 0

        if (
            self._last_printed_at_s is not None
            and observed_at_s - self._last_printed_at_s < self.interval_s
        ):
            self._suppressed_count += 1
            return None

        suppressed_count = self._suppressed_count
        self._last_printed_at_s = observed_at_s
        self._suppressed_count = 0
        return suppressed_count

    def reset(self) -> None:
        self._last_key = None
        self._last_printed_at_s = None
        self._suppressed_count = 0


def _print_observation(
    observation: PerceptionResult,
    *,
    log_gate: _ObservationLogGate | None = None,
) -> None:
    if log_gate is not None and not log_gate.should_print(observation):
        return
    emit_log(
        owner="perception.camera",
        event_type="observation",
        data=observation.to_dict(),
    )


def _print_outcome(outcome: DecisionOutcome) -> None:
    emit_log(
        owner="runtime.skill",
        event_type="decision_outcome",
        data=outcome.to_dict(),
    )


def _print_events(events: tuple[WorldEvent, ...]) -> None:
    for event in events:
        emit_log(
            owner="perception.detector",
            event_type="world_event",
            data={"event": event.to_dict()},
        )


def _print_policy_outcome(outcome: VisionPolicyOutcome) -> None:
    if (
        outcome.decision.action in {"continue", "ignore"}
        and not outcome.executed
        and outcome.skill_result is None
        and not outcome.speech_spoken
        and outcome.suppressed_reason is None
    ):
        return
    emit_log(
        owner="runtime.skill",
        event_type="vision_policy_outcome",
        data=outcome.to_dict(),
    )


@dataclass(slots=True)
class _DepthSafetyGate:
    stop_distance_m: float = 0.4
    release_distance_m: float = 0.55
    latched: bool = False

    def __post_init__(self) -> None:
        if self.stop_distance_m <= 0:
            raise ValueError("safety stop distance must be greater than zero")
        if self.release_distance_m <= self.stop_distance_m:
            raise ValueError(
                "safety release distance must be greater than stop distance"
            )

    def update(self, distance_m: float | None) -> str | None:
        if distance_m is not None and distance_m <= self.stop_distance_m:
            if not self.latched:
                self.latched = True
                return "triggered"
            return None
        if (
            self.latched
            and distance_m is not None
            and distance_m >= self.release_distance_m
        ):
            self.latched = False
            return "released"
        return None


async def run_perception_loop(
    *,
    camera: RealSensePersonDetector,
    decision_loop: AutonomousDecisionLoop,
    once: bool = False,
    once_wait_s: float = 8.0,
    observation_interval_s: float = 5.0,
    verbose_observations: bool = False,
) -> int:
    log_gate = _ObservationLogGate(
        interval_s=observation_interval_s,
        verbose=verbose_observations,
    )
    await decision_loop.start_workers()
    try:
        observation = await asyncio.to_thread(camera.capture)
        events = await decision_loop.observe(observation)
        _print_observation(observation, log_gate=log_gate)
        _print_events(events)
        if once:
            if events:
                try:
                    failed = False
                    async with asyncio.timeout(once_wait_s):
                        for _ in events:
                            outcome = await decision_loop.next_outcome()
                            _print_outcome(outcome)
                            failed = failed or (
                                outcome.skill_result is not None
                                and not outcome.skill_result.success
                            )
                    return int(failed)
                except TimeoutError:
                    for error in decision_loop.drain_errors():
                        emit_log(
                            owner="agent.event_policy",
                            event_type="decision_error",
                            level="error",
                            data={"message": error},
                            stream=sys.stderr,
                        )
                    return 1
            return 0

        while True:
            for error in decision_loop.drain_errors():
                emit_log(
                    owner="agent.event_policy",
                    event_type="decision_error",
                    level="error",
                    data={"message": error},
                    stream=sys.stderr,
                )
            for outcome in decision_loop.drain_outcomes():
                _print_outcome(outcome)
            observation = await asyncio.to_thread(camera.capture)
            events = await decision_loop.observe(observation)
            _print_observation(observation, log_gate=log_gate)
            _print_events(events)
    finally:
        await decision_loop.stop_workers()


async def run_vision_perception_loop(
    *,
    camera: RealSensePersonDetector,
    policy_worker: VisionPolicyWorker,
    video_buffer: VideoBuffer,
    safety_gate: _DepthSafetyGate,
    once: bool = False,
    once_wait_s: float = 30.0,
    observation_interval_s: float = 5.0,
    verbose_observations: bool = False,
) -> int:
    log_gate = _ObservationLogGate(
        interval_s=observation_interval_s,
        verbose=verbose_observations,
        immediate_state_changes=False,
    )
    error_log_gate = _RepeatedErrorLogGate()
    await policy_worker.start()
    try:
        started = asyncio.get_running_loop().time()
        while True:
            frame = await asyncio.to_thread(camera.capture_frame)
            video_buffer.push(frame)
            policy_worker.observe_frame(frame)
            _print_observation(frame.observation, log_gate=log_gate)

            safety_transition = safety_gate.update(frame.nearest_obstacle_distance_m)
            if safety_transition == "triggered":
                policy_worker.set_safety_latched(True)
                interrupted = await policy_worker.stop_locomotion_for_safety(
                    "depth safety stop",
                )
                emit_log(
                    owner="perception.safety",
                    event_type="safety_stop",
                    level="warning",
                    data={
                        "distance_m": frame.nearest_obstacle_distance_m,
                        "threshold_m": safety_gate.stop_distance_m,
                        "interrupted_active_behavior": interrupted,
                        "scope": "mobile_base",
                        "upper_body_actions_preserved": True,
                    },
                )
            elif safety_transition == "released":
                policy_worker.set_safety_latched(False)
                emit_log(
                    owner="perception.safety",
                    event_type="safety_released",
                    data={
                        "distance_m": frame.nearest_obstacle_distance_m,
                        "release_m": safety_gate.release_distance_m,
                    },
                )

            policy_decisions = policy_worker.drain_policy_decisions()
            if policy_decisions:
                error_log_gate.reset()
            for policy_decision in policy_decisions:
                emit_log(
                    owner="agent.vision_policy",
                    event_type="vision_decision",
                    data=policy_decision.to_dict(),
                )
            policy_outcomes = policy_worker.drain_outcomes()
            for outcome in policy_outcomes:
                _print_policy_outcome(outcome)
            policy_errors = policy_worker.drain_errors()
            for error in policy_errors:
                suppressed_count = error_log_gate.admit(error.stage, error.message)
                if suppressed_count is None:
                    continue
                owner = (
                    "agent.vision_policy"
                    if error.stage == "decision"
                    else "runtime.skill"
                )
                error_data: dict[str, object] = {
                    "stage": error.stage,
                    "message": error.message,
                }
                if suppressed_count:
                    error_data["repeated_errors_suppressed"] = suppressed_count
                emit_log(
                    owner=owner,
                    event_type="vision_policy_error",
                    level="error",
                    data=error_data,
                    stream=sys.stderr,
                )

            if once and policy_errors:
                return 1
            if once and asyncio.get_running_loop().time() - started >= once_wait_s:
                return 1
            if once and policy_outcomes:
                return int(
                    any(
                        outcome.skill_result is not None
                        and not outcome.skill_result.success
                        for outcome in policy_outcomes
                    )
                )
    finally:
        await policy_worker.stop()


async def _run(args: argparse.Namespace) -> int:
    hardware_robot: HardwareRobot | None = None
    audio: UnitreeAudioOutput | None = None
    vision_agent: VisionDecisionAgent | None = None
    robot: RobotAdapter
    robot_model: RobotModel = args.robot
    if args.hardware:
        hardware_robot = create_hardware_robot(
            robot_model,
            network_interface=args.network,
            domain_id=args.domain_id,
        )
        robot = hardware_robot
    else:
        robot = create_simulated_robot(robot_model)

    runtime = SkillRuntime(robot)
    include_operator_only = getattr(args, "include_operator_only_skills", False)
    if robot_model == "go2":
        register_go2_skills(runtime, include_operator_only=include_operator_only)
    else:
        register_g1_skills(runtime, include_operator_only=include_operator_only)
    camera = RealSensePersonDetector(
        serial=args.camera_serial,
        width=args.width,
        height=args.height,
        fps=args.fps,
        detection_fps=args.detection_fps,
        frame_timeout_ms=args.frame_timeout_ms,
        min_score=args.min_score,
        max_distance_m=args.max_distance_m,
        bridge_python=args.camera_python,
    )
    world_state = WorldState(
        absence_reset_s=args.absence_reset_s,
    )
    event_detector = EventDetector(
        too_close_distance_m=args.too_close_m,
        too_close_release_m=args.too_close_release_m,
    )

    selected_model = args.model or (
        DEFAULT_UNIFOLM_MODEL
        if args.policy == "vision" and args.vision_backend == "unifolm"
        else DEFAULT_LLAMA_CPP_MODEL
        if args.policy == "vision" and args.vision_backend == "llamacpp"
        else DEFAULT_VISION_MODEL
        if args.policy == "vision"
        else os.getenv("OLLAMA_MODEL", "qwen3:1.7b")
    )

    emit_log(
        owner="app.perception",
        event_type="startup",
        data={
            "policy": args.policy,
            "vision_backend": (
                args.vision_backend if args.policy == "vision" else None
            ),
            "model": selected_model,
            "camera_serial": args.camera_serial,
            "hardware": args.hardware,
            "robot_model": robot_model,
            "vision_social_profile": args.vision_social_profile
            if args.vision_task == "social"
            else None,
            "vision_thinking_disabled": args.vision_disable_thinking,
            "vision_frame_count": (
                args.vision_frame_count if args.policy == "vision" else None
            ),
            "max_decision_age_s": (
                args.max_decision_age_s if args.policy == "vision" else None
            ),
            "owners": list(_LOG_OWNERS),
            "log_schema": "g1agent.log.v1",
        },
    )

    try:
        # Open the camera bridge before model runtimes start worker threads. Starting
        # its system-Python subprocess from a thread after that can deadlock.
        camera.open()
        emit_log(
            owner="perception.camera",
            event_type="camera_ready",
            data={
                "source": f"realsense:{args.camera_serial or 'D435i'}",
                "width": args.width,
                "height": args.height,
                "fps": args.fps,
            },
        )

        decision_agent: EventDecisionAgent | None = None
        if args.policy == "event":
            decision_agent = EventDecisionAgent(
                model_name=selected_model,
                base_url=args.ollama_url,
                timeout_s=args.decision_timeout_s,
                skill_catalog=runtime.registry.list(),
            )
        else:
            if args.vision_backend == "ollama":
                vision_invoker = OllamaVisionInvoker(
                    selected_model,
                    base_url=args.ollama_url,
                    output_schema=(
                        TaskDrivenObservation.model_json_schema()
                        if args.vision_task == "social"
                        else None
                    ),
                    max_new_tokens=args.vision_max_new_tokens,
                    constrain_json=args.vision_json_mode == "schema",
                    think=False if args.vision_disable_thinking else None,
                )
            elif args.vision_backend == "cuda":
                vision_invoker = CudaVisionInvoker(
                    selected_model,
                    max_new_tokens=args.vision_max_new_tokens,
                    python_executable=args.vision_python,
                    packages_path=(
                        Path(args.vision_packages) if args.vision_packages else None
                    ),
                    startup_timeout_s=max(120.0, args.vision_timeout_s),
                    inference_timeout_s=args.vision_timeout_s,
                )
            elif args.vision_backend == "unifolm":
                vision_invoker = UnifolmVisionInvoker(
                    selected_model,
                    base_url=args.vision_url or DEFAULT_UNIFOLM_URL,
                    max_new_tokens=args.vision_max_new_tokens,
                    timeout_s=args.vision_timeout_s,
                )
            elif args.vision_backend == "llamacpp":
                llama_output_schema = (
                    TaskDrivenObservation.model_json_schema()
                    if args.vision_task == "social"
                    else None
                )
                vision_invoker = LlamaCppVisionInvoker(
                    selected_model,
                    base_url=args.vision_url or DEFAULT_LLAMA_CPP_URL,
                    max_new_tokens=args.vision_max_new_tokens,
                    timeout_s=args.vision_timeout_s,
                    constrain_json=args.vision_json_mode != "prompt",
                    output_schema=llama_output_schema,
                )
            else:
                vision_invoker = None
            agent_class = (
                SocialVisionAgent
                if args.vision_task == "social"
                else VisionDecisionAgent
            )
            vision_agent = agent_class(
                model_name=selected_model,
                invoker=vision_invoker,
                timeout_s=args.vision_timeout_s,
                goal=args.vision_goal,
                **(
                    {
                        "prompt_profile": args.vision_social_profile,
                        "generate_speech": args.vision_generate_speech,
                    }
                    if args.vision_task == "social"
                    else {}
                ),
            )

        if vision_agent is not None:
            emit_log(
                owner="agent.vision_policy",
                event_type="model_loading",
                data={
                    "model": vision_agent.model_name,
                    "vision_task": args.vision_task,
                },
            )
            await vision_agent.warmup()
            ready_data: dict[str, object] = {
                "model": vision_agent.model_name,
                "backend": args.vision_backend,
            }
            if isinstance(
                vision_invoker,
                (CudaVisionInvoker, UnifolmVisionInvoker, LlamaCppVisionInvoker),
            ):
                ready_data.update(vision_invoker.backend_info)
            emit_log(
                owner="agent.vision_policy",
                event_type="model_ready",
                data=ready_data,
            )
        if hardware_robot is not None:
            await hardware_robot.connect()
            emit_log(
                owner="robot.adapter",
                event_type="robot_connected",
                data={
                    "network": args.network,
                    "domain_id": args.domain_id,
                    "robot_model": robot_model,
                },
            )
            if robot_model == "go2":
                if not args.no_audio:
                    emit_log(
                        owner="robot.audio",
                        event_type="audio_skipped",
                        data={
                            "reason": "go2_has_no_g1_audio_client",
                            "speaker_id": args.speaker_id,
                        },
                    )
            elif not args.no_audio:
                audio = UnitreeAudioOutput(
                    hardware_robot,
                    speaker_id=args.speaker_id,
                )
                await audio.connect()
                emit_log(
                    owner="robot.audio",
                    event_type="audio_ready",
                    data={"speaker_id": args.speaker_id},
                )
        if decision_agent is not None:
            decision_loop = AutonomousDecisionLoop(
                runtime,
                decision_agent,
                world_state=world_state,
                event_detector=event_detector,
                speech=audio,
            )
            return await run_perception_loop(
                camera=camera,
                decision_loop=decision_loop,
                once=args.once,
                once_wait_s=args.decision_timeout_s + 1.0,
                observation_interval_s=args.observation_interval_s,
                verbose_observations=args.verbose_observations,
            )

        if vision_agent is None:
            raise RuntimeError("vision policy was not initialized")
        if args.vision_capture_dir and args.vision_backend != "ollama":
            raise ValueError(
                "vision capture currently requires --vision-backend ollama"
            )
        video_buffer = VideoBuffer(
            window_s=args.video_window_s,
            max_frames=max(1, math.ceil(args.fps * args.video_window_s)),
        )
        policy_worker = VisionPolicyWorker(
            runtime,
            vision_agent,
            video_buffer,
            speech=audio,
            interval_s=args.vision_interval_s,
            frame_count=args.vision_frame_count,
            action_cooldown_s=args.action_cooldown_s,
            max_decision_age_s=args.max_decision_age_s,
            capture=(
                VisionCapture(args.vision_capture_dir, args.vision_capture_limit)
                if args.vision_capture_dir
                else None
            ),
            rotation_deg=args.vision_rotation_deg,
        )
        return await run_vision_perception_loop(
            camera=camera,
            policy_worker=policy_worker,
            video_buffer=video_buffer,
            safety_gate=_DepthSafetyGate(
                stop_distance_m=args.safety_stop_m,
                release_distance_m=args.safety_release_m,
            ),
            once=args.once,
            once_wait_s=args.vision_timeout_s + 1.0,
            observation_interval_s=args.observation_interval_s,
            verbose_observations=args.verbose_observations,
        )
    finally:
        try:
            try:
                if vision_agent is not None:
                    await vision_agent.close()
            except Exception as exc:  # noqa: BLE001 - continue shutdown
                emit_log(
                    owner="agent.cuda_worker",
                    event_type="shutdown_error",
                    level="error",
                    data={"message": str(exc)},
                    stream=sys.stderr,
                )
            finally:
                camera.close()
        finally:
            try:
                if audio is not None:
                    await audio.close()
            finally:
                if hardware_robot is not None:
                    await hardware_robot.close()


def main() -> int:
    configure_structured_logging()
    try:
        return asyncio.run(_run(parse_args()))
    except KeyboardInterrupt:
        return 130
    except (
        AudioOutputError,
        DecisionAgentError,
        PerceptionError,
        RobotCommandError,
        RuntimeError,
        ValueError,
    ) as exc:
        emit_log(
            owner="app.perception",
            event_type="fatal_error",
            level="error",
            data={"message": str(exc)},
            stream=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
