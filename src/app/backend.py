"""Stateful console service shared by the FastAPI routes and WebSocket feed."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from adapters import AudioOutputError, UnitreeAudioOutput
from adapters.langchain import SkillToolObserver
from agent import AgentError, RobotAgent
from agent.service import system_prompt_for
from agent.social_vision import SocialVisionAgent
from agent.unifolm_vision import UnifolmVisionInvoker
from agent.vision_policy import OllamaVisionInvoker, VisionPolicyWorker
from core.runtime import SkillRuntime
from perception import (
    CameraFrame,
    PerceptionError,
    RealSensePersonDetector,
    VideoBuffer,
)
from robot import (
    HardwareRobot,
    RobotAdapter,
    RobotCommandError,
    RobotModel,
    create_hardware_robot,
    create_simulated_robot,
)
from skills import register_g1_skills, register_go2_skills

from .perception import _DepthSafetyGate


def _to_camel(value: str) -> str:
    head, *tail = value.split("_")
    return head + "".join(part.capitalize() for part in tail)


class ApiModel(BaseModel):
    """Use frontend field names on the wire while retaining Python names."""

    model_config = ConfigDict(
        alias_generator=_to_camel,
        populate_by_name=True,
        serialize_by_alias=True,
    )


class ConsoleLog(ApiModel):
    id: str
    time: str
    timestamp: datetime
    level: Literal["DEBUG", "INFO", "WARN", "ERROR"]
    source: str
    message: str


class ToolCall(ApiModel):
    name: str
    payload: str
    arguments: dict[str, object] = Field(default_factory=dict)
    result: dict[str, object] = Field(default_factory=dict)


class RobotView(ApiModel):
    mode: Literal["simulation", "hardware"]
    connected: bool
    details: dict[str, object] = Field(default_factory=dict)


class CameraView(ApiModel):
    source: Literal["demo", "local"]
    label: str
    status: Literal["idle", "starting", "ready", "error"]
    frame_available: bool
    frame_url: str = "/api/v1/camera/frame.jpg"
    frame_version: int = 0
    width: int = 640
    height: int = 480
    fps: int = 30
    observation: dict[str, object] | None = None
    error: str | None = None


class ConsoleSnapshot(ApiModel):
    backend: bool
    starting: bool
    busy: bool
    prompt_saved: bool
    system_prompt: str
    session_id: str
    task_id: str | None
    camera_source: Literal["demo", "local"]
    model_status: str
    skill_status: str
    skill_name: str
    progress: int
    progress_text: str
    active_step: int
    current_task: str
    model_output: str
    model_duration: float
    latency: int | None
    task_count: int
    robot: RobotView
    camera: CameraView
    tools: list[ToolCall]
    logs: list[ConsoleLog]


class ConsoleEvent(ApiModel):
    type: str
    timestamp: datetime
    data: dict[str, object]


class ChatAgent(Protocol):
    async def chat(self, text: str) -> str: ...

    def reset(self) -> None: ...


type AgentFactory = Callable[[SkillRuntime, str, SkillToolObserver], ChatAgent]


@dataclass(frozen=True, slots=True)
class BackendConfig:
    hardware: bool = False
    robot_model: RobotModel = "g1"
    network_interface: str = ""
    domain_id: int = 0
    include_operator_only_skills: bool = False
    model_name: str | None = None
    ollama_url: str | None = None
    audio_enabled: bool = True
    speaker_id: int = 0
    camera_source: Literal["demo", "local"] = "demo"
    camera_serial: str | None = None
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 30
    camera_detection_fps: float = 5.0
    vision_model: str = "qwen3.5:9b"
    vision_backend: Literal["ollama", "unifolm"] = "ollama"
    vision_url: str = "http://127.0.0.1:11435"
    vision_rotation_deg: int = 180
    vision_max_age_s: float = 5.0
    # Match run-remote-vision.sh; configurable for recognition/latency replay.
    vision_window_s: float = 0.8
    vision_frame_count: int = 3

    def __post_init__(self) -> None:
        if self.vision_rotation_deg not in (0, 90, 180, 270):
            raise ValueError("invalid vision rotation")
        if self.camera_detection_fps <= 0:
            raise ValueError("camera detection FPS must be positive")
        if (
            self.vision_max_age_s <= 0
            or self.vision_window_s <= 0
            or self.vision_frame_count < 2
        ):
            raise ValueError("invalid vision window or freshness configuration")


class BackendNotRunning(RuntimeError):
    """Raised when a request requires an active console session."""


class TaskConflict(RuntimeError):
    """Raised when a second task is submitted while one is active."""


class EventHub:
    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue[ConsoleEvent]] = set()

    def subscribe(self) -> asyncio.Queue[ConsoleEvent]:
        queue: asyncio.Queue[ConsoleEvent] = asyncio.Queue(maxsize=100)
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[ConsoleEvent]) -> None:
        self._subscribers.discard(queue)

    def publish(self, event: ConsoleEvent) -> None:
        for queue in tuple(self._subscribers):
            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            queue.put_nowait(event)


class ConsoleBackend(SkillToolObserver):
    """Own the Agent, SkillRuntime, robot connection, camera, and UI state."""

    def __init__(
        self,
        config: BackendConfig | None = None,
        *,
        robot: RobotAdapter | None = None,
        agent_factory: AgentFactory | None = None,
        camera_factory: Callable[[], RealSensePersonDetector] | None = None,
        vision_agent_factory: Callable[[str], SocialVisionAgent] | None = None,
    ) -> None:
        self.config = config or BackendConfig()
        self.hardware_robot: HardwareRobot | None = None
        if robot is not None:
            self.robot = robot
        elif self.config.hardware:
            self.hardware_robot = create_hardware_robot(
                self.config.robot_model,
                network_interface=self.config.network_interface,
                domain_id=self.config.domain_id,
            )
            self.robot = self.hardware_robot
        else:
            self.robot = create_simulated_robot(self.config.robot_model)

        self.runtime = SkillRuntime(self.robot)
        if self.config.robot_model == "go2":
            register_go2_skills(
                self.runtime,
                include_operator_only=self.config.include_operator_only_skills,
            )
        else:
            register_g1_skills(
                self.runtime,
                include_operator_only=self.config.include_operator_only_skills,
            )
        self._agent_factory = agent_factory or self._build_agent
        self._camera_factory = camera_factory or self._build_camera
        self._vision_agent_factory = vision_agent_factory or self._build_vision_agent
        self._vision_worker: VisionPolicyWorker | None = None
        self._video_buffer = self._new_video_buffer()
        self._safety_gate = _DepthSafetyGate()
        self._agent: ChatAgent | None = None
        self._audio: UnitreeAudioOutput | None = None
        self._camera: RealSensePersonDetector | None = None
        self._camera_task: asyncio.Task[None] | None = None
        self._safety_stop_task: asyncio.Task[None] | None = None
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._active_task: asyncio.Task[None] | None = None
        self._lifecycle_lock = asyncio.Lock()
        self._task_lock = asyncio.Lock()
        self.events = EventHub()

        self.backend = False
        self.starting = False
        self.busy = False
        self.prompt_saved = True
        self.system_prompt = system_prompt_for(self.config.robot_model)
        self.session_id = "—"
        self.task_id: str | None = None
        self.camera_source: Literal["demo", "local"] = self.config.camera_source
        self.model_status = "待命"
        self.skill_status = "IDLE"
        self.skill_name = "等待调度"
        self.progress = 0
        self.progress_text = "等待执行"
        self.active_step = -1
        self.current_task = ""
        self.model_output = ""
        self.model_duration_s = 0.0
        self.latency_ms: int | None = None
        self.task_count = 0
        self.robot_connected = not self.config.hardware
        self.robot_details: dict[str, object] = {}
        self.camera_status: Literal["idle", "starting", "ready", "error"] = (
            "ready" if self.camera_source == "demo" else "idle"
        )
        self.camera_error: str | None = None
        self.latest_frame: bytes | None = None
        self.latest_observation: dict[str, object] | None = None
        self.frame_version = 0
        self.tools: list[ToolCall] = []
        self.logs: list[ConsoleLog] = []

    def _new_video_buffer(self) -> VideoBuffer:
        return VideoBuffer(
            window_s=self.config.vision_window_s,
            max_frames=max(
                self.config.vision_frame_count,
                round(self.config.camera_fps * self.config.vision_window_s),
            ),
        )

    def _build_vision_agent(self, instruction: str) -> SocialVisionAgent:
        if self.config.vision_backend == "unifolm":
            invoker = UnifolmVisionInvoker(
                self.config.vision_model,
                base_url=self.config.vision_url,
                max_new_tokens=96,
                timeout_s=120,
            )
        else:
            invoker = OllamaVisionInvoker(
                self.config.vision_model,
                base_url=self.config.vision_url,
                constrain_json=False,
                max_new_tokens=256,
                think=False,
            )
        return SocialVisionAgent(
            model_name=self.config.vision_model,
            prompt_profile="egocentric",
            generate_speech=True,
            task_context=f"{self.system_prompt}\nCurrent task: {instruction}",
            timeout_s=120,
            invoker=invoker,
        )

    def _build_agent(
        self,
        runtime: SkillRuntime,
        system_prompt: str,
        observer: SkillToolObserver,
    ) -> ChatAgent:
        return RobotAgent(
            runtime,
            model_name=self.config.model_name,
            base_url=self.config.ollama_url,
            system_prompt=system_prompt,
            tool_observer=observer,
        )

    def _build_camera(self) -> RealSensePersonDetector:
        return RealSensePersonDetector(
            serial=self.config.camera_serial,
            width=self.config.camera_width,
            height=self.config.camera_height,
            fps=self.config.camera_fps,
            detection_fps=self.config.camera_detection_fps,
            rgb_rotation_deg=self.config.vision_rotation_deg,
        )

    async def start(self) -> ConsoleSnapshot:
        async with self._lifecycle_lock:
            if self.backend:
                return self.snapshot()
            self.starting = True
            await self._log("INFO", "backend", "正在初始化后端服务。")
            self._emit_state()
            try:
                if self.hardware_robot is not None:
                    await self.hardware_robot.connect()
                    self.robot_connected = True
                    if self.config.audio_enabled and self.config.robot_model != "go2":
                        self._audio = UnitreeAudioOutput(
                            self.hardware_robot,
                            speaker_id=self.config.speaker_id,
                        )
                        await self._audio.connect()
                    elif self.config.audio_enabled:
                        await self._log(
                            "WARN",
                            "audio",
                            "Go2 不使用 G1 AudioClient TTS，语音输出已禁用。",
                        )
                else:
                    state = await self.robot.get_state()
                    self.robot_connected = state.connected
                    self.robot_details = dict(state.details)

                self._agent = self._agent_factory(
                    self.runtime,
                    self.system_prompt,
                    self,
                )
                self.backend = True
                self.starting = False
                self.session_id = uuid.uuid4().hex[:6].upper()
                self.model_status = "待命"
                self.skill_status = "IDLE"
                self._heartbeat_task = asyncio.create_task(
                    self._heartbeat_loop(),
                    name="g1-console-heartbeat",
                )
                if self.camera_source == "local":
                    try:
                        await self._start_camera()
                    except PerceptionError:
                        # The optional D435i may be absent while the Agent and
                        # robot console remain otherwise usable.
                        pass
                mode = "真机" if self.config.hardware else "模拟"
                model_label = "Go2" if self.config.robot_model == "go2" else "G1"
                await self._log(
                    "INFO",
                    "backend",
                    f"后端已启动 · {model_label} · {mode}模式。",
                )
                await self._log("INFO", "agent", "Agent 与 SkillRuntime 已就绪。")
                self._emit_state()
                return self.snapshot()
            except Exception as exc:
                self.backend = False
                self.starting = False
                self.robot_connected = False
                await self._log("ERROR", "backend", f"后端启动失败：{exc}")
                self._emit_state()
                await self._close_resources()
                raise

    async def stop(self) -> ConsoleSnapshot:
        async with self._lifecycle_lock:
            await self.cancel_task("后端服务已停止")
            await self._close_resources()
            self.backend = False
            self.starting = False
            self.latency_ms = None
            self.robot_connected = not self.config.hardware
            self.model_status = "已停止"
            self.skill_status = "STOPPED"
            await self._log("WARN", "backend", "后端服务已停止。")
            self._emit_state()
            return self.snapshot()

    async def _close_resources(self) -> None:
        heartbeat = self._heartbeat_task
        self._heartbeat_task = None
        if heartbeat is not None and heartbeat is not asyncio.current_task():
            heartbeat.cancel()
            await asyncio.gather(heartbeat, return_exceptions=True)
        await self._stop_camera()
        safety_stop = self._safety_stop_task
        if safety_stop is not None and safety_stop is not asyncio.current_task():
            await asyncio.gather(safety_stop, return_exceptions=True)
        if self._audio is not None:
            await self._audio.close()
            self._audio = None
        if self.hardware_robot is not None:
            await self.hardware_robot.close()
        self._agent = None

    async def update_system_prompt(self, prompt: str) -> ConsoleSnapshot:
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("system prompt must not be empty")
        if self.busy:
            raise TaskConflict("cannot update the system prompt during a task")
        self.prompt_saved = False
        self.system_prompt = prompt
        if self.backend:
            self._agent = self._agent_factory(self.runtime, prompt, self)
        self.prompt_saved = True
        await self._log(
            "INFO",
            "config",
            f"系统提示词已更新（{len(prompt)} 字）。",
        )
        self._emit_state()
        return self.snapshot()

    async def set_camera_source(self, source: str) -> ConsoleSnapshot:
        if self.busy:
            raise TaskConflict("stop the current task before switching cameras")
        normalized = "local" if source in {"local", "d435i"} else "demo"
        if normalized == self.camera_source and (
            normalized == "demo" or self._camera_task is not None
        ):
            return self.snapshot()
        if normalized == "demo":
            await self._stop_camera()
            self.camera_source = "demo"
            self.camera_status = "ready"
            self.camera_error = None
            self.latest_frame = None
            self.latest_observation = None
            await self._log("INFO", "camera", "已切换至模拟视频源。")
        else:
            self.camera_source = "local"
            if self.backend:
                await self._start_camera()
            else:
                self.camera_status = "idle"
            await self._log("INFO", "camera", "已选择 USB RealSense D435i。")
        self._emit_state()
        return self.snapshot()

    async def _start_camera(self) -> None:
        await self._stop_camera()
        self.camera_status = "starting"
        self.camera_error = None
        self._emit_state()
        camera = self._camera_factory()
        try:
            await asyncio.to_thread(camera.open)
        except Exception as exc:
            self.camera_status = "error"
            self.camera_error = str(exc)
            await self._log("ERROR", "camera", f"D435i 启动失败：{exc}")
            raise
        self._camera = camera
        self.camera_status = "ready"
        self._camera_task = asyncio.create_task(
            self._camera_loop(camera),
            name="g1-console-camera",
        )

    async def _stop_camera(self) -> None:
        task = self._camera_task
        self._camera_task = None
        if task is not None and task is not asyncio.current_task():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        camera = self._camera
        self._camera = None
        if camera is not None:
            await asyncio.to_thread(camera.close)
        if self.camera_source == "local":
            self.camera_status = "idle"
        self.latest_frame = None
        self.latest_observation = None
        self._video_buffer = self._new_video_buffer()

    async def _camera_loop(self, camera: RealSensePersonDetector) -> None:
        try:
            while True:
                frame = await asyncio.to_thread(camera.capture_frame)
                if (
                    getattr(camera, "rgb_rotation_deg", 0)
                    == self.config.vision_rotation_deg
                ):
                    self.latest_frame = self._frame_jpeg(frame)
                else:
                    self.latest_frame = await asyncio.to_thread(
                        self._vision_frame_jpeg, frame
                    )
                # The preview and model receive the same oriented JPEG bytes.
                frame = replace(frame, rgb=self.latest_frame)
                self._video_buffer.push(frame)
                transition = self._safety_gate.update(frame.nearest_obstacle_distance_m)
                worker = self._vision_worker
                if worker is not None:
                    worker.observe_frame(frame)
                    worker.set_safety_latched(self._safety_gate.latched)
                    if transition == "triggered":
                        self._schedule_depth_safety_stop(worker)
                self.latest_observation = {
                    **frame.observation.to_dict(),
                    "nearest_obstacle_distance_m": (frame.nearest_obstacle_distance_m),
                }
                self.frame_version += 1
                self.camera_status = "ready"
                self.camera_error = None
                self._emit(
                    "camera",
                    {
                        "frameVersion": self.frame_version,
                        "observation": self.latest_observation,
                    },
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - keep the camera worker alive
            self.camera_status = "error"
            self.camera_error = str(exc)
            self.latest_frame = None
            await self._log("ERROR", "camera", f"D435i 采集失败：{exc}")
            self._emit_state()

    def _schedule_depth_safety_stop(self, worker: VisionPolicyWorker) -> None:
        task = self._safety_stop_task
        if task is not None and not task.done():
            return
        self._safety_stop_task = asyncio.create_task(
            self._handle_depth_safety_stop(worker),
            name="g1-console-depth-safety-stop",
        )

    async def _handle_depth_safety_stop(
        self,
        worker: VisionPolicyWorker,
    ) -> None:
        current = asyncio.current_task()
        try:
            await self._log(
                "WARN",
                "perception.safety",
                "深度安全停止：仅限制底盘，非手臂安全保证。",
            )
            await worker.stop_locomotion_for_safety("console depth safety stop")
        except Exception as exc:  # noqa: BLE001 - isolate the camera producer
            await self._log(
                "ERROR",
                "perception.safety",
                f"深度安全停止执行失败：{exc}",
            )
        finally:
            if self._safety_stop_task is current:
                self._safety_stop_task = None

    def _vision_frame_jpeg(self, frame: CameraFrame) -> bytes:
        data = self._frame_jpeg(frame)
        if not self.config.vision_rotation_deg:
            return data
        import io

        from PIL import Image

        with Image.open(io.BytesIO(data)) as image:
            image = image.convert("RGB").rotate(
                -self.config.vision_rotation_deg, expand=True
            )
            output = io.BytesIO()
            image.save(output, format="JPEG", quality=90)
            return output.getvalue()

    @staticmethod
    def _frame_jpeg(frame: CameraFrame) -> bytes:
        if isinstance(frame.rgb, bytes):
            return frame.rgb
        if isinstance(frame.rgb, (bytearray, memoryview)):
            return bytes(frame.rgb)
        try:
            import cv2  # type: ignore[import-not-found]

            bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
            encoded, payload = cv2.imencode(".jpg", bgr)
        except Exception as exc:
            raise PerceptionError(f"could not encode D435i RGB frame: {exc}") from exc
        if not encoded:
            raise PerceptionError("could not encode D435i RGB frame")
        return bytes(payload)

    async def submit_task(
        self,
        instruction: str,
        *,
        camera_source: str | None = None,
    ) -> ConsoleSnapshot:
        instruction = instruction.strip()
        if not instruction:
            raise ValueError("task instruction must not be empty")
        if not self.backend or self._agent is None:
            raise BackendNotRunning("backend session is not running")
        async with self._task_lock:
            if self.busy:
                raise TaskConflict("another task is already running")
            if camera_source is not None:
                await self.set_camera_source(camera_source)
            if self.camera_source == "local" and self.camera_status != "ready":
                raise PerceptionError("本地相机未就绪，不能启动视觉任务")
            self.task_id = uuid.uuid4().hex
            self.busy = True
            self.current_task = instruction
            self.model_output = "已收到指令。\n"
            self.model_status = "生成中"
            self.skill_status = "RUNNING"
            self.skill_name = "等待 Agent 调度"
            self.progress = 10
            self.progress_text = "正在感知环境"
            self.active_step = 0
            self.model_duration_s = 0.0
            self.tools.clear()
            self._active_task = asyncio.create_task(
                self._run_task(self.task_id, instruction),
                name=f"g1-console-task-{self.task_id[:8]}",
            )
        await self._log("INFO", "agent", f"收到任务：{instruction}")
        self._emit_state()
        return self.snapshot()

    async def _run_task(self, task_id: str, instruction: str) -> None:
        started = time.monotonic()
        try:
            if self.camera_source == "local":
                await self._run_vision_task(instruction)
                return
            camera_result = {
                "source": self.camera_source,
                "mode": "d435i" if self.camera_source == "local" else "simulated",
                "frame_available": (
                    self.latest_frame is not None
                    if self.camera_source == "local"
                    else True
                ),
                "observation": self.latest_observation,
            }
            self._record_tool(
                "camera.get_frame", {"source": self.camera_source}, camera_result
            )
            await self._log("INFO", "tools", "camera.get_frame → 返回成功")
            self.progress = 40
            self.progress_text = "正在规划任务"
            self.active_step = 1
            self._emit_state()

            agent = self._agent
            if agent is None:
                raise BackendNotRunning("Agent is not initialized")
            reply = await agent.chat(instruction)
            self.model_output += f"\n{reply}"
            if self._audio is not None:
                try:
                    await self._audio.speak(reply)
                except AudioOutputError as exc:
                    await self._log("WARN", "audio", f"语音播报失败：{exc}")

            if self.task_id != task_id:
                return
            self.progress = 100
            self.progress_text = "执行完成"
            self.active_step = 3
            self.model_status = "已完成"
            if self.skill_status == "RUNNING":
                self.skill_status = "DONE"
            self.busy = False
            self.task_count += 1
            await self._log("INFO", "agent", "任务执行完成。")
        except asyncio.CancelledError:
            if self.task_id == task_id:
                self.busy = False
                self.model_status = "已停止"
                self.skill_status = "STOPPED"
                self.progress_text = "任务已停止"
            raise
        except (AgentError, BackendNotRunning, RobotCommandError, ValueError) as exc:
            if self.task_id == task_id:
                self.busy = False
                self.model_status = "失败"
                self.skill_status = "FAILED"
                self.progress_text = "执行失败"
                self.model_output += f"\n\n执行失败：{exc}"
            await self._log("ERROR", "agent", f"任务执行失败：{exc}")
        except Exception as exc:  # noqa: BLE001 - keep the API worker alive
            if self.task_id == task_id:
                self.busy = False
                self.model_status = "失败"
                self.skill_status = "FAILED"
                self.progress_text = "执行失败"
                self.model_output += f"\n\n执行失败：{exc}"
            await self._log("ERROR", "agent", f"任务执行异常：{exc}")
        finally:
            self.model_duration_s = round(time.monotonic() - started, 3)
            if self._active_task is asyncio.current_task():
                self._active_task = None
            self._emit_state()

    async def _run_vision_task(self, instruction: str) -> None:
        """Continuous, cancellable vision task, reusing the CLI execution boundary."""
        agent = self._vision_agent_factory(instruction)
        worker = VisionPolicyWorker(
            self.runtime,
            agent,
            self._video_buffer,
            speech=self._audio,
            frame_count=self.config.vision_frame_count,
            max_decision_age_s=self.config.vision_max_age_s,
            interval_s=0.5,
        )
        self._vision_worker = worker
        try:
            self.model_status = "加载视觉模型"
            self.model_output = f"视觉交互 · {self.config.vision_model}\n任务：{instruction}\n持续运行，点击停止任务结束。"
            self._emit_state()
            await agent.warmup()
            worker.set_safety_latched(self._safety_gate.latched)
            # Wait for the initial window, but never start on an unavailable camera.
            deadline = time.monotonic() + self.config.vision_max_age_s
            while (
                len(self._video_buffer) < 2
                and self.camera_status == "ready"
                and time.monotonic() < deadline
            ):
                await asyncio.sleep(0.05)
            await self._check_vision_camera()
            await worker.start()
            self.task_count += 1
            await self._log(
                "INFO",
                "vision",
                "已接入实时RGB窗口与SkillRuntime；范围：握手/挥手/击掌。",
            )
            while True:
                await self._check_vision_camera()
                self.model_status = "持续视觉交互"
                self.progress_text = "正在观察手势 · 停止任务可结束"
                self.skill_name = (
                    "执行视觉技能" if worker.active_behavior else "等待确认手势"
                )
                self.progress = 40
                self.active_step = 1
                for record in worker.drain_policy_decisions():
                    payload = record.to_dict()
                    payload["input_preprocessing"] = {
                        "rotation_deg": self.config.vision_rotation_deg
                    }
                    self._record_tool(
                        "vision.decide", {"instruction": instruction}, payload
                    )
                    self.model_output = json.dumps(
                        payload, ensure_ascii=False, indent=2, default=str
                    )
                    self.model_duration_s = (
                        float(record.model_metrics.get("round_trip_s", 0))
                        if record.model_metrics
                        else 0
                    )
                    self.skill_status = "RUNNING" if worker.active_behavior else "IDLE"
                for outcome in worker.drain_outcomes():
                    self._record_tool("vision.outcome", {}, outcome.to_dict())
                    if outcome.skill_result is not None:
                        await self.after_skill(
                            outcome.decision.skill or "vision",
                            outcome.decision.arguments,
                            outcome.skill_result.to_dict(),
                        )
                        await self._log(
                            "INFO", "audio", f"TTS调用成功返回：{outcome.speech_spoken}"
                        )
                    elif outcome.suppressed_reason:
                        await self._log("INFO", "vision", outcome.suppressed_reason)
                errors = worker.drain_errors()
                if errors:
                    raise RuntimeError(
                        f"视觉任务错误：{errors[0].stage}: {errors[0].message}"
                    )
                self._emit_state()
                await asyncio.sleep(0.2)
        finally:
            try:
                await worker.stop()
            finally:
                self._vision_worker = None
                await agent.close()

    async def _check_vision_camera(self) -> None:
        frames = self._video_buffer.sample(1)
        if (
            self.camera_status != "ready"
            or not frames
            or time.monotonic() - frames[-1].observed_at_s
            > self.config.vision_max_age_s
        ):
            raise PerceptionError("相机中断或画面过期，视觉任务已停止")

    async def cancel_task(self, reason: str = "用户停止了任务") -> ConsoleSnapshot:
        task = self._active_task
        if task is None or task.done():
            return self.snapshot()
        self._active_task = None
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        try:
            await self.robot.stop()
        except (RobotCommandError, RuntimeError) as exc:
            await self._log("ERROR", "executor", f"停止机器人失败：{exc}")
        self.busy = False
        self.skill_status = "STOPPED"
        self.model_status = "已停止"
        self.progress_text = "任务已停止"
        self.model_output += f"\n\n执行已中断：{reason}。"
        await self._log("WARN", "executor", reason)
        self._emit_state()
        return self.snapshot()

    async def execute_skill(
        self,
        skill_name: str,
        arguments: dict[str, object],
    ) -> dict[str, object]:
        if not self.backend:
            raise BackendNotRunning("backend session is not running")
        if self.busy:
            raise TaskConflict("another task is already running")
        await self.before_skill(skill_name, arguments)
        result = await self.runtime.execute(skill_name, **arguments)
        payload = result.to_dict()
        await self.after_skill(skill_name, arguments, payload)
        return payload

    async def clear_logs(self) -> ConsoleSnapshot:
        self.logs.clear()
        self._emit_state()
        return self.snapshot()

    async def before_skill(
        self,
        skill_name: str,
        arguments: dict[str, object],
    ) -> None:
        self.skill_name = skill_name
        self.skill_status = "RUNNING"
        self.progress = max(self.progress, 70)
        self.progress_text = "正在执行技能"
        self.active_step = 2
        await self._log("INFO", "executor", f"{skill_name} 开始执行。")
        self._emit_state()

    async def after_skill(
        self,
        skill_name: str,
        arguments: dict[str, object],
        result: dict[str, object],
    ) -> None:
        self._record_tool(skill_name, arguments, result)
        success = result.get("success") is True
        self.skill_status = "DONE" if success else "FAILED"
        level: Literal["INFO", "ERROR"] = "INFO" if success else "ERROR"
        message = "执行完成" if success else f"执行失败：{result.get('message', '')}"
        await self._log(level, "executor", f"{skill_name} {message}。")
        self._emit_state()

    def _record_tool(
        self,
        name: str,
        arguments: dict[str, object],
        result: dict[str, object],
    ) -> None:
        payload = json.dumps(
            {"arguments": arguments, "result": result},
            ensure_ascii=False,
            indent=2,
            default=str,
        )
        self.tools.append(
            ToolCall(
                name=name,
                payload=payload,
                arguments=arguments,
                result=result,
            )
        )
        if len(self.tools) > 100:
            del self.tools[:-100]

    async def _heartbeat_loop(self) -> None:
        while True:
            started = time.monotonic()
            try:
                state = await self.robot.get_state()
                self.robot_connected = state.connected
                self.robot_details = dict(state.details)
                self.latency_ms = max(
                    0,
                    round((time.monotonic() - started) * 1000),
                )
                self._emit(
                    "heartbeat",
                    {
                        "latencyMs": self.latency_ms,
                        "connected": self.robot_connected,
                        "executor": "running" if self.busy else "idle",
                    },
                )
            except Exception as exc:  # noqa: BLE001 - report transient SDK errors
                self.robot_connected = False
                self.latency_ms = None
                await self._log("WARN", "heartbeat", f"机器人状态读取失败：{exc}")
            await asyncio.sleep(2.0)

    async def _log(
        self,
        level: Literal["DEBUG", "INFO", "WARN", "ERROR"],
        source: str,
        message: str,
    ) -> None:
        now = datetime.now(UTC)
        entry = ConsoleLog(
            id=uuid.uuid4().hex,
            time=now.astimezone().strftime("%H:%M:%S"),
            timestamp=now,
            level=level,
            source=source,
            message=message,
        )
        self.logs.append(entry)
        if len(self.logs) > 500:
            del self.logs[:-500]
        self._emit("log", entry.model_dump(mode="json", by_alias=True))

    def _emit_state(self) -> None:
        self._emit(
            "state",
            self.snapshot().model_dump(mode="json", by_alias=True),
        )

    def _emit(self, event_type: str, data: dict[str, object]) -> None:
        self.events.publish(
            ConsoleEvent(
                type=event_type,
                timestamp=datetime.now(UTC),
                data=data,
            )
        )

    def snapshot(self) -> ConsoleSnapshot:
        return ConsoleSnapshot(
            backend=self.backend,
            starting=self.starting,
            busy=self.busy,
            prompt_saved=self.prompt_saved,
            system_prompt=self.system_prompt,
            session_id=self.session_id,
            task_id=self.task_id,
            camera_source=self.camera_source,
            model_status=self.model_status,
            skill_status=self.skill_status,
            skill_name=self.skill_name,
            progress=self.progress,
            progress_text=self.progress_text,
            active_step=self.active_step,
            current_task=self.current_task,
            model_output=self.model_output,
            model_duration=self.model_duration_s,
            latency=self.latency_ms,
            task_count=self.task_count,
            robot=RobotView(
                mode="hardware" if self.config.hardware else "simulation",
                connected=self.robot_connected,
                details={
                    "robot_model": self.config.robot_model,
                    **self.robot_details,
                },
            ),
            camera=CameraView(
                source=self.camera_source,
                label=(
                    "USB RealSense D435i"
                    if self.camera_source == "local"
                    else "模拟视频源"
                ),
                status=self.camera_status,
                frame_available=(
                    self.latest_frame is not None
                    if self.camera_source == "local"
                    else True
                ),
                frame_version=self.frame_version,
                width=self.config.camera_width,
                height=self.config.camera_height,
                fps=self.config.camera_fps,
                observation=self.latest_observation,
                error=self.camera_error,
            ),
            tools=list(self.tools),
            logs=list(self.logs),
        )

    def skill_catalog(self) -> list[dict[str, object]]:
        return [
            {
                "name": skill.metadata.name,
                "description": skill.metadata.description,
                "version": skill.metadata.version,
                "tags": list(skill.metadata.tags),
                "requiredResources": list(skill.metadata.required_resources),
                "timeoutS": skill.metadata.timeout_s,
                "interruptible": skill.metadata.interruptible,
                "argumentsSchema": skill.args_model.model_json_schema(),
            }
            for skill in self.runtime.registry.list()
        ]
