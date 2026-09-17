"""FastAPI control plane for the G1 Flutter console."""

from __future__ import annotations

import argparse
import os
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Literal

import uvicorn
from fastapi import FastAPI, HTTPException, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import Field

from agent import DEFAULT_UNIFOLM_MODEL
from perception import PerceptionError

from .backend import (
    ApiModel,
    BackendConfig,
    BackendNotRunning,
    ConsoleBackend,
    ConsoleSnapshot,
    TaskConflict,
)


class HealthResponse(ApiModel):
    status: Literal["ready", "stopped", "starting", "degraded"]
    backend: bool
    hardware: bool
    robot_connected: bool
    camera_status: str


class SystemPromptUpdate(ApiModel):
    system_prompt: str = Field(min_length=1)


class TaskRequest(ApiModel):
    instruction: str = Field(min_length=1)
    camera_source: Literal["demo", "local"] | None = None
    task_mode: Literal["text", "gesture"] | None = None
    # Accepted for older clients; ignored. Gesture skills map 1:1 from vision.
    wave_response: Literal["wave", "heart"] | None = None


class CancelTaskRequest(ApiModel):
    reason: str = "用户停止了任务"


class SkillExecuteRequest(ApiModel):
    arguments: dict[str, object] = Field(default_factory=dict)


class CameraSourceUpdate(ApiModel):
    source: Literal["demo", "local"]


class SkillExecutionResponse(ApiModel):
    result: dict[str, object]
    console: ConsoleSnapshot


def create_app(
    config: BackendConfig | None = None,
    *,
    backend: ConsoleBackend | None = None,
    auto_start: bool = True,
) -> FastAPI:
    """Create an API instance around one long-lived console backend."""

    console = backend or ConsoleBackend(config)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.console_backend = console
        if auto_start:
            try:
                await console.start()
            except Exception as exc:  # noqa: BLE001 - preserve the retry API
                # Startup errors already enter the console log. Keep the API
                # online so the frontend can show the failure and retry.
                app.state.startup_error = str(exc)
        try:
            yield
        finally:
            if console.backend or console.starting:
                await console.stop()

    app = FastAPI(
        title="G1 Agent Console API",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.console_backend = console
    app.add_middleware(
        CORSMiddleware,
        # The console can be served from another machine on the robot's trusted
        # LAN. The API does not use browser credentials or cookies.
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/v1/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        if console.starting:
            status: Literal["ready", "stopped", "starting", "degraded"] = "starting"
        elif console.backend and console.robot_connected:
            status = "ready"
        elif console.backend:
            status = "degraded"
        else:
            status = "stopped"
        return HealthResponse(
            status=status,
            backend=console.backend,
            hardware=console.config.hardware,
            robot_connected=console.robot_connected,
            camera_status=console.camera_status,
        )

    @app.get("/api/v1/console", response_model=ConsoleSnapshot)
    async def get_console() -> ConsoleSnapshot:
        return console.snapshot()

    @app.post("/api/v1/session/start", response_model=ConsoleSnapshot)
    async def start_session() -> ConsoleSnapshot:
        try:
            return await console.start()
        except (PerceptionError, RuntimeError) as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    @app.post("/api/v1/session/stop", response_model=ConsoleSnapshot)
    async def stop_session() -> ConsoleSnapshot:
        return await console.stop()

    @app.put("/api/v1/config/system-prompt", response_model=ConsoleSnapshot)
    async def update_system_prompt(body: SystemPromptUpdate) -> ConsoleSnapshot:
        try:
            return await console.update_system_prompt(body.system_prompt)
        except TaskConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/v1/tasks", response_model=ConsoleSnapshot, status_code=202)
    async def submit_task(body: TaskRequest) -> ConsoleSnapshot:
        try:
            return await console.submit_task(
                body.instruction,
                camera_source=body.camera_source,
                task_mode=body.task_mode,
                wave_response=body.wave_response,
            )
        except TaskConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except BackendNotRunning as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except PerceptionError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/v1/tasks/current/cancel", response_model=ConsoleSnapshot)
    async def cancel_task(body: CancelTaskRequest | None = None) -> ConsoleSnapshot:
        reason = body.reason if body is not None else "用户停止了任务"
        return await console.cancel_task(reason)

    @app.get("/api/v1/skills")
    async def get_skills() -> dict[str, list[dict[str, object]]]:
        return {"skills": console.skill_catalog()}

    @app.post(
        "/api/v1/skills/{skill_name}/execute",
        response_model=SkillExecutionResponse,
    )
    async def execute_skill(
        skill_name: str,
        body: SkillExecuteRequest,
    ) -> SkillExecutionResponse:
        try:
            result = await console.execute_skill(skill_name, body.arguments)
        except TaskConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except BackendNotRunning as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return SkillExecutionResponse(result=result, console=console.snapshot())

    @app.put("/api/v1/camera/source", response_model=ConsoleSnapshot)
    async def update_camera_source(body: CameraSourceUpdate) -> ConsoleSnapshot:
        try:
            return await console.set_camera_source(body.source)
        except TaskConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except PerceptionError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    @app.get("/api/v1/camera/frame.jpg")
    async def camera_frame() -> Response:
        frame = console.latest_frame
        if frame is None:
            raise HTTPException(status_code=404, detail="camera frame is not available")
        return Response(
            content=frame,
            media_type="image/jpeg",
            headers={
                "Cache-Control": "no-store, max-age=0",
                "X-Frame-Version": str(console.frame_version),
            },
        )

    @app.delete("/api/v1/logs", response_model=ConsoleSnapshot)
    async def clear_logs() -> ConsoleSnapshot:
        return await console.clear_logs()

    @app.websocket("/api/v1/events")
    async def events(websocket: WebSocket) -> None:
        await websocket.accept()
        queue = console.events.subscribe()
        try:
            await websocket.send_json(
                {
                    "type": "state",
                    "timestamp": datetime.now(UTC).isoformat(),
                    "data": console.snapshot().model_dump(mode="json", by_alias=True),
                }
            )
            while True:
                event = await queue.get()
                await websocket.send_json(event.model_dump(mode="json", by_alias=True))
        except WebSocketDisconnect:
            pass
        finally:
            console.events.unsubscribe(queue)

    return app


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the G1 console FastAPI server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--hardware", action="store_true")
    parser.add_argument(
        "--robot",
        choices=("g1", "go2"),
        default="g1",
        help="robot model to assemble; go2 uses SportClient and a reduced skill catalog",
    )
    parser.add_argument("--network", default="")
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument("--model")
    parser.add_argument("--ollama-url")
    parser.add_argument("--no-audio", action="store_true")
    parser.add_argument("--speaker-id", type=int, default=0)
    parser.add_argument(
        "--host-audio-device",
        default=os.getenv("G1_AUDIO_DEVICE"),
        help="ALSA/Pulse device for Go2 host speaker TTS (external speaker)",
    )
    parser.add_argument(
        "--host-tts-voice",
        default="cmn",
        help="espeak-ng voice for host speaker TTS (cmn=普通话)",
    )
    parser.add_argument(
        "--camera-source",
        choices=("demo", "local"),
        default="demo",
    )
    parser.add_argument("--camera-serial")
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument(
        "--camera-detection-fps",
        type=float,
        default=5.0,
        help="person detection rate; RGB preview continues at --camera-fps",
    )
    parser.add_argument(
        "--vision-backend", choices=("ollama", "unifolm"), default="ollama"
    )
    parser.add_argument("--vision-model")
    parser.add_argument("--vision-url", default="http://127.0.0.1:11435")
    parser.add_argument("--vision-window-s", type=float, default=0.8)
    parser.add_argument("--vision-frame-count", type=int, default=3)
    parser.add_argument(
        "--vision-rotation-deg",
        type=int,
        choices=(0, 90, 180, 270),
        default=0,
        help="rotate RGB for preview and VLM; use 180 if the camera is mounted upside-down",
    )
    parser.add_argument("--include-operator-only-skills", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    config = BackendConfig(
        hardware=args.hardware,
        robot_model=args.robot,
        network_interface=args.network,
        domain_id=args.domain_id,
        include_operator_only_skills=args.include_operator_only_skills,
        model_name=args.model,
        ollama_url=args.ollama_url,
        audio_enabled=not args.no_audio,
        speaker_id=args.speaker_id,
        host_audio_device=args.host_audio_device,
        host_tts_voice=args.host_tts_voice,
        camera_source=args.camera_source,
        camera_serial=args.camera_serial,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        camera_fps=args.camera_fps,
        camera_detection_fps=args.camera_detection_fps,
        vision_model=(
            args.vision_model
            or (
                DEFAULT_UNIFOLM_MODEL
                if args.vision_backend == "unifolm"
                else "qwen3.5:9b"
            )
        ),
        vision_backend=args.vision_backend,
        vision_url=args.vision_url,
        vision_window_s=args.vision_window_s,
        vision_frame_count=args.vision_frame_count,
        vision_rotation_deg=args.vision_rotation_deg,
    )
    uvicorn.run(create_app(config), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
