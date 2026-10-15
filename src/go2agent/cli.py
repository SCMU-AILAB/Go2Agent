"""`go2agent` command line.

go2agent run --server 192.168.31.143 --robot go2 --camera realsense
go2agent run --server 192.168.31.143 --robot sim --camera video:clip.mp4 --task "跟着穿黑衣服的人"
go2agent replay runs/20261014-180000/thinks/0003.json --server 192.168.31.143
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path

from go2agent.audio.mic import Microphone
from go2agent.audio.speaker import Speaker
from go2agent.behaviors.base import BehaviorEnv
from go2agent.behaviors.geometry import CameraModel
from go2agent.brain.brain import Brain
from go2agent.brain.client import VLMClient, VLMConfig, VLMError
from go2agent.brain.prompt import ImageSizes, system_message
from go2agent.brain.schema import decision_json_schema
from go2agent.robot.base import Robot
from go2agent.robot.go2_actions import GO2_ACTIONS
from go2agent.runtime import Runtime
from go2agent.safety import SafetyConfig, SafetyGate
from go2agent.sensors.base import Camera
from go2agent.timeline import Timeline, load_think, summarize

log = logging.getLogger("go2agent")

SERVICES_PORT = 8100  # GPU server: perception, voice and the VLM proxy (/v1)


def _vlm_url(args: argparse.Namespace) -> str:
    return args.vlm_url or f"http://{args.server}:{SERVICES_PORT}/v1"


def _make_robot(args: argparse.Namespace) -> Robot:
    if args.robot == "sim":
        from go2agent.robot.sim import SimRobot

        return SimRobot()
    from go2agent.robot.go2 import Go2Config, Go2Robot

    return Go2Robot(Go2Config(network_interface=args.network))


def _make_camera(args: argparse.Namespace) -> tuple[Camera, CameraModel]:
    spec: str = args.camera
    if spec == "go2front":
        from go2agent.sensors.go2_front import GO2_FRONT_MODEL, Go2FrontCamera

        # With the real Go2 the driver owns the DDS channel; with the simulated robot the camera opens it.
        camera = Go2FrontCamera(network_interface=args.network, owns_channel=args.robot != "go2")
        return camera, GO2_FRONT_MODEL
    return _make_plain_camera(args), CameraModel()


def _make_closeup(args: argparse.Namespace) -> Camera | None:
    if args.closeup == "none":
        return None
    from go2agent.sensors.realsense import RealSenseCamera

    return RealSenseCamera(rgb_rotation_deg=args.camera_rotation)


def _make_plain_camera(args: argparse.Namespace) -> Camera:
    spec: str = args.camera
    if spec == "realsense":
        from go2agent.sensors.realsense import RealSenseCamera

        return RealSenseCamera(rgb_rotation_deg=args.camera_rotation)
    from go2agent.sensors.video import VideoCamera

    if spec.startswith("device:"):
        return VideoCamera(int(spec.removeprefix("device:")))
    if spec.startswith("video:"):
        return VideoCamera(spec.removeprefix("video:"), loop=True)
    raise SystemExit(f"unknown --camera {spec!r}; use go2front, realsense, video:PATH or device:N")


async def _run(args: argparse.Namespace) -> None:
    import uvicorn

    from go2agent.api.server import create_app
    from go2agent.perception.remote import RemoteTracker

    robot = _make_robot(args)
    safety = SafetyGate(robot, SafetyConfig(allow_dangerous=args.allow_dangerous))
    camera, camera_model = _make_camera(args)
    env = BehaviorEnv(camera_model, {a.name: a for a in safety.allowed_actions()})
    tracker = RemoteTracker(args.perception_url or f"ws://{args.server}:{SERVICES_PORT}/track")
    client = VLMClient(VLMConfig(base_url=_vlm_url(args), model=args.model, timeout=args.vlm_timeout))
    # The wide front camera shows people smaller, so give the brain a larger newest frame.
    sizes = ImageSizes(newest=640, history=320) if args.camera == "go2front" else ImageSizes()
    brain = Brain(client, env, tracker, sizes)
    timeline = Timeline(None if args.no_log else Path(args.log_dir))
    voice_base = f"http://{args.server}:{SERVICES_PORT}"
    speaker = Speaker(f"{voice_base}/tts", device=_device(args.speaker_device)) if args.voice else None
    rt = Runtime(
        robot,
        camera,
        tracker,
        brain,
        safety,
        timeline,
        task=args.task,
        speaker=speaker.say if speaker else None,
        closeup=_make_closeup(args),
    )
    mic: Microphone | None = None
    if speaker is not None:
        spk = speaker
        mic = Microphone(
            f"{voice_base}/asr", rt.hear, device=_device(args.mic_device), muted=lambda: spk.playing
        )

    server = uvicorn.Server(
        uvicorn.Config(create_app(rt), host=args.api_host, port=args.api_port, log_level="warning")
    )
    try:
        await rt.start()  # a failure here (e.g. no camera) still runs the cleanup below
        if mic:
            try:
                mic.start()
            except Exception as e:  # voice is optional: the agent keeps running without it
                log.warning("microphone unavailable, continuing without voice input: %s", e)
                mic = None
        log.info("api on http://%s:%d", args.api_host, args.api_port)
        await server.serve()
    finally:
        if mic:
            await mic.close()
        if speaker:
            await speaker.close()
        await rt.stop()


def _device(spec: str | None) -> str | int | None:
    """Audio device by index or name substring."""
    if spec is None or not spec.isdigit():
        return spec
    return int(spec)


async def _replay(args: argparse.Namespace) -> None:
    messages, recorded = load_think(Path(args.think))
    actions = tuple(a for a in GO2_ACTIONS if args.allow_dangerous or "dangerous" not in a.tags)
    if args.current_prompt:
        messages[0]["content"] = system_message(actions)
    client = VLMClient(VLMConfig(base_url=_vlm_url(args), model=args.model, timeout=args.vlm_timeout))
    print("recorded:", recorded.get("raw") or recorded.get("error"))
    try:
        for i in range(args.n):
            t0 = time.monotonic()
            try:
                reply = await client.complete_json(messages, decision_json_schema(actions))
                print(f"[{i + 1}] {reply.latency_ms:.0f}ms {json.dumps(reply.data, ensure_ascii=False)}")
            except VLMError as e:
                print(f"[{i + 1}] {(time.monotonic() - t0) * 1000:.0f}ms error: {e}")
    finally:
        await client.close()


async def _eval(args: argparse.Namespace) -> None:
    from go2agent.brain.evaluate import run_cases, summarize

    actions = tuple(a for a in GO2_ACTIONS if args.allow_dangerous or "dangerous" not in a.tags)
    client = VLMClient(VLMConfig(base_url=_vlm_url(args), model=args.model, timeout=args.vlm_timeout))
    prompt = system_message(actions) if args.current_prompt else None
    try:
        results = await run_cases(Path(args.cases), client, decision_json_schema(actions), prompt)
    finally:
        await client.close()
    print(summarize(results))


def main() -> None:
    p = argparse.ArgumentParser(prog="go2agent")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--server", default="127.0.0.1", help="GPU server host (VLM + perception + voice)")
        sp.add_argument("--vlm-url", help=f"override, default http://SERVER:{SERVICES_PORT}/v1")
        sp.add_argument("--model", default="brain", help="served model name")
        sp.add_argument("--vlm-timeout", type=float, default=3.0)
        sp.add_argument("--allow-dangerous", action="store_true", help="let the brain use flips/jumps")

    run = sub.add_parser("run", help="run the agent")
    common(run)
    run.add_argument("--robot", choices=["sim", "go2"], default="sim")
    run.add_argument("--network", default="eth0", help="Go2 network interface")
    run.add_argument("--camera", default="go2front", help="go2front | realsense | video:PATH | device:N")
    run.add_argument("--camera-rotation", type=int, default=0, choices=[0, 90, 180, 270])
    run.add_argument(
        "--closeup",
        choices=["none", "realsense"],
        default="none",
        help="second camera the brain also sees (the upward-looking D435i shows people up close)",
    )
    run.add_argument("--perception-url", help=f"override, default ws://SERVER:{SERVICES_PORT}/track")
    run.add_argument("--task", default="", help="initial task")
    run.add_argument(
        "--voice",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="microphone + speaker via the GPU server's ASR/TTS",
    )
    run.add_argument("--mic-device", help="input device index or name")
    run.add_argument("--speaker-device", help="output device index or name")
    run.add_argument("--api-host", default="0.0.0.0")
    run.add_argument("--api-port", type=int, default=8080)
    run.add_argument("--log-dir", default="runs")
    run.add_argument("--no-log", action="store_true")

    rp = sub.add_parser("replay", help="re-send a recorded think to the VLM")
    common(rp)
    rp.add_argument("think", help="runs/<run>/thinks/NNNN.json")
    rp.add_argument("-n", type=int, default=1, help="repeat to measure latency / variance")
    rp.add_argument("--current-prompt", action="store_true", help="swap in the current system prompt")

    ev = sub.add_parser("eval", help="score the brain on labeled recorded thinks")
    common(ev)
    ev.add_argument("cases", help="cases.jsonl, see go2agent.brain.evaluate")
    ev.add_argument("--current-prompt", action="store_true", help="swap in the current system prompt")

    tl = sub.add_parser("timeline", help="print a run's timeline (decisions, behaviors, tracks)")
    tl.add_argument("run", nargs="?", help="runs/<run> (default: the newest run in --log-dir)")
    tl.add_argument("--log-dir", default="runs")
    tl.add_argument("--since", type=float, default=0.0, help="skip the first N seconds")
    tl.add_argument("--all", action="store_true", help="include perception health and speech records")

    args = p.parse_args()
    if args.cmd == "timeline":
        run = Path(args.run) if args.run else max(Path(args.log_dir).iterdir())
        print(f"run: {run}")
        print("\n".join(summarize(run, args.since, args.all)))
        return
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    asyncio.run({"run": _run, "replay": _replay, "eval": _eval}[args.cmd](args))
