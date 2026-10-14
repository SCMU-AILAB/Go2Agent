"""GPU-server entry points.

    go2agent-server all                      perception + ASR + TTS + VLM lifecycle/proxy on :8100
    go2agent-server model load|unload|status manage the VLM of a running server (default http://127.0.0.1:8100)

Model files live under --models-dir (default $GO2AGENT_MODELS_DIR or /models), one directory per family:
    yolo/yolo11s.pt   faster-whisper/<name>/   piper/<voice>.onnx   Qwen3-VL/<name>/
A file that is missing there falls back to the hub name (downloaded on first use).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI

from go2agent.server import asr, models, perception, tts

DEFAULT_PORT = 8100
DEFAULT_VLM = "Qwen3-VL-8B-Instruct-FP8"


def _models_dir() -> Path:
    return Path(os.environ.get("GO2AGENT_MODELS_DIR", "/models"))


def in_models(models_dir: Path, rel: str, fallback: str) -> str:
    """`models_dir/rel` if it exists, else `fallback` (a hub name)."""
    path = models_dir / rel
    return str(path) if path.exists() else fallback


def create_all_app(
    *,
    model: str = "yolo11s.pt",
    classes: Sequence[str] = ("person",),
    device: str | None = None,
    asr_model: str = "small",
    asr_device: str = "auto",
    asr_compute_type: str | None = None,
    voice: str = "zh_CN-huayan-medium",
    model_dir: str = "models/piper",
    vlm: models.VLLMManager | None = None,
    detector_factory: Callable[[], perception.Detector] | None = None,
    asr_engine_factory: Callable[[], asr.ASREngine] | None = None,
    tts_engine_factory: Callable[[], tts.TTSEngine] | None = None,
) -> FastAPI:
    app = perception.create_app(model, classes, device=device, detector_factory=detector_factory)
    app.title = "Go2Agent GPU services"
    app.include_router(
        asr.create_router(asr_model, asr_device, asr_compute_type, engine_factory=asr_engine_factory)
    )
    app.include_router(tts.create_router(voice, model_dir, engine_factory=tts_engine_factory))
    if vlm is not None:
        app.include_router(models.create_router(vlm))
    return app


async def _serve(app: FastAPI, host: str, port: int, vlm: models.VLLMManager | None) -> None:
    server = uvicorn.Server(uvicorn.Config(app, host=host, port=port))
    try:
        await server.serve()
    finally:
        if vlm is not None:
            await vlm.close()  # never leave vLLM holding the GPU


def _model_command(args: argparse.Namespace) -> None:
    url = args.url.rstrip("/")
    with httpx.Client(timeout=None if args.action == "load" else 10) as http:
        if args.action == "status":
            r = http.get(f"{url}/models")
        else:
            body = {"model": args.model} if args.action == "load" and args.model else None
            r = http.post(f"{url}/models/brain/{args.action}", json=body)
    print(json.dumps(r.json(), ensure_ascii=False, indent=2))
    if r.status_code != 200:
        sys.exit(1)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="go2agent-server")
    subparsers = parser.add_subparsers(dest="service", required=True)
    for name, port in (("all", DEFAULT_PORT), ("perception", DEFAULT_PORT), ("asr", 8101), ("tts", 8102)):
        command = subparsers.add_parser(name, help=f"Run {name} services" if name == "all" else f"Run {name}")
        command.add_argument("--host", default="0.0.0.0")
        command.add_argument("--port", type=int, default=port)
        command.add_argument("--models-dir", type=Path, default=_models_dir())
        if name in ("all", "perception"):
            command.add_argument("--model", help="YOLO weights (default: <models-dir>/yolo/yolo11s.pt)")
            command.add_argument("--classes", nargs="+", default=["person"], help="COCO class names")
            command.add_argument(
                "--device", default=None, help="Perception: cuda, mps, or cpu (default: auto)"
            )
        if name in ("all", "asr"):
            prefix = "asr-" if name == "all" else ""
            command.add_argument(
                f"--{prefix}model",
                default="large-v3-turbo",
                help="faster-whisper name; <models-dir>/faster-whisper/<name> if present",
            )
            command.add_argument(f"--{prefix}device", choices=("auto", "cuda", "cpu"), default="auto")
            command.add_argument(f"--{prefix}compute-type", default=None)
        if name in ("all", "tts"):
            command.add_argument("--voice", default="zh_CN-huayan-medium")
            command.add_argument("--model-dir", help="Piper voices (default: <models-dir>/piper)")
        if name == "all":
            command.add_argument(
                "--vlm-model",
                default=DEFAULT_VLM,
                help="<models-dir>/Qwen3-VL/<name> if present, else a hub repo id",
            )
            command.add_argument("--vlm-gpu-util", type=float, default=0.70)
            command.add_argument("--vlm-port", type=int, default=8001, help="internal vLLM port")
            command.add_argument("--no-vlm", action="store_true", help="do not offer the VLM lifecycle")
    mc = subparsers.add_parser("model", help="load / unload / status of the VLM on a running server")
    mc.add_argument("action", choices=("status", "load", "unload"))
    mc.add_argument("--url", default=f"http://127.0.0.1:{DEFAULT_PORT}")
    mc.add_argument("--model", help="load: hub repo id or path (default: the server's --vlm-model)")

    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or (arguments[0].startswith("-") and arguments[0] not in ("-h", "--help")):
        arguments.insert(0, "all")
    args = parser.parse_args(arguments)
    if args.service == "model":
        _model_command(args)
        return
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    md: Path = args.models_dir
    if hasattr(args, "model_dir"):
        args.model_dir = args.model_dir or str(md / "piper")
    vlm = None
    if args.service == "all":
        if not args.no_vlm:
            vlm_model = args.vlm_model
            if "/" not in vlm_model:
                vlm_model = in_models(md, f"Qwen3-VL/{vlm_model}", f"Qwen/{vlm_model}")
            config = models.VLLMConfig(
                model=vlm_model, port=args.vlm_port, gpu_memory_utilization=args.vlm_gpu_util
            )
            vlm = models.VLLMManager(config, Path("logs/vllm.log"))
        app = create_all_app(
            model=args.model or in_models(md, "yolo/yolo11s.pt", "yolo11s.pt"),
            classes=args.classes,
            device=args.device,
            asr_model=in_models(md, f"faster-whisper/{args.asr_model}", args.asr_model),
            asr_device=args.asr_device,
            asr_compute_type=args.asr_compute_type,
            voice=args.voice,
            model_dir=args.model_dir,
            vlm=vlm,
        )
    elif args.service == "perception":
        model = args.model or in_models(md, "yolo/yolo11s.pt", "yolo11s.pt")
        app = perception.create_app(model=model, classes=args.classes, device=args.device)
    elif args.service == "asr":
        model = in_models(md, f"faster-whisper/{args.model}", args.model)
        app = asr.create_app(model=model, device=args.device, compute_type=args.compute_type)
    else:
        app = tts.create_app(voice=args.voice, model_dir=args.model_dir)
    asyncio.run(_serve(app, args.host, args.port, vlm))


if __name__ == "__main__":
    main()
