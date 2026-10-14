"""GPU-server service entry points."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable, Sequence

import uvicorn
from fastapi import FastAPI

from go2agent.server import asr, perception, tts


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
    detector_factory: Callable[[], perception.Detector] | None = None,
    asr_engine_factory: Callable[[], asr.ASREngine] | None = None,
    tts_engine_factory: Callable[[], tts.TTSEngine] | None = None,
) -> FastAPI:
    app = perception.create_app(model, classes, device=device, detector_factory=detector_factory)
    app.title = "Go2Agent perception and voice"
    app.include_router(
        asr.create_router(asr_model, asr_device, asr_compute_type, engine_factory=asr_engine_factory)
    )
    app.include_router(tts.create_router(voice, model_dir, engine_factory=tts_engine_factory))
    return app


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="go2agent-server")
    subparsers = parser.add_subparsers(dest="service", required=True)
    for name, port in (("all", 8100), ("perception", 8100), ("asr", 8101), ("tts", 8102)):
        command = subparsers.add_parser(name, help=f"Run {name} services" if name == "all" else f"Run {name}")
        command.add_argument("--host", default="0.0.0.0")
        command.add_argument("--port", type=int, default=port)
        if name in ("all", "perception"):
            command.add_argument("--model", default="yolo11s.pt")
            command.add_argument("--classes", nargs="+", default=["person"], help="COCO class names")
            command.add_argument(
                "--device", default=None, help="Perception: cuda, mps, or cpu (default: auto)"
            )
        if name in ("all", "asr"):
            prefix = "asr-" if name == "all" else ""
            command.add_argument(f"--{prefix}model", default="small")
            command.add_argument(f"--{prefix}device", choices=("auto", "cuda", "cpu"), default="auto")
            command.add_argument(f"--{prefix}compute-type", default=None)
        if name in ("all", "tts"):
            command.add_argument("--voice", default="zh_CN-huayan-medium")
            command.add_argument("--model-dir", default="models/piper")
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or (arguments[0].startswith("-") and arguments[0] not in ("-h", "--help")):
        arguments.insert(0, "all")
    args = parser.parse_args(arguments)
    logging.basicConfig(level=logging.INFO)
    if args.service == "all":
        app = create_all_app(
            model=args.model,
            classes=args.classes,
            device=args.device,
            asr_model=args.asr_model,
            asr_device=args.asr_device,
            asr_compute_type=args.asr_compute_type,
            voice=args.voice,
            model_dir=args.model_dir,
        )
    elif args.service == "perception":
        app = perception.create_app(model=args.model, classes=args.classes, device=args.device)
    elif args.service == "asr":
        app = asr.create_app(model=args.model, device=args.device, compute_type=args.compute_type)
    else:
        app = tts.create_app(voice=args.voice, model_dir=args.model_dir)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
