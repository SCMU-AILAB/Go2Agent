"""GPU-server service entry points."""

from __future__ import annotations

import argparse
import logging

import uvicorn

from go2agent.server.perception import create_app


def main() -> None:
    parser = argparse.ArgumentParser(prog="go2agent-server")
    subparsers = parser.add_subparsers(dest="service", required=True)
    perception = subparsers.add_parser("perception", help="Run YOLO/ByteTrack perception")
    perception.add_argument("--host", default="0.0.0.0")
    perception.add_argument("--port", type=int, default=8100)
    perception.add_argument("--model", default="yolo11s.pt")
    perception.add_argument("--classes", nargs="+", default=["person"], help="COCO class names")
    perception.add_argument("--device", default=None, help="Default: cuda, then mps, then cpu")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.service == "perception":
        uvicorn.run(
            create_app(model=args.model, classes=args.classes, device=args.device),
            host=args.host,
            port=args.port,
        )


if __name__ == "__main__":
    main()
