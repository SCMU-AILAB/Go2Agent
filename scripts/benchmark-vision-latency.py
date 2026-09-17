"""Measure Ollama image request latency without a camera or robot connection.

Synthetic inputs measure performance only, never gesture accuracy. Use --images
with consecutive real camera JPEGs for representative replay. NDJSON goes to stdout.
"""

import argparse
import base64
import concurrent.futures
import json
import math
import random
import statistics
import struct
import time
import urllib.request
import zlib
from pathlib import Path
from typing import Any

GESTURE_EVIDENCE = {
    "handshake": "offered_hand",
    "wave": "side_to_side",
    "high_five": "raised_palm",
    "none": "none",
    "uncertain": "ambiguous",
}


def synthetic_image(seed: int) -> bytes:
    """Standard-library PNG; fresh pixels prevent identical-image cache hits."""
    rng = random.Random(seed)
    width, height = 640, 480
    raw = b"".join(b"\0" + rng.randbytes(width * 3) for _ in range(height))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data))
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def validate_response(raw: str) -> tuple[bool, bool]:
    try:
        parsed: Any = json.loads(raw)
    except (TypeError, ValueError):
        return False, False
    if not isinstance(parsed, dict):
        return False, False
    required = {
        "gesture",
        "hand_visible",
        "directed_at_robot",
        "present_in_latest",
        "evidence",
    }
    schema_valid = required <= parsed.keys() and all(
        isinstance(parsed.get(field), bool)
        for field in ("hand_visible", "directed_at_robot", "present_in_latest")
    )
    gesture = parsed.get("gesture")
    consistent = (
        schema_valid
        and gesture in GESTURE_EVIDENCE
        and parsed.get("evidence") == GESTURE_EVIDENCE[gesture]
        and (
            gesture not in ("none", "uncertain")
            or parsed.get("speech") is None
        )
    )
    return schema_valid, consistent


def request(
    args: argparse.Namespace,
    count: int,
    index: int,
    images: list[bytes],
    prompt: str,
) -> dict[str, object]:
    frames = (
        images
        if images
        else [synthetic_image(index * 100 + i) for i in range(count)]
    )
    if images:
        frames = [
            images[round(i * (len(images) - 1) / max(1, count - 1))]
            for i in range(count)
        ]
    options = {"temperature": 0, "num_predict": 256}
    if args.context:
        options["num_ctx"] = args.context
    payload = json.dumps(
        {
            "model": args.model,
            "stream": False,
            "think": False,
            "format": "json",
            "keep_alive": "30m",
            "options": options,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [base64.b64encode(frame).decode() for frame in frames],
                }
            ],
        }
    ).encode()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    started = time.monotonic()
    request_value = urllib.request.Request(
        args.url.rstrip("/") + "/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with opener.open(request_value, timeout=120) as response:
        result = json.load(response)
    elapsed = time.monotonic() - started
    raw = result.get("message", {}).get("content", "")
    schema_valid, semantic_consistent = validate_response(raw)
    row = {
        "type": "request",
        "frames": count,
        "index": index,
        "wall_s": round(elapsed, 4),
        "bytes": len(payload),
        "done": result.get("done"),
        "done_reason": result.get("done_reason"),
        "schema_valid": schema_valid,
        "semantic_consistent": semantic_consistent,
        "input_tokens": result.get("prompt_eval_count"),
        "output_tokens": result.get("eval_count"),
        "response": raw,
    }
    for key in ("total", "load", "prompt_eval", "eval"):
        row[key + "_s"] = round(result.get(key + "_duration", 0) / 1e9, 4)
    row["server_other_s"] = round(
        row["total_s"]
        - row["load_s"]
        - row["prompt_eval_s"]
        - row["eval_s"],
        4,
    )
    row["transport_residual_s"] = round(elapsed - row["total_s"], 4)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:11435")
    parser.add_argument("--model", default="qwen3.5:9b")
    parser.add_argument("--frames", type=int, nargs="+", default=[8, 3])
    parser.add_argument("--requests", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--context", type=int, default=0)
    parser.add_argument("--prompt", type=Path)
    parser.add_argument("--images", type=Path, nargs="+")
    args = parser.parse_args()
    if min(*args.frames, args.requests, args.concurrency) <= 0:
        parser.error("frames, requests and concurrency must be positive")
    images = [p.read_bytes() for p in args.images] if args.images else []
    prompt = args.prompt.read_text() if args.prompt else (
        "These are chronological camera images. Classify only the human gesture "
        "still visible in the last image: wave, handshake, high_five, none or uncertain. "
        "Noise or no visible human hand means none. Never invent a person. "
        "Return JSON only with gesture, hand_visible (boolean), directed_at_robot "
        "(boolean), present_in_latest (boolean), evidence (side_to_side, offered_hand, "
        "raised_palm, none or ambiguous), speech (null for none/uncertain)."
    )
    print(
        json.dumps(
            {
                "type": "config",
                "model": args.model,
                "context": args.context,
                "concurrency": args.concurrency,
                "input": (
                    "replay" if images else "synthetic_noise_not_accuracy_eval"
                ),
            }
        ),
        flush=True,
    )
    for count in args.frames:
        warmup = request(args, count, 0, images, prompt)
        warmup["type"] = "warmup"
        print(json.dumps(warmup), flush=True)
        started = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            rows = list(
                pool.map(
                    lambda i, frame_count=count: request(
                        args, frame_count, i, images, prompt
                    ),
                    range(1, args.requests + 1),
                )
            )
        duration = time.monotonic() - started
        for row in rows:
            print(json.dumps(row), flush=True)
        times = sorted(row["wall_s"] for row in rows)
        completed = [
            row
            for row in rows
            if row["done"] and row["done_reason"] != "length"
        ]
        print(
            json.dumps(
                {
                    "type": "summary",
                    "frames": count,
                    "concurrency": args.concurrency,
                    "count": len(rows),
                    "median_s": statistics.median(times),
                    "p95_s": times[math.ceil(len(times) * 0.95) - 1],
                    "requests_per_s": len(rows) / duration,
                    "schema_completed": sum(
                        bool(row["schema_valid"]) for row in completed
                    ),
                    "semantically_consistent": sum(
                        bool(row["semantic_consistent"]) for row in completed
                    ),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
