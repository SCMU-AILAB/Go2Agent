import json
from typing import Any

import httpx
import numpy as np
import pytest

from go2agent.core.types import Frame


class FakeClock:
    def __init__(self, t: float = 100.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


def make_frame(stamp: float, seq: int = 0, depth: float | None = None, w: int = 64, h: int = 48) -> Frame:
    rgb = np.zeros((h, w, 3), np.uint8)
    d = None if depth is None else np.full((h, w), depth, np.float32)
    return Frame(seq=seq, stamp=stamp, rgb=rgb, depth=d)


def sse_response(content: str, status: int = 200, chunk: int = 5) -> httpx.Response:
    """An OpenAI-style streamed chat completion that delivers `content` in small chunks."""
    if status != 200:
        return httpx.Response(status, text=content)
    events: list[dict[str, Any]] = [
        {"choices": [{"index": 0, "delta": {"content": content[i : i + chunk]}}]}
        for i in range(0, len(content), chunk)
    ]
    events.append({"choices": [], "usage": {"prompt_tokens": 100, "completion_tokens": len(events)}})
    body = "".join(f"data: {json.dumps(e, ensure_ascii=False)}\n\n" for e in events) + "data: [DONE]\n\n"
    return httpx.Response(200, content=body.encode(), headers={"content-type": "text/event-stream"})


def compact(data: object) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))
