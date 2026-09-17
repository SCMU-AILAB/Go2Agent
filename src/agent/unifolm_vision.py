"""HTTP client for a remote, persistent Unitree UnifoLM-ER vision server."""

from __future__ import annotations

import asyncio
import base64
import json
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence

from .decision import DecisionAgentError
from .vision_policy import OllamaVisionInvoker

DEFAULT_UNIFOLM_MODEL = "unitreerobotics/UnifoLM-ER-1"
DEFAULT_UNIFOLM_URL = "http://127.0.0.1:8011"


class UnifolmVisionInvoker:
    """Send latest-only visual policy windows to a resident 4090 model."""

    def __init__(
        self,
        model_name: str = DEFAULT_UNIFOLM_MODEL,
        *,
        base_url: str = DEFAULT_UNIFOLM_URL,
        max_new_tokens: int = 96,
        timeout_s: float = 10.0,
    ) -> None:
        if max_new_tokens <= 0:
            raise ValueError("max new tokens must be greater than zero")
        if timeout_s <= 0:
            raise ValueError("UnifoLM timeout must be greater than zero")
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.max_new_tokens = max_new_tokens
        self.timeout_s = timeout_s
        self._request_id = 0
        self._lock = asyncio.Lock()
        self._closed = False
        self._backend_info: dict[str, object] = {}
        self._last_metrics: dict[str, object] = {}

    @property
    def backend_info(self) -> Mapping[str, object]:
        return dict(self._backend_info)

    @property
    def last_metrics(self) -> Mapping[str, object]:
        return dict(self._last_metrics)

    async def warmup(self) -> None:
        if self._closed:
            raise DecisionAgentError("UnifoLM vision invoker is closed")
        response = await asyncio.to_thread(self._request_json, "GET", "/health", None)
        if response.get("status") != "ready":
            raise DecisionAgentError(
                f"UnifoLM server is not ready: {response.get('status', 'unknown')}"
            )
        self._backend_info = dict(response)

    async def ainvoke(self, frames: Sequence[object], prompt: str) -> object:
        if not frames:
            raise ValueError("UnifoLM vision requires at least one frame")
        if self._closed:
            raise DecisionAgentError("UnifoLM vision invoker is closed")

        started = time.monotonic()
        encoded, image_bytes = await asyncio.to_thread(self._encode_frames, frames)
        encoded_at = time.monotonic()
        async with self._lock:
            self._request_id += 1
            request_id = self._request_id
            response = await asyncio.to_thread(
                self._request_json,
                "POST",
                "/v1/vision/invoke",
                {
                    "request_id": request_id,
                    "model": self.model_name,
                    "frames": encoded,
                    "prompt": prompt,
                    "max_new_tokens": self.max_new_tokens,
                },
            )
        finished = time.monotonic()
        if response.get("request_id") != request_id:
            raise DecisionAgentError("UnifoLM server returned a mismatched request id")
        output = response.get("output")
        if not isinstance(output, str):
            raise DecisionAgentError("UnifoLM server returned an invalid output")
        server_metrics = response.get("metrics")
        self._last_metrics = {
            **(dict(server_metrics) if isinstance(server_metrics, Mapping) else {}),
            "remote_request_id": request_id,
            "client_encode_s": round(encoded_at - started, 4),
            "round_trip_s": round(finished - started, 4),
            "http_round_trip_s": round(finished - encoded_at, 4),
            "image_bytes": image_bytes,
            "frame_count": len(frames),
        }
        return output

    @staticmethod
    def _encode_frames(frames: Sequence[object]) -> tuple[list[str], int]:
        raw = [OllamaVisionInvoker._as_bytes(frame) for frame in frames]
        return [base64.b64encode(frame).decode("ascii") for frame in raw], sum(
            len(frame) for frame in raw
        )

    async def close(self) -> None:
        self._closed = True

    def _request_json(
        self,
        method: str,
        path: str,
        payload: Mapping[str, object] | None,
    ) -> dict[str, object]:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + path,
            data=body,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                decoded = json.loads(response.read())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise DecisionAgentError(f"UnifoLM request failed: {exc}") from exc
        if not isinstance(decoded, dict):
            raise DecisionAgentError("UnifoLM server returned a non-object response")
        return decoded
