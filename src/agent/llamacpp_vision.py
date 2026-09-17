"""OpenAI-compatible llama.cpp client for a local quantized vision model."""

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

DEFAULT_LLAMA_CPP_MODEL = "UnifoLM-ER-1-Q4_K_M"
DEFAULT_LLAMA_CPP_URL = "http://127.0.0.1:8012"


class LlamaCppVisionInvoker:
    """Send visual-policy windows to a resident local ``llama-server``."""

    def __init__(
        self,
        model_name: str = DEFAULT_LLAMA_CPP_MODEL,
        *,
        base_url: str = DEFAULT_LLAMA_CPP_URL,
        max_new_tokens: int = 96,
        timeout_s: float = 30.0,
        constrain_json: bool = True,
    ) -> None:
        if max_new_tokens <= 0:
            raise ValueError("max new tokens must be greater than zero")
        if timeout_s <= 0:
            raise ValueError("llama.cpp timeout must be greater than zero")
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.max_new_tokens = max_new_tokens
        self.timeout_s = timeout_s
        self.constrain_json = constrain_json
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
            raise DecisionAgentError("llama.cpp vision invoker is closed")
        response = await asyncio.to_thread(self._request_json, "GET", "/health", None)
        if response.get("status") not in {"ok", "ready"}:
            raise DecisionAgentError(
                f"llama.cpp server is not ready: {response.get('status', 'unknown')}"
            )
        self._backend_info = {
            **response,
            "model": self.model_name,
            "backend": "llama.cpp",
        }

    async def ainvoke(self, frames: Sequence[object], prompt: str) -> object:
        if not frames:
            raise ValueError("llama.cpp vision requires at least one frame")
        if self._closed:
            raise DecisionAgentError("llama.cpp vision invoker is closed")

        started = time.monotonic()
        raw_frames = await asyncio.to_thread(
            lambda: [OllamaVisionInvoker._as_bytes(frame) for frame in frames]
        )
        encoded_at = time.monotonic()
        content: list[dict[str, object]] = [
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/jpeg;base64,"
                    + base64.b64encode(frame).decode("ascii")
                },
            }
            for frame in raw_frames
        ]
        content.append({"type": "text", "text": prompt})

        async with self._lock:
            self._request_id += 1
            request_id = self._request_id
            payload: dict[str, object] = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": content}],
                "temperature": 0,
                "max_tokens": self.max_new_tokens,
                "stream": False,
            }
            if self.constrain_json:
                payload["response_format"] = {"type": "json_object"}
            response = await asyncio.to_thread(
                self._request_json,
                "POST",
                "/v1/chat/completions",
                payload,
            )
        finished = time.monotonic()

        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise DecisionAgentError("llama.cpp returned no completion choice")
        choice = choices[0]
        if not isinstance(choice, Mapping):
            raise DecisionAgentError("llama.cpp returned an invalid completion choice")
        if choice.get("finish_reason") == "length":
            raise DecisionAgentError(
                "llama.cpp exhausted the output token budget; refusing truncated decision"
            )
        message = choice.get("message")
        if not isinstance(message, Mapping) or not isinstance(
            message.get("content"), str
        ):
            raise DecisionAgentError("llama.cpp returned an invalid message")

        metrics: dict[str, object] = {
            "remote_request_id": request_id,
            "client_encode_s": round(encoded_at - started, 4),
            "round_trip_s": round(finished - started, 4),
            "http_round_trip_s": round(finished - encoded_at, 4),
            "image_bytes": sum(len(frame) for frame in raw_frames),
            "frame_count": len(raw_frames),
        }
        usage = response.get("usage")
        if isinstance(usage, Mapping):
            metrics["input_tokens"] = usage.get("prompt_tokens")
            metrics["generated_tokens"] = usage.get("completion_tokens")
        timings = response.get("timings")
        if isinstance(timings, Mapping):
            prompt_ms = timings.get("prompt_ms")
            predicted_ms = timings.get("predicted_ms")
            if isinstance(prompt_ms, (int, float)):
                metrics["prompt_eval_s"] = round(prompt_ms / 1000, 4)
            if isinstance(predicted_ms, (int, float)):
                metrics["eval_s"] = round(predicted_ms / 1000, 4)
            if isinstance(prompt_ms, (int, float)) or isinstance(
                predicted_ms, (int, float)
            ):
                inference_s = (float(prompt_ms or 0) + float(predicted_ms or 0)) / 1000
                metrics["inference_s"] = round(inference_s, 4)
                metrics["network_rtt_s"] = round(
                    max(0.0, finished - encoded_at - inference_s), 4
                )
        self._last_metrics = metrics
        return str(message["content"])

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
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", errors="replace")[:500]
            except OSError:
                detail = str(exc)
            raise DecisionAgentError(
                f"llama.cpp request failed with HTTP {exc.code}: {detail}"
            ) from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise DecisionAgentError(f"llama.cpp request failed: {exc}") from exc
        if not isinstance(decoded, dict):
            raise DecisionAgentError("llama.cpp server returned a non-object response")
        return decoded
