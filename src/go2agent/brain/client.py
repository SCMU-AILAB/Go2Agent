"""OpenAI-compatible chat client for the VLM (vLLM, SGLang, llama.cpp server, ...)."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass(frozen=True, slots=True)
class VLMConfig:
    base_url: str = "http://127.0.0.1:8100/v1"
    model: str = "brain"  # served-model-name, see go2agent.server.models
    api_key: str = "EMPTY"
    timeout: float = 3.0
    max_tokens: int = 300
    temperature: float = 0.0


@dataclass(frozen=True, slots=True)
class VLMReply:
    data: dict[str, Any]
    raw: str
    latency_ms: float
    usage: dict[str, Any]


class VLMError(Exception):
    pass


class VLMClient:
    def __init__(self, config: VLMConfig, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.config = config
        self._http = httpx.AsyncClient(
            base_url=config.base_url,
            headers={"Authorization": f"Bearer {config.api_key}"},
            timeout=config.timeout,
            transport=transport,
            trust_env=False,  # LAN service: never route through http_proxy from the shell profile
        )

    async def close(self) -> None:
        await self._http.aclose()

    def stream_json(self, messages: list[dict[str, Any]], schema: dict[str, Any]) -> VLMStream:
        """Stream a schema-constrained completion (OpenAI SSE, `stream: true`).

        Iterate the returned VLMStream for text chunks as they arrive. The whole stream must finish
        within `config.timeout` (else VLMError("timeout ...")); connection, HTTP-status and malformed-SSE
        problems raise VLMError like complete_json. After iteration, `stream.ttft_ms` is the time to the
        first non-empty text chunk and `stream.latency_ms` the time to the end of the stream;
        `stream.usage` is filled when the server sends it (request `stream_options.include_usage`).
        """
        return VLMStream(self, messages, schema)

    async def complete_json(self, messages: list[dict[str, Any]], schema: dict[str, Any]) -> VLMReply:
        c = self.config
        body = {
            "model": c.model,
            "messages": messages,
            "max_tokens": c.max_tokens,
            "temperature": c.temperature,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "decision", "schema": schema, "strict": True},
            },
        }
        t0 = time.monotonic()
        try:
            r = await self._http.post("/chat/completions", json=body)
        except httpx.TimeoutException as e:
            raise VLMError(f"timeout after {c.timeout}s") from e
        except httpx.HTTPError as e:
            raise VLMError(f"unreachable: {e}") from e
        latency = (time.monotonic() - t0) * 1000
        if r.status_code != 200:
            raise VLMError(f"http {r.status_code}: {r.text[:200]}")
        payload = r.json()
        raw = payload["choices"][0]["message"].get("content") or ""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise VLMError(f"invalid json: {raw[:200]}") from e
        return VLMReply(data, raw, latency, payload.get("usage", {}))


class VLMStream:
    """Async iterator of text chunks for one streamed completion. See VLMClient.stream_json."""

    def __init__(self, client: VLMClient, messages: list[dict[str, Any]], schema: dict[str, Any]) -> None:
        self.ttft_ms: float | None = None
        self.latency_ms: float | None = None
        self.usage: dict[str, Any] = {}
        self._client = client
        self._messages = messages
        self._schema = schema
        self._started = False

    def __aiter__(self) -> AsyncIterator[str]:
        if self._started:
            raise RuntimeError("VLMStream can only be iterated once")
        self._started = True
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[str]:
        c = self._client.config
        body = {
            "model": c.model,
            "messages": self._messages,
            "max_tokens": c.max_tokens,
            "temperature": c.temperature,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "decision", "schema": self._schema, "strict": True},
            },
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        t0 = time.monotonic()
        try:
            request = self._client._http.build_request("POST", "/chat/completions", json=body)
            async with self._timeout(t0):
                response = await self._client._http.send(request, stream=True)
            try:
                if response.status_code != 200:
                    async with self._timeout(t0):
                        await response.aread()
                    raise VLMError(f"http {response.status_code}: {response.text[:200]}")
                lines = response.aiter_lines()
                while True:
                    try:
                        async with self._timeout(t0):
                            line = await anext(lines)
                    except StopAsyncIteration:
                        break
                    if not line.startswith("data:"):
                        continue
                    raw = line[5:].strip()
                    if raw == "[DONE]":
                        return
                    content = self._content(raw)
                    if content:
                        if self.ttft_ms is None:
                            self.ttft_ms = (time.monotonic() - t0) * 1000
                        yield content
            finally:
                await response.aclose()
        except (TimeoutError, httpx.TimeoutException) as e:
            raise VLMError(f"timeout after {c.timeout}s") from e
        except httpx.HTTPError as e:
            raise VLMError(f"unreachable: {e}") from e
        finally:
            self.latency_ms = (time.monotonic() - t0) * 1000

    def _timeout(self, t0: float) -> asyncio.Timeout:
        # Keep one deadline, but never leave a cancellation scope active while the consumer handles a yield.
        remaining = self._client.config.timeout - (time.monotonic() - t0)
        if remaining <= 0:
            raise TimeoutError
        return asyncio.timeout(remaining)

    def _content(self, raw: str) -> str:
        try:
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise ValueError("chunk must be an object")
            if payload.get("usage") is not None:
                usage = payload["usage"]
                if not isinstance(usage, dict):
                    raise ValueError("usage must be an object")
                self.usage = usage
            choices = payload["choices"]
            if not isinstance(choices, list):
                raise ValueError("choices must be an array")
            if not choices:
                return ""
            content = choices[0]["delta"].get("content")
            if content is None:
                return ""
            if not isinstance(content, str):
                raise ValueError("content must be a string")
            return content
        except (ValueError, KeyError, TypeError, AttributeError) as e:
            raise VLMError(f"invalid SSE data: {raw[:200]}") from e
