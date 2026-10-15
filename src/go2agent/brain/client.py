"""OpenAI-compatible chat client for the VLM (vLLM, SGLang, llama.cpp server, ...)."""

from __future__ import annotations

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
        raise NotImplementedError

    def __aiter__(self) -> AsyncIterator[str]:
        raise NotImplementedError
