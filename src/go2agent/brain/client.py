"""OpenAI-compatible chat client for the VLM (vLLM, SGLang, llama.cpp server, ...)."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass(frozen=True, slots=True)
class VLMConfig:
    base_url: str = "http://127.0.0.1:8000/v1"
    model: str = "Qwen/Qwen3-VL-8B-Instruct-AWQ"
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
        )

    async def close(self) -> None:
        await self._http.aclose()

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
