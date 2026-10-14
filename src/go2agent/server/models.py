"""VLM lifecycle on the GPU server: vLLM runs as a managed child process, loaded and unloaded on request.

Nothing loads automatically. `unload` ends the process so all of its GPU memory is returned, which lets the
GPU be shared with other workloads. The OpenAI-compatible API is proxied under /v1 on the services port, so
the dog needs a single port for perception, voice and the VLM.

    GET  /models                 {"brain": {...status...}}
    POST /models/brain/load      optional JSON {"model": "<hf repo or path>"}; returns once ready
    POST /models/brain/unload
    *    /v1/...                 proxied to vLLM; 503 while not loaded
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

log = logging.getLogger(__name__)


class ModelState(StrEnum):
    UNLOADED = "unloaded"
    LOADING = "loading"
    READY = "ready"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class VLLMConfig:
    model: str = "Qwen/Qwen3-VL-8B-Instruct-FP8"
    served_name: str = "brain"
    port: int = 8001  # internal; clients use the proxy on the services port
    gpu_memory_utilization: float = 0.70  # leave room for perception and ASR on the same GPU
    max_model_len: int = 8192
    max_num_seqs: int = 2
    max_images: int = 4
    load_timeout_s: float = 900.0
    extra_args: tuple[str, ...] = ()

    def command(self, model: str) -> list[str]:
        return [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", model,
            "--served-model-name", self.served_name,
            "--host", "127.0.0.1", "--port", str(self.port),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            "--max-model-len", str(self.max_model_len),
            "--max-num-seqs", str(self.max_num_seqs),
            "--limit-mm-per-prompt", f'{{"image": {self.max_images}, "video": 0}}',
            "--enable-prefix-caching",
            *self.extra_args,
        ]  # fmt: skip


@dataclass(slots=True)
class ModelStatus:
    state: ModelState = ModelState.UNLOADED
    model: str | None = None
    pid: int | None = None
    since: float | None = None  # wall time of the last state change
    load_seconds: float | None = None
    error: str | None = None
    log_tail: list[str] = field(default_factory=list)


def _die_with_parent() -> None:
    """Linux: the child gets SIGTERM if the server process dies, so no orphan keeps the GPU."""
    if sys.platform.startswith("linux"):
        import ctypes

        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG


class VLLMManager:
    def __init__(
        self, config: VLLMConfig, log_path: Path, command: Callable[[str], list[str]] | None = None
    ) -> None:
        self.config = config
        self.log_path = log_path
        self._command = command or config.command
        self._proc: subprocess.Popen[bytes] | None = None
        self._status = ModelStatus()
        self._lock = asyncio.Lock()
        self.proxy = httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=2.0))

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.config.port}"

    @property
    def ready(self) -> bool:
        return self._status.state is ModelState.READY and self._proc is not None and self._proc.poll() is None

    def status(self) -> dict[str, Any]:
        if (
            self._status.state is ModelState.READY
            and self._proc is not None
            and self._proc.poll() is not None
        ):
            self._set(ModelState.FAILED, error=f"vLLM exited with code {self._proc.returncode}")
        st = asdict(self._status)
        st["log_tail"] = self._read_tail() if self._status.state is not ModelState.READY else []
        return st

    async def load(self, model: str | None = None) -> dict[str, Any]:
        async with self._lock:
            model = model or self.config.model
            if self.ready and self._status.model == model:
                return self.status()
            if self._proc is not None:
                await self._terminate()
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            logf = self.log_path.open("ab")
            log.info("loading %s", model)
            t0 = time.monotonic()
            self._proc = subprocess.Popen(
                self._command(model),
                stdout=logf,
                stderr=subprocess.STDOUT,
                preexec_fn=_die_with_parent,
                env=os.environ.copy(),
            )
            logf.close()
            self._set(ModelState.LOADING, model=model, pid=self._proc.pid)
            error = await self._wait_ready()
            if error:
                await self._terminate()
                self._set(ModelState.FAILED, model=model, error=error)
            else:
                self._set(ModelState.READY, model=model, pid=self._proc.pid)
                self._status.load_seconds = round(time.monotonic() - t0, 1)
                log.info("%s ready in %.0fs", model, self._status.load_seconds)
            return self.status()

    async def unload(self) -> dict[str, Any]:
        async with self._lock:
            if self._proc is not None:
                await self._terminate()
            self._set(ModelState.UNLOADED)
            return self.status()

    async def close(self) -> None:
        await self.unload()
        await self.proxy.aclose()

    async def _wait_ready(self) -> str | None:
        deadline = time.monotonic() + self.config.load_timeout_s
        async with httpx.AsyncClient(timeout=2.0) as http:
            while time.monotonic() < deadline:
                assert self._proc is not None
                if self._proc.poll() is not None:
                    return f"vLLM exited with code {self._proc.returncode} while loading"
                with contextlib.suppress(httpx.HTTPError):
                    if (await http.get(f"{self.base_url}/health")).status_code == 200:
                        return None
                await asyncio.sleep(1.0)
        return f"not ready after {self.config.load_timeout_s:.0f}s"

    async def _terminate(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None or proc.poll() is not None:
            return
        log.info("stopping vLLM pid %d", proc.pid)
        proc.terminate()
        try:
            await asyncio.to_thread(proc.wait, 30)
        except subprocess.TimeoutExpired:
            proc.kill()
            await asyncio.to_thread(proc.wait, 10)

    def _set(self, state: ModelState, **fields: Any) -> None:
        self._status = ModelStatus(state=state, since=time.time(), **fields)

    def _read_tail(self) -> list[str]:
        try:
            with self.log_path.open("rb") as f:
                f.seek(max(0, f.seek(0, os.SEEK_END) - 8192))
                lines = f.read().decode(errors="replace").splitlines()
        except OSError:
            return []
        return lines[-15:]


class LoadBody(BaseModel):
    model: str | None = None


_HOP_HEADERS = {"host", "content-length", "connection", "transfer-encoding", "keep-alive"}


def create_router(manager: VLLMManager) -> APIRouter:
    """Routes for lifecycle and the /v1 proxy. The owner must `await manager.close()` on shutdown."""
    router = APIRouter()
    proxy = manager.proxy

    @router.get("/models")
    async def models() -> dict[str, Any]:
        return {manager.config.served_name: manager.status()}

    @router.post(f"/models/{manager.config.served_name}/load")
    async def load(body: LoadBody | None = None) -> dict[str, Any]:
        return await manager.load(body.model if body else None)

    @router.post(f"/models/{manager.config.served_name}/unload")
    async def unload() -> dict[str, Any]:
        return await manager.unload()

    @router.api_route("/v1/{path:path}", methods=["GET", "POST"])
    async def v1(path: str, request: Request) -> Response:
        if not manager.ready:
            state = manager.status()["state"]
            return JSONResponse(
                {"error": f"model {manager.config.served_name} not loaded (state: {state})"}, status_code=503
            )
        headers = {k: v for k, v in request.headers.items() if k.lower() not in _HOP_HEADERS}
        try:
            r = await proxy.request(
                request.method,
                f"{manager.base_url}/v1/{path}",
                content=await request.body(),
                headers=headers,
                params=request.query_params,
            )
        except httpx.HTTPError as e:
            return JSONResponse({"error": f"vLLM unreachable: {e}"}, status_code=502)
        return Response(r.content, status_code=r.status_code, media_type=r.headers.get("content-type"))

    return router
