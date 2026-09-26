"""Persistent bridge to the Jetson system-Python CUDA VLM worker."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import queue
import subprocess
import threading
import time
from collections import deque
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TextIO, cast


class CudaVisionWorkerError(RuntimeError):
    """Raised when the persistent CUDA worker cannot serve a request."""


class CudaVisionInvoker:
    """Run SmolVLM2 in the Jetson CUDA-enabled Python 3.8 environment."""

    def __init__(
        self,
        model_name: str,
        *,
        max_new_tokens: int = 40,
        python_executable: str | None = None,
        packages_path: Path | None = None,
        worker_path: Path | None = None,
        startup_timeout_s: float = 120.0,
        inference_timeout_s: float = 120.0,
    ) -> None:
        if max_new_tokens <= 0:
            raise ValueError("max new tokens must be greater than zero")
        if startup_timeout_s <= 0 or inference_timeout_s <= 0:
            raise ValueError("CUDA worker timeouts must be greater than zero")
        project_root = Path(__file__).resolve().parents[2]
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.python_executable = python_executable or os.getenv(
            "GO2_VISION_PYTHON", "/usr/bin/python3"
        )
        self.packages_path = packages_path or project_root / ".cuda-packages"
        self.worker_path = worker_path or Path(__file__).with_name(
            "_smolvlm_cuda_worker.py"
        )
        self.startup_timeout_s = startup_timeout_s
        self.inference_timeout_s = inference_timeout_s
        self._process: subprocess.Popen[str] | None = None
        self._lines: queue.Queue[str | None] = queue.Queue()
        self._stdout_thread: threading.Thread | None = None
        self._diagnostics: deque[str] = deque(maxlen=20)
        self._operation_lock = threading.RLock()
        self._request_id = 0
        self._backend_info: dict[str, object] = {}
        self._last_metrics: dict[str, object] = {}
        self._logger = logging.getLogger("agent.cuda_worker")
        self._async_lock = asyncio.Lock()
        self._closed = False

    @property
    def opened(self) -> bool:
        return self._process is not None and self._process.poll() is None

    @property
    def backend_info(self) -> Mapping[str, object]:
        return dict(self._backend_info)

    @property
    def last_metrics(self) -> Mapping[str, object]:
        return dict(self._last_metrics)

    async def warmup(self) -> None:
        # Jetson system libraries can deadlock when Popen forks from a worker
        # thread. Startup is intentionally synchronous and happens only once.
        async with self._async_lock:
            if self._closed:
                raise CudaVisionWorkerError("CUDA worker invoker is closed")
            self.open()

    async def ainvoke(self, frames: Sequence[object], prompt: str) -> object:
        if not frames:
            raise ValueError("CUDA vision worker requires at least one frame")
        if self._closed:
            raise CudaVisionWorkerError("CUDA worker invoker is closed")
        encoded_frames = self._encode_frames(frames)
        async with self._async_lock:
            if not self.opened:
                self.open()
            self._request_id += 1
            request_id = self._request_id
            self._send(
                {
                    "command": "invoke",
                    "request_id": request_id,
                    "frames": list(encoded_frames),
                    "prompt": prompt,
                }
            )
            response = await self._read_response_async(
                expected_type="result",
                request_id=request_id,
                timeout_s=self.inference_timeout_s,
            )
            output = response.get("output")
            if not isinstance(output, str):
                raise CudaVisionWorkerError("CUDA worker returned invalid output")
            metrics = response.get("metrics")
            self._last_metrics = (
                dict(metrics) if isinstance(metrics, dict) else {}
            )
            return output

    async def close(self) -> None:
        if self._closed:
            return
        async with self._async_lock:
            process = self._process
            if process is None:
                self._closed = True
                return
            close_error: CudaVisionWorkerError | None = None
            if process.poll() is None:
                try:
                    self._send({"command": "close"})
                    await self._read_response_async(
                        expected_type="closed",
                        request_id=None,
                        timeout_s=15.0,
                        discard_types={"result", "error"},
                    )
                except CudaVisionWorkerError as exc:
                    close_error = exc
            self._shutdown_process()
            self._closed = True
            if close_error is not None:
                raise close_error

    def open(self) -> None:
        with self._operation_lock:
            if self.opened:
                return
            if not self.packages_path.is_dir():
                raise CudaVisionWorkerError(
                    f"CUDA package directory does not exist: {self.packages_path}"
                )
            if not self.worker_path.is_file():
                raise CudaVisionWorkerError(
                    f"CUDA worker script does not exist: {self.worker_path}"
                )

            environment = os.environ.copy()
            existing_pythonpath = environment.get("PYTHONPATH")
            environment["PYTHONPATH"] = os.pathsep.join(
                value
                for value in (str(self.packages_path), existing_pythonpath)
                if value
            )
            environment.update(
                {
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                    "HF_HUB_DISABLE_TELEMETRY": "1",
                    "PYTHONUNBUFFERED": "1",
                }
            )
            command = [
                self.python_executable,
                "-u",
                str(self.worker_path),
                "--model",
                self.model_name,
                "--max-new-tokens",
                str(self.max_new_tokens),
            ]
            try:
                process = subprocess.Popen(
                    command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    env=environment,
                )
            except OSError as exc:
                raise CudaVisionWorkerError(
                    f"could not start CUDA worker with {self.python_executable}: {exc}"
                ) from exc

            if process.stdin is None or process.stdout is None:
                process.kill()
                process.wait()
                raise CudaVisionWorkerError("CUDA worker pipes are unavailable")

            self._process = process
            self._lines = queue.Queue()
            self._diagnostics.clear()
            self._stdout_thread = threading.Thread(
                target=self._read_stdout,
                args=(process.stdout,),
                name="cuda-vision-worker-output",
                daemon=True,
            )
            self._stdout_thread.start()

            try:
                ready = self._read_response(
                    expected_type="ready",
                    request_id=None,
                    timeout_s=self.startup_timeout_s,
                )
                info = ready.get("backend")
                self._backend_info = dict(info) if isinstance(info, dict) else {}
            except Exception:
                self._shutdown_process()
                raise

    def _send(self, payload: Mapping[str, object]) -> None:
        process = self._process
        if process is None or process.stdin is None or process.poll() is not None:
            raise CudaVisionWorkerError(self._exit_message())
        try:
            process.stdin.write(json.dumps(payload, ensure_ascii=True) + "\n")
            process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise CudaVisionWorkerError(self._exit_message()) from exc

    def _read_response(
        self,
        *,
        expected_type: str,
        request_id: int | None,
        timeout_s: float,
    ) -> dict[str, object]:
        deadline = time.monotonic() + timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise CudaVisionWorkerError(
                    f"CUDA worker timed out waiting for {expected_type}"
                )
            try:
                line = self._lines.get(timeout=remaining)
            except queue.Empty as exc:
                raise CudaVisionWorkerError(
                    f"CUDA worker timed out waiting for {expected_type}"
                ) from exc
            if line is None:
                raise CudaVisionWorkerError(self._exit_message())
            stripped = line.strip()
            if not stripped:
                continue
            try:
                decoded = json.loads(stripped)
            except json.JSONDecodeError:
                self._diagnostics.append(stripped)
                self._logger.warning(stripped)
                continue
            if not isinstance(decoded, dict):
                self._diagnostics.append(stripped)
                self._logger.warning(stripped)
                continue
            response = cast(dict[str, object], decoded)
            response_type = response.get("type")
            if response_type == "error":
                raise CudaVisionWorkerError(
                    str(response.get("message", "unknown CUDA worker error"))
                )
            if response_type != expected_type:
                raise CudaVisionWorkerError(
                    "CUDA worker returned "
                    f"{response_type!r} while waiting for {expected_type!r}"
                )
            if request_id is not None and response.get("request_id") != request_id:
                raise CudaVisionWorkerError(
                    "CUDA worker returned a mismatched request identifier"
                )
            return response

    async def _read_response_async(
        self,
        *,
        expected_type: str,
        request_id: int | None,
        timeout_s: float,
        discard_types: set[str] | None = None,
    ) -> dict[str, object]:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        while True:
            try:
                line = self._lines.get_nowait()
            except queue.Empty:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise CudaVisionWorkerError(
                        f"CUDA worker timed out waiting for {expected_type}"
                    )
                await asyncio.sleep(min(0.01, remaining))
                continue
            if line is None:
                raise CudaVisionWorkerError(self._exit_message())
            stripped = line.strip()
            if not stripped:
                continue
            try:
                decoded = json.loads(stripped)
            except json.JSONDecodeError:
                self._diagnostics.append(stripped)
                self._logger.warning(stripped)
                continue
            if not isinstance(decoded, dict):
                self._diagnostics.append(stripped)
                self._logger.warning(stripped)
                continue
            response = cast(dict[str, object], decoded)
            response_type = response.get("type")
            response_id = response.get("request_id")
            if (
                request_id is not None
                and isinstance(response_id, int)
                and response_id < request_id
            ):
                continue
            if isinstance(response_type, str) and response_type in (
                discard_types or set()
            ):
                continue
            if response_type == "error":
                raise CudaVisionWorkerError(
                    str(response.get("message", "unknown CUDA worker error"))
                )
            if response_type != expected_type:
                raise CudaVisionWorkerError(
                    "CUDA worker returned "
                    f"{response_type!r} while waiting for {expected_type!r}"
                )
            if request_id is not None and response.get("request_id") != request_id:
                raise CudaVisionWorkerError(
                    "CUDA worker returned a mismatched request identifier"
                )
            return response

    def _exit_message(self) -> str:
        process = self._process
        return_code = process.poll() if process is not None else None
        message = f"CUDA worker exited unexpectedly (code {return_code})"
        if self._diagnostics:
            message += ": " + " | ".join(self._diagnostics)
        return message

    def _shutdown_process(self) -> None:
        process = self._process
        self._process = None
        self._backend_info = {}
        self._last_metrics = {}
        if process is None:
            return
        if process.stdin is not None:
            try:
                process.stdin.close()
            except OSError:
                pass
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if process.stdout is not None:
            process.stdout.close()

    def _read_stdout(self, stream: TextIO) -> None:
        try:
            for line in stream:
                self._lines.put(line)
        finally:
            self._lines.put(None)

    @classmethod
    def _encode_frames(cls, frames: Sequence[object]) -> tuple[str, ...]:
        return tuple(
            base64.b64encode(cls._as_jpeg_bytes(frame)).decode("ascii")
            for frame in frames
        )

    @staticmethod
    def _as_jpeg_bytes(frame: object) -> bytes:
        if isinstance(frame, bytes):
            return frame
        if isinstance(frame, (bytearray, memoryview)):
            return bytes(frame)
        from PIL import Image

        if not hasattr(frame, "__array_interface__"):
            raise TypeError(
                f"unsupported CUDA video frame type: {type(frame).__name__}"
            )
        image = Image.fromarray(frame).convert("RGB")
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=80)
        return output.getvalue()
