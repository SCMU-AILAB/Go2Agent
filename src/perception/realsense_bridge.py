"""Persistent subprocess bridge for system-installed RealSense bindings."""

from __future__ import annotations

import base64
import json
import os
import queue
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from typing import TextIO, cast

from .models import CameraFrame, PerceptionResult


class RealSenseBridgeError(RuntimeError):
    """Raised when the RealSense worker cannot serve a request."""


class RealSenseBridge:
    """Run camera capture in a Python interpreter with compatible bindings."""

    def __init__(
        self,
        *,
        serial: str | None,
        width: int,
        height: int,
        fps: int,
        detection_fps: float = 5.0,
        frame_timeout_ms: int,
        min_score: float,
        max_distance_m: float | None,
        rgb_rotation_deg: int = 0,
        python_executable: str | None = None,
        worker_path: Path | None = None,
    ) -> None:
        if rgb_rotation_deg not in (0, 90, 180, 270):
            raise ValueError("RGB rotation must be 0, 90, 180 or 270")
        self.serial = serial
        self.width = width
        self.height = height
        self.fps = fps
        self.detection_fps = detection_fps
        self.frame_timeout_ms = frame_timeout_ms
        self.min_score = min_score
        self.max_distance_m = max_distance_m
        self.rgb_rotation_deg = rgb_rotation_deg
        self.python_executable = python_executable or os.getenv(
            "G1_REALSENSE_PYTHON", "/usr/bin/python3"
        )
        self.worker_path = worker_path or Path(__file__).with_name(
            "_realsense_worker.py"
        )
        self._process: subprocess.Popen[str] | None = None
        self._lines: queue.Queue[str | None] = queue.Queue()
        self._reader_thread: threading.Thread | None = None
        self._diagnostics: deque[str] = deque(maxlen=20)

    @property
    def opened(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def open(self) -> None:
        if self.opened:
            return

        command = [
            self.python_executable,
            "-u",
            str(self.worker_path),
            "--width",
            str(self.width),
            "--height",
            str(self.height),
            "--fps",
            str(self.fps),
            "--detection-fps",
            str(self.detection_fps),
            "--frame-timeout-ms",
            str(self.frame_timeout_ms),
            "--min-score",
            str(self.min_score),
            "--rgb-rotation-deg",
            str(self.rgb_rotation_deg),
        ]
        if self.serial:
            command.extend(("--serial", self.serial))
        if self.max_distance_m is not None:
            command.extend(("--max-distance-m", str(self.max_distance_m)))

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
            )
        except OSError as exc:
            raise RealSenseBridgeError(
                f"could not start RealSense worker with {self.python_executable}: {exc}"
            ) from exc

        if process.stdout is None or process.stdin is None:
            process.kill()
            process.wait()
            raise RealSenseBridgeError("RealSense worker pipes are unavailable")

        self._process = process
        self._lines = queue.Queue()
        self._diagnostics.clear()
        self._reader_thread = threading.Thread(
            target=self._read_stdout,
            args=(process.stdout,),
            name="realsense-worker-output",
            daemon=True,
        )
        self._reader_thread.start()

        try:
            self._read_response(
                expected_type="ready",
                timeout_s=max(10.0, self.frame_timeout_ms / 1000.0 + 5.0),
            )
        except Exception:
            self._shutdown_process()
            raise

    def capture(self) -> PerceptionResult:
        return self.capture_frame(include_rgb=False).observation

    def capture_frame(self, *, include_rgb: bool = True) -> CameraFrame:
        if not self.opened:
            raise RealSenseBridgeError("RealSense worker is not running")
        self._send({"command": "capture", "include_rgb": include_rgb})
        response = self._read_response(
            expected_type="observation",
            timeout_s=self.frame_timeout_ms / 1000.0 + 10.0,
        )
        payload = response.get("result")
        if not isinstance(payload, dict):
            raise RealSenseBridgeError(
                "RealSense worker returned an invalid observation"
            )

        try:
            distance_value = payload.get("nearest_person_distance_m")
            center_value = payload.get("person_center_x")
            confidence_value = payload.get("confidence")
            observed_at_s = float(payload["observed_at_s"])
            observation = PerceptionResult(
                observed_at_s=observed_at_s,
                person_count=int(payload["person_count"]),
                nearest_person_distance_m=(
                    float(distance_value) if distance_value is not None else None
                ),
                person_center_x=(
                    float(center_value) if center_value is not None else None
                ),
                confidence=(
                    float(confidence_value) if confidence_value is not None else None
                ),
                source=str(payload["source"]),
            )
            obstacle_value = payload.get("nearest_obstacle_distance_m")
            encoded_rgb = payload.get("rgb_jpeg_base64")
            if include_rgb and not isinstance(encoded_rgb, str):
                raise ValueError("worker did not return an RGB frame")
            return CameraFrame(
                observed_at_s=observed_at_s,
                rgb=(base64.b64decode(encoded_rgb) if encoded_rgb else None),
                depth=None,
                observation=observation,
                nearest_obstacle_distance_m=(
                    float(obstacle_value) if obstacle_value is not None else None
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RealSenseBridgeError(
                f"RealSense worker returned an invalid observation: {exc}"
            ) from exc

    def close(self) -> None:
        process = self._process
        if process is None:
            return

        close_error: RealSenseBridgeError | None = None
        if process.poll() is None:
            try:
                self._send({"command": "close"})
                self._read_response(expected_type="closed", timeout_s=3.0)
            except RealSenseBridgeError as exc:
                close_error = exc
        self._shutdown_process()
        if close_error is not None:
            raise close_error

    def _send(self, payload: dict[str, object]) -> None:
        process = self._process
        if process is None or process.stdin is None or process.poll() is not None:
            raise RealSenseBridgeError(self._exit_message())
        try:
            process.stdin.write(json.dumps(payload, ensure_ascii=True) + "\n")
            process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise RealSenseBridgeError(self._exit_message()) from exc

    def _read_response(
        self,
        *,
        expected_type: str,
        timeout_s: float,
    ) -> dict[str, object]:
        deadline = time.monotonic() + timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RealSenseBridgeError(
                    f"RealSense worker timed out waiting for {expected_type}"
                )
            try:
                line = self._lines.get(timeout=remaining)
            except queue.Empty as exc:
                raise RealSenseBridgeError(
                    f"RealSense worker timed out waiting for {expected_type}"
                ) from exc
            if line is None:
                raise RealSenseBridgeError(self._exit_message())

            stripped = line.strip()
            if not stripped:
                continue
            try:
                decoded = json.loads(stripped)
            except json.JSONDecodeError:
                self._diagnostics.append(stripped)
                continue
            if not isinstance(decoded, dict):
                self._diagnostics.append(stripped)
                continue
            response = cast(dict[str, object], decoded)
            response_type = response.get("type")
            if response_type == "error":
                raise RealSenseBridgeError(
                    str(response.get("message", "unknown RealSense worker error"))
                )
            if response_type != expected_type:
                raise RealSenseBridgeError(
                    "RealSense worker returned "
                    f"{response_type!r} while waiting for {expected_type!r}"
                )
            return response

    def _exit_message(self) -> str:
        process = self._process
        return_code = process.poll() if process is not None else None
        message = f"RealSense worker exited unexpectedly (code {return_code})"
        if self._diagnostics:
            message += ": " + " | ".join(self._diagnostics)
        return message

    def _shutdown_process(self) -> None:
        process = self._process
        self._process = None
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
