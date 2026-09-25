"""Direct USB Intel RealSense D435i person detector."""

from __future__ import annotations

import importlib
import logging
import math
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast

from .models import CameraFrame, PerceptionResult
from .realsense_bridge import RealSenseBridge, RealSenseBridgeError

logger = logging.getLogger(__name__)


class PerceptionError(RuntimeError):
    """Raised when camera capture or person detection fails."""


class _RealSenseFrame(Protocol):
    def get_data(self) -> object: ...

    def get_width(self) -> int: ...

    def get_height(self) -> int: ...


class _RealSenseDepthFrame(_RealSenseFrame, Protocol):
    def get_distance(self, x: int, y: int) -> float: ...


class _RealSenseFrames(Protocol):
    def get_color_frame(self) -> _RealSenseFrame | None: ...

    def get_depth_frame(self) -> _RealSenseDepthFrame | None: ...


class _RealSensePipeline(Protocol):
    def start(self, config: object) -> object: ...

    def wait_for_frames(self, timeout_ms: int) -> _RealSenseFrames: ...

    def stop(self) -> None: ...


class _RealSenseConfig(Protocol):
    def enable_device(self, serial: str) -> None: ...

    def enable_stream(self, *args: object) -> None: ...


class _RealSenseAlign(Protocol):
    def process(self, frames: _RealSenseFrames) -> _RealSenseFrames: ...


class _RealSenseStreamValues(Protocol):
    color: object
    depth: object


class _RealSenseFormatValues(Protocol):
    bgr8: object
    z16: object


class _RealSenseApi(Protocol):
    pipeline: Callable[[], _RealSensePipeline]
    config: Callable[[], _RealSenseConfig]
    align: Callable[[object], _RealSenseAlign]
    stream: _RealSenseStreamValues
    format: _RealSenseFormatValues


class _NumpyApi(Protocol):
    def asanyarray(self, value: object) -> object: ...

    def rot90(self, image: object, k: int) -> object: ...


class _HogDescriptor(Protocol):
    def setSVMDetector(self, detector: object) -> None: ...

    def detectMultiScale(
        self,
        image: object,
        *,
        winStride: tuple[int, int],
        padding: tuple[int, int],
        scale: float,
    ) -> tuple[Sequence[Sequence[int]], Sequence[float]]: ...


class _OpenCvApi(Protocol):
    HOGDescriptor: Callable[[], _HogDescriptor]
    HOGDescriptor_getDefaultPeopleDetector: Callable[[], object]


@dataclass(frozen=True, slots=True)
class RealSenseBindings:
    realsense: _RealSenseApi
    numpy: _NumpyApi
    opencv: _OpenCvApi


class RealSensePersonDetector:
    """Capture aligned color/depth frames and detect people in RGB images."""

    def __init__(
        self,
        *,
        serial: str | None = None,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        detection_fps: float = 5.0,
        frame_timeout_ms: int = 5000,
        min_score: float = 0.5,
        max_distance_m: float | None = 4.0,
        rgb_rotation_deg: int = 0,
        bindings: RealSenseBindings | None = None,
        bridge_python: str | None = None,
    ) -> None:
        if width <= 0 or height <= 0 or fps <= 0:
            raise ValueError("camera width, height, and fps must be positive")
        if detection_fps <= 0:
            raise ValueError("detection_fps must be positive")
        if frame_timeout_ms <= 0:
            raise ValueError("frame_timeout_ms must be positive")
        if max_distance_m is not None and max_distance_m <= 0:
            raise ValueError("max_distance_m must be positive")
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
        self._bindings = bindings
        self._bridge_python = bridge_python
        self._bridge: RealSenseBridge | None = None
        self._pipeline: _RealSensePipeline | None = None
        self._align: _RealSenseAlign | None = None
        self._hog: _HogDescriptor | None = None
        self._last_detection_at_s: float | None = None
        self._cached_rectangles: Sequence[Sequence[int]] = ()
        self._cached_scores: Sequence[float] = ()
        self._lock = threading.Lock()

    @property
    def opened(self) -> bool:
        return self._pipeline is not None or (
            self._bridge is not None and self._bridge.opened
        )

    def open(self) -> None:
        with self._lock:
            if self.opened:
                return
            if self._bindings is None:
                try:
                    bindings = self._load_bindings()
                except PerceptionError as native_error:
                    self._open_bridge(native_error)
                    return
            else:
                bindings = self._bindings
            self._open_native(bindings)

    def _open_native(self, bindings: RealSenseBindings) -> None:
        pipeline = bindings.realsense.pipeline()
        config = bindings.realsense.config()
        if self.serial:
            config.enable_device(self.serial)
        config.enable_stream(
            bindings.realsense.stream.depth,
            self.width,
            self.height,
            bindings.realsense.format.z16,
            self.fps,
        )
        config.enable_stream(
            bindings.realsense.stream.color,
            self.width,
            self.height,
            bindings.realsense.format.bgr8,
            self.fps,
        )

        try:
            pipeline.start(config)
            align = bindings.realsense.align(bindings.realsense.stream.color)
            hog = bindings.opencv.HOGDescriptor()
            hog.setSVMDetector(bindings.opencv.HOGDescriptor_getDefaultPeopleDetector())
        except Exception as exc:
            try:
                pipeline.stop()
            except Exception:
                logger.exception("failed to stop RealSense pipeline after open failure")
            raise PerceptionError(f"failed to open RealSense D435i: {exc}") from exc

        self._bindings = bindings
        self._pipeline = pipeline
        self._align = align
        self._hog = hog
        self._reset_detection_cache()

    def _open_bridge(self, native_error: PerceptionError) -> None:
        bridge = RealSenseBridge(
            serial=self.serial,
            width=self.width,
            height=self.height,
            fps=self.fps,
            detection_fps=self.detection_fps,
            frame_timeout_ms=self.frame_timeout_ms,
            min_score=self.min_score,
            max_distance_m=self.max_distance_m,
            rgb_rotation_deg=self.rgb_rotation_deg,
            python_executable=self._bridge_python,
        )
        try:
            bridge.open()
        except RealSenseBridgeError as bridge_error:
            raise PerceptionError(
                "D435i bindings failed in both the project and system Python. "
                f"Project Python: {native_error}. System Python bridge: {bridge_error}"
            ) from bridge_error
        logger.warning(
            "native RealSense bindings unavailable (%s); using %s bridge",
            native_error,
            bridge.python_executable,
        )
        self._bridge = bridge

    def close(self) -> None:
        with self._lock:
            bridge = self._bridge
            self._bridge = None
            if bridge is not None:
                try:
                    bridge.close()
                except RealSenseBridgeError as exc:
                    raise PerceptionError(
                        f"failed to stop RealSense D435i: {exc}"
                    ) from exc
                return
            pipeline = self._pipeline
            self._pipeline = None
            self._align = None
            self._hog = None
            self._reset_detection_cache()
            if pipeline is None:
                return
            try:
                pipeline.stop()
            except Exception as exc:
                raise PerceptionError(f"failed to stop RealSense D435i: {exc}") from exc

    def capture(self) -> PerceptionResult:
        return self.capture_frame(include_rgb=False).observation

    def capture_frame(self, *, include_rgb: bool = True) -> CameraFrame:
        with self._lock:
            bridge = self._bridge
            if bridge is not None:
                try:
                    return bridge.capture_frame(include_rgb=include_rgb)
                except RealSenseBridgeError as exc:
                    raise PerceptionError(f"D435i capture failed: {exc}") from exc
            pipeline = self._pipeline
            align = self._align
            hog = self._hog
            bindings = self._bindings
            if pipeline is None or align is None or hog is None or bindings is None:
                raise PerceptionError("RealSense D435i is not open")

            try:
                frames = pipeline.wait_for_frames(self.frame_timeout_ms)
                aligned_frames = align.process(frames)
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()
                if color_frame is None or depth_frame is None:
                    raise PerceptionError("D435i returned an incomplete frame set")

                image = bindings.numpy.asanyarray(color_frame.get_data())
                detection_at_s = time.monotonic()
                if self._should_detect(detection_at_s):
                    rectangles, scores = hog.detectMultiScale(
                        image,
                        winStride=(8, 8),
                        padding=(8, 8),
                        scale=1.05,
                    )
                    self._cached_rectangles = rectangles
                    self._cached_scores = scores
                    self._last_detection_at_s = detection_at_s
                else:
                    rectangles = self._cached_rectangles
                    scores = self._cached_scores
                observed_at_s = time.monotonic()
                observation = self._build_result(
                    color_frame,
                    depth_frame,
                    rectangles,
                    scores,
                    observed_at_s=observed_at_s,
                )
                rgb_image = None
                if include_rgb:
                    output_image = image
                    if self.rgb_rotation_deg:
                        rotations = {90: 3, 180: 2, 270: 1}
                        output_image = bindings.numpy.rot90(
                            image, rotations[self.rgb_rotation_deg]
                        )
                    rgb_image = self._to_rgb_image(output_image)
                return CameraFrame(
                    observed_at_s=observed_at_s,
                    rgb=rgb_image,
                    depth=(
                        self._copy_image(
                            bindings.numpy.asanyarray(depth_frame.get_data())
                        )
                        if include_rgb
                        else None
                    ),
                    observation=observation,
                    nearest_obstacle_distance_m=(
                        self._sample_obstacle_distance(
                            depth_frame,
                            color_frame.get_width(),
                            color_frame.get_height(),
                        )
                        if include_rgb
                        else None
                    ),
                )
            except PerceptionError:
                raise
            except Exception as exc:
                raise PerceptionError(f"D435i capture failed: {exc}") from exc

    def _should_detect(self, now_s: float) -> bool:
        last_detection_at_s = self._last_detection_at_s
        return last_detection_at_s is None or (
            now_s - last_detection_at_s >= 1.0 / self.detection_fps
        )

    def _reset_detection_cache(self) -> None:
        self._last_detection_at_s = None
        self._cached_rectangles = ()
        self._cached_scores = ()

    def _build_result(
        self,
        color_frame: _RealSenseFrame,
        depth_frame: _RealSenseDepthFrame,
        rectangles: Sequence[Sequence[int]],
        scores: Sequence[float],
        *,
        observed_at_s: float | None = None,
    ) -> PerceptionResult:
        accepted_scores: list[float] = []
        distances: list[float] = []
        target: tuple[float, float] | None = None
        width = color_frame.get_width()
        height = color_frame.get_height()

        for rectangle, score_value in zip(rectangles, scores, strict=False):
            score = float(score_value)
            if score < self.min_score or len(rectangle) < 4:
                continue
            x, y, box_width, box_height = (
                int(rectangle[0]),
                int(rectangle[1]),
                int(rectangle[2]),
                int(rectangle[3]),
            )
            center_x = min(max(x + box_width // 2, 0), width - 1)
            center_y = min(max(y + box_height // 2, 0), height - 1)
            distance = float(depth_frame.get_distance(center_x, center_y))
            if distance > 0:
                if self.max_distance_m is not None and distance > self.max_distance_m:
                    continue
                distances.append(distance)
                if target is None or distance < target[0]:
                    target = (distance, (center_x + 0.5) / width)
            accepted_scores.append(score)

        confidence = (
            self._score_to_confidence(max(accepted_scores)) if accepted_scores else None
        )
        return PerceptionResult(
            observed_at_s=(
                time.monotonic() if observed_at_s is None else observed_at_s
            ),
            person_count=len(accepted_scores),
            nearest_person_distance_m=min(distances) if distances else None,
            person_center_x=target[1] if target is not None else None,
            confidence=confidence,
            source=f"realsense:{self.serial or 'D435i'}",
        )

    @staticmethod
    def _sample_obstacle_distance(
        depth_frame: _RealSenseDepthFrame,
        width: int,
        height: int,
    ) -> float | None:
        distances: list[float] = []
        for row in range(7):
            y = round(height * (0.2 + row * 0.1))
            for column in range(9):
                x = round(width * (0.2 + column * 0.075))
                distance = float(
                    depth_frame.get_distance(
                        min(max(x, 0), width - 1),
                        min(max(y, 0), height - 1),
                    )
                )
                if distance > 0:
                    distances.append(distance)
        if not distances:
            return None
        distances.sort()
        return distances[max(0, round((len(distances) - 1) * 0.1))]

    @staticmethod
    def _copy_image(image: object) -> object:
        copy = getattr(image, "copy", None)
        return copy() if callable(copy) else image

    @classmethod
    def _to_rgb_image(cls, image: object) -> object:
        try:
            converted = cast(Any, image)[..., ::-1]
        except (IndexError, TypeError):
            return cls._copy_image(image)
        return cls._copy_image(converted)

    @staticmethod
    def _score_to_confidence(score: float) -> float:
        if score >= 0:
            return 1.0 / (1.0 + math.exp(-score))
        exp_score = math.exp(score)
        return exp_score / (1.0 + exp_score)

    @staticmethod
    def _load_bindings() -> RealSenseBindings:
        try:
            realsense_module = importlib.import_module("pyrealsense2")
            numpy_module = importlib.import_module("numpy")
            opencv_module = importlib.import_module("cv2")
        except ImportError as exc:
            raise PerceptionError(
                f"native D435i dependencies are unavailable or ABI-incompatible: {exc}"
            ) from exc
        return RealSenseBindings(
            realsense=cast(_RealSenseApi, cast(object, realsense_module)),
            numpy=cast(_NumpyApi, cast(object, numpy_module)),
            opencv=cast(_OpenCvApi, cast(object, opencv_module)),
        )
