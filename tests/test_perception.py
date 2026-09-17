from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

from app.perception import (
    _DepthSafetyGate,
    _ObservationLogGate,
    _RepeatedErrorLogGate,
    parse_args,
)
from perception import (
    EventDetector,
    PerceptionError,
    PerceptionResult,
    RealSensePersonDetector,
    WorldEventType,
    WorldState,
)
from perception.realsense import RealSenseBindings


class VisionProfileArgumentTests(unittest.TestCase):
    def test_new_profile_and_rollback_thinking_flags(self):
        with patch("sys.argv", ["perception", "--vision-social-profile", "egocentric", "--vision-disable-thinking"]):
            args = parse_args()
            self.assertEqual(args.vision_social_profile, "egocentric")
            self.assertTrue(args.vision_disable_thinking)
            self.assertFalse(args.hardware)
        with patch("sys.argv", ["perception", "--vision-disable-thinking", "--no-vision-disable-thinking"]):
            self.assertFalse(parse_args().vision_disable_thinking)


def observation(
    at_s: float,
    *,
    people: int,
    distance_m: float | None = None,
) -> PerceptionResult:
    return PerceptionResult(
        observed_at_s=at_s,
        person_count=people,
        nearest_person_distance_m=distance_m,
    )


class EventDetectorTests(unittest.TestCase):
    def test_person_entry_is_emitted_once_until_person_leaves(self) -> None:
        state = WorldState(absence_reset_s=2.0)
        detector = EventDetector()

        entered = detector.update(
            observation(1.0, people=1, distance_m=2.0),
            state,
        )
        repeated = detector.update(
            observation(1.1, people=1, distance_m=2.0),
            state,
        )
        brief_miss = detector.update(observation(2.0, people=0), state)
        left = detector.update(observation(3.1, people=0), state)

        self.assertEqual([event.type for event in entered], [WorldEventType.PERSON_ENTERED])
        self.assertEqual(entered[0].data["distance_m"], 2.0)
        self.assertEqual(repeated, ())
        self.assertEqual(brief_miss, ())
        self.assertEqual([event.type for event in left], [WorldEventType.PERSON_LEFT])

    def test_too_close_event_uses_hysteresis(self) -> None:
        state = WorldState()
        detector = EventDetector(
            too_close_distance_m=0.8,
            too_close_release_m=1.0,
        )
        detector.update(observation(1.0, people=1, distance_m=2.0), state)

        first = detector.update(
            observation(2.0, people=1, distance_m=0.7),
            state,
        )
        repeated = detector.update(
            observation(2.1, people=1, distance_m=0.6),
            state,
        )
        detector.update(observation(3.0, people=1, distance_m=1.1), state)
        second = detector.update(
            observation(4.0, people=1, distance_m=0.7),
            state,
        )

        self.assertEqual([event.type for event in first], [WorldEventType.PERSON_TOO_CLOSE])
        self.assertEqual(repeated, ())
        self.assertEqual([event.type for event in second], [WorldEventType.PERSON_TOO_CLOSE])

    def test_entry_and_too_close_are_both_emitted_for_close_new_person(self) -> None:
        state = WorldState()
        detector = EventDetector()

        events = detector.update(
            observation(1.0, people=1, distance_m=0.6),
            state,
        )

        self.assertEqual(
            [event.type for event in events],
            [
                WorldEventType.PERSON_ENTERED,
                WorldEventType.PERSON_TOO_CLOSE,
            ],
        )

    def test_out_of_order_observation_is_rejected(self) -> None:
        state = WorldState()
        detector = EventDetector()
        detector.update(observation(2.0, people=0), state)

        with self.assertRaisesRegex(ValueError, "time ordered"):
            detector.update(observation(1.0, people=0), state)


class FakeColorFrame:
    def get_data(self) -> object:
        return [[0]]

    def get_width(self) -> int:
        return 640

    def get_height(self) -> int:
        return 480


class FakeDepthFrame(FakeColorFrame):
    def __init__(self, distances: dict[tuple[int, int], float]) -> None:
        self.distances = distances
        self.queries: list[tuple[int, int]] = []

    def get_distance(self, x: int, y: int) -> float:
        self.queries.append((x, y))
        return self.distances.get((x, y), 2.0)


class FakeFrames:
    def __init__(self, depth: FakeDepthFrame) -> None:
        self.color = FakeColorFrame()
        self.depth = depth

    def get_color_frame(self) -> FakeColorFrame:
        return self.color

    def get_depth_frame(self) -> FakeDepthFrame:
        return self.depth


class FakePipeline:
    def __init__(self, frames: FakeFrames) -> None:
        self.frames = frames
        self.started_with: object | None = None
        self.timeout_calls: list[int] = []
        self.stop_count = 0

    def start(self, config: object) -> object:
        self.started_with = config
        return object()

    def wait_for_frames(self, timeout_ms: int) -> FakeFrames:
        self.timeout_calls.append(timeout_ms)
        return self.frames

    def stop(self) -> None:
        self.stop_count += 1


class FakeConfig:
    def __init__(self) -> None:
        self.serials: list[str] = []
        self.streams: list[tuple[object, ...]] = []

    def enable_device(self, serial: str) -> None:
        self.serials.append(serial)

    def enable_stream(self, *args: object) -> None:
        self.streams.append(args)


class FakeAlign:
    def __init__(self) -> None:
        self.inputs: list[FakeFrames] = []

    def process(self, frames: FakeFrames) -> FakeFrames:
        self.inputs.append(frames)
        return frames


class FakeHog:
    def __init__(
        self,
        rectangles: list[tuple[int, int, int, int]],
        scores: list[float],
    ) -> None:
        self.rectangles = rectangles
        self.scores = scores
        self.detector: object | None = None
        self.call_count = 0

    def setSVMDetector(self, detector: object) -> None:
        self.detector = detector

    def detectMultiScale(
        self,
        image: object,
        *,
        winStride: tuple[int, int],
        padding: tuple[int, int],
        scale: float,
    ) -> tuple[list[tuple[int, int, int, int]], list[float]]:
        self.call_count += 1
        return self.rectangles, self.scores


class FakeNumpy:
    def __init__(self) -> None:
        self.rotations: list[int] = []

    def asanyarray(self, value: object) -> object:
        return value

    def rot90(self, image: object, k: int) -> object:
        self.rotations.append(k)
        return image


def make_bindings(
    *,
    pipeline: FakePipeline,
    config: FakeConfig,
    align: FakeAlign,
    hog: FakeHog,
) -> RealSenseBindings:
    realsense = SimpleNamespace(
        pipeline=lambda: pipeline,
        config=lambda: config,
        align=lambda stream: align,
        stream=SimpleNamespace(depth="depth", color="color"),
        format=SimpleNamespace(z16="z16", bgr8="bgr8"),
    )
    opencv = SimpleNamespace(
        HOGDescriptor=lambda: hog,
        HOGDescriptor_getDefaultPeopleDetector=lambda: "detector",
    )
    namespace = SimpleNamespace(
        realsense=realsense,
        numpy=FakeNumpy(),
        opencv=opencv,
    )
    return cast(RealSenseBindings, namespace)


class RealSensePersonDetectorTests(unittest.TestCase):
    def test_capture_aligns_depth_and_filters_distant_people(self) -> None:
        depth = FakeDepthFrame({(30, 50): 1.5, (120, 130): 6.0})
        frames = FakeFrames(depth)
        pipeline = FakePipeline(frames)
        config = FakeConfig()
        align = FakeAlign()
        hog = FakeHog(
            [(10, 20, 40, 60), (100, 100, 40, 60)],
            [1.0, 2.0],
        )
        detector = RealSensePersonDetector(
            serial="camera-1",
            frame_timeout_ms=1234,
            min_score=0.5,
            max_distance_m=4.0,
            bindings=make_bindings(
                pipeline=pipeline,
                config=config,
                align=align,
                hog=hog,
            ),
        )

        detector.open()
        result = detector.capture()
        detector.close()

        self.assertEqual(config.serials, ["camera-1"])
        self.assertEqual(
            config.streams,
            [
                ("depth", 640, 480, "z16", 30),
                ("color", 640, 480, "bgr8", 30),
            ],
        )
        self.assertEqual(pipeline.timeout_calls, [1234])
        self.assertEqual(align.inputs, [frames])
        self.assertEqual(depth.queries, [(30, 50), (120, 130)])
        self.assertEqual(result.person_count, 1)
        self.assertEqual(result.nearest_person_distance_m, 1.5)
        self.assertEqual(result.source, "realsense:camera-1")
        self.assertIsNotNone(result.confidence)
        self.assertEqual(pipeline.stop_count, 1)

    def test_capture_requires_open_camera(self) -> None:
        detector = RealSensePersonDetector(bindings=cast(RealSenseBindings, object()))

        with self.assertRaisesRegex(PerceptionError, "not open"):
            detector.capture()

    def test_capture_frame_keeps_rgb_depth_and_safety_distance(self) -> None:
        depth = FakeDepthFrame({})
        detector = RealSensePersonDetector(
            bindings=make_bindings(
                pipeline=FakePipeline(FakeFrames(depth)),
                config=FakeConfig(),
                align=FakeAlign(),
                hog=FakeHog([], []),
            )
        )

        detector.open()
        frame = detector.capture_frame()
        detector.close()

        self.assertIsNotNone(frame.rgb)
        self.assertIsNotNone(frame.depth)
        self.assertEqual(frame.nearest_obstacle_distance_m, 2.0)
        self.assertEqual(frame.observation.person_count, 0)

    def test_capture_frame_applies_output_rotation_before_encoding(self) -> None:
        depth = FakeDepthFrame({})
        bindings = make_bindings(
            pipeline=FakePipeline(FakeFrames(depth)),
            config=FakeConfig(),
            align=FakeAlign(),
            hog=FakeHog([], []),
        )
        detector = RealSensePersonDetector(
            rgb_rotation_deg=180,
            bindings=bindings,
        )

        detector.open()
        detector.capture_frame()
        detector.close()

        self.assertEqual(bindings.numpy.rotations, [2])

    def test_capture_throttles_expensive_detection_without_throttling_frames(
        self,
    ) -> None:
        depth = FakeDepthFrame({(30, 50): 1.5})
        pipeline = FakePipeline(FakeFrames(depth))
        hog = FakeHog([(10, 20, 40, 60)], [1.0])
        detector = RealSensePersonDetector(
            detection_fps=5.0,
            bindings=make_bindings(
                pipeline=pipeline,
                config=FakeConfig(),
                align=FakeAlign(),
                hog=hog,
            ),
        )

        detector.open()
        with patch(
            "perception.realsense.time.monotonic",
            side_effect=(1.0, 1.01, 1.05, 1.06, 1.25, 1.26),
        ):
            frames = [detector.capture_frame() for _ in range(3)]
        detector.close()

        self.assertEqual(len(pipeline.timeout_calls), 3)
        self.assertEqual(hog.call_count, 2)
        self.assertTrue(all(frame.observation.person_count == 1 for frame in frames))

    def test_close_is_idempotent(self) -> None:
        depth = FakeDepthFrame({})
        pipeline = FakePipeline(FakeFrames(depth))
        detector = RealSensePersonDetector(
            bindings=make_bindings(
                pipeline=pipeline,
                config=FakeConfig(),
                align=FakeAlign(),
                hog=FakeHog([], []),
            )
        )

        detector.open()
        detector.close()
        detector.close()

        self.assertEqual(pipeline.stop_count, 1)


class ObservationLogGateTests(unittest.TestCase):
    def test_unchanged_observations_are_rate_limited(self) -> None:
        gate = _ObservationLogGate(interval_s=5.0)
        self.assertTrue(gate.should_print(observation(1.0, people=0)))
        self.assertFalse(gate.should_print(observation(2.0, people=0)))
        self.assertTrue(gate.should_print(observation(6.0, people=0)))

    def test_person_state_changes_are_printed_immediately(self) -> None:
        gate = _ObservationLogGate(interval_s=5.0)
        self.assertTrue(gate.should_print(observation(1.0, people=0)))
        self.assertTrue(gate.should_print(observation(1.1, people=1, distance_m=2.0)))
        self.assertFalse(gate.should_print(observation(1.2, people=1, distance_m=2.0)))

    def test_verbose_mode_prints_every_observation(self) -> None:
        gate = _ObservationLogGate(interval_s=5.0, verbose=True)
        self.assertTrue(gate.should_print(observation(1.0, people=0)))
        self.assertTrue(gate.should_print(observation(1.1, people=0)))

    def test_state_changes_can_be_periodic_only_for_noisy_detectors(self) -> None:
        gate = _ObservationLogGate(
            interval_s=5.0,
            immediate_state_changes=False,
        )

        self.assertTrue(gate.should_print(observation(1.0, people=0)))
        self.assertFalse(gate.should_print(observation(1.1, people=1)))
        self.assertFalse(gate.should_print(observation(1.2, people=0)))
        self.assertTrue(gate.should_print(observation(6.0, people=1)))


class RepeatedErrorLogGateTests(unittest.TestCase):
    def test_repeated_errors_are_rate_limited_and_counted(self) -> None:
        gate = _RepeatedErrorLogGate(interval_s=5.0)

        self.assertEqual(gate.admit("decision", "failed", now_s=1.0), 0)
        self.assertIsNone(gate.admit("decision", "failed", now_s=2.0))
        self.assertIsNone(gate.admit("decision", "failed", now_s=3.0))
        self.assertEqual(gate.admit("decision", "failed", now_s=6.0), 2)

    def test_changed_error_and_recovery_emit_immediately(self) -> None:
        gate = _RepeatedErrorLogGate(interval_s=5.0)

        self.assertEqual(gate.admit("decision", "first", now_s=1.0), 0)
        self.assertEqual(gate.admit("execution", "second", now_s=1.1), 0)
        gate.reset()
        self.assertEqual(gate.admit("decision", "first", now_s=1.2), 0)


class DepthSafetyGateTests(unittest.TestCase):
    def test_gate_latches_until_release_distance(self) -> None:
        gate = _DepthSafetyGate(stop_distance_m=0.4, release_distance_m=0.55)

        self.assertEqual(gate.update(0.35), "triggered")
        self.assertIsNone(gate.update(0.45))
        self.assertEqual(gate.update(0.6), "released")
