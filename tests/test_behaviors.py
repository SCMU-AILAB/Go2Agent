import math

import numpy as np
from conftest import make_frame

from go2agent.behaviors.base import BehaviorEnv, Status, Step, Tick
from go2agent.behaviors.geometry import CameraModel, bearing, distance
from go2agent.behaviors.primitives import BEHAVIORS
from go2agent.core.types import Frame, RobotState, Track, Tracks, Velocity
from go2agent.robot.go2_actions import GO2_ACTIONS

ENV = BehaviorEnv(CameraModel(), {a.name: a for a in GO2_ACTIONS})


def make(name, **args):
    cls = BEHAVIORS[name]
    return cls(cls.args_model(**args), ENV)


def tick(now, tracks=(), depth=None, healthy=True, busy=False):
    frame = make_frame(now, depth=depth)
    return Tick(now, frame, Tracks(0, now, tuple(tracks)), healthy, RobotState(now, True, busy=busy))


def vel(step: Step) -> Velocity:
    assert step.velocity is not None
    return step.velocity


def person(tid=1, cx=0.5, h=0.6):
    return Track(tid, "person", (cx - 0.1, 0.2, cx + 0.1, 0.2 + h), 0.9)


def test_bearing_sign_and_center():
    cam = CameraModel()
    assert bearing((0.4, 0, 0.6, 1), cam) == 0
    assert bearing((0.0, 0, 0.2, 1), cam) > 0  # target on the left -> positive
    assert math.isclose(bearing((0.9, 0, 1.1, 1), cam), -cam.hfov / 2, rel_tol=1e-6)


def test_distance_prefers_depth_then_height():
    t = person()
    depth = np.full((48, 64), 2.5, np.float32)
    assert distance(t, Frame(0, 0, np.zeros((48, 64, 3), np.uint8), depth), CameraModel()) == 2.5
    far, near = distance(person(h=0.3), None, CameraModel()), distance(person(h=0.8), None, CameraModel())
    assert far is not None and near is not None and far > near > 0
    assert distance(Track(2, "chair", (0.4, 0.4, 0.6, 0.6), 0.9), None, CameraModel()) is None


def test_follow_moves_toward_far_target_and_turns_toward_it():
    b = make("follow", target=1)
    s = b.tick(tick(0, [person(cx=0.4)], depth=3.0))
    assert s.status is Status.RUNNING and vel(s).vx > 0 and vel(s).wz > 0


def test_follow_holds_at_distance():
    s = make("follow", target=1, distance=1.2).tick(tick(0, [person()], depth=1.2))
    assert vel(s).vx == 0 and abs(vel(s).wz) < 1e-9


def test_follow_waits_then_fails_when_target_lost():
    b = make("follow", target=1)
    b.tick(tick(0, [person()], depth=2.0))
    s = b.tick(tick(0.5))
    assert s.status is Status.RUNNING and s.velocity is None
    assert b.tick(tick(0.8, [person()], depth=2.0)).status is Status.RUNNING  # re-acquired
    b.tick(tick(1.0))
    assert b.tick(tick(2.1)).reason == "target_lost"


def test_target_behaviors_fail_when_perception_down():
    for name in ("follow", "approach", "face"):
        assert (
            make(name, target=1).tick(tick(0, [person()], healthy=False)).reason == "perception_unavailable"
        )


def test_approach_succeeds_when_close_and_centered():
    b = make("approach", target=1, stop_distance=0.8)
    assert vel(b.tick(tick(0, [person()], depth=2.0))).vx > 0
    assert b.tick(tick(1, [person()], depth=0.85)).status is Status.SUCCEEDED


def test_face_turns_then_succeeds():
    b = make("face", target=1)
    s = b.tick(tick(0, [person(cx=0.9)]))
    assert vel(s).wz < 0 and vel(s).vx == 0
    assert b.tick(tick(1, [person(cx=0.5)])).status is Status.SUCCEEDED


def test_search_succeeds_on_new_track_and_times_out():
    b = make("search", direction="right", timeout=2)
    s = b.tick(tick(0, [person(1)]))
    assert vel(s).wz < 0
    s = b.tick(tick(1, [person(1), person(7, cx=0.8)]))
    assert s.status is Status.SUCCEEDED and s.reason == "found:7"
    b2 = make("search", timeout=2)
    b2.tick(tick(0))
    assert b2.tick(tick(2.5)).reason == "not_found"


def test_move_translates_then_turns():
    b = make("move", dx=0.6, dyaw=0.6)
    assert vel(b.tick(tick(0))).vx > 0
    assert vel(b.tick(tick(2.5))).wz > 0
    assert b.tick(tick(3.1)).status is Status.SUCCEEDED


def test_perform_starts_once_and_waits_for_duration():
    b = make("perform", action="hello")
    assert b.tick(tick(0)).perform == "hello"
    s = b.tick(tick(1))
    assert s.status is Status.RUNNING and s.perform is None
    assert b.tick(tick(5.0, busy=True)).status is Status.RUNNING
    assert b.tick(tick(5.1)).status is Status.SUCCEEDED
    assert make("perform", action="moonwalk").tick(tick(0)).reason == "unknown_action"


def test_perform_fails_when_the_robot_reports_the_action_failed():
    b = make("perform", action="hello")
    b.tick(tick(0))
    t = tick(0.5)
    failed = Tick(
        t.now, t.frame, t.tracks, t.tracker_healthy, RobotState(0.5, True, action_error="status 3104")
    )
    step = b.tick(failed)
    assert step.status is Status.FAILED and "3104" in step.reason
