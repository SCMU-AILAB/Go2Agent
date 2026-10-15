"""Exercise the real driver with a fake SportClient and DDS subscription."""

import math
import threading
import time
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from go2agent.core.types import Velocity
from go2agent.robot.go2 import ACTION_METHODS, Go2Bindings, Go2Config, Go2Robot
from go2agent.robot.go2_actions import GO2_ACTIONS


def eventually(predicate, timeout=2):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            pytest.fail("Driver did not reach expected state")
        time.sleep(0.005)


def telemetry():
    return SimpleNamespace(
        mode=3,
        gait_type=1,
        position=[1.0, 2.0, 0.3],
        velocity=[0.1, 0.0, 0.0],
        error_code=0,
        imu_state=SimpleNamespace(rpy=[0.0, 0.0, 0.4]),
    )


class FakeClient:
    def __init__(self):
        self.calls = []
        self.block = None
        self.entered, self.release = threading.Event(), threading.Event()
        self.status = {}
        self.errors: dict[str, Exception] = {}
        self.timeouts: list[float] = []
        self.initialized = 0
        self.started_at: dict[str, float] = {}
        self.finished_at: dict[str, float] = {}

    def set_timeout(self, timeout):
        self.timeout = timeout
        self.timeouts.append(timeout)

    def init(self):
        self.initialized += 1

    def __getattr__(self, method):
        if method not in {*ACTION_METHODS.values(), "move", "stop_move"}:
            raise AttributeError(method)

        def call(*args):
            self.calls.append((method, args, threading.get_ident()))
            self.started_at[method] = time.monotonic()
            if self.block == method:
                self.entered.set()
                assert self.release.wait(3)
            self.finished_at[method] = time.monotonic()
            if method in self.errors:
                raise self.errors[method]
            return self.status.get(method, 0)

        return call

    def count(self, method):
        return sum(name == method for name, _, _ in self.calls)


@pytest.fixture
def rig(clock):
    client, action_client, lifecycle = FakeClient(), FakeClient(), []
    channel = SimpleNamespace(
        initialize=lambda domain, interface: lifecycle.append(("initialize", domain, interface)),
        release=lambda: lifecycle.append("release"),
    )

    def subscriber(callback):
        return SimpleNamespace(
            init_channel=lambda: callback(telemetry()),
            close_channel=lambda: lifecycle.append("close_channel"),
        )

    clients = iter((client, action_client))
    robot = Go2Robot(bindings=Go2Bindings(channel, lambda: next(clients), subscriber), clock=clock)
    robot.connect()
    yield robot, client, action_client, lifecycle
    client.release.set()
    action_client.release.set()
    robot.close()


def test_telemetry_liveness_snapshot_and_lifecycle(rig, clock):
    robot, client, action_client, lifecycle = rig
    assert lifecycle == [("initialize", 0, "eth0")]
    assert client.calls == [] and client.timeout == 0.2
    assert client is not action_client
    assert client.initialized == action_client.initialized == 1
    robot.connect()
    assert client.initialized == action_client.initialized == 1
    message = telemetry()
    robot._on_state(message)
    message.position[0] = 100
    state = robot.state()
    assert state.connected and not state.busy
    assert state.details["mode"] == 3 and state.details["gait_type"] == 1
    assert state.details["position"] == (1.0, 2.0, 0.3)
    assert state.details["velocity"] == (0.1, 0.0, 0.0)
    assert state.details["error_code"] == 0 and state.details["telemetry_age"] == 0
    clock.advance(2.01)
    assert not robot.state().connected
    robot.velocity(Velocity(0.1))
    assert "stale" in robot.state().details["command_refusal"]
    robot.close()
    assert not robot.state().connected
    assert lifecycle[-2:] == ["close_channel", "release"]
    robot.close()
    assert lifecycle.count("release") == 1


def test_velocity_nonblocking_and_zero_sent_once(rig):
    robot, client, _, _ = rig
    robot.velocity(Velocity())
    robot.velocity(Velocity())
    eventually(lambda: client.count("stop_move") == 1 and not robot.state().details["stopping"])
    client.block = "move"
    started = time.monotonic()
    robot.velocity(Velocity(0.2, -0.1, 0.3))
    assert time.monotonic() - started < 0.1
    assert client.entered.wait(1)
    assert client.calls[-1][1] == (0.2, -0.1, 0.3)
    assert client.calls[-1][2] != threading.get_ident()
    robot.velocity(Velocity())
    robot.velocity(Velocity())
    client.release.set()
    eventually(lambda: client.count("stop_move") == 2 and not robot.state().details["stopping"])
    robot.stop()
    time.sleep(0.03)
    assert client.count("stop_move") == 2


@pytest.mark.parametrize("spec", GO2_ACTIONS, ids=lambda spec: spec.name)
def test_every_catalog_action_maps_and_busy_lasts_for_duration(rig, clock, spec):
    robot, client, action_client, _ = rig
    assert set(ACTION_METHODS) == {action.name for action in GO2_ACTIONS}
    assert robot.actions() == GO2_ACTIONS
    started = time.monotonic()
    assert robot.perform(spec.name) is None
    assert time.monotonic() - started < 0.1
    assert robot.state().busy
    eventually(lambda: action_client.count(ACTION_METHODS[spec.name]) == 1 and not robot._action_running)
    assert client.timeouts == [0.2] and client.calls == []
    assert action_client.timeout == spec.duration + robot.config.action_timeout_margin_s
    clock.advance(spec.duration - 0.01)
    assert robot.state().busy
    clock.advance(0.02)
    assert not robot.state().busy


def test_busy_until_rpc_finishes_and_zero_does_not_cancel_action(rig, clock):
    robot, client, action_client, _ = rig
    action_client.block = "hello"
    robot.perform("hello")
    assert action_client.entered.wait(1)
    robot.velocity(Velocity())
    assert client.count("stop_move") == 0
    clock.advance(10)
    robot._on_state(telemetry())
    assert robot.state().busy
    assert "busy" in str(robot.perform("sit"))
    assert "busy" in robot.state().details["command_refusal"]
    action_client.release.set()
    eventually(lambda: not robot.state().busy)
    assert action_client.count("sit") == 0


def test_stop_discards_pending_actions_and_follows_inflight_rpc(rig):
    robot, client, action_client, _ = rig
    client.block = "move"
    robot.velocity(Velocity(0.2))
    assert client.entered.wait(1)
    robot.perform("hello")
    robot.stop()
    robot.stop()
    robot.perform("sit")
    assert "stopping" in robot.state().details["command_refusal"]
    assert client.count("stop_move") == 0
    client.release.set()
    eventually(lambda: client.count("stop_move") == 1 and not robot.state().details["stopping"])
    assert action_client.count("hello") == 0 and not robot.state().busy
    assert [name for name, _, _ in client.calls] == ["move", "stop_move"]


def test_zero_then_nonzero_during_stop_sends_stop_then_latest_move(rig):
    robot, client, _, _ = rig
    client.block = "stop_move"
    robot.velocity(Velocity())
    assert client.entered.wait(1)
    robot.velocity(Velocity(0.1))
    robot.velocity(Velocity(wz=0.2))
    assert robot.state().details["stopping"]
    assert client.count("move") == 0
    client.release.set()
    eventually(lambda: client.count("move") == 1)
    assert [(name, args) for name, args, _ in client.calls] == [
        ("stop_move", ()),
        ("move", (0.0, 0.0, 0.2)),
    ]
    # The resumed move must invalidate the previous successful stop.
    robot.velocity(Velocity())
    eventually(lambda: client.count("stop_move") == 2 and not robot.state().details["stopping"])
    robot.velocity(Velocity())
    assert client.count("stop_move") == 2


@pytest.mark.parametrize("rpc_running", [True, False])
def test_velocity_during_native_action_is_dropped_without_log_spam(rig, clock, caplog, rpc_running):
    robot, client, action_client, _ = rig
    action_client.block = "hello"
    robot.perform("hello")
    assert action_client.entered.wait(1)
    if not rpc_running:
        action_client.release.set()
        eventually(lambda: not robot._action_running)
    assert robot.state().busy
    with caplog.at_level("DEBUG", logger="go2agent.robot.go2"):
        for _ in range(20):
            robot.velocity(Velocity(wz=0.2))
    assert "busy" in robot.state().details["command_refusal"]
    assert len(caplog.records) <= 1
    assert all(record.levelname == "DEBUG" for record in caplog.records)
    action_client.release.set()
    eventually(lambda: not robot._action_running)
    clock.advance(next(spec.duration for spec in GO2_ACTIONS if spec.name == "hello") + 0.1)
    robot._on_state(telemetry())
    assert not robot.state().busy
    assert robot._pending_velocity is None
    assert [name for name, _, _ in action_client.calls] == ["hello"]
    assert client.calls == []
    robot.velocity(Velocity(0.1))
    eventually(lambda: client.count("move") == 1)
    assert client.calls[-1][1] == (0.1, 0.0, 0.0)
    assert robot.state().details["command_refusal"] is None


def test_stale_velocity_and_action_are_dropped_without_replay(rig, clock):
    robot, client, _, _ = rig
    clock.advance(2.01)
    robot.velocity(Velocity(wz=0.2))
    robot.perform("hello")
    assert not robot.state().connected
    assert "stale" in robot.state().details["command_refusal"]
    assert robot._pending_velocity is None and robot._pending_action is None
    robot._on_state(telemetry())
    robot.velocity(Velocity(0.1))
    eventually(lambda: client.count("move") == 1)
    assert [(name, args) for name, args, _ in client.calls] == [("move", (0.1, 0.0, 0.0))]


def test_disconnected_commands_are_dropped():
    robot = Go2Robot()
    robot.velocity(Velocity(0.1))
    assert "not connected" in str(robot.perform("hello"))
    robot.velocity(Velocity())
    assert not robot.state().connected
    assert "not connected" in robot.state().details["command_refusal"]
    assert robot._pending_velocity is None and robot._pending_action is None


def test_unavailable_known_action_is_refused(rig):
    robot, client, action_client, _ = rig
    action_client.hello = None
    assert robot.perform("hello") is not None
    assert "do not provide hello" in robot.state().details["command_refusal"]
    assert not robot.state().busy and not client.calls and not action_client.calls


def test_command_expiry_sends_stop_without_control_loop(rig, clock):
    robot, client, _, _ = rig
    robot.velocity(Velocity(0.2))
    eventually(lambda: client.count("move") == 1)
    clock.advance(0.6)
    eventually(lambda: client.count("stop_move") == 1)


@pytest.mark.parametrize("status", [1, 3104, None, True])
def test_action_sdk_errors_are_reported_without_stopping_motion(rig, status):
    robot, client, action_client, _ = rig
    action_client.status["hello"] = status
    robot.perform("hello")
    eventually(lambda: robot.state().action_error is not None)
    state = robot.state()
    assert "SDK status" in str(state.action_error)
    assert state.details["last_error"] == state.action_error
    assert not robot.state().busy
    time.sleep(0.03)
    assert client.calls == []
    robot.velocity(Velocity(0.1))
    eventually(lambda: client.count("move") == 1)
    assert client.count("stop_move") == 0


def test_stop_reaches_motion_client_while_action_rpc_is_blocked(rig, clock):
    robot, motion, action, _ = rig
    action.block = "hello"
    assert robot.perform("hello") is None
    assert action.entered.wait(1)
    started = time.monotonic()
    robot.stop()
    robot.velocity(Velocity(0.2))
    assert time.monotonic() - started < 0.1
    eventually(lambda: motion.count("stop_move") == 1 and not robot.state().details["stopping"])
    assert robot.state().busy and robot._action_running
    assert not action.release.is_set()
    assert motion.count("move") == 0  # Velocity stays non-blocking but is dropped throughout busy.
    assert robot._pending_velocity is None
    assert motion.calls[0][2] != action.calls[0][2]
    assert motion.timeouts == [0.2]
    robot.stop()
    assert "busy" in str(robot.perform("sit"))
    clock.advance(10)
    robot._on_state(telemetry())
    assert robot.state().busy  # An RPC can outlast the estimated duration.
    action.release.set()
    eventually(lambda: not robot.state().busy)
    robot.velocity(Velocity(0.2))
    eventually(lambda: motion.count("move") == 1)
    assert motion.count("stop_move") == 1
    assert motion.finished_at["stop_move"] < action.finished_at["hello"] < motion.started_at["move"]


def test_stop_preserves_minimum_action_duration(rig, clock):
    robot, motion, action, _ = rig
    action.block = "hello"
    robot.perform("hello")
    assert action.entered.wait(1)
    robot.stop()
    eventually(lambda: motion.count("stop_move") == 1 and not robot.state().details["stopping"])
    action.release.set()
    eventually(lambda: not robot._action_running)
    duration = next(spec.duration for spec in GO2_ACTIONS if spec.name == "hello")
    clock.advance(duration - 0.01)
    assert robot.state().busy
    clock.advance(0.02)
    assert not robot.state().busy


def test_perform_while_moving_waits_for_stop_completion(rig):
    robot, motion, action, _ = rig
    robot.velocity(Velocity(0.2))
    eventually(lambda: motion.count("move") == 1)
    motion.block = "stop_move"
    started = time.monotonic()
    assert robot.perform("hello") is None
    assert time.monotonic() - started < 0.1
    assert robot.state().busy
    assert motion.entered.wait(1)
    assert action.calls == []  # Entering stop_move alone is not sufficient.
    assert "busy" in str(robot.perform("sit"))
    motion.release.set()
    eventually(lambda: action.count("hello") == 1)
    assert motion.finished_at["stop_move"] <= action.started_at["hello"]
    assert [name for name, _, _ in motion.calls] == ["move", "stop_move"]
    assert [name for name, _, _ in action.calls] == ["hello"]


@pytest.mark.parametrize("failure", [3104, RuntimeError("stop disconnected")])
def test_failed_prerequisite_stop_refuses_action(rig, failure):
    robot, motion, action, _ = rig
    if isinstance(failure, Exception):
        motion.errors["stop_move"] = failure
    else:
        motion.status["stop_move"] = failure
    robot.velocity(Velocity(0.2))
    eventually(lambda: motion.count("move") == 1)
    assert robot.perform("hello") is None
    eventually(lambda: robot.state().action_error is not None)
    state = robot.state()
    assert "stop_move failed" in str(state.action_error)
    assert state.details["last_error"] == state.action_error
    assert not state.busy
    assert action.calls == []
    assert motion.count("stop_move") == 1


@pytest.mark.parametrize("blocked_method", ["move", "stop_move"])
def test_wait_for_stop_is_bounded_and_never_replays_action(rig, blocked_method):
    robot, motion, action, _ = rig
    robot.config = Go2Config(action_stop_timeout_s=0.05)
    motion.block = blocked_method
    robot.velocity(Velocity(0.2))
    eventually(lambda: motion.count("move") == 1)
    assert robot.perform("hello") is None
    assert motion.entered.wait(1)
    eventually(lambda: robot.state().action_error is not None, timeout=1)
    state = robot.state()
    assert "timed out waiting for stop_move" in str(state.action_error)
    assert state.details["last_error"] == state.action_error
    assert not state.busy and state.details["stopping"]
    assert action.calls == []
    motion.release.set()
    eventually(lambda: not robot.state().details["stopping"])
    robot.velocity(Velocity(0.1))
    eventually(lambda: motion.count("move") == 2)
    assert action.calls == []


def test_stale_telemetry_after_prerequisite_stop_refuses_action(rig, clock):
    robot, motion, action, _ = rig
    robot.velocity(Velocity(0.2))
    eventually(lambda: motion.count("move") == 1)
    motion.block = "stop_move"
    robot.perform("hello")
    assert motion.entered.wait(1)
    clock.advance(3)
    motion.release.set()
    eventually(lambda: robot.state().action_error is not None)
    assert "stale" in str(robot.state().action_error)
    assert not robot.state().busy
    robot._on_state(telemetry())
    assert action.calls == []


@pytest.mark.parametrize("error", [TimeoutError(), RuntimeError("action disconnected")])
def test_action_exception_is_reported_and_clears_on_next_action(rig, error):
    robot, motion, action, _ = rig
    action.errors["hello"] = error
    robot.perform("hello")
    eventually(lambda: robot.state().action_error is not None)
    state = robot.state()
    assert state.action_error == (str(error) or type(error).__name__)
    assert state.details["last_error"] == state.action_error
    assert not state.busy
    assert robot.perform("sit") is None
    eventually(lambda: action.count("sit") == 1)
    assert robot.state().action_error is None
    assert action.timeout == 2.0 + robot.config.action_timeout_margin_s
    assert motion.timeouts == [0.2] and motion.calls == []


def test_action_failure_does_not_disturb_inflight_stop(rig):
    robot, motion, action, _ = rig
    action.block = "hello"
    action.status["hello"] = 3104
    robot.perform("hello")
    assert action.entered.wait(1)
    motion.block = "stop_move"
    robot.stop()
    assert motion.entered.wait(1)
    action.release.set()
    eventually(lambda: robot.state().action_error is not None)
    assert not robot.state().busy
    assert robot.state().details["stopping"]
    robot.velocity(Velocity(0.1))
    assert motion.count("move") == 0
    motion.release.set()
    eventually(lambda: motion.count("move") == 1)
    assert [name for name, _, _ in motion.calls] == ["stop_move", "move"]


@pytest.mark.parametrize("failure", [3104, RuntimeError("move disconnected")])
def test_motion_failures_still_stop_motion(rig, failure):
    robot, motion, action, _ = rig
    if isinstance(failure, Exception):
        motion.errors["move"] = failure
    else:
        motion.status["move"] = failure
    robot.velocity(Velocity(0.1))
    eventually(lambda: motion.count("stop_move") == 1 and not robot.state().details["stopping"])
    assert robot.state().details["last_error"] is not None
    assert robot.state().action_error is None
    assert not robot.state().busy and action.calls == []


def test_close_stops_motion_then_waits_for_action_before_releasing_dds(rig):
    robot, motion, action, lifecycle = rig
    action.block = "hello"
    robot.perform("hello")
    assert action.entered.wait(1)
    motion_worker, action_worker = robot._worker, robot._action_worker
    errors: list[Exception] = []

    def close():
        try:
            robot.close()
        except Exception as exc:
            errors.append(exc)

    closer = threading.Thread(target=close)
    closer.start()
    try:
        eventually(lambda: motion.count("stop_move") == 1 and not robot.state().details["stopping"])
        assert closer.is_alive()
        assert not robot.state().connected
        assert "release" not in lifecycle and "close_channel" not in lifecycle
        assert "not connected" in str(robot.perform("sit"))
    finally:
        action.release.set()
        closer.join(timeout=2)
    assert not closer.is_alive() and errors == []
    assert motion_worker is not None and not motion_worker.is_alive()
    assert action_worker is not None and not action_worker.is_alive()
    assert lifecycle[-2:] == ["close_channel", "release"]
    assert not robot.state().busy


def test_close_retains_dds_if_action_worker_has_not_finished(rig, monkeypatch):
    robot, motion, action, lifecycle = rig
    action.block = "hello"
    robot.perform("hello")
    assert action.entered.wait(1)
    robot.stop()
    eventually(lambda: motion.count("stop_move") == 1 and not robot.state().details["stopping"])
    worker = robot._action_worker
    assert worker is not None
    with monkeypatch.context() as patch:
        patch.setattr(worker, "join", lambda timeout: None)
        with pytest.raises(TimeoutError, match="DDS remains open"):
            robot.close()
    assert "release" not in lifecycle and "close_channel" not in lifecycle
    action.release.set()
    robot.close()
    assert not worker.is_alive()
    assert lifecycle[-2:] == ["close_channel", "release"]


def test_invalid_inputs(rig):
    robot, _, _, _ = rig
    with pytest.raises(ValueError):
        robot.perform("unsupported")
    with pytest.raises(ValueError):
        robot.velocity(Velocity(float("nan")))
    with pytest.raises(ValueError):
        Go2Config(timeout_s=-1)
    with pytest.raises(ValueError):
        Go2Config(domain_id=True)
    with pytest.raises(ValueError):
        Go2Config(action_stop_timeout_s=0)
    with pytest.raises(ValueError):
        Go2Config(action_timeout_margin_s=float("inf"))


def test_action_client_init_failure_releases_dds():
    motion, action = FakeClient(), FakeClient()
    clients = iter((motion, action))
    lifecycle = []

    def fail():
        raise RuntimeError("action init failed")

    action.init = fail
    bindings = Go2Bindings(
        SimpleNamespace(initialize=lambda *_: None, release=lambda: lifecycle.append("release")),
        lambda: next(clients),
        lambda _: pytest.fail("Must initialize both clients before subscribing"),
    )
    robot = Go2Robot(bindings=bindings)
    with pytest.raises(RuntimeError, match="action init failed"):
        robot.connect()
    assert lifecycle == ["release"]
    assert not robot.state().connected
    assert robot._worker is None and robot._action_worker is None


def test_subscription_failure_cleans_up_channel():
    cleanup = []

    def fail():
        raise RuntimeError("subscription failed")

    bindings = Go2Bindings(
        SimpleNamespace(initialize=lambda *_: None, release=lambda: cleanup.append("release")),
        FakeClient,
        lambda _: SimpleNamespace(init_channel=fail, close_channel=lambda: cleanup.append("close")),
    )
    robot = Go2Robot(bindings=bindings)
    with pytest.raises(RuntimeError, match="subscription"):
        robot.connect()
    assert cleanup == ["close", "release"]
    assert not robot.state().connected


def test_lidar_height_map_and_pose_become_obstacle_clearances(clock):
    lifecycle: list[object] = []
    channel = SimpleNamespace(initialize=lambda *_: None, release=lambda: lifecycle.append("release"))
    handlers: dict[str, Callable[[Any], None]] = {}

    def state_subscriber(callback):
        return SimpleNamespace(init_channel=lambda: callback(telemetry()), close_channel=lambda: None)

    def obstacle_subscribers(on_height_map, on_pose):
        handlers.update(height_map=on_height_map, pose=on_pose)
        return [
            SimpleNamespace(init_channel=lambda: None, close_channel=lambda: lifecycle.append("obstacles"))
        ]

    bindings = Go2Bindings(channel, FakeClient, state_subscriber, obstacle_subscribers)
    robot = Go2Robot(bindings=bindings, clock=clock)
    robot.connect()
    try:
        assert robot.obstacles() is None
        grid = np.full((100, 100), -0.03, np.float32)
        grid[63:67, 48:52] = 0.4  # map x ~= 0.8..1.0, y ~= -0.1..0.1
        height_map = SimpleNamespace(
            data=grid.ravel().tolist(), width=100, height=100, resolution=0.06, origin=[-3.0, -3.0]
        )
        handlers["height_map"](height_map)
        assert robot.obstacles() is None  # no pose yet: cannot place the map relative to the body
        pose = SimpleNamespace(
            pose=SimpleNamespace(
                position=SimpleNamespace(x=0.0, y=0.0, z=0.3), orientation=SimpleNamespace(x=0, y=0, z=0, w=1)
            )
        )
        handlers["pose"](pose)
        handlers["height_map"](height_map)
        obstacles = robot.obstacles()
        assert obstacles is not None and obstacles.stamp == clock()
        assert 0.3 < obstacles.front < 0.6 and math.isinf(obstacles.back)
    finally:
        robot.close()
    assert "obstacles" in lifecycle
