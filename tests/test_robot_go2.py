"""Exercise the real driver with a fake SportClient and DDS subscription."""

import threading
import time
from types import SimpleNamespace

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

    def set_timeout(self, timeout):
        self.timeout = timeout

    def init(self):
        pass

    def __getattr__(self, method):
        if method not in {*ACTION_METHODS.values(), "move", "stop_move"}:
            raise AttributeError(method)

        def call(*args):
            self.calls.append((method, args, threading.get_ident()))
            if self.block == method:
                self.entered.set()
                assert self.release.wait(3)
            return self.status.get(method, 0)

        return call

    def count(self, method):
        return sum(name == method for name, _, _ in self.calls)


@pytest.fixture
def rig(clock):
    client, lifecycle = FakeClient(), []
    channel = SimpleNamespace(
        initialize=lambda domain, interface: lifecycle.append(("initialize", domain, interface)),
        release=lambda: lifecycle.append("release"),
    )

    def subscriber(callback):
        return SimpleNamespace(
            init_channel=lambda: callback(telemetry()),
            close_channel=lambda: lifecycle.append("close_channel"),
        )

    robot = Go2Robot(bindings=Go2Bindings(channel, lambda: client, subscriber), clock=clock)
    robot.connect()
    yield robot, client, lifecycle
    client.release.set()
    robot.close()


def test_telemetry_liveness_snapshot_and_lifecycle(rig, clock):
    robot, client, lifecycle = rig
    assert lifecycle == [("initialize", 0, "eth0")]
    assert client.calls == [] and client.timeout == 0.2
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
    robot, client, _ = rig
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
    robot, client, _ = rig
    assert set(ACTION_METHODS) == {action.name for action in GO2_ACTIONS}
    assert robot.actions() == GO2_ACTIONS
    started = time.monotonic()
    assert robot.perform(spec.name) is None
    assert time.monotonic() - started < 0.1
    assert robot.state().busy
    eventually(lambda: client.count(ACTION_METHODS[spec.name]) == 1 and not robot._action_running)
    clock.advance(spec.duration - 0.01)
    assert robot.state().busy
    clock.advance(0.02)
    assert not robot.state().busy


def test_busy_until_rpc_finishes_and_zero_does_not_cancel_action(rig, clock):
    robot, client, _ = rig
    client.block = "hello"
    robot.perform("hello")
    assert client.entered.wait(1)
    robot.velocity(Velocity())
    assert client.count("stop_move") == 0
    clock.advance(10)
    robot._on_state(telemetry())
    assert robot.state().busy
    assert "busy" in robot.perform("sit")
    assert "busy" in robot.state().details["command_refusal"]
    client.release.set()
    eventually(lambda: not robot.state().busy)
    assert client.count("sit") == 0


def test_stop_discards_pending_actions_and_follows_inflight_rpc(rig):
    robot, client, _ = rig
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
    assert client.count("hello") == 0 and not robot.state().busy
    assert [name for name, _, _ in client.calls] == ["move", "stop_move"]


def test_zero_then_nonzero_during_stop_sends_stop_then_latest_move(rig):
    robot, client, _ = rig
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
    robot, client, _ = rig
    client.block = "hello"
    robot.perform("hello")
    assert client.entered.wait(1)
    if not rpc_running:
        client.release.set()
        eventually(lambda: not robot._action_running)
    assert robot.state().busy
    with caplog.at_level("DEBUG", logger="go2agent.robot.go2"):
        for _ in range(20):
            robot.velocity(Velocity(wz=0.2))
    assert "busy" in robot.state().details["command_refusal"]
    assert len(caplog.records) <= 1
    assert all(record.levelname == "DEBUG" for record in caplog.records)
    client.release.set()
    eventually(lambda: not robot._action_running)
    clock.advance(3.1)
    robot._on_state(telemetry())
    assert not robot.state().busy
    assert robot._pending_velocity is None
    assert [name for name, _, _ in client.calls] == ["hello"]
    robot.velocity(Velocity(0.1))
    eventually(lambda: client.count("move") == 1)
    assert client.calls[-1][1] == (0.1, 0.0, 0.0)
    assert robot.state().details["command_refusal"] is None


def test_stale_velocity_and_action_are_dropped_without_replay(rig, clock):
    robot, client, _ = rig
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
    assert "not connected" in robot.perform("hello")
    robot.velocity(Velocity())
    assert not robot.state().connected
    assert "not connected" in robot.state().details["command_refusal"]
    assert robot._pending_velocity is None and robot._pending_action is None


def test_unavailable_known_action_is_refused(rig):
    robot, client, _ = rig
    client.hello = None
    assert robot.perform("hello") is not None
    assert "do not provide hello" in robot.state().details["command_refusal"]
    assert not robot.state().busy and not client.calls


def test_command_expiry_sends_stop_without_control_loop(rig, clock):
    robot, client, _ = rig
    robot.velocity(Velocity(0.2))
    eventually(lambda: client.count("move") == 1)
    clock.advance(0.6)
    eventually(lambda: client.count("stop_move") == 1)


@pytest.mark.parametrize("status", [1, None, True])
def test_sdk_errors_are_reported_and_stop_motion(rig, status):
    robot, client, _ = rig
    client.status["hello"] = status
    robot.perform("hello")
    eventually(lambda: client.count("stop_move") == 1)
    assert "SDK status" in robot.state().details["last_error"]
    assert not robot.state().busy


def test_invalid_inputs(rig):
    robot, _, _ = rig
    with pytest.raises(ValueError):
        robot.perform("unsupported")
    with pytest.raises(ValueError):
        robot.velocity(Velocity(float("nan")))
    with pytest.raises(ValueError):
        Go2Config(timeout_s=-1)
    with pytest.raises(ValueError):
        Go2Config(domain_id=True)


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
