import math

from conftest import make_frame

from go2agent.core.types import EventKind, Obstacles, Velocity
from go2agent.robot.sim import SimRobot
from go2agent.safety import SafetyConfig, SafetyGate


def gate(clock, **cfg):
    robot = SimRobot(clock=clock)
    robot.connect()
    return robot, SafetyGate(robot, SafetyConfig(**cfg), clock=clock)


def test_clamps_speed(clock):
    robot, g = gate(clock)
    r = g.apply(Velocity(2.0, -1.0, 3.0), make_frame(clock()))
    assert r.applied == Velocity(0.5, -0.3, 0.8)
    assert "speed_limit" in r.reasons
    assert robot.cmd == r.applied


def test_stale_frame_stops(clock):
    _, g = gate(clock)
    frame = make_frame(clock())
    clock.advance(0.6)
    r = g.apply(Velocity(0.3), frame)
    assert r.applied.is_zero and r.reasons == ("stale_frame",)
    assert g.apply(Velocity(0.3), None).applied.is_zero


def test_obstacle_blocks_forward_only_and_emits_once(clock):
    _, g = gate(clock)
    near = make_frame(clock(), depth=0.3)
    r = g.apply(Velocity(0.3, 0.0, 0.5), near)
    assert r.applied == Velocity(0.0, 0.0, 0.5)
    assert [e.kind for e in r.events] == [EventKind.OBSTACLE_CLOSE]
    r2 = g.apply(Velocity(0.3), near)
    assert r2.events == ()
    assert g.apply(Velocity(-0.2), near).applied == Velocity(-0.2)
    assert g.apply(Velocity(0.3), make_frame(clock(), depth=2.0)).applied == Velocity(0.3)


def test_estop_latches_until_release(clock):
    robot, g = gate(clock)
    f = make_frame(clock())
    g.apply(Velocity(0.3), f)
    g.estop()
    assert robot.cmd.is_zero
    assert g.apply(Velocity(0.3), f).applied.is_zero
    assert g.perform("hello") == "estop"
    g.release()
    assert g.apply(Velocity(0.3), f).applied == Velocity(0.3)


def test_dangerous_actions_refused_unless_enabled(clock):
    robot, g = gate(clock)
    assert "dangerous" in str(g.perform("front_flip"))
    assert str(g.perform("nope")).startswith("unknown")
    assert g.perform("hello") is None and robot.performed == ["hello"]
    assert all("dangerous" not in a.tags for a in g.allowed_actions())
    _, g2 = gate(clock, allow_dangerous=True)
    assert g2.perform("front_flip") is None


def test_idle_does_not_spam_zero_commands(clock):
    robot, g = gate(clock)
    calls = []
    robot.velocity = calls.append  # type: ignore[method-assign]
    f = make_frame(clock())
    g.apply(None, f)
    g.apply(Velocity(0.2), f)
    g.apply(None, f)
    g.apply(None, f)
    assert calls == [Velocity(0.2), Velocity()]


def test_watchdog_stops_stalled_motion(clock):
    robot, g = gate(clock)
    g.apply(Velocity(0.3), make_frame(clock()))
    clock.advance(0.1)
    assert not g.check_watchdog()
    clock.advance(0.2)
    assert g.check_watchdog()
    assert robot.cmd.is_zero
    assert not g.check_watchdog()


def test_perform_refused_when_robot_busy_or_disconnected(clock):
    robot, g = gate(clock)
    assert g.perform("hello") is None
    assert g.perform("heart") == "robot_busy"
    clock.advance(5.1)  # hello takes ~5 s
    assert g.perform("heart") is None
    robot.connected = False
    clock.advance(3.1)
    assert g.perform("hello") == "robot_not_connected"


def test_lidar_slows_then_stops_forward_and_allows_other_directions(clock):
    robot, g = gate(clock)
    f = make_frame(clock())
    robot.sim_obstacles = Obstacles(clock(), front=0.45, left=math.inf, back=math.inf, right=0.10)
    r = g.apply(Velocity(0.5, 0.0, 0.3), f)
    assert abs(r.applied.vx - 0.2) < 1e-9 and r.applied.wz == 0.3  # 1.0 * (0.45 - 0.25)
    assert "obstacle_slowdown" in r.reasons and "obstacle" not in r.reasons
    robot.sim_obstacles = Obstacles(clock(), front=0.2, left=math.inf, back=math.inf, right=0.10)
    r = g.apply(Velocity(0.5, -0.2, 0.0), f)
    assert r.applied == Velocity(0.0, 0.0, 0.0) and "obstacle" in r.reasons
    assert sorted(e.data["direction"] for e in r.events) == ["front", "right"]
    assert g.apply(Velocity(0.5, -0.2, 0.0), f).events == ()  # reported once while blocked
    assert g.apply(Velocity(-0.2, 0.2, 0.0), f).applied == Velocity(-0.2, 0.2, 0.0)


def test_stale_obstacle_sensing_blocks_translation_not_turning(clock):
    robot, g = gate(clock)
    robot.sim_obstacles = Obstacles(clock(), math.inf, math.inf, math.inf, math.inf)
    clock.advance(0.6)
    r = g.apply(Velocity(0.3, 0.1, 0.4), make_frame(clock()))
    assert r.applied == Velocity(0.0, 0.0, 0.4) and "obstacle_sensor_stale" in r.reasons
