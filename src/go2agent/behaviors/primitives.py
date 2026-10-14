"""The behavior primitives the brain can plan with."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import Field

from go2agent.behaviors.base import Args, Behavior, Status, Step, Tick
from go2agent.behaviors.geometry import bearing, distance
from go2agent.core.types import Track, Velocity

LOST_GRACE_S = 1.0  # tracker gets this long to re-acquire a target before the behavior fails
K_YAW = 1.5
K_DIST = 0.8
MAX_VX = 0.5
MAX_WZ = 0.8


def _clamp(v: float, lim: float) -> float:
    return max(-lim, min(lim, v))


class _TargetBehavior(Behavior):
    """Shared target lookup with a grace period for brief tracker dropouts."""

    def __init__(self, args, env) -> None:
        super().__init__(args, env)
        self._lost_since: float | None = None

    def _target(self, t: Tick) -> Track | Step:
        """The target track, or the Step to return when it is not visible."""
        if not t.tracker_healthy:
            return Step.fail("perception_unavailable")
        track = t.tracks.get(self.args.target) if t.tracks else None
        if track is not None:
            self._lost_since = None
            return track
        if self._lost_since is None:
            self._lost_since = t.now
        if t.now - self._lost_since > LOST_GRACE_S:
            return Step.fail("target_lost")
        return Step()  # hold still while waiting for re-acquisition

    def _turn(self, track: Track) -> tuple[float, float]:
        """(bearing, wz) toward the target."""
        b = bearing(track.box, self.env.camera)
        return b, _clamp(K_YAW * b, MAX_WZ)


class FollowArgs(Args):
    target: int = Field(description="track id of the target to follow")
    distance: float = Field(1.2, ge=0.6, le=3.0, description="distance to keep, meters")


class Follow(_TargetBehavior):
    name = "follow"
    description = "Keep following a tracked target at a distance until replaced, or until the target is lost."
    Args = FollowArgs

    def step(self, t: Tick) -> Step:
        track = self._target(t)
        if isinstance(track, Step):
            return track
        b, wz = self._turn(track)
        d = distance(track, t.frame, self.env.camera)
        vx = 0.0 if d is None else _clamp(K_DIST * (d - self.args.distance), MAX_VX)
        if vx > 0:
            vx *= max(0.0, 1 - abs(b) / 0.6)  # turn first when the target is far off-center
        if abs(vx) < 0.05:
            vx = 0.0
        return Step(velocity=Velocity(vx, 0.0, wz))


class ApproachArgs(Args):
    target: int = Field(description="track id of the target to walk up to")
    stop_distance: float = Field(0.8, ge=0.5, le=3.0, description="stop this far from the target, meters")


class Approach(_TargetBehavior):
    name = "approach"
    description = "Walk up to a tracked target and stop in front of it, facing it."
    Args = ApproachArgs

    def step(self, t: Tick) -> Step:
        track = self._target(t)
        if isinstance(track, Step):
            return track
        b, wz = self._turn(track)
        d = distance(track, t.frame, self.env.camera)
        if d is None:
            return Step.fail("distance_unknown")
        gap = d - self.args.stop_distance
        if gap <= 0.1 and abs(b) < 0.15:
            return Step.ok()
        vx = _clamp(K_DIST * gap, MAX_VX) if gap > 0.1 else 0.0
        vx *= max(0.0, 1 - abs(b) / 0.6)
        return Step(velocity=Velocity(max(vx, 0.0), 0.0, wz))


class FaceArgs(Args):
    target: int = Field(description="track id of the target to turn toward")


class Face(_TargetBehavior):
    name = "face"
    description = "Turn in place until a tracked target is centered in view."
    Args = FaceArgs

    def step(self, t: Tick) -> Step:
        track = self._target(t)
        if isinstance(track, Step):
            return track
        b, wz = self._turn(track)
        if abs(b) < 0.08:
            return Step.ok()
        return Step(velocity=Velocity(0.0, 0.0, wz))


class SearchArgs(Args):
    direction: Literal["left", "right"] = "left"
    label: str = Field("person", description="object class to look for")
    timeout: float = Field(8.0, ge=1.0, le=30.0, description="seconds before giving up")


class Search(Behavior):
    name = "search"
    description = (
        "Rotate in place until a new object of `label` comes into view (succeeds with its id), "
        "or fail with not_found after `timeout`."
    )
    Args = SearchArgs

    def __init__(self, args, env) -> None:
        super().__init__(args, env)
        self._known: set[int] | None = None

    def step(self, t: Tick) -> Step:
        if not t.tracker_healthy:
            return Step.fail("perception_unavailable")
        ids = {tr.id for tr in t.tracks.items if tr.label == self.args.label} if t.tracks else set()
        if self._known is None:
            self._known = ids
        new = sorted(ids - self._known)
        if new:
            return Step.ok(reason=f"found:{new[0]}")
        if self.elapsed(t) > self.args.timeout:
            return Step.fail("not_found")
        return Step(velocity=Velocity(0.0, 0.0, 0.5 if self.args.direction == "left" else -0.5))


class MoveArgs(Args):
    dx: float = Field(0.0, ge=-1.0, le=3.0, description="meters forward (negative = backward)")
    dy: float = Field(0.0, ge=-2.0, le=2.0, description="meters to the left (negative = right)")
    dyaw: float = Field(0.0, ge=-math.pi, le=math.pi, description="radians to turn, positive = left")


class Move(Behavior):
    """Open-loop relative move: translate, then rotate, at fixed speeds."""

    name = "move"
    description = "Move by a relative offset in the robot frame, then turn by dyaw."
    Args = MoveArgs
    SPEED = 0.3
    TURN = 0.6

    def step(self, t: Tick) -> Step:
        a = self.args
        dist = math.hypot(a.dx, a.dy)
        t_move = dist / self.SPEED
        t_turn = abs(a.dyaw) / self.TURN
        e = self.elapsed(t)
        if e < t_move:
            return Step(velocity=Velocity(a.dx / t_move, a.dy / t_move, 0.0))
        if e < t_move + t_turn:
            return Step(velocity=Velocity(0.0, 0.0, math.copysign(self.TURN, a.dyaw)))
        return Step.ok()


class PerformArgs(Args):
    action: str = Field(description="native action name from the action list")


class Perform(Behavior):
    name = "perform"
    description = "Run one native action from the action list and wait for it to finish."
    Args = PerformArgs

    def __init__(self, args, env) -> None:
        super().__init__(args, env)
        self._sent = False

    def step(self, t: Tick) -> Step:
        spec = self.env.actions.get(self.args.action)
        if spec is None:
            return Step.fail("unknown_action")
        if not self._sent:
            self._sent = True
            return Step(perform=spec.name)
        if self.elapsed(t) >= spec.duration and not t.robot.busy:
            return Step.ok()
        return Step(Status.RUNNING)


class WaitArgs(Args):
    seconds: float = Field(ge=0.0, le=60.0)


class Wait(Behavior):
    name = "wait"
    description = "Stay still for some seconds."
    Args = WaitArgs

    def step(self, t: Tick) -> Step:
        return Step.ok() if self.elapsed(t) >= self.args.seconds else Step()


class StopArgs(Args):
    pass


class Stop(Behavior):
    name = "stop"
    description = "Stop moving and clear any queued behaviors."
    Args = StopArgs

    def step(self, t: Tick) -> Step:
        return Step.ok()


BEHAVIORS: dict[str, type[Behavior]] = {
    b.name: b for b in (Follow, Approach, Face, Search, Move, Perform, Wait, Stop)
}
TARGET_BEHAVIORS = frozenset({"follow", "approach", "face"})
