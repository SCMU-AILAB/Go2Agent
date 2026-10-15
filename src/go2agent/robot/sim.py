"""Simulated robot: integrates velocity into a planar pose and pretends to run native actions."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable

from go2agent.core.types import ActionSpec, Obstacles, RobotState, Velocity
from go2agent.robot.base import Robot
from go2agent.robot.go2_actions import GO2_ACTIONS


class SimRobot(Robot):
    def __init__(
        self, actions: tuple[ActionSpec, ...] = GO2_ACTIONS, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self._actions = actions
        self._clock = clock
        self._lock = threading.Lock()
        self.connected = False
        self.x = self.y = self.yaw = 0.0
        self.cmd = Velocity()
        self.performed: list[str] = []
        self.sim_obstacles: Obstacles | None = None  # set by tests / scenarios; None = no sensing
        self._cmd_stamp = 0.0
        self._last_step = 0.0
        self._busy_until = 0.0

    def connect(self) -> None:
        self.connected = True
        self._last_step = self._clock()

    def close(self) -> None:
        self.stop()
        self.connected = False

    def velocity(self, cmd: Velocity) -> None:
        with self._lock:
            self._step()
            self.cmd = cmd
            self._cmd_stamp = self._clock()

    def perform(self, action: str) -> str | None:
        spec = next((a for a in self._actions if a.name == action), None)
        if spec is None:
            raise ValueError(f"unknown action {action!r}")
        with self._lock:
            self._step()
            self.cmd = Velocity()
            self.performed.append(action)
            self._busy_until = self._clock() + spec.duration
        return None

    def stop(self) -> None:
        with self._lock:
            self._step()
            self.cmd = Velocity()
            self._busy_until = 0.0

    def state(self) -> RobotState:
        with self._lock:
            self._step()
            now = self._clock()
            return RobotState(
                stamp=now,
                connected=self.connected,
                busy=now < self._busy_until,
                details={
                    "x": self.x,
                    "y": self.y,
                    "yaw": self.yaw,
                    "vx": self.cmd.vx,
                    "vy": self.cmd.vy,
                    "wz": self.cmd.wz,
                },
            )

    def actions(self) -> tuple[ActionSpec, ...]:
        return self._actions

    def obstacles(self) -> Obstacles | None:
        return self.sim_obstacles

    def _step(self) -> None:
        now = self._clock()
        dt = max(0.0, now - self._last_step)
        self._last_step = now
        # Like Go2, a velocity command expires if not refreshed.
        if now - self._cmd_stamp > 0.5:
            self.cmd = Velocity()
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        self.x += (self.cmd.vx * c - self.cmd.vy * s) * dt
        self.y += (self.cmd.vx * s + self.cmd.vy * c) * dt
        self.yaw += self.cmd.wz * dt
