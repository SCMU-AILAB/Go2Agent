"""Runtime: wires sensors, tracker, executor, safety and brain; runs the control loop and the brain loop."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from go2agent.behaviors.base import Tick
from go2agent.brain.brain import Brain, ThinkResult
from go2agent.brain.prompt import Observation
from go2agent.core.types import TRIGGER_EVENTS, Event, EventKind, Frame, Tracks
from go2agent.executor import Executor
from go2agent.perception.base import Tracker
from go2agent.robot.base import Robot
from go2agent.safety import SafetyGate
from go2agent.sensors.base import Camera
from go2agent.timeline import Timeline

log = logging.getLogger(__name__)

ESTOP_WORDS = re.compile(r"(停下|停止|别动|不许动|站住|^停|stop)", re.IGNORECASE)
# Triggers that are worth a think even without a task.
_TASKLESS_TRIGGERS = frozenset({EventKind.TASK, EventKind.USER_SPEECH})


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    control_hz: float = 20.0
    periodic_s: float = 4.0  # re-check interval while a behavior runs
    idle_watch_s: float = 1.5  # look interval while idle with a task and someone in view
    keyframes: int = 3
    keyframe_spacing_s: float = 0.7
    track_lost_s: float = 1.5  # id unseen this long -> TRACK_LOST
    brain_down_s: float = 10.0  # brain failing this long -> stop and say so
    recent_events: int = 8
    history: int = 4


Speaker = Callable[[str], Awaitable[None] | None]


class Runtime:
    def __init__(
        self,
        robot: Robot,
        camera: Camera,
        tracker: Tracker,
        brain: Brain,
        safety: SafetyGate,
        timeline: Timeline,
        config: RuntimeConfig | None = None,
        speaker: Speaker | None = None,
        task: str = "",
    ) -> None:
        self.robot = robot
        self.camera = camera
        self.tracker = tracker
        self.brain = brain
        self.safety = safety
        self.timeline = timeline
        self.config = config or RuntimeConfig()
        self.speaker = speaker
        self.task = task
        self.executor = Executor(safety.perform)

        self._frames: deque[Frame] = deque(maxlen=int(self.config.control_hz * 4))
        self._last_frame_seq = -1
        self._seen: dict[int, tuple[str, float]] = {}  # track id -> (label, last seen)
        self._tracks_seq = -1
        self._tracker_healthy: bool | None = None
        self._events: deque[Event] = deque(maxlen=64)
        self._pending: list[Event] = []
        self._wake = asyncio.Event()
        self._last_think = 0.0
        self._behavior_since: float | None = None
        self._brain_failing_since: float | None = None
        self._brain_down_announced = False
        self._rejection: str | None = None
        self._history: deque[str] = deque(maxlen=self.config.history)
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()
        self._tasks: list[asyncio.Task] = []
        self._background: set[asyncio.Future] = set()
        self.thinking = False
        self.last_decision: dict[str, Any] | None = None

    # --- public control ---------------------------------------------------------------------------

    def set_task(self, text: str, source: str = "app") -> None:
        """Explicit new task (from the app). Releases a latched e-stop."""
        self.task = text.strip()
        if not self.task:
            self.executor.clear()
        self.safety.release()
        self.emit(Event(EventKind.TASK, time.monotonic(), {"text": self.task, "source": source}))

    def hear(self, text: str) -> None:
        """A transcribed utterance from the microphone."""
        text = text.strip()
        if not text:
            return
        if ESTOP_WORDS.search(text):
            self.estop("voice")
        self.emit(Event(EventKind.USER_SPEECH, time.monotonic(), {"text": text}))

    def estop(self, source: str) -> None:
        self.executor.clear()
        self.safety.estop()
        self.emit(Event(EventKind.ESTOP, time.monotonic(), {"source": source}))

    def emit(self, e: Event) -> None:
        self._events.append(e)
        self._publish(self.timeline.record(e.kind.value, e.data, e.stamp))
        # After an e-stop only speech reaches the brain (it may carry a new task).
        if e.kind in TRIGGER_EVENTS and (not self.safety.estopped or e.kind is EventKind.USER_SPEECH):
            self._pending.append(e)
            self._wake.set()

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=256)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[dict[str, Any]]) -> None:
        self._subscribers.discard(q)

    def latest_frame(self) -> Frame | None:
        return self._frames[-1] if self._frames else None

    def status(self) -> dict[str, Any]:
        st = self.robot.state()
        tracks = self.tracker.latest()
        return {
            "task": self.task,
            "estopped": self.safety.estopped,
            "thinking": self.thinking,
            "executor": self.executor.snapshot(),
            "robot": {"connected": st.connected, "busy": st.busy, **st.details},
            "tracker_healthy": self.tracker.healthy,
            "tracks": [{"id": t.id, "label": t.label, "box": t.box} for t in tracks.items] if tracks else [],
            "last_decision": self.last_decision,
            "run_dir": str(self.timeline.dir) if self.timeline.dir else None,
        }

    # --- lifecycle --------------------------------------------------------------------------------

    async def start(self) -> None:
        self.robot.connect()
        self.camera.start()
        await self.tracker.start()
        self.safety.start_watchdog()
        self._tasks = [
            asyncio.create_task(self._control_loop(), name="control"),
            asyncio.create_task(self._brain_loop(), name="brain"),
        ]
        if self.task:
            self.set_task(self.task, source="startup")

    async def stop(self) -> None:
        for t in self._tasks:
            t.cancel()
        for t in self._tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await t
        self.safety.stop_watchdog()
        self.robot.stop()
        await self.tracker.close()
        self.camera.close()
        self.robot.close()
        await self.brain.client.close()
        self.timeline.close()

    # --- control loop -----------------------------------------------------------------------------

    async def _control_loop(self) -> None:
        period = 1.0 / self.config.control_hz
        next_t = time.monotonic()
        while True:
            try:
                self.control_step(time.monotonic())
            except Exception:
                log.exception("control step failed; stopping robot")
                self.executor.clear()
                self.robot.stop()
            next_t += period
            delay = next_t - time.monotonic()
            if delay < 0:
                next_t = time.monotonic()
                delay = 0
            await asyncio.sleep(delay)

    def control_step(self, now: float) -> None:
        frame = self.camera.latest()
        if frame is not None and frame.seq != self._last_frame_seq:
            self._last_frame_seq = frame.seq
            self._frames.append(frame)
            self.tracker.submit(frame)
        tracks = self.tracker.latest()
        healthy = self.tracker.healthy
        if healthy != self._tracker_healthy:
            self._tracker_healthy = healthy
            self.emit(Event(EventKind.SERVICE, now, {"service": "perception", "ok": healthy}))
        self._update_tracks(tracks, now)

        out = self.executor.tick(Tick(now, frame, tracks, healthy, self.robot.state()))
        result = self.safety.apply(out.velocity, frame)
        for e in (*out.events, *result.events):
            self.emit(e)
        if self.executor.current is None:
            self._behavior_since = None
        elif self._behavior_since is None:
            self._behavior_since = now

    def _update_tracks(self, tracks: Tracks | None, now: float) -> None:
        if tracks is None or tracks.frame_seq == self._tracks_seq:
            return
        self._tracks_seq = tracks.frame_seq
        for t in tracks.items:
            if t.id not in self._seen:
                self.emit(Event(EventKind.TRACK_APPEARED, now, {"id": t.id, "label": t.label}))
            self._seen[t.id] = (t.label, now)
        for tid, (label, last) in list(self._seen.items()):
            if now - last > self.config.track_lost_s:
                del self._seen[tid]
                self.emit(Event(EventKind.TRACK_LOST, now, {"id": tid, "label": label}))

    # --- brain loop -------------------------------------------------------------------------------

    async def _brain_loop(self) -> None:
        while True:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._wake.wait(), timeout=0.25)
            self._wake.clear()
            now = time.monotonic()
            triggers, self._pending = self._pending, []
            if not triggers and self._periodic_due(now):
                triggers = [Event(EventKind.PERIODIC, now)]
            if not triggers or not self._should_think(triggers):
                continue
            try:
                await self.think(triggers)
            except Exception:
                log.exception("think failed")

    def _periodic_due(self, now: float) -> bool:
        """Re-check while a behavior runs; while idle, keep watching as long as someone is in view
        (a person already in view who starts waving produces no event)."""
        if not self.task or self.safety.estopped:
            return False
        if self.executor.current is not None or self.executor.queue:
            interval = self.config.periodic_s
        elif self._seen:
            interval = self.config.idle_watch_s
        else:
            return False
        return now - self._last_think >= interval

    def _should_think(self, triggers: list[Event]) -> bool:
        if self.task:
            return True
        return any(e.kind in _TASKLESS_TRIGGERS for e in triggers)

    async def think(self, triggers: list[Event]) -> ThinkResult:
        obs = self.observe(triggers, time.monotonic())
        self.thinking = True
        try:
            res = await self.brain.think(obs)
        finally:
            self.thinking = False
            self._last_think = time.monotonic()
        await self._apply(res, triggers)
        return res

    def observe(self, triggers: list[Event], now: float) -> Observation:
        tracks = self.tracker.latest()
        trigger_ids = {id(e) for e in triggers}
        recent = [
            e
            for e in self._events
            if id(e) not in trigger_ids and e.kind not in (EventKind.BRAIN_DECISION, EventKind.SERVICE)
        ]
        ex = self.executor.snapshot()
        ex["running_s"] = None if self._behavior_since is None else now - self._behavior_since
        return Observation(
            now=now,
            task=self.task,
            triggers=tuple(triggers),
            recent_events=tuple(recent[-self.config.recent_events :]),
            executor=ex,
            frames=self._keyframes(tracks),
            tracks=tracks,
            robot=self.robot.state(),
            history=tuple(self._history),
            rejection=self._rejection,
        )

    def _keyframes(self, tracks: Tracks | None) -> tuple[Frame, ...]:
        """Frames spaced keyframe_spacing_s apart, ending at the frame the tracks belong to."""
        if not self._frames:
            return ()
        frames = list(self._frames)
        newest = next((f for f in reversed(frames) if tracks and f.seq == tracks.frame_seq), frames[-1])
        picked = [newest]
        for f in reversed(frames):
            if len(picked) == self.config.keyframes:
                break
            if picked[-1].stamp - f.stamp >= self.config.keyframe_spacing_s:
                picked.append(f)
        return tuple(reversed(picked))

    async def _apply(self, res: ThinkResult, triggers: list[Event]) -> None:
        now = time.monotonic()
        think_id = self.timeline.save_think(res.messages, {"raw": res.raw, "error": res.error})
        if res.error is not None:
            self._on_brain_error(res.error, now)
            return
        self._brain_failing_since = None
        self._brain_down_announced = False
        self._rejection = res.rejection
        d = res.decision
        if d is None:
            return
        if self.safety.estopped and any(e.kind is EventKind.USER_SPEECH for e in triggers) and d.task:
            self.safety.release()  # a spoken new task after an e-stop resumes operation
        if d.task:
            self.task = d.task
        if not self.safety.estopped:
            self.executor.submit(res.behaviors, res.mode)  # type: ignore[arg-type]
        plan = [repr(b) for b in res.behaviors]
        self.last_decision = {
            "thought": d.thought,
            "say": d.say,
            "task": d.task,
            "mode": res.mode,
            "plan": plan,
            "rejection": res.rejection,
            "latency_ms": round(res.latency_ms),
            "think_id": think_id,
        }
        self._history.append(f"{d.thought} -> {res.mode} {plan}" + (f" 说「{d.say}」" if d.say else ""))
        self.emit(Event(EventKind.BRAIN_DECISION, now, self.last_decision))
        if d.say:
            self._spawn(self.say(d.say))

    def _on_brain_error(self, error: str, now: float) -> None:
        log.warning("brain call failed: %s", error)
        self.emit(Event(EventKind.BRAIN_ERROR, now, {"reason": error}))
        if self._brain_failing_since is None:
            self._brain_failing_since = now
        if not self._brain_down_announced and now - self._brain_failing_since >= self.config.brain_down_s:
            self._brain_down_announced = True
            self.executor.clear()
            self._spawn(self.say("我暂时没法思考，先停下来。"))

    async def say(self, text: str) -> None:
        self.emit(Event(EventKind.SAY, time.monotonic(), {"text": text}))
        if self.speaker is None:
            return
        try:
            r = self.speaker(text)
            if r is not None:
                await r
        except Exception:
            log.exception("speaker failed")

    def _spawn(self, coro: Awaitable[None]) -> None:
        """Fire-and-forget so speaking never blocks the brain loop."""
        task = asyncio.ensure_future(coro)
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    def _publish(self, rec: dict[str, Any]) -> None:
        for q in list(self._subscribers):
            if q.full():
                with contextlib.suppress(asyncio.QueueEmpty):
                    q.get_nowait()
            q.put_nowait(rec)
