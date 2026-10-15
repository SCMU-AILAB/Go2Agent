"""Brain: one think = observation -> streamed VLM decision -> validated behaviors, delivered as they complete.

The decision streams in field order (mode, new_task, plan, say, thought). Each plan step is validated and
handed to the sink the moment it is complete, so the robot starts the first step while the model is still
writing the rest. A rejected step stops the plan there; steps already handed over keep running.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from pydantic import ValidationError

from go2agent.behaviors.base import Behavior, BehaviorEnv
from go2agent.behaviors.primitives import BEHAVIORS, TARGET_BEHAVIORS
from go2agent.brain.client import VLMClient, VLMError
from go2agent.brain.prompt import ImageSizes, Observation, build_messages
from go2agent.brain.schema import BoxTarget, Decision, decision_json_schema
from go2agent.brain.stream import (
    DecisionStream,
    DoneEvent,
    ModeEvent,
    NewTaskEvent,
    PlanDoneEvent,
    SayEvent,
    StepEvent,
    StreamError,
    StreamEvent,
)
from go2agent.perception.base import Tracker

log = logging.getLogger(__name__)

SubmitMode = Literal["replace", "append"]


class Rejected(Exception):
    pass


class DecisionSink(Protocol):
    """Receives a decision while it is generated, in field order."""

    def new_task(self, text: str) -> None: ...

    def step(self, behavior: Behavior[Any], mode: SubmitMode) -> None:
        """`replace` for the first step of a replace decision, `append` for every other step."""
        ...

    def say(self, text: str) -> None: ...


class _NullSink:
    def new_task(self, text: str) -> None: ...
    def step(self, behavior: Behavior[Any], mode: SubmitMode) -> None: ...
    def say(self, text: str) -> None: ...


@dataclass(slots=True)
class ThinkResult:
    decision: Decision | None
    behaviors: list[Behavior[Any]] = field(default_factory=list)  # steps handed to the sink
    rejection: str | None = None  # part of the decision was invalid; fed back next think
    error: str | None = None  # the VLM call or stream failed
    latency_ms: float = 0.0  # request -> complete decision
    ttft_ms: float | None = None  # request -> first text
    first_step_ms: float | None = None  # request -> first step handed over (what the robot feels)
    messages: list[dict[str, Any]] = field(default_factory=list)
    raw: str = ""

    @property
    def mode(self) -> str:
        if self.decision is None or not self.behaviors:
            return "keep"
        return self.decision.mode


class Brain:
    def __init__(
        self,
        client: VLMClient,
        env: BehaviorEnv,
        tracker: Tracker | None = None,
        sizes: ImageSizes | None = None,
    ) -> None:
        self.client = client
        self.env = env
        self.tracker = tracker
        self.sizes = sizes or ImageSizes()
        self.actions = tuple(env.actions.values())
        self.schema = decision_json_schema(self.actions)

    async def think(self, obs: Observation, sink: DecisionSink | None = None) -> ThinkResult:
        messages = build_messages(obs, self.actions, self.env.camera, self.sizes)
        run = _Think(self, obs, sink or _NullSink(), ThinkResult(None, messages=messages))
        stream = self.client.stream_json(messages, self.schema)
        parser = DecisionStream()
        try:
            async for chunk in stream:
                for event in parser.feed(chunk):
                    await run.on(event)
            await run.on(parser.finish())
        except VLMError as e:
            run.res.error = str(e)
        except StreamError as e:
            run.res.error = f"invalid output stream: {e}"
        run.res.raw = parser.text
        run.res.ttft_ms = stream.ttft_ms
        run.res.latency_ms = (time.monotonic() - run.t0) * 1000
        if run.res.rejection:
            log.info("decision rejected: %s", run.res.rejection)
        return run.res

    async def resolve_step(self, i: int, step: dict[str, Any], obs: Observation) -> Behavior[Any]:
        """Validate plan step `i` against the observation; Rejected carries a reason for the model."""
        name = step.get("behavior")
        cls = BEHAVIORS.get(name) if isinstance(name, str) else None
        if cls is None:
            raise Rejected(f"第{i + 1}步：没有行为 {name!r}")
        raw_args = step.get("args")
        args: dict[str, Any] = dict(raw_args) if isinstance(raw_args, dict) else {}
        if name in TARGET_BEHAVIORS and "target" in args:
            args["target"] = await self._target(args["target"], obs, i)
        if name == "perform" and args.get("action") not in self.env.actions:
            raise Rejected(f"第{i + 1}步：动作 {args.get('action')!r} 不在可用动作中")
        try:
            return cls(cls.args_model.model_validate(args), self.env)
        except ValidationError as e:
            raise Rejected(f"第{i + 1}步 {name} 参数错误：{_short(e)}") from e

    async def _target(self, target: Any, obs: Observation, i: int) -> int:
        visible = {t.id for t in obs.tracks.items} if obs.tracks else set()
        if isinstance(target, int) and not isinstance(target, bool):
            if target not in visible:
                raise Rejected(
                    f"第{i + 1}步：目标 #{target} 不在当前画面中，可见目标：{sorted(visible) or '无'}"
                )
            return target
        try:
            bt = BoxTarget.model_validate(target)
        except ValidationError as e:
            raise Rejected(f"第{i + 1}步：target 必须是编号或 box 对象") from e
        if self.tracker is None or not obs.frames:
            raise Rejected("当前无法按 box 建立新目标，请使用编号")
        x1, y1, x2, y2 = bt.box
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise Rejected(f"第{i + 1}步：box 坐标必须是 0~1 且 x1<x2、y1<y2")
        try:
            return await self.tracker.add_box(obs.frames[-1].seq, bt.box, bt.label)
        except Exception as e:  # tracker errors are reported back to the model, never crash the loop
            raise Rejected(f"第{i + 1}步：无法跟踪该 box（{e}）") from e


class _Think:
    """State of one streamed decision."""

    def __init__(self, brain: Brain, obs: Observation, sink: DecisionSink, res: ThinkResult) -> None:
        self.brain, self.obs, self.sink, self.res = brain, obs, sink, res
        self.t0 = time.monotonic()
        self.mode: str | None = None

    async def on(self, event: StreamEvent) -> None:
        match event:
            case ModeEvent(mode=mode):
                self.mode = mode
            case NewTaskEvent(text=text) if text.strip():
                self.sink.new_task(text.strip())
            case StepEvent(index=i, step=step) if (
                self.mode in ("replace", "append") and not self.res.rejection
            ):
                try:
                    behavior = await self.brain.resolve_step(i, step, self.obs)
                except Rejected as e:
                    self.res.rejection = str(e)
                    return
                first = not self.res.behaviors
                self.sink.step(behavior, "replace" if first and self.mode == "replace" else "append")
                self.res.behaviors.append(behavior)
                if first:
                    self.res.first_step_ms = (time.monotonic() - self.t0) * 1000
            case PlanDoneEvent(count=0) if self.mode in ("replace", "append"):
                self.res.rejection = f"mode={self.mode} 但 plan 为空"
            case SayEvent(text=text) if text.strip():
                self.sink.say(text.strip())
            case DoneEvent(data=data):
                try:
                    self.res.decision = Decision.model_validate(data)
                except ValidationError as e:
                    self.res.rejection = self.res.rejection or f"输出格式错误：{_short(e)}"
            case _:
                pass


def _short(e: ValidationError) -> str:
    return "; ".join(f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors()[:3])
