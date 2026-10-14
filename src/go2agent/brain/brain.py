"""Brain: one think = observation -> VLM -> validated behaviors."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from go2agent.behaviors.base import Behavior, BehaviorEnv
from go2agent.behaviors.primitives import BEHAVIORS, TARGET_BEHAVIORS
from go2agent.brain.client import VLMClient, VLMError
from go2agent.brain.prompt import ImageSizes, Observation, build_messages
from go2agent.brain.schema import BoxTarget, Decision, decision_json_schema
from go2agent.perception.base import Tracker

log = logging.getLogger(__name__)


class Rejected(Exception):
    pass


@dataclass(slots=True)
class ThinkResult:
    decision: Decision | None
    behaviors: list[Behavior] = field(default_factory=list)
    rejection: str | None = None  # the decision was invalid; fed back next think
    error: str | None = None  # the VLM call itself failed
    latency_ms: float = 0.0
    messages: list[dict[str, Any]] = field(default_factory=list)
    raw: str = ""

    @property
    def mode(self) -> str:
        if self.decision is None or self.rejection or not self.behaviors:
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

    async def think(self, obs: Observation) -> ThinkResult:
        messages = build_messages(obs, self.actions, self.env.camera, self.sizes)
        try:
            reply = await self.client.complete_json(messages, self.schema)
        except VLMError as e:
            return ThinkResult(None, error=str(e), messages=messages)
        res = ThinkResult(None, latency_ms=reply.latency_ms, messages=messages, raw=reply.raw)
        try:
            res.decision = decision = Decision.model_validate(reply.data)
            res.behaviors = await self.resolve(decision, obs)
        except ValidationError as e:
            res.rejection = f"输出格式错误：{_short(e)}"
        except Rejected as e:
            res.rejection = str(e)
        if res.rejection:
            res.behaviors = []
            log.info("decision rejected: %s", res.rejection)
        return res

    async def resolve(self, d: Decision, obs: Observation) -> list[Behavior]:
        if d.mode == "keep":
            return []
        if not d.plan:
            raise Rejected(f"mode={d.mode} 但 plan 为空")
        visible = {t.id for t in obs.tracks.items} if obs.tracks else set()
        out: list[Behavior] = []
        for i, step in enumerate(d.plan):
            cls = BEHAVIORS.get(step.behavior)
            if cls is None:
                raise Rejected(f"第{i + 1}步：没有行为 {step.behavior!r}")
            args = dict(step.args)
            if step.behavior in TARGET_BEHAVIORS and "target" in args:
                args["target"] = await self._target(args["target"], visible, obs, i)
            if step.behavior == "perform" and args.get("action") not in self.env.actions:
                raise Rejected(f"第{i + 1}步：动作 {args.get('action')!r} 不在可用动作中")
            try:
                out.append(cls(cls.args_model.model_validate(args), self.env))
            except ValidationError as e:
                raise Rejected(f"第{i + 1}步 {step.behavior} 参数错误：{_short(e)}") from e
        return out

    async def _target(self, target: Any, visible: set[int], obs: Observation, i: int) -> int:
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


def _short(e: ValidationError) -> str:
    return "; ".join(f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors()[:3])
