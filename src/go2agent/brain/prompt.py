"""Builds the VLM messages for one think."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from go2agent.behaviors.geometry import CameraModel, bearing, distance
from go2agent.behaviors.primitives import BEHAVIORS
from go2agent.brain.marks import data_url, draw_marks, jpeg_bytes, resize_long_side
from go2agent.core.types import ActionSpec, Event, Frame, RobotState, Tracks

SYSTEM_PROMPT = """你是一只四足机器狗的大脑。你通过相机持续观察周围，根据「任务」决定接下来做什么。

你不直接控制腿：你输出「行为」，由本地控制器高频闭环执行（跟随、走近、转向等会自己持续修正）。
你只在有新情况时被唤醒：新任务、有人说话、行为完成或失败、目标出现或丢失、障碍、定期复查。

图像说明：多张图按时间从旧到新排列，最后一张是当前画面。当前画面中被跟踪的目标画了带编号的框，
编号就是 target 的取值。需要的目标没有编号框时，target 可以给 {"box":[x1,y1,x2,y2],"label":"..."}，
坐标是当前画面上 0~1 的归一化值。

输出规则：
- mode=keep：当前行为仍然符合任务，就保持不变（plan 留空）。这是最常见的选择，不要无故打断正在进行的行为。
- mode=replace：情况变了或当前行为不对，用 plan 替换当前行为和队列。
- mode=append：把 plan 接在当前行为之后。
- plan 可以是多步，例如先 approach 再 perform。最多 4 步。
- 不要重复刚做过的表演动作，除非任务或对方明确要求。
- say：只有真正需要说话时才填，一两句、口语化、和用户同语言；否则留空字符串。say 和行为同时进行。
- thought：一句话说明你看到了什么、为什么这样决定。
- task：只有当用户说的话给出了新任务或修改了任务时，填写更新后的完整任务（会替换当前任务）；
  闲聊、打招呼、回答问题时留空字符串。
- 没有任务、也没有需要回应的人时，保持安静（keep 或 stop）。
- 「上次输出被拒绝」说明你上次的输出有错，按原因修正。"""


def _behavior_doc() -> str:
    lines = []
    for b in BEHAVIORS.values():
        props = b.Args.model_json_schema().get("properties", {})
        args = ", ".join(
            f"{k}"
            + (f"={v['default']}" if "default" in v else "")
            + (f" ({v['description']})" if "description" in v else "")
            for k, v in props.items()
        )
        lines.append(f"- {b.name}({args}): {b.description}")
    return "\n".join(lines)


def system_message(actions: tuple[ActionSpec, ...]) -> str:
    acts = "\n".join(f"- {a.name}: {a.description}" for a in actions)
    return f"{SYSTEM_PROMPT}\n\n可用行为：\n{_behavior_doc()}\n\nperform 可用的动作：\n{acts}"


@dataclass(frozen=True, slots=True)
class Observation:
    """Everything the brain sees in one think."""

    now: float
    task: str
    triggers: tuple[Event, ...]
    recent_events: tuple[Event, ...]
    executor: dict[str, Any]
    frames: tuple[Frame, ...]  # oldest -> newest; newest gets marks
    tracks: Tracks | None
    robot: RobotState
    history: tuple[str, ...] = ()  # short summaries of previous decisions, oldest first
    rejection: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def _ago(now: float, stamp: float) -> str:
    return f"{max(0.0, now - stamp):.1f}s前"


def _event_line(now: float, e: Event) -> str:
    data = ", ".join(f"{k}={v}" for k, v in e.data.items() if k != "args" or v)
    return f"{_ago(now, e.stamp)} {e.kind.value}" + (f" ({data})" if data else "")


def _targets(obs: Observation, cam: CameraModel) -> str:
    if obs.tracks is None or not obs.tracks.items:
        return "（当前画面没有被跟踪的目标）"
    newest = obs.frames[-1] if obs.frames else None
    out = []
    for t in obs.tracks.items:
        deg = math.degrees(bearing(t.box, cam))
        side = "正前方" if abs(deg) < 8 else (f"左{deg:.0f}°" if deg > 0 else f"右{-deg:.0f}°")
        d = distance(t, newest, cam)
        dist = f"约{d:.1f}m" if d is not None else "距离未知"
        out.append(f"#{t.id} {t.label} {side} {dist}")
    return "；".join(out)


def observation_text(obs: Observation, cam: CameraModel) -> str:
    ex = obs.executor
    current = ex.get("current") or "无（空闲）"
    if ex.get("current") and ex.get("running_s") is not None:
        current += f"，已运行 {ex['running_s']:.1f}s"
    parts = [
        f"任务：{obs.task or '（无）'}",
        "唤醒原因：" + ("；".join(_event_line(obs.now, e) for e in obs.triggers) or "定期复查"),
        f"当前行为：{current}" + (f"；队列：{', '.join(ex['queue'])}" if ex.get("queue") else ""),
        f"可见目标：{_targets(obs, cam)}",
        f"机器人：{'执行原生动作中' if obs.robot.busy else '就绪'}"
        + ("" if obs.robot.connected else "（未连接）"),
    ]
    if obs.recent_events:
        parts.append("最近事件：\n" + "\n".join("  " + _event_line(obs.now, e) for e in obs.recent_events))
    if obs.history:
        parts.append("你之前的决策：\n" + "\n".join("  " + h for h in obs.history))
    if obs.rejection:
        parts.append(f"上次输出被拒绝：{obs.rejection}")
    return "\n".join(parts)


def build_messages(
    obs: Observation, actions: tuple[ActionSpec, ...], cam: CameraModel, image_long_side: int = 512
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for i, f in enumerate(obs.frames):
        newest = i == len(obs.frames) - 1
        rgb = draw_marks(f.rgb, obs.tracks) if newest else f.rgb
        rgb = resize_long_side(rgb, image_long_side)
        content.append({"type": "image_url", "image_url": {"url": data_url(jpeg_bytes(rgb))}})
    content.append({"type": "text", "text": observation_text(obs, cam)})
    return [
        {"role": "system", "content": system_message(actions)},
        {"role": "user", "content": content},
    ]
