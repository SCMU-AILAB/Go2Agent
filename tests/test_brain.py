import base64
import json
from typing import Any

import cv2
import httpx
import numpy as np
import pytest
from conftest import compact, make_frame, sse_response

from go2agent.behaviors.base import Behavior, BehaviorEnv
from go2agent.behaviors.geometry import CameraModel
from go2agent.brain.brain import Brain, SubmitMode
from go2agent.brain.client import VLMClient, VLMConfig
from go2agent.brain.prompt import Observation, build_messages, observation_text
from go2agent.brain.schema import decision_json_schema
from go2agent.core.types import Box, Event, EventKind, Frame, RobotState, Track, Tracks
from go2agent.perception.base import Tracker
from go2agent.robot.go2_actions import GO2_ACTIONS

SAFE = tuple(a for a in GO2_ACTIONS if "dangerous" not in a.tags)
ENV = BehaviorEnv(CameraModel(), {a.name: a for a in SAFE})


class FakeTracker(Tracker):
    def __init__(self) -> None:
        self.calls: list[tuple[int, Box, str]] = []

    async def start(self) -> None: ...
    async def close(self) -> None: ...
    def submit(self, frame: Frame) -> None: ...

    def latest(self) -> Tracks | None:
        return None

    @property
    def healthy(self) -> bool:
        return True

    async def add_box(self, frame_seq: int, box: Box, label: str) -> int:
        self.calls.append((frame_seq, box, label))
        return 42


def vlm(reply: dict[str, Any] | str, status: int = 200, seen: list[Any] | None = None) -> VLMClient:
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(json.loads(request.content))
        return sse_response(reply if isinstance(reply, str) else compact(reply), status)

    return VLMClient(VLMConfig(base_url="http://vlm/v1"), transport=httpx.MockTransport(handler))


def obs(**kw):
    frame = make_frame(10.0, seq=5)
    base: dict[str, Any] = dict(
        now=10.0,
        task="跟着穿黑衣服的人",
        triggers=(Event(EventKind.TASK, 10.0, {"text": "跟着穿黑衣服的人"}),),
        recent_events=(),
        executor={"current": None, "queue": []},
        frames=(frame,),
        tracks=Tracks(5, 10.0, (Track(3, "person", (0.4, 0.2, 0.6, 0.9), 0.9),)),
        robot=RobotState(10.0, True),
    )
    base.update(kw)
    return Observation(**base)


def decision(mode="replace", plan=(), say="", new_task=""):
    return {"mode": mode, "new_task": new_task, "plan": list(plan), "say": say, "thought": "t"}


async def test_valid_plan_becomes_behaviors_and_request_is_constrained():
    seen = []
    plan = [
        {"behavior": "approach", "args": {"target": 3}},
        {"behavior": "perform", "args": {"action": "hello"}},
    ]
    brain = Brain(vlm(decision(plan=plan, say="你好"), seen=seen), ENV)
    res = await brain.think(obs())
    assert res.rejection is None and res.mode == "replace"
    assert [repr(b) for b in res.behaviors] == [
        "approach(target=3, stop_distance=0.8)",
        "perform(action='hello')",
    ]
    req = seen[0]
    assert req["stream"] is True
    assert req["response_format"]["type"] == "json_schema"
    assert req["messages"][1]["content"][0]["image_url"]["url"].startswith("data:image/jpeg;base64,")


async def test_keep_yields_no_behaviors():
    res = await Brain(vlm(decision(mode="keep")), ENV).think(obs())
    assert res.mode == "keep" and res.behaviors == [] and res.rejection is None


@pytest.mark.parametrize(
    ("plan", "fragment"),
    [
        ([], "plan 为空"),
        ([{"behavior": "follow", "args": {"target": 9}}], "#9 不在当前画面"),
        ([{"behavior": "fly", "args": {}}], "没有行为"),
        ([{"behavior": "perform", "args": {"action": "front_flip"}}], "不在可用动作"),
        ([{"behavior": "follow", "args": {"target": 3, "distance": 10}}], "参数错误"),
    ],
)
async def test_invalid_plans_are_rejected_as_keep(plan, fragment):
    res = await Brain(vlm(decision(plan=plan)), ENV).think(obs())
    assert res.mode == "keep" and res.behaviors == []
    assert fragment in res.rejection


async def test_box_target_creates_track():
    tracker = FakeTracker()
    plan = [{"behavior": "follow", "args": {"target": {"box": [0.1, 0.1, 0.3, 0.8], "label": "person"}}}]
    res = await Brain(vlm(decision(plan=plan)), ENV, tracker=tracker).think(obs())
    assert res.behaviors[0].args.model_dump()["target"] == 42
    assert tracker.calls == [(5, (0.1, 0.1, 0.3, 0.8), "person")]


async def test_vlm_failures_are_errors_not_exceptions():
    assert "http 500" in str((await Brain(vlm({}, status=500), ENV).think(obs())).error)
    assert "invalid output stream" in str((await Brain(vlm("not json"), ENV).think(obs())).error)


def test_schema_lists_only_allowed_actions():
    schema = decision_json_schema(SAFE)
    perform = next(
        s
        for s in schema["properties"]["plan"]["items"]["anyOf"]
        if s["properties"]["behavior"]["const"] == "perform"
    )
    enum = perform["properties"]["args"]["properties"]["action"]["enum"]
    assert "hello" in enum and "front_flip" not in enum


def test_observation_text_mentions_targets_and_rejection():
    text = observation_text(obs(rejection="坏输出"), CameraModel())
    assert "#3 person 正前方" in text and "上次输出被拒绝：坏输出" in text


def test_history_frames_are_smaller_than_the_newest():
    frames = (make_frame(9.0, seq=3, w=640, h=480), make_frame(10.0, seq=5, w=640, h=480))
    msgs = build_messages(obs(frames=frames), SAFE, CameraModel())
    widths = []
    for part in msgs[1]["content"][:-1]:
        raw = base64.b64decode(part["image_url"]["url"].split(",", 1)[1])
        image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        assert image is not None
        widths.append(image.shape[1])
    assert widths == [256, 512]


class RecordingSink:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def new_task(self, text: str) -> None:
        self.calls.append(("new_task", text))

    def step(self, behavior: Behavior[Any], mode: SubmitMode) -> None:
        self.calls.append((mode, repr(behavior)))

    def say(self, text: str) -> None:
        self.calls.append(("say", text))


async def test_steps_task_and_speech_are_delivered_in_order_as_they_complete():
    sink = RecordingSink()
    plan = [
        {"behavior": "approach", "args": {"target": 3}},
        {"behavior": "perform", "args": {"action": "hello"}},
    ]
    res = await Brain(vlm(decision(plan=plan, say="你好", new_task="和人打招呼")), ENV).think(obs(), sink)
    assert sink.calls == [
        ("new_task", "和人打招呼"),
        ("replace", "approach(target=3, stop_distance=0.8)"),
        ("append", "perform(action='hello')"),
        ("say", "你好"),
    ]
    assert res.first_step_ms is not None and res.ttft_ms is not None
    assert res.first_step_ms <= res.latency_ms


async def test_append_decision_appends_every_step():
    sink = RecordingSink()
    await Brain(vlm(decision(mode="append", plan=[{"behavior": "wait", "args": {"seconds": 1}}])), ENV).think(
        obs(), sink
    )
    assert sink.calls == [("append", "wait(seconds=1.0)")]


async def test_rejected_step_stops_the_plan_but_keeps_earlier_steps():
    sink = RecordingSink()
    plan = [
        {"behavior": "approach", "args": {"target": 3}},
        {"behavior": "follow", "args": {"target": 9}},
        {"behavior": "perform", "args": {"action": "hello"}},
    ]
    res = await Brain(vlm(decision(plan=plan)), ENV).think(obs(), sink)
    assert [c[0] for c in sink.calls] == ["replace"]
    assert res.mode == "replace" and len(res.behaviors) == 1
    assert "#9 不在当前画面" in str(res.rejection)


async def test_broken_stream_keeps_steps_already_delivered():
    sink = RecordingSink()
    full = compact(
        decision(
            plan=[{"behavior": "face", "args": {"target": 3}}, {"behavior": "wait", "args": {"seconds": 1}}]
        )
    )
    cut = full[: full.index('{"behavior":"wait"') + 5]
    res = await Brain(vlm(cut), ENV).think(obs(), sink)
    assert sink.calls == [("replace", "face(target=3)")]
    assert res.error is not None and len(res.behaviors) == 1
