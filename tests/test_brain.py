import json
from typing import Any

import httpx
import pytest
from conftest import make_frame

from go2agent.behaviors.base import BehaviorEnv
from go2agent.behaviors.geometry import CameraModel
from go2agent.brain.brain import Brain
from go2agent.brain.client import VLMClient, VLMConfig
from go2agent.brain.prompt import Observation, observation_text
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


def vlm(reply: dict | str, status=200, seen=None):
    def handler(request: httpx.Request):
        if seen is not None:
            seen.append(json.loads(request.content))
        content = reply if isinstance(reply, str) else json.dumps(reply)
        return httpx.Response(status, json={"choices": [{"message": {"content": content}}], "usage": {}})

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


def decision(mode="replace", plan=(), say=""):
    return {"thought": "t", "say": say, "new_task": "", "mode": mode, "plan": list(plan)}


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
    assert "invalid json" in str((await Brain(vlm("not json"), ENV).think(obs())).error)


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
