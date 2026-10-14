import asyncio
import json

import httpx
from conftest import make_frame

from go2agent.behaviors.base import BehaviorEnv
from go2agent.behaviors.geometry import CameraModel
from go2agent.brain.brain import Brain
from go2agent.brain.client import VLMClient, VLMConfig
from go2agent.core.types import EventKind, Track, Tracks
from go2agent.robot.sim import SimRobot
from go2agent.runtime import Runtime, RuntimeConfig
from go2agent.safety import SafetyGate
from go2agent.timeline import Timeline, load_think


class StepCamera:
    def __init__(self, clock):
        self.clock, self.seq = clock, 0

    def start(self): ...
    def close(self): ...

    def latest(self):
        self.seq += 1
        return make_frame(self.clock(), seq=self.seq, depth=2.5)


PERSON = Track(3, "person", (0.4, 0.2, 0.6, 0.9), 0.9)


class StaticTracker:
    def __init__(self, cam, items=(PERSON,)):
        self.cam, self.items, self.healthy = cam, items, True

    async def start(self): ...
    async def close(self): ...
    def submit(self, frame): ...

    def latest(self):
        return Tracks(self.cam.seq, self.cam.clock(), tuple(self.items))


def make_runtime(clock, replies, tmp_path=None, **cfg):
    seen = []

    def handler(req):
        seen.append(json.loads(req.content))
        r = replies.pop(0) if replies else {"thought": "", "say": "", "task": "", "mode": "keep", "plan": []}
        if isinstance(r, int):
            return httpx.Response(r)
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(r)}}]})

    robot = SimRobot(clock=clock)
    cam = StepCamera(clock)
    tracker = StaticTracker(cam)
    gate = SafetyGate(robot, clock=clock)
    env = BehaviorEnv(CameraModel(), {a.name: a for a in gate.allowed_actions()})
    brain = Brain(VLMClient(VLMConfig(base_url="http://vlm/v1"), httpx.MockTransport(handler)), env, tracker)
    said = []
    rt = Runtime(
        robot, cam, tracker, brain, gate, Timeline(tmp_path), RuntimeConfig(**cfg), speaker=said.append
    )
    robot.connect()
    return rt, robot, tracker, seen, said


def follow_reply(say=""):
    return {
        "thought": "看到黑衣服的人",
        "say": say,
        "task": "",
        "mode": "replace",
        "plan": [{"behavior": "follow", "args": {"target": 3}}],
    }


async def test_task_to_motion(clock, tmp_path):
    rt, robot, _, seen, said = make_runtime(clock, [follow_reply(say="好的")], tmp_path)
    rt.control_step(clock())
    rt.set_task("跟着穿黑衣服的人")
    triggers, rt._pending = rt._pending, []
    res = await rt.think(triggers)
    await asyncio.sleep(0)
    assert res.mode == "replace" and said == ["好的"]
    assert "跟着穿黑衣服的人" in seen[0]["messages"][1]["content"][-1]["text"]
    clock.advance(0.05)
    rt.control_step(clock())
    assert robot.cmd.vx > 0  # person at 2.5m, keep 1.2m -> walk forward
    kinds = [
        json.loads(line)["kind"] for line in (rt.timeline.dir / "timeline.jsonl").read_text().splitlines()
    ]
    assert "task" in kinds and "brain_decision" in kinds
    messages, _ = load_think(rt.timeline.dir / "thinks" / "0001.json")
    assert messages[1]["content"][0]["image_url"]["url"].startswith("data:image/jpeg")


async def test_voice_estop_stops_and_blocks_new_plans(clock):
    rt, robot, _, _, _ = make_runtime(clock, [follow_reply(), follow_reply()])
    rt.control_step(clock())
    rt.set_task("跟着我")
    await rt.think(rt._pending)
    rt._pending = []
    rt.control_step(clock())
    assert robot.cmd.vx > 0
    rt.hear("停下！")
    assert rt.safety.estopped and robot.cmd.is_zero and rt.executor.idle
    assert [e.kind for e in rt._pending] == [EventKind.USER_SPEECH]
    await rt.think(rt._pending)
    assert rt.executor.idle  # brain answered with a plan but no new task -> still stopped


async def test_tracks_appear_and_get_lost(clock):
    rt, _, tracker, _, _ = make_runtime(clock, [], track_lost_s=1.0)
    rt.control_step(clock())
    assert [(e.kind, e.data["id"]) for e in rt._events if e.kind is EventKind.TRACK_APPEARED] == [
        (EventKind.TRACK_APPEARED, 3)
    ]
    tracker.items = ()
    clock.advance(1.5)
    rt.control_step(clock())
    assert any(e.kind is EventKind.TRACK_LOST for e in rt._events)


async def test_brain_outage_stops_after_grace(clock):
    rt, _, _, _, said = make_runtime(clock, [follow_reply(), 500, 500], brain_down_s=0.0)
    rt.control_step(clock())
    rt.set_task("跟着我")
    await rt.think(rt._pending)
    assert not rt.executor.idle
    await rt.think([])
    await asyncio.sleep(0)
    assert rt.executor.idle and said == ["我暂时没法思考，先停下来。"]


async def test_no_task_ignores_track_events(clock):
    rt, _, _, seen, _ = make_runtime(clock, [])
    rt.control_step(clock())
    assert not rt._should_think(rt._pending)
    rt.hear("你好")
    assert rt._should_think(rt._pending)


async def test_periodic_watch_when_idle_only_with_someone_in_view(clock):
    rt, _, tracker, _, _ = make_runtime(clock, [], idle_watch_s=1.5, periodic_s=4.0, track_lost_s=1.0)
    rt.task = "有人招手就打招呼"
    rt.control_step(clock())
    rt._last_think = clock()
    clock.advance(1.0)
    assert not rt._periodic_due(clock())
    clock.advance(0.6)
    assert rt._periodic_due(clock())  # idle, person #3 in view
    tracker.items = ()
    clock.advance(1.5)
    rt.control_step(clock())
    assert not rt._periodic_due(clock())  # nobody in view -> save the GPU
