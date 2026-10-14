from test_behaviors import make, person, tick

from go2agent.core.types import EventKind
from go2agent.executor import Executor


def ex(refuse=None):
    started = []

    def perform(action):
        started.append(action)
        return refuse

    return Executor(perform), started


def test_runs_plan_in_order_and_reports_done():
    e, started = ex()
    e.submit([make("perform", action="hello"), make("wait", seconds=1)], "replace")
    e.tick(tick(0))
    assert started == ["hello"]
    out = e.tick(tick(3.1))
    assert [ev.kind for ev in out.events] == [EventKind.BEHAVIOR_DONE]
    assert out.events[0].data["behavior"] == "perform" and not out.events[0].data["queue_empty"]
    e.tick(tick(3.2))
    out = e.tick(tick(4.3))
    assert out.events[0].data["queue_empty"] and e.idle


def test_keep_does_nothing_replace_preempts_append_queues():
    e, _ = ex()
    e.submit([make("follow", target=1)], "replace")
    e.tick(tick(0, [person()], depth=2.0))
    e.submit([make("wait", seconds=1)], "keep")
    assert e.current is not None and e.current.name == "follow" and not e.queue
    e.submit([make("wait", seconds=1)], "append")
    assert len(e.queue) == 1
    e.submit([make("face", target=1)], "replace")
    assert e.current is None and [b.name for b in e.queue] == ["face"]


def test_stop_clears_everything():
    e, _ = ex()
    e.submit([make("follow", target=1), make("wait", seconds=1)], "replace")
    e.tick(tick(0, [person()], depth=2.0))
    e.submit([make("stop")], "append")
    assert e.idle


def test_failure_clears_rest_of_plan():
    e, _ = ex()
    e.submit([make("approach", target=1), make("perform", action="hello")], "replace")
    out = e.tick(tick(0, healthy=False))
    assert out.events[0].kind is EventKind.BEHAVIOR_FAILED
    assert out.events[0].data["reason"] == "perception_unavailable" and e.idle


def test_refused_action_fails_behavior():
    e, _ = ex(refuse="estop")
    e.submit([make("perform", action="hello")], "replace")
    out = e.tick(tick(0))
    assert out.events[0].kind is EventKind.BEHAVIOR_FAILED and out.events[0].data["reason"] == "estop"
