import json

import httpx

from go2agent.brain.client import VLMClient, VLMConfig
from go2agent.brain.evaluate import check, run_cases, summarize
from go2agent.timeline import Timeline


def test_check():
    out = {
        "mode": "replace",
        "say": "",
        "plan": [
            {"behavior": "follow", "args": {"target": 3}},
            {"behavior": "perform", "args": {"action": "hello"}},
        ],
    }
    assert (
        check(
            {
                "mode": ["replace", "append"],
                "behavior": "follow",
                "target": 3,
                "action": "hello",
                "say": False,
            },
            out,
        )
        == []
    )
    assert len(check({"mode": "keep", "target": 4, "say": True}, out)) == 3


async def test_run_cases_end_to_end(tmp_path):
    tl = Timeline(tmp_path)
    assert tl.dir is not None
    msgs = [
        {"role": "system", "content": "old"},
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
    ]
    tl.save_think(msgs, {"raw": ""})
    think = tl.dir / "thinks" / "0001.json"
    cases = tmp_path / "cases.jsonl"
    rel = think.relative_to(tmp_path)
    cases.write_text(
        json.dumps({"think": str(rel), "expect": {"mode": "keep"}})
        + "\n"
        + json.dumps({"think": str(rel), "expect": {"mode": "replace"}, "note": "n"})
        + "\n"
    )
    seen = []

    def handler(req):
        seen.append(json.loads(req.content))
        reply = {"thought": "", "say": "", "new_task": "", "mode": "keep", "plan": []}
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(reply)}}]})

    client = VLMClient(VLMConfig(base_url="http://vlm/v1"), httpx.MockTransport(handler))
    results = await run_cases(cases, client, {}, system_prompt="new")
    assert [r.ok for r in results] == [True, False]
    assert seen[0]["messages"][0]["content"] == "new"
    assert summarize(results).startswith("1/2 passed")


def test_timeline_summary_lists_decisions_and_hides_health_flips(tmp_path):
    from go2agent.timeline import summarize

    tl = Timeline(tmp_path)
    tl.record("task", {"text": "hi"}, mono=10.0)
    tl.record("service", {"service": "perception", "ok": False}, mono=10.5)
    tl.record(
        "brain_decision", {"mode": "keep", "plan": [], "thought": "等待", "think_id": "0001"}, mono=11.0
    )
    tl.close()
    assert tl.dir is not None
    lines = summarize(tl.dir)
    assert (
        len(lines) == 2
        and "task" in lines[0]
        and '"thought": "等待"' in lines[1]
        and '"plan"' not in lines[1]
    )
    assert len(summarize(tl.dir, everything=True)) == 3
    assert len(summarize(tl.dir, since=0.8)) == 1
