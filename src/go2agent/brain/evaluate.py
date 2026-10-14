"""Offline brain evaluation: replay recorded thinks against the VLM and score them against labels.

Cases file (JSONL), paths relative to the cases file:
    {"think": "runs/20261014-180000/thinks/0003.json",
     "expect": {"mode": ["replace", "append"], "behavior": "follow", "target": 3},
     "note": "黑衣服的人走进画面，应开始跟随"}

`expect` keys (all optional, all must hold):
    mode       str or list of allowed modes
    behavior   name of the first plan step
    target     target id of the first plan step
    action     perform action anywhere in the plan
    say        true = must speak, false = must stay silent
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from go2agent.brain.client import VLMClient, VLMError
from go2agent.timeline import load_think


@dataclass(slots=True)
class CaseResult:
    think: str
    ok: bool
    latency_ms: float | None
    failures: list[str]
    output: dict[str, Any] | None
    note: str = ""


def check(expect: dict[str, Any], out: dict[str, Any]) -> list[str]:
    """Return the list of violated expectations (empty = pass)."""
    fails = []
    plan = out.get("plan") or []
    first = plan[0] if plan else {}
    if "mode" in expect:
        allowed = expect["mode"] if isinstance(expect["mode"], list) else [expect["mode"]]
        if out.get("mode") not in allowed:
            fails.append(f"mode {out.get('mode')} not in {allowed}")
    if "behavior" in expect and first.get("behavior") != expect["behavior"]:
        fails.append(f"behavior {first.get('behavior')} != {expect['behavior']}")
    if "target" in expect and first.get("args", {}).get("target") != expect["target"]:
        fails.append(f"target {first.get('args', {}).get('target')} != {expect['target']}")
    if "action" in expect and not any(s.get("args", {}).get("action") == expect["action"] for s in plan):
        fails.append(f"action {expect['action']} not in plan")
    if "say" in expect and bool(out.get("say")) != expect["say"]:
        fails.append("should speak" if expect["say"] else f"should stay silent, said {out.get('say')!r}")
    return fails


async def run_cases(
    cases_path: Path, client: VLMClient, schema: dict[str, Any], system_prompt: str | None = None
) -> list[CaseResult]:
    results = []
    for line in cases_path.read_text().splitlines():
        if not line.strip():
            continue
        case = json.loads(line)
        think = cases_path.parent / case["think"]
        messages, _ = load_think(think)
        if system_prompt is not None:
            messages[0]["content"] = system_prompt
        try:
            reply = await client.complete_json(messages, schema)
        except VLMError as e:
            results.append(
                CaseResult(case["think"], False, None, [f"vlm error: {e}"], None, case.get("note", ""))
            )
            continue
        fails = check(case.get("expect", {}), reply.data)
        results.append(
            CaseResult(case["think"], not fails, reply.latency_ms, fails, reply.data, case.get("note", ""))
        )
    return results


def summarize(results: list[CaseResult]) -> str:
    if not results:
        return "no cases"
    lat = sorted(r.latency_ms for r in results if r.latency_ms is not None)
    passed = sum(r.ok for r in results)
    lines = [f"{passed}/{len(results)} passed ({passed / len(results):.0%})"]
    if lat:
        p90 = lat[min(len(lat) - 1, int(len(lat) * 0.9))]
        lines.append(f"latency p50 {statistics.median(lat):.0f}ms  p90 {p90:.0f}ms  max {lat[-1]:.0f}ms")
    for r in results:
        if not r.ok:
            lines.append(f"FAIL {r.think}  {'; '.join(r.failures)}" + (f"  ({r.note})" if r.note else ""))
    return "\n".join(lines)
