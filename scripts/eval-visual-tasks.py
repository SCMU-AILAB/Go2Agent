#!/usr/bin/env python3
"""Compare visual decision routes on recorded cases. Never execute any skill."""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

from app.backend import BackendConfig, ConsoleBackend
from perception import CameraFrame, PerceptionResult
from robot import RobotState


async def evaluate(cases, factory, *, root=Path("."), max_age_s=2.0):
    results = []
    for case in cases:
        agent = factory(case["instruction"])
        started = time.monotonic()
        try:
            spec = await agent.prepare_task(case.get("catalog", ()))
            await agent.warmup()
            planned = time.monotonic()
            observation = PerceptionResult(
                observed_at_s=planned, **case.get("perception", {})
            )
            frames = [
                CameraFrame(
                    observed_at_s=planned + index * 0.000001,
                    rgb=(root / path).read_bytes(),
                    depth=None,
                    observation=observation,
                    nearest_obstacle_distance_m=case.get("obstacle_m"),
                )
                for index, path in enumerate(case["frames"])
            ]
            decision = await agent.decide(
                frames,
                RobotState(hardware=False, connected=True),
                case.get("catalog", ()),
            )
            finished = time.monotonic()
            matched = decision.action == case[
                "expected_action"
            ] and decision.skill == case.get("expected_skill")
            results.append(
                {
                    "id": case["id"],
                    "matched": matched,
                    "decision": decision.to_dict(),
                    "spec": spec.model_dump(),
                    "planning_s": round(planned - started, 4),
                    "decision_s": round(finished - planned, 4),
                    "stale": finished - planned > max_age_s,
                    "metrics": dict(agent.last_metrics),
                }
            )
        except Exception as exc:  # noqa: BLE001 - report all cases
            results.append(
                {
                    "id": case["id"],
                    "matched": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        finally:
            await agent.close()
    durations = sorted(r["decision_s"] for r in results if "decision_s" in r)
    return {
        "case_count": len(results),
        "correct_count": sum(r["matched"] for r in results),
        "stale_count": sum(r.get("stale", False) for r in results),
        "decision_p50_s": statistics.median(durations) if durations else None,
        "decision_p95_s": durations[max(0, int(len(durations) * 0.95 + 0.999) - 1)]
        if durations
        else None,
        "cases": results,
    }


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--vision-url", default="http://127.0.0.1:8011")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11435")
    parser.add_argument("--decision-model", default="qwen3.5:9b")
    parser.add_argument(
        "--route", choices=("grounded", "direct", "both"), default="both"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = json.loads(args.manifest.read_text())
    report = {}
    routes = ("grounded", "direct") if args.route == "both" else (args.route,)
    for route in routes:
        backend = ConsoleBackend(
            BackendConfig(
                hardware=False,
                robot_model="go2",
                audio_enabled=False,
                vision_backend="unifolm",
                vision_decision_route=route,
                vision_model="unitreerobotics/UnifoLM-ER-1",
                vision_url=args.vision_url,
                model_name=args.decision_model,
                ollama_url=args.ollama_url,
            )
        )
        configured = [
            {**case, "catalog": backend.runtime.registry.list()} for case in cases
        ]
        report[route] = await evaluate(
            configured, backend._build_vision_agent, root=args.manifest.parent
        )
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(
        json.dumps(
            {
                route: {k: v for k, v in summary.items() if k != "cases"}
                for route, summary in report.items()
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
