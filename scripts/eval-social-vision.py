"""Offline evaluation through the production classifier. Never executes skills."""
import argparse
import asyncio
import hashlib
import json
import statistics
import time
from collections import Counter
from pathlib import Path

from agent.social_vision import SocialVisionAgent
from agent.vision_policy import DEFAULT_VISION_GOAL, OllamaVisionInvoker
from perception import CameraFrame, PerceptionResult
from robot import RobotState
from skills import build_go2_all_skills


class RecordingInvoker(OllamaVisionInvoker):
    async def ainvoke(self, frames, prompt):
        self.raw = None
        self.prompt = prompt
        self.raw = await super().ainvoke(frames, prompt)
        return self.raw


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--profile", choices=("legacy", "egocentric"), default="egocentric")
    parser.add_argument("--no-think", action="store_true")
    parser.add_argument("--speech", action="store_true", help="evaluate model-generated speech as well as gestures")
    parser.add_argument("--task", default=DEFAULT_VISION_GOAL)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    invoker = RecordingInvoker(args.model, base_url="http://127.0.0.1:11435",
                               constrain_json=False, max_new_tokens=256 if args.speech else 160,
                               think=False if args.no_think else None)
    # Ignore process-wide web proxies for the local SSH tunnel.
    import ollama
    invoker._client = ollama.AsyncClient(host="http://127.0.0.1:11435", trust_env=False)
    agent = SocialVisionAgent(invoker=invoker, model_name=args.model,
                              prompt_profile=args.profile, generate_speech=args.speech,
                              task_context=args.task, timeout_s=90)
    rows = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        for case in manifest["cases"]:
            folder = Path(manifest["capture_root"]) / case["session"] / case["window"]
            meta = json.loads((folder / "input.json").read_text())
            frames = []
            for f in meta["frames"]:
                data = (folder / f["file"]).read_bytes()
                if hashlib.sha256(data).hexdigest() != f["sha256"]:
                    raise ValueError(f"capture digest mismatch: {folder}")
                frames.append(CameraFrame(observed_at_s=f["observed_at_s"], rgb=data,
                    depth=None, observation=PerceptionResult(observed_at_s=f["observed_at_s"], source="offline")))
            started = time.monotonic()
            row = {"case": case, "model": args.model, "profile": args.profile}
            try:
                result = await agent.decide(frames, RobotState(hardware=False, connected=True),
                                            build_go2_all_skills())
                predicted = result.skill if result.action in ("execute_skill", "execute_and_speak") else "none"
                row.update(decision=result.model_dump(), predicted=predicted,
                           correct=predicted in case["allowed_decisions"],
                           metrics=dict(agent.last_metrics), raw=invoker.raw,
                           prompt_sha256=hashlib.sha256(invoker.prompt.encode()).hexdigest())
            except Exception as exc:
                row.update(error=str(exc), correct=False, predicted="error", raw=invoker.raw)
            row["elapsed_s"] = round(time.monotonic()-started, 3)
            output.write(json.dumps(row, ensure_ascii=False)+"\n")
            output.flush()
            rows.append(row)
            print(case["window"], case["allowed_decisions"], row["predicted"], row["elapsed_s"], flush=True)
        summary = {"total": len(rows), "correct": sum(r["correct"] for r in rows),
                   "errors": sum("error" in r for r in rows),
                   "median_s": statistics.median(r["elapsed_s"] for r in rows),
                   "false_actions": sum(r["predicted"] not in ("none", "error")
                                        and "none" in r["case"]["allowed_decisions"]
                                        and not r["correct"] for r in rows),
                   "predictions": dict(Counter(r["predicted"] for r in rows))}
        output.write(json.dumps({"summary": summary})+"\n")
        print(summary, flush=True)


if __name__ == "__main__":
    asyncio.run(main())
