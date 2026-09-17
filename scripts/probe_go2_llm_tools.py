"""Offline-safe Chinese tool-selection probe against the host Ollama model.

Uses SimulatedRobotAdapter only — never connects DDS or drives the physical dog.
"""

from __future__ import annotations

import asyncio
import json
import os

from adapters.langchain import SkillToolObserver, build_langchain_tools
from agent.service import GO2_SYSTEM_PROMPT
from core.runtime import SkillRuntime
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_ollama import ChatOllama
from robot import SimulatedRobotAdapter
from skills import register_go2_skills

BASE_URL = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11435")
MODEL = os.environ.get("OLLAMA_MODEL", "qwen3.5:9b")

CASES = [
    ("给我比个心", "heart"),
    ("伸个懒腰", "stretch"),
    ("坐下", "sit"),
    ("站起来", "stand_up"),
    ("跳一段舞蹈一", "dance1"),
    ("打个招呼", "hello"),
    ("趴下", "stand_down"),
    ("向左转", "turn_left"),
]


class Recorder(SkillToolObserver):
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def before_skill(self, skill_name: str, arguments: dict) -> None:
        self.calls.append(skill_name)

    async def after_skill(self, skill_name: str, arguments: dict, result: dict) -> None:
        return None


async def run_case(agent, recorder: Recorder, instruction: str, expected: str) -> dict:
    recorder.calls.clear()
    try:
        output = await agent.ainvoke(
            {"messages": [HumanMessage(content=instruction)]}
        )
        messages = output.get("messages") if isinstance(output, dict) else None
        reply = ""
        if messages:
            for message in reversed(messages):
                if isinstance(message, AIMessage) and str(message.content).strip():
                    reply = str(message.content).strip()
                    break
        tools = list(recorder.calls)
        ok = expected in tools
        return {
            "instruction": instruction,
            "expected": expected,
            "tools": tools,
            "ok": ok,
            "reply": reply[:160],
        }
    except Exception as exc:  # noqa: BLE001 - report and continue
        return {
            "instruction": instruction,
            "expected": expected,
            "tools": list(recorder.calls),
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


async def main() -> int:
    runtime = SkillRuntime(SimulatedRobotAdapter())
    register_go2_skills(runtime)
    recorder = Recorder()
    tools = build_langchain_tools(runtime, observer=recorder)
    model = ChatOllama(
        model=MODEL,
        base_url=BASE_URL,
        temperature=0.0,
        client_kwargs={"trust_env": False},
    )
    agent = create_agent(model=model, tools=tools, system_prompt=GO2_SYSTEM_PROMPT)
    results = []
    for instruction, expected in CASES:
        result = await run_case(agent, recorder, instruction, expected)
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    passed = sum(1 for item in results if item["ok"])
    summary = {
        "model": MODEL,
        "base_url": BASE_URL,
        "passed": passed,
        "total": len(results),
        "failures": [r["instruction"] for r in results if not r["ok"]],
    }
    print(json.dumps({"summary": summary}, ensure_ascii=False), flush=True)
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
