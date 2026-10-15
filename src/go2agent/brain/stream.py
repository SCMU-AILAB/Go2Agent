"""Incremental parsing of the brain's streamed JSON decision.

The decision schema fixes the field order: mode, plan, say, new_task, thought (see brain/schema.py).
A field is complete once the next field has started (or the object is closed); a plan step is complete once
the next step has started or the plan array is closed. That lets the runtime act on `mode` and the first
plan step while the rest of the answer is still being generated.

    parser = DecisionStream()
    async for chunk in client.stream_json(messages, schema):
        for event in parser.feed(chunk):
            ...  # ModeEvent, StepEvent, SayEvent, NewTaskEvent, ThoughtEvent, DoneEvent
    parser.finish()  # raises StreamError if the text never formed a complete decision
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

FIELD_ORDER = ("mode", "plan", "say", "new_task", "thought")


class StreamError(Exception):
    """The streamed text is not (the prefix of) a valid decision object."""


@dataclass(frozen=True, slots=True)
class ModeEvent:
    mode: str  # "keep" | "replace" | "append"


@dataclass(frozen=True, slots=True)
class StepEvent:
    index: int  # 0-based position in plan
    step: dict[str, Any]  # {"behavior": str, "args": {...}}, complete


@dataclass(frozen=True, slots=True)
class PlanDoneEvent:
    count: int  # number of steps in the closed plan array


@dataclass(frozen=True, slots=True)
class SayEvent:
    text: str


@dataclass(frozen=True, slots=True)
class NewTaskEvent:
    text: str


@dataclass(frozen=True, slots=True)
class ThoughtEvent:
    text: str


@dataclass(frozen=True, slots=True)
class DoneEvent:
    data: dict[str, Any]  # the complete decision object
    raw: str  # the complete text


StreamEvent = ModeEvent | StepEvent | PlanDoneEvent | SayEvent | NewTaskEvent | ThoughtEvent | DoneEvent


class DecisionStream:
    """Feed text chunks as they arrive; each call returns the events that became complete.

    Guarantees:
    - Each event is emitted exactly once, in FIELD_ORDER; steps in index order; DoneEvent last.
    - Chunk boundaries can fall anywhere (inside keys, strings, escapes, numbers, multi-byte text).
    - Tolerates whitespace between tokens (the server may or may not strip it).
    - Raises StreamError from feed() as soon as the text cannot be a valid prefix of a JSON object, and from
      finish() if the object was never closed.
    """

    def __init__(self) -> None:
        raise NotImplementedError

    def feed(self, chunk: str) -> list[StreamEvent]:
        raise NotImplementedError

    def finish(self) -> DoneEvent:
        """Call after the stream ends. Returns the DoneEvent (also emitted by feed when the object closed)."""
        raise NotImplementedError

    @property
    def text(self) -> str:
        """Everything fed so far."""
        raise NotImplementedError
