"""Incremental parsing of the brain's streamed JSON decision.

The decision schema fixes the field order: mode, plan, say, new_task, thought (see brain/schema.py).
A field is complete once the next field has started (or the object is closed); a plan step is complete once
the next step has started or the plan array is closed. That lets the runtime act on `mode` and the first
plan step while the rest of the answer is still being generated.

    parser = DecisionStream()
    async for chunk in client.stream_json(messages, schema):
        for event in parser.feed(chunk):
            ...  # ModeEvent, StepEvent, PlanDoneEvent, SayEvent, NewTaskEvent, ThoughtEvent, DoneEvent
    parser.finish()  # raises StreamError if the text never formed a complete decision
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from pydantic_core import from_json

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


class _Incomplete(Exception):
    """A valid JSON prefix ended before its next required character."""


class _Prefix:
    """Check strict JSON syntax and locate decision boundaries before partial deserialization.

    from_json(allow_partial=True) can ignore invalid suffixes after complete values. This scan also
    distinguishes a closed plan from a closed nested array, even when strings contain delimiters.
    """

    def __init__(self, text: str) -> None:
        self.text = text
        self.pos = 0
        self.fields: list[str] = []
        self.plan_closed = False
        self.closed = False
        try:
            self._expect("{")
            self._object(root=True)
            self.closed = True
            if self.text[self.pos :].strip(" \r\n\t"):
                raise StreamError("unexpected text after decision object")
        except _Incomplete:
            pass

    def _char(self) -> str:
        if self.pos == len(self.text):
            raise _Incomplete
        return self.text[self.pos]

    def _peek(self) -> str:
        while self._char() in " \r\n\t":
            self.pos += 1
        return self._char()

    def _expect(self, char: str) -> None:
        if self._peek() != char:
            raise StreamError(f"expected {char!r} at character {self.pos}")
        self.pos += 1

    def _string(self) -> str:
        self._expect('"')
        start = self.pos - 1
        while True:
            char = self._char()
            self.pos += 1
            if char == '"':
                return json.loads(self.text[start : self.pos])
            if ord(char) < 32:
                raise StreamError(f"invalid control character at character {self.pos - 1}")
            if char == "\\":
                escape = self._char()
                self.pos += 1
                if escape == "u":
                    for _ in range(4):
                        if self._char() not in "0123456789abcdefABCDEF":
                            raise StreamError(f"invalid Unicode escape at character {self.pos}")
                        self.pos += 1
                elif escape not in '"\\/bfnrt':
                    raise StreamError(f"invalid escape at character {self.pos - 1}")

    def _object(self, *, root: bool = False) -> None:
        if self._peek() == "}":
            self.pos += 1
            return
        while True:
            key = self._string()
            if root:
                if len(self.fields) >= len(FIELD_ORDER) or key != FIELD_ORDER[len(self.fields)]:
                    raise StreamError(f"expected decision fields in order {FIELD_ORDER}, got {key!r}")
                self.fields.append(key)
            self._expect(":")
            self._value(plan=root and key == "plan")
            if self._peek() == "}":
                self.pos += 1
                return
            self._expect(",")

    def _array(self, *, plan: bool) -> None:
        if self._peek() != "]":
            while True:
                self._value()
                if self._peek() == "]":
                    break
                self._expect(",")
        self.pos += 1
        if plan:
            self.plan_closed = True

    def _digits(self) -> None:
        if self._char() not in "0123456789":
            raise StreamError(f"expected a digit at character {self.pos}")
        while self.pos < len(self.text) and self.text[self.pos] in "0123456789":
            self.pos += 1

    def _number(self) -> None:
        if self._char() == "-":
            self.pos += 1
        if self._char() == "0":
            self.pos += 1
        else:
            self._digits()
        if self.pos < len(self.text) and self.text[self.pos] == ".":
            self.pos += 1
            self._digits()
        if self.pos < len(self.text) and self.text[self.pos] in "eE":
            self.pos += 1
            if self._char() in "+-":
                self.pos += 1
            self._digits()

    def _value(self, *, plan: bool = False) -> None:
        char = self._peek()
        if char == "{":
            self.pos += 1
            self._object()
        elif char == "[":
            self.pos += 1
            self._array(plan=plan)
        elif char == '"':
            self._string()
        elif char in "-0123456789":
            self._number()
        elif char in "tfn":
            literal = {"t": "true", "f": "false", "n": "null"}[char]
            for expected in literal:
                if self._char() != expected:
                    raise StreamError(f"invalid literal at character {self.pos}")
                self.pos += 1
        else:
            raise StreamError(f"unexpected character {char!r} at character {self.pos}")


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
        self._text = ""
        self._emitted: set[str] = set()
        self._steps = 0
        self._done: DoneEvent | None = None

    def feed(self, chunk: str) -> list[StreamEvent]:
        self._text += chunk
        prefix = _Prefix(self._text)
        if not self._text.strip(" \r\n\t"):
            return []
        try:
            data: dict[str, Any] = from_json(self._text, allow_partial=True)
        except ValueError as e:
            raise StreamError(f"invalid json: {e}") from e
        if prefix.closed and tuple(prefix.fields) != FIELD_ORDER:
            raise StreamError("incomplete decision fields")
        events: list[StreamEvent] = []
        for i, field in enumerate(FIELD_ORDER):
            if field in self._emitted:
                continue
            complete = prefix.closed or len(prefix.fields) > i + 1
            if field == "plan":
                plan = data.get("plan", [])
                if not isinstance(plan, list):
                    raise StreamError("plan must be an array")
                count = len(plan) if prefix.plan_closed else max(0, len(plan) - 1)
                for index in range(self._steps, count):
                    step = plan[index]
                    if (
                        not isinstance(step, dict)
                        or not isinstance(step.get("behavior"), str)
                        or not isinstance(step.get("args"), dict)
                    ):
                        raise StreamError("plan steps must contain behavior and args")
                    events.append(StepEvent(index, step))
                self._steps = count
                if not prefix.plan_closed:
                    break
                events.append(PlanDoneEvent(count))
            else:
                if not complete:
                    break
                value = data.get(field)
                if not isinstance(value, str):
                    raise StreamError(f"{field} must be a string")
                if field == "mode":
                    if value not in ("keep", "replace", "append"):
                        raise StreamError(f"invalid mode: {value!r}")
                    events.append(ModeEvent(value))
                elif field == "say":
                    events.append(SayEvent(value))
                elif field == "new_task":
                    events.append(NewTaskEvent(value))
                else:
                    events.append(ThoughtEvent(value))
            self._emitted.add(field)
        if prefix.closed and self._done is None:
            self._done = DoneEvent(data, self._text)
            events.append(self._done)
        return events

    def finish(self) -> DoneEvent:
        """Call after the stream ends. Returns the DoneEvent (also emitted by feed when the object closed)."""
        if self._done is None:
            raise StreamError("incomplete decision: object was never closed")
        return self._done

    @property
    def text(self) -> str:
        """Everything fed so far."""
        return self._text
