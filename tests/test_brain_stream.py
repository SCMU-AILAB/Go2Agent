import asyncio
import json
import random
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest

from go2agent.brain.client import VLMClient, VLMConfig, VLMError
from go2agent.brain.stream import (
    FIELD_ORDER,
    DecisionStream,
    DoneEvent,
    ModeEvent,
    NewTaskEvent,
    PlanDoneEvent,
    SayEvent,
    StepEvent,
    StreamError,
    StreamEvent,
    ThoughtEvent,
)


def decision(*, keep: bool = False) -> dict[str, Any]:
    return {
        "mode": "keep" if keep else "replace",
        "new_task": '跟着穿黑衣服的人，听到"停"就停止。',
        "plan": []
        if keep
        else [
            {
                "behavior": "follow",
                "args": {
                    "target": {"box": [0.1, 0.2, 0.3, 0.9], "label": '人 { [ \\" ] } 😀'},
                    "distance": 1.25e-5,
                    "metadata": {"visible": True, "lost": False, "extra": None},
                },
            },
            {"behavior": "perform", "args": {"action": "hello"}},
        ],
        "say": '好的，听到"停"就停。\\\n',
        "thought": "目标在画面中。",
    }


def expected_events(data: dict[str, Any], raw: str) -> list[StreamEvent]:
    return [
        ModeEvent(data["mode"]),
        NewTaskEvent(data["new_task"]),
        *[StepEvent(i, step) for i, step in enumerate(data["plan"])],
        PlanDoneEvent(len(data["plan"])),
        SayEvent(data["say"]),
        ThoughtEvent(data["thought"]),
        DoneEvent(data, raw),
    ]


@pytest.mark.parametrize("keep", [False, True])
@pytest.mark.parametrize("pretty", [False, True])
@pytest.mark.parametrize("ensure_ascii", [False, True])
def test_decision_all_chunk_boundaries(keep: bool, pretty: bool, ensure_ascii: bool) -> None:
    data = decision(keep=keep)
    raw = json.dumps(data, ensure_ascii=ensure_ascii, indent=2 if pretty else None, separators=(",", ":"))
    expected = expected_events(data, raw)
    rng = random.Random(2357)
    chunkings = [[raw], list(raw)]
    for _ in range(60):
        chunks: list[str] = []
        start = 0
        while start < len(raw):
            end = start + rng.randint(1, 40)
            chunks.append(raw[start:end])
            start = end
        chunkings.append(chunks)
    for chunks in chunkings:
        parser = DecisionStream()
        assert parser.feed("") == []
        events = [event for chunk in chunks for event in parser.feed(chunk)]
        assert events == expected
        assert parser.finish() is events[-1]
        assert parser.finish() is parser.finish()
        assert parser.text == raw
        assert parser.feed("") == []
    assert FIELD_ORDER == ("mode", "new_task", "plan", "say", "thought")


def test_events_arrive_at_field_and_plan_boundaries_without_partial_args() -> None:
    parser = DecisionStream()
    assert parser.feed(' \n {"mode":"append"') == []
    assert parser.feed(', "new_task"') == [ModeEvent("append")]
    assert parser.feed(':"走近目标", "plan"') == [NewTaskEvent("走近目标")]
    assert parser.feed(':[{"behavior":"follow","args":{"target":{"box":[0.1,0.2,0.3,0.4]') == []
    assert parser.feed(',"label":"带有 ] } 的名字"}') == []
    assert parser.feed(',"distance":1') == []
    assert parser.feed(".25e") == []
    assert parser.feed("-2}}") == []
    assert parser.feed(",") == []
    first = {
        "behavior": "follow",
        "args": {"target": {"box": [0.1, 0.2, 0.3, 0.4], "label": "带有 ] } 的名字"}, "distance": 0.0125},
    }
    assert parser.feed("{") == [StepEvent(0, first)]
    assert parser.feed('"behavior":"perform","args":{"action":"hello"}}') == []
    second = {"behavior": "perform", "args": {"action": "hello"}}
    assert parser.feed("]") == [StepEvent(1, second), PlanDoneEvent(2)]
    assert parser.feed(',"say":"你好"') == []
    assert parser.feed(',"thought"') == [SayEvent("你好")]
    assert parser.feed(':"跟随目标"') == []
    final = parser.feed("}")
    assert final == [ThoughtEvent("跟随目标"), DoneEvent(json.loads(parser.text), parser.text)]
    assert parser.finish() is final[-1]
    assert parser.feed(" \r\n\t") == []


@pytest.mark.parametrize(
    ("prefix", "bad"),
    [
        ("", "x"),
        (" \n", "["),
        ("{", "x"),
        ('{"mode"', "x"),
        ('{"mode":', "!"),
        ('{"mode":"keep"', "x"),
        ('{"mode":t', "x"),
        ('{"mode":true', "x"),
        ('{"mode":1', "x"),
        ('{"mode":0', "1"),
        ('{"mode":1.', "x"),
        ('{"mode":1e+', "x"),
        ('{"mode":"a', "\n"),
        ('{"mode":"a\\', "q"),
        ('{"mode":"a\\u', "X"),
        ('{"mode":"a\\u12', "X"),
        ('{"mode":"keep",', "}"),
        ('{"mode":"keep","new_task":"","plan":[', ","),
        ('{"mode":"keep","new_task":"","plan":[{},', "]"),
    ],
)
def test_invalid_prefix_raises_on_first_bad_character(prefix: str, bad: str) -> None:
    parser = DecisionStream()
    for char in prefix:
        parser.feed(char)
    with pytest.raises(StreamError):
        parser.feed(bad)


@pytest.mark.parametrize("same_chunk", [False, True])
def test_trailing_garbage_is_rejected(same_chunk: bool) -> None:
    parser = DecisionStream()
    raw = json.dumps(decision())
    if same_chunk:
        with pytest.raises(StreamError):
            parser.feed(raw + "x")
    else:
        parser.feed(raw)
        with pytest.raises(StreamError):
            parser.feed("x")


@pytest.mark.parametrize("raw", ["", " \n", "{", '{"mode":"keep"', json.dumps(decision())[:-1]])
def test_finish_requires_closed_object(raw: str) -> None:
    parser = DecisionStream()
    parser.feed(raw)
    with pytest.raises(StreamError, match="never closed"):
        parser.finish()


@pytest.mark.parametrize(
    "raw",
    [
        "{}",
        '{"new_task":"","mode":"keep"}',
        '{"mode":"keep","mode":"keep"}',
        '{"mode":"wrong","new_task":""}',
        '{"mode":null,"new_task":""}',
        '{"mode":"keep","new_task":null,"plan":[]}',
        '{"mode":"keep","new_task":"","plan":{}}',
        '{"mode":"keep","new_task":"","plan":[{}]}',
        json.dumps({**decision(), "say": None}),
        json.dumps({**decision(), "thought": 1}),
    ],
)
def test_invalid_decisions_are_rejected(raw: str) -> None:
    with pytest.raises(StreamError):
        DecisionStream().feed(raw)


class SSEBody(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes], *, delay: float = 0.0) -> None:
        self.chunks = chunks
        self.delay = delay
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self.chunks:
            if self.delay:
                await asyncio.sleep(self.delay)
            yield chunk

    async def aclose(self) -> None:
        self.closed = True


def sse(payload: dict[str, Any]) -> bytes:
    return ("data: " + json.dumps(payload, ensure_ascii=False) + "\r\n\r\n").encode()


def text_chunk(content: str | None) -> bytes:
    return sse({"choices": [{"delta": {"content": content}}], "usage": None})


async def test_vlm_stream_request_text_usage_and_timing() -> None:
    raw = json.dumps(decision(), ensure_ascii=False)
    fragments = [raw[:50], raw[50:100], raw[100:]]
    usage = {"prompt_tokens": 42, "completion_tokens": 20, "total_tokens": 62}
    wire = (
        b": keepalive\r\nevent: message\r\n\r\n"
        + sse({"choices": [{"delta": {"role": "assistant"}}]})
        + text_chunk("")
        + text_chunk(None)
        + b"".join(text_chunk(fragment) for fragment in fragments)
        + sse({"choices": [{"delta": {}, "finish_reason": "stop"}]})
        + sse({"choices": [], "usage": usage})
        + b"data: [DONE]\r\n\r\ndata: invalid after done\r\n"
    )
    # Split even the SSE delimiters and UTF-8 sequences across network reads.
    body = SSEBody([wire[i : i + 1] for i in range(len(wire))])
    messages = [{"role": "user", "content": "跟随那个人"}]
    schema = {"type": "object"}
    config = VLMConfig(
        base_url="http://vlm/v1", model="test", api_key="test-key", max_tokens=444, temperature=0.2
    )
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, stream=body, headers={"Content-Type": "text/event-stream"})

    client = VLMClient(config, transport=httpx.MockTransport(handler))
    try:
        stream = client.stream_json(messages, schema)
        assert stream.ttft_ms is None and stream.latency_ms is None and stream.usage == {}
        assert [chunk async for chunk in stream] == fragments
        assert "".join(fragments) == raw
        assert stream.usage == usage
        assert stream.ttft_ms is not None and stream.latency_ms is not None
        assert 0 <= stream.ttft_ms <= stream.latency_ms
        assert body.closed
        assert len(seen) == 1
        assert seen[0].method == "POST"
        assert str(seen[0].url) == "http://vlm/v1/chat/completions"
        assert seen[0].headers["Authorization"] == "Bearer test-key"
        assert json.loads(seen[0].content) == {
            "model": config.model,
            "messages": messages,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "decision", "schema": schema, "strict": True},
            },
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        with pytest.raises(RuntimeError, match="once"):
            stream.__aiter__()
    finally:
        await client.close()


async def test_vlm_iteration_is_claimed_before_first_read() -> None:
    client = VLMClient(VLMConfig())
    try:
        stream = client.stream_json([], {})
        stream.__aiter__()
        with pytest.raises(RuntimeError, match="once"):
            stream.__aiter__()
    finally:
        await client.close()


@pytest.mark.parametrize("ending", [b"data:[DONE]\n\n", b""])
async def test_vlm_stream_without_text(ending: bytes) -> None:
    client = VLMClient(
        VLMConfig(), transport=httpx.MockTransport(lambda _: httpx.Response(200, content=ending))
    )
    try:
        stream = client.stream_json([], {})
        assert [chunk async for chunk in stream] == []
        assert stream.ttft_ms is None
        assert stream.latency_ms is not None
        assert stream.usage == {}
    finally:
        await client.close()


async def test_vlm_http_error_reads_streamed_body() -> None:
    body = SSEBody([b"model ", b"unavailable"])
    client = VLMClient(VLMConfig(), transport=httpx.MockTransport(lambda _: httpx.Response(503, stream=body)))
    try:
        stream = client.stream_json([], {})
        with pytest.raises(VLMError, match="http 503: model unavailable"):
            _ = [chunk async for chunk in stream]
        assert body.closed
        assert stream.ttft_ms is None and stream.latency_ms is not None
    finally:
        await client.close()


@pytest.mark.parametrize(
    "line",
    [
        "data: {broken",
        "data:",
        "data: []",
        "data: {}",
        'data: {"choices":null}',
        'data: {"choices":[null]}',
        'data: {"choices":[{"delta":null}]}',
        'data: {"choices":[{"delta":{"content":42}}]}',
        'data: {"choices":[],"usage":42}',
    ],
)
async def test_vlm_malformed_data_is_error(line: str) -> None:
    body = SSEBody([(line + "\n\n").encode()])
    client = VLMClient(VLMConfig(), transport=httpx.MockTransport(lambda _: httpx.Response(200, stream=body)))
    try:
        stream = client.stream_json([], {})
        with pytest.raises(VLMError, match="invalid SSE data"):
            _ = [chunk async for chunk in stream]
        assert body.closed
        assert stream.latency_ms is not None
    finally:
        await client.close()


async def test_vlm_timeout_covers_entire_stream() -> None:
    body = SSEBody([text_chunk(str(i)) for i in range(30)], delay=0.01)
    client = VLMClient(
        VLMConfig(timeout=0.1), transport=httpx.MockTransport(lambda _: httpx.Response(200, stream=body))
    )
    try:
        stream = client.stream_json([], {})
        received: list[str] = []
        with pytest.raises(VLMError, match="timeout after 0.1s"):
            async for chunk in stream:
                received.append(chunk)
        assert 0 < len(received) < 30
        assert stream.ttft_ms is not None and stream.latency_ms is not None
        assert stream.latency_ms >= stream.ttft_ms
        assert body.closed
    finally:
        await client.close()


async def test_vlm_timeout_includes_connection() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        await asyncio.sleep(1)
        return httpx.Response(200, content=b"data: [DONE]\n\n")

    client = VLMClient(VLMConfig(timeout=0.01), transport=httpx.MockTransport(handler))
    try:
        stream = client.stream_json([], {})
        with pytest.raises(VLMError, match="timeout"):
            _ = [chunk async for chunk in stream]
        assert stream.ttft_ms is None and stream.latency_ms is not None
    finally:
        await client.close()


async def test_vlm_deadline_does_not_cancel_consumer_processing() -> None:
    body = SSEBody([text_chunk("first"), text_chunk("second")])
    client = VLMClient(
        VLMConfig(timeout=0.02), transport=httpx.MockTransport(lambda _: httpx.Response(200, stream=body))
    )
    try:
        stream = client.stream_json([], {})
        iterator = aiter(stream)
        assert await anext(iterator) == "first"
        await asyncio.sleep(0.04)
        with pytest.raises(VLMError, match="timeout"):
            await anext(iterator)
        assert body.closed
        assert stream.latency_ms is not None
    finally:
        await client.close()


async def test_vlm_cancellation_propagates_and_closes_response() -> None:
    body = SSEBody([text_chunk("first")], delay=1)
    requested = asyncio.Event()

    def handler(_: httpx.Request) -> httpx.Response:
        requested.set()
        return httpx.Response(200, stream=body)

    client = VLMClient(VLMConfig(), transport=httpx.MockTransport(handler))
    try:
        stream = client.stream_json([], {})

        async def consume() -> None:
            _ = [chunk async for chunk in stream]

        task = asyncio.create_task(consume())
        await requested.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert body.closed
        assert stream.ttft_ms is None and stream.latency_ms is not None
    finally:
        await client.close()


@pytest.mark.parametrize("error_type", [httpx.ConnectError, httpx.ReadError, httpx.ReadTimeout])
@pytest.mark.parametrize("during_body", [False, True])
async def test_vlm_httpx_errors_are_mapped(error_type: type[httpx.HTTPError], during_body: bool) -> None:
    class BrokenBody(httpx.AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield text_chunk("first")
            raise error_type("disconnected")

    def handler(_: httpx.Request) -> httpx.Response:
        if not during_body:
            raise error_type("disconnected")
        return httpx.Response(200, stream=BrokenBody())

    client = VLMClient(VLMConfig(), transport=httpx.MockTransport(handler))
    try:
        stream = client.stream_json([], {})
        match = "timeout" if issubclass(error_type, httpx.TimeoutException) else "unreachable: disconnected"
        with pytest.raises(VLMError, match=match):
            _ = [chunk async for chunk in stream]
        assert stream.latency_ms is not None
        assert (stream.ttft_ms is not None) == during_body
    finally:
        await client.close()
