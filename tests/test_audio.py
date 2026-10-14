"""Voice segmentation, thread boundaries and bounded playback without hardware."""

import asyncio
import io
import json
import sys
import threading
import wave
from types import SimpleNamespace

import httpx
import numpy as np
import pytest

from go2agent.audio.mic import EnergyVAD, Microphone, VADConfig, encode_wav
from go2agent.audio.speaker import Speaker, play_wav, select_device


def samples(seconds, amplitude=0.1):
    return np.full(round(seconds * 16000), amplitude, dtype=np.float32)


def utterance():
    return np.concatenate((samples(0.2), samples(0.6, 0)))


async def eventually(predicate, timeout=2):
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.005)


def test_vad_start_silence_and_onset_preservation():
    vad = EnergyVAD()
    assert not vad.feed(samples(0.1))
    assert not vad.feed(samples(0.6, 0)), "Short noises must not start an utterance"
    assert not vad.feed(samples(0.15))
    assert not vad.feed(samples(0.59, 0))
    result = vad.feed(samples(0.01, 0))
    assert len(result) == 1 and len(result[0]) == round(0.75 * 16000)
    np.testing.assert_allclose(result[0][:2400], 0.1)


def test_vad_arbitrary_blocks_and_maximum_duration():
    vad = EnergyVAD()
    audio = utterance()
    result = []
    for i in range(0, len(audio), 73):
        result.extend(vad.feed(audio[i : i + 73]))
    assert len(result) == 1
    np.testing.assert_array_equal(result[0], audio)
    result = vad.feed(samples(11))
    assert len(result) == 1 and len(result[0]) == 160000
    assert len(vad.feed(samples(0.6, 0))[0]) == 25600


@pytest.mark.parametrize(
    "kwargs", [{"threshold": 2}, {"silence_s": 0}, {"max_s": 0.1}, {"start_s": float("nan")}]
)
def test_vad_invalid_config(kwargs):
    with pytest.raises(ValueError):
        VADConfig(**kwargs)


class FakeStream:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.closed = self.stopped = False

    def start(self):
        pass

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True

    def feed(self, audio):
        self.kwargs["callback"](audio.reshape(-1, 1), len(audio), None, None)


async def test_microphone_sends_wav_and_delivers_on_event_loop():
    requests, delivered = [], []
    loop_thread = threading.get_ident()

    def asr(request):
        requests.append((request, threading.get_ident()))
        with wave.open(io.BytesIO(request.content), "rb") as wav:
            assert (wav.getnchannels(), wav.getframerate(), wav.getsampwidth()) == (1, 16000, 2)
            assert wav.getnframes() == len(utterance())
        return httpx.Response(200, json={"text": " 停下 "})

    with httpx.Client(transport=httpx.MockTransport(asr)) as client:
        mic = Microphone(
            "http://fake/asr",
            lambda text: delivered.append((text, threading.get_ident())),
            client=client,
            stream_factory=FakeStream,
            device="USB microphone",
        )
        mic.start()
        stream = mic._stream
        try:
            assert stream.kwargs["device"] == "USB microphone"
            assert stream.kwargs["samplerate"] == 16000 and stream.kwargs["channels"] == 1
            await asyncio.to_thread(stream.feed, utterance())
            await eventually(lambda: bool(delivered))
            assert delivered == [("停下", loop_thread)]
            assert requests[0][1] != loop_thread
            assert requests[0][0].url.params["language"] == "zh"
        finally:
            await mic.close()
        assert stream.closed and stream.stopped


async def test_microphone_mutes_and_waits_300ms_after_playback(clock):
    muted, delivered = [True], []
    with httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json={"text": "hello"}))
    ) as client:
        mic = Microphone(
            "http://fake/asr",
            delivered.append,
            muted=lambda: muted[0],
            clock=clock,
            client=client,
            stream_factory=FakeStream,
        )
        mic.start()
        try:
            stream = mic._stream
            stream.feed(utterance())
            muted[0] = False
            stream.feed(utterance())  # Starts the tail after playback ends.
            clock.advance(0.29)
            stream.feed(utterance())
            assert mic._utterances.empty()
            clock.advance(0.02)
            stream.feed(samples(0.1))
            muted[0] = True
            stream.feed(samples(0.1))  # Discards the incomplete utterance.
            muted[0] = False
            stream.feed(samples(0.01, 0))
            clock.advance(0.31)
            stream.feed(samples(0.05))
            stream.feed(samples(0.6, 0))
            assert mic._utterances.empty()
            stream.feed(utterance())
            await eventually(lambda: delivered == ["hello"])
        finally:
            await mic.close()


async def test_microphone_close_suppresses_late_transcription():
    entered, release = threading.Event(), threading.Event()
    delivered = []

    def asr(_):
        entered.set()
        assert release.wait(2)
        return httpx.Response(200, json={"text": "late"})

    with httpx.Client(transport=httpx.MockTransport(asr)) as client:
        mic = Microphone("http://fake/asr", delivered.append, client=client, stream_factory=FakeStream)
        mic.start()
        mic._stream.feed(utterance())
        try:
            await eventually(entered.is_set)
            closing = asyncio.create_task(mic.close())
            await eventually(lambda: not mic._running.is_set())
            release.set()
            await closing
            await asyncio.sleep(0)
            assert not delivered
        finally:
            release.set()
            await mic.close()


async def test_speaker_queue_drops_oldest_and_plays_on_worker():
    requested, played, threads = [], [], []
    entered, release = threading.Event(), threading.Event()

    def tts(request):
        text = json.loads(request.content)["text"]
        requested.append(text)
        return httpx.Response(200, content=text.encode(), headers={"content-type": "audio/wav"})

    def player(audio, device, stopped):
        played.append(audio.decode())
        threads.append(threading.get_ident())
        assert device == 7
        if audio == b"one":
            entered.set()
            while not release.wait(0.01) and not stopped.is_set():
                pass

    async with httpx.AsyncClient(transport=httpx.MockTransport(tts)) as client:
        speaker = Speaker("http://fake/tts", client=client, player=player, device=7)
        tasks = [asyncio.create_task(speaker.say("one"))]
        try:
            await eventually(entered.is_set)
            assert speaker.playing
            for text in ("two", "three", "four"):
                tasks.append(asyncio.create_task(speaker.say(text)))
                await asyncio.sleep(0)
            await eventually(tasks[1].done)
            assert speaker._queue.qsize() == 2
            release.set()
            await asyncio.wait_for(asyncio.gather(*tasks), 2)
            assert requested == played == ["one", "three", "four"]
            assert all(thread != threading.get_ident() for thread in threads)
            assert not speaker.playing
        finally:
            release.set()
            await speaker.close()


@pytest.mark.parametrize("abort_error", [False, True])
async def test_speaker_close_stops_playback_and_cancels_pending(abort_error):
    entered = threading.Event()

    def player(audio, device, stopped):
        entered.set()
        assert stopped.wait(2)
        if abort_error:
            raise RuntimeError("Audio device closed during playback")

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, content=b"audio"))
    ) as client:
        speaker = Speaker("http://fake/tts", client=client, player=player)
        active = asyncio.create_task(speaker.say("active"))
        await eventually(entered.is_set)
        pending = asyncio.create_task(speaker.say("pending"))
        await asyncio.sleep(0)
        await speaker.close()
        await asyncio.gather(active, pending, return_exceptions=True)
        assert active.cancelled() and pending.cancelled() and not speaker.playing
        await speaker.close()
        with pytest.raises(RuntimeError, match="closed"):
            await speaker.say("after close")


async def test_speaker_recovers_after_tts_error():
    played = []

    def tts(request):
        return httpx.Response(503 if json.loads(request.content)["text"] == "fail" else 200, content=b"ok")

    async with httpx.AsyncClient(transport=httpx.MockTransport(tts)) as client:
        speaker = Speaker("http://fake/tts", client=client, player=lambda data, *_: played.append(data))
        try:
            with pytest.raises(httpx.HTTPStatusError):
                await speaker.say("fail")
            assert not speaker.playing
            await speaker.say("works")
            assert played == [b"ok"]
        finally:
            await speaker.close()


def test_device_selection_and_pcm_playback(monkeypatch):
    opened, blocks = [], []

    class Output:
        def __init__(self, **kwargs):
            opened.append(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def write(self, block):
            blocks.append(block)

    devices = [
        {"name": "HDMI", "max_output_channels": 2, "max_input_channels": 0},
        {"name": "USB sound card", "max_output_channels": 2, "max_input_channels": 1},
    ]
    sd = SimpleNamespace(query_devices=lambda: devices, RawOutputStream=Output)
    monkeypatch.setitem(sys.modules, "sounddevice", sd)
    assert select_device(sd, None, "input") == 1
    assert select_device(sd, "3", "output") == 3
    assert select_device(sd, "named speaker", "output") == "named speaker"
    play_wav(encode_wav(samples(0.2)), None, threading.Event())
    assert opened[0]["device"] == 1 and opened[0]["samplerate"] == 16000
    assert len(b"".join(blocks)) == 6400
