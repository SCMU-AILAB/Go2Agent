"""Voice HTTP endpoints, engine adapters and combined routing without model weights."""

import asyncio
import io
import sys
import threading
import wave
from types import SimpleNamespace

import cv2
import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from go2agent.audio.mic import encode_wav
from go2agent.perception.protocol import FrameHeader, TrackMsg, TracksMsg
from go2agent.server import asr, cli, tts


class FakeASR:
    def __init__(self):
        self.calls = []

    def transcribe(self, samples, language):
        self.calls.append((samples, language, threading.get_ident()))
        return "停止"


class FakeTTS:
    def __init__(self):
        self.calls = []

    def synthesize(self, text):
        self.calls.append((text, threading.get_ident()))
        return encode_wav(np.zeros(1600, np.float32))


class EmptyDetector:
    def track(self, bgr: np.ndarray) -> list[TrackMsg]:
        return []


def test_combined_endpoints_and_perception_on_same_app():
    recognizer, synthesizer = FakeASR(), FakeTTS()
    app = cli.create_all_app(
        detector_factory=EmptyDetector,
        asr_engine_factory=lambda: recognizer,
        tts_engine_factory=lambda: synthesizer,
    )
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        payload = encode_wav(np.full(1600, 0.2, np.float32))
        response = client.post("/asr", content=payload, headers={"content-type": "audio/wav"})
        assert response.status_code == 200
        assert response.json()["text"] == "停止" and response.json()["ms"] >= 0
        assert recognizer.calls[0][1] == "zh"
        assert recognizer.calls[0][0].dtype == np.float32
        np.testing.assert_allclose(recognizer.calls[0][0], 0.2, atol=0.0001)
        assert client.post("/asr?language=en", content=payload).status_code == 200
        assert recognizer.calls[-1][1] == "en"
        response = client.post("/tts", json={"text": "  你好  "})
        assert response.status_code == 200 and response.headers["content-type"] == "audio/wav"
        assert synthesizer.calls[0][0] == "你好"
        with wave.open(io.BytesIO(response.content), "rb") as wav:
            assert wav.getnframes() == 1600
        with client.websocket_connect("/track") as ws:
            ws.send_text(FrameHeader(seq=1, stamp=1).model_dump_json())
            ok, jpeg = cv2.imencode(".jpg", np.zeros((20, 20, 3), np.uint8))
            assert ok
            ws.send_bytes(jpeg.tobytes())
            assert TracksMsg.model_validate_json(ws.receive_text()).seq == 1


@pytest.mark.parametrize(
    "rate,channels,width,count", [(8000, 1, 2, 20), (16000, 2, 2, 20), (16000, 1, 1, 20), (16000, 1, 2, 0)]
)
def test_asr_rejects_wrong_audio_format(rate, channels, width, count):
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(width)
        wav.setframerate(rate)
        wav.writeframes(bytes(count * channels * width))
    with TestClient(asr.create_app(engine_factory=FakeASR)) as client:
        assert client.post("/asr", content=output.getvalue()).status_code == 400


def test_malformed_inputs_and_missing_engines():
    def unavailable():
        raise ImportError("not installed")

    with TestClient(asr.create_app(engine_factory=unavailable)) as client:
        assert client.get("/health").status_code == 200
        assert client.post("/asr", content=b"not audio").status_code == 400
        audio = encode_wav(np.zeros(1600, np.float32))
        assert client.post("/asr", content=audio[:-1]).status_code == 400
        assert client.post("/asr?language=", content=audio).status_code == 400
        assert client.post("/asr", content=audio).status_code == 503
    with TestClient(tts.create_app(engine_factory=unavailable)) as client:
        assert client.get("/health").status_code == 200
        assert client.post("/tts", json={"text": " "}).status_code == 400
        assert client.post("/tts", json={}).status_code == 422
        assert client.post("/tts", json={"text": "你好"}).status_code == 503


async def test_slow_engine_does_not_block_health_and_is_loaded_once():
    entered, release = threading.Event(), threading.Event()
    loaded, threads = [], []

    class SlowEngine:
        def synthesize(self, text):
            threads.append(threading.get_ident())
            entered.set()
            assert release.wait(2)
            return b"RIFF"

    def factory():
        loaded.append(True)
        return SlowEngine()

    app = tts.create_app(engine_factory=factory)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
        pending = asyncio.create_task(client.post("/tts", json={"text": "hi"}))
        try:
            async with asyncio.timeout(1):
                while not entered.is_set():
                    await asyncio.sleep(0.005)
            assert (await asyncio.wait_for(client.get("/health"), 0.2)).status_code == 200
        finally:
            release.set()
        assert (await pending).status_code == 200
        assert (await client.post("/tts", json={"text": "again"})).status_code == 200
    assert loaded == [True] and all(thread != threading.get_ident() for thread in threads)


@pytest.mark.parametrize("cuda,device,compute", [(True, "cuda", "float16"), (False, "cpu", "int8")])
def test_faster_whisper_defaults_and_consumes_segments(monkeypatch, cuda, device, compute):
    loaded, transcribed = [], []

    class Whisper:
        def __init__(self, model, **kwargs):
            loaded.append((model, kwargs))

        def transcribe(self, audio, **kwargs):
            transcribed.append(kwargs)
            return iter([SimpleNamespace(text="你"), SimpleNamespace(text="好")]), None

    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=Whisper))
    monkeypatch.setitem(sys.modules, "ctranslate2", SimpleNamespace(get_cuda_device_count=lambda: int(cuda)))
    engine = asr.FasterWhisperEngine()
    assert loaded == [("small", {"device": device, "compute_type": compute})]
    assert engine.transcribe(np.zeros(160, np.float32), "zh") == "你好"
    assert transcribed[0]["language"] == "zh"


def test_piper_voice_path_and_wav_output(monkeypatch, tmp_path):
    loaded = []

    def synthesize(text, wav):
        assert text == "你好"
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(22050)
        wav.writeframes(bytes(200))

    def load(path, config_path):
        loaded.append((path, config_path))
        return SimpleNamespace(synthesize_wav=synthesize)

    monkeypatch.setitem(sys.modules, "piper", SimpleNamespace(PiperVoice=SimpleNamespace(load=load)))
    model = tmp_path / "zh_CN-huayan-medium.onnx"
    config = tmp_path / "zh_CN-huayan-medium.onnx.json"
    model.touch()
    config.write_text("{}")
    engine = tts.PiperEngine(model_dir=tmp_path)
    assert loaded == [(str(model), str(config))]
    with wave.open(io.BytesIO(engine.synthesize("你好")), "rb") as wav:
        assert wav.getframerate() == 22050 and wav.getnframes() == 100
    config.unlink()
    with pytest.raises(FileNotFoundError, match="Piper requires"):
        tts.PiperEngine(model_dir=tmp_path)


VLM_PATHS = {"/models", "/models/brain/load", "/models/brain/unload", "/v1/{path}"}


@pytest.mark.parametrize(
    "arguments,paths,port",
    [
        ([], {"/track", "/asr", "/tts", "/health", *VLM_PATHS}, 8100),
        (["all", "--port", "9000", "--no-vlm"], {"/track", "/asr", "/tts", "/health"}, 9000),
        (["perception"], {"/track", "/health"}, 8100),
        (["asr", "--model", "tiny", "--device", "cpu"], {"/asr", "/health"}, 8101),
        (["tts", "--voice", "custom", "--model-dir", "/tmp/voices"], {"/tts", "/health"}, 8102),
    ],
)
def test_cli_commands(monkeypatch, arguments, paths, port):
    served = []

    async def fake_serve(app, host, port, vlm):
        served.append((app, host, port))

    monkeypatch.setattr(cli, "_serve", fake_serve)
    cli.main(arguments)
    app, host, served_port = served[0]
    assert (host, served_port) == ("0.0.0.0", port)
    actual = set(app.openapi()["paths"])
    if any(getattr(route, "path", None) == "/track" for route in app.routes):
        actual.add("/track")
    assert actual == paths
