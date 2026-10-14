"""WAV transcription service with a replaceable synchronous ASR engine."""

from __future__ import annotations

import asyncio
import io
import logging
import threading
import time
import wave
from collections.abc import Callable
from typing import Protocol

import numpy as np
from fastapi import APIRouter, FastAPI, HTTPException, Request

logger = logging.getLogger(__name__)


class ASREngine(Protocol):
    def transcribe(self, samples: np.ndarray, language: str) -> str: ...


class FasterWhisperEngine:
    def __init__(self, model: str = "small", device: str = "auto", compute_type: str | None = None) -> None:
        import ctranslate2
        from faster_whisper import WhisperModel

        if device == "auto":
            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        self._model = WhisperModel(
            model, device=device, compute_type=compute_type or ("float16" if device == "cuda" else "int8")
        )

    def transcribe(self, samples: np.ndarray, language: str) -> str:
        segments, _ = self._model.transcribe(
            samples, language=language, beam_size=1, condition_on_previous_text=False
        )
        return "".join(segment.text for segment in segments).strip()


def decode_wav(data: bytes) -> np.ndarray:
    """Accept exactly PCM16 mono at 16 kHz and return float32 samples."""
    try:
        with wave.open(io.BytesIO(data), "rb") as wav:
            if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getcomptype()) != (
                1,
                2,
                16000,
                "NONE",
            ):
                raise ValueError("Expected 16 kHz mono 16-bit PCM WAV")
            count = wav.getnframes()
            pcm = wav.readframes(count)
            if count == 0 or len(pcm) != count * 2:
                raise ValueError("WAV must contain complete, nonempty audio")
    except (wave.Error, EOFError) as exc:
        raise ValueError("Invalid WAV audio") from exc
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0


def create_router(
    model: str = "small",
    device: str = "auto",
    compute_type: str | None = None,
    *,
    engine_factory: Callable[[], ASREngine] | None = None,
) -> APIRouter:
    router = APIRouter()
    factory = engine_factory or (lambda: FasterWhisperEngine(model, device, compute_type))
    engine: ASREngine | None = None
    lock = threading.Lock()

    def transcribe(samples: np.ndarray, language: str) -> str:
        nonlocal engine
        with lock:
            if engine is None:
                engine = factory()
            return engine.transcribe(samples, language)

    @router.post("/asr")
    async def asr(request: Request, language: str = "zh") -> dict[str, str | float]:
        try:
            samples = decode_wav(await request.body())
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not language.strip():
            raise HTTPException(status_code=400, detail="language must not be empty")
        started = time.monotonic()
        try:
            text = await asyncio.to_thread(transcribe, samples, language)
        except Exception as exc:
            logger.exception("ASR engine failed")
            raise HTTPException(status_code=503, detail="ASR engine unavailable") from exc
        return {"text": text, "ms": (time.monotonic() - started) * 1000}

    return router


def create_app(
    model: str = "small",
    device: str = "auto",
    compute_type: str | None = None,
    *,
    engine_factory: Callable[[], ASREngine] | None = None,
) -> FastAPI:
    app = FastAPI(title="Go2Agent ASR")
    app.include_router(create_router(model, device, compute_type, engine_factory=engine_factory))

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
