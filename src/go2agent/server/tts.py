"""Piper speech service. Place the voice's .onnx and .onnx.json in model_dir."""

from __future__ import annotations

import asyncio
import io
import logging
import threading
import wave
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from fastapi import APIRouter, FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class TTSEngine(Protocol):
    def synthesize(self, text: str) -> bytes: ...


class PiperEngine:
    def __init__(self, voice: str = "zh_CN-huayan-medium", model_dir: str | Path = "models/piper") -> None:
        from piper import PiperVoice

        model = Path(model_dir).expanduser() / f"{voice}.onnx"
        config = model.with_suffix(".onnx.json")
        if not model.is_file() or not config.is_file():
            raise FileNotFoundError(f"Piper requires {model} and {config}")
        self._voice = PiperVoice.load(str(model), config_path=str(config))

    def synthesize(self, text: str) -> bytes:
        output = io.BytesIO()
        with wave.open(output, "wb") as wav:
            self._voice.synthesize_wav(text, wav)
        return output.getvalue()


class SpeechRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)


def create_router(
    voice: str = "zh_CN-huayan-medium",
    model_dir: str | Path = "models/piper",
    *,
    engine_factory: Callable[[], TTSEngine] | None = None,
) -> APIRouter:
    router = APIRouter()
    factory = engine_factory or (lambda: PiperEngine(voice, model_dir))
    engine: TTSEngine | None = None
    lock = threading.Lock()

    def synthesize(text: str) -> bytes:
        nonlocal engine
        with lock:
            if engine is None:
                engine = factory()
            return engine.synthesize(text)

    @router.post("/tts")
    async def tts(request: SpeechRequest) -> Response:
        text = request.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="text must not be blank")
        try:
            audio = await asyncio.to_thread(synthesize, text)
        except Exception as exc:
            logger.exception("TTS engine failed")
            raise HTTPException(status_code=503, detail="TTS engine unavailable") from exc
        return Response(content=audio, media_type="audio/wav")

    return router


def create_app(
    voice: str = "zh_CN-huayan-medium",
    model_dir: str | Path = "models/piper",
    *,
    engine_factory: Callable[[], TTSEngine] | None = None,
) -> FastAPI:
    app = FastAPI(title="Go2Agent TTS")
    app.include_router(create_router(voice, model_dir, engine_factory=engine_factory))

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
