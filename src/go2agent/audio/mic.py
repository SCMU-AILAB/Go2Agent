"""Energy VAD and asynchronous delivery of external-microphone speech to Runtime.hear."""

from __future__ import annotations

import asyncio
import io
import logging
import math
import queue
import threading
import time
import wave
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

import httpx
import numpy as np

from go2agent.audio.speaker import AudioDevice, select_device

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class VADConfig:
    threshold: float = 0.015  # RMS of normalized float32 audio.
    start_s: float = 0.15
    silence_s: float = 0.6
    max_s: float = 10.0

    def __post_init__(self) -> None:
        values = (self.threshold, self.start_s, self.silence_s, self.max_s)
        if not all(math.isfinite(v) and v > 0 for v in values):
            raise ValueError("VAD settings must be positive and finite")
        if self.threshold > 1 or self.max_s < self.start_s:
            raise ValueError("VAD requires threshold <= 1 and max_s >= start_s")


class EnergyVAD:
    """Segment normalized mono audio in 10 ms windows, preserving the speech onset."""

    sample_rate = 16000
    window = 160

    def __init__(self, config: VADConfig | None = None) -> None:
        self.config = config or VADConfig()
        self.reset()

    def reset(self) -> None:
        self._remainder = np.empty(0, np.float32)
        self._parts: list[np.ndarray] = []
        self._samples = 0
        self._silence = 0
        self._active = False

    def feed(self, samples: np.ndarray) -> list[np.ndarray]:
        values = np.asarray(samples, dtype=np.float32).reshape(-1)
        data = np.concatenate((self._remainder, values))
        count = len(data) // self.window * self.window
        self._remainder = data[count:].copy()
        utterances: list[np.ndarray] = []
        for offset in range(0, count, self.window):
            chunk = data[offset : offset + self.window]
            loud = float(np.sqrt(np.mean(chunk * chunk))) >= self.config.threshold
            if not self._active and not loud:
                self._parts.clear()
                self._samples = 0
                continue
            self._parts.append(chunk.copy())
            self._samples += len(chunk)
            if not self._active:
                self._active = self._samples >= math.ceil(self.config.start_s * self.sample_rate)
            self._silence = 0 if loud else self._silence + len(chunk)
            ended = self._active and self._silence >= math.ceil(self.config.silence_s * self.sample_rate)
            if ended or self._samples >= math.ceil(self.config.max_s * self.sample_rate):
                utterances.append(np.concatenate(self._parts))
                self._parts.clear()
                self._samples = self._silence = 0
                self._active = False
        return utterances


def encode_wav(samples: np.ndarray) -> bytes:
    output = io.BytesIO()
    pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2")
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(pcm.tobytes())
    return output.getvalue()


class Microphone:
    """start() on the asyncio loop; await close(). Audio/HTTP never block that loop.

    PortAudio callbacks only run VAD and enqueue complete utterances. One worker
    sends WAVs to ASR, retaining at most two pending utterances. Muting discards a
    partial utterance and holds for 300 ms after the last playing observation.
    """

    def __init__(
        self,
        asr_url: str,
        on_text: Callable[[str], None],
        *,
        device: AudioDevice = None,
        muted: Callable[[], bool] = lambda: False,
        vad: VADConfig | None = None,
        language: str = "zh",
        mute_tail_s: float = 0.3,
        timeout: float = 10.0,
        client: httpx.Client | None = None,
        stream_factory: Callable[..., Any] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not math.isfinite(mute_tail_s) or mute_tail_s < 0:
            raise ValueError("mute_tail_s must be nonnegative and finite")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be positive and finite")
        self.asr_url, self.on_text = asr_url, on_text
        self.device, self.language = device, language
        self._muted, self._clock = muted, clock
        self._vad = EnergyVAD(vad)
        self._mute_tail_s, self._timeout = mute_tail_s, timeout
        self._muted_until = 0.0
        self._was_muted = False
        self._client = client
        self._stream_factory = stream_factory
        self._stream: Any | None = None
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._running = threading.Event()
        self._utterances: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=2)
        self._generation = 0

    def start(self) -> None:
        if self._running.is_set():
            return
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("Previous microphone worker is still stopping")
        self._loop = asyncio.get_running_loop()
        device = self.device
        factory = self._stream_factory
        if factory is None:
            import sounddevice as sd

            factory = sd.InputStream
            device = select_device(sd, device, "input")
        self._vad.reset()
        self._muted_until = 0.0
        self._was_muted = False
        self._utterances = queue.Queue(maxsize=2)
        self._generation += 1
        stream = factory(
            samplerate=16000,
            channels=1,
            dtype="float32",
            blocksize=320,
            device=device,
            callback=self._on_audio,
        )
        self._running.set()
        try:
            stream.start()
        except Exception:
            self._running.clear()
            stream.close()
            raise
        self._stream = stream
        self._thread = threading.Thread(target=self._recognize, name="microphone-asr", daemon=True)
        self._thread.start()

    async def close(self) -> None:
        self._running.clear()
        try:
            if self._stream is not None:
                stream, self._stream = self._stream, None
                try:
                    await asyncio.to_thread(stream.stop)
                finally:
                    await asyncio.to_thread(stream.close)
        finally:
            self._enqueue(None)
            if self._thread is not None:
                await asyncio.to_thread(self._thread.join, self._timeout + 1)
                if self._thread.is_alive():
                    raise TimeoutError("Microphone ASR request did not finish")
                self._thread = None

    def _on_audio(self, data: np.ndarray, frames: int, timing: Any, status: Any) -> None:
        if not self._running.is_set():
            return
        try:
            now = self._clock()
            if self._muted():
                self._was_muted = True
                self._vad.reset()
                return
            if self._was_muted:
                self._muted_until = now + self._mute_tail_s
                self._was_muted = False
            if now < self._muted_until or status:
                self._vad.reset()
                return
            for utterance in self._vad.feed(data):
                self._enqueue(utterance)
        except Exception:
            self._vad.reset()
            logger.exception("Microphone audio callback failed")

    def _enqueue(self, utterance: np.ndarray | None) -> None:
        while True:
            try:
                self._utterances.put_nowait(utterance)
                return
            except queue.Full:
                with suppress(queue.Empty):
                    self._utterances.get_nowait()

    def _recognize(self) -> None:
        generation = self._generation
        client = self._client or httpx.Client(timeout=self._timeout)
        try:
            while self._running.is_set():
                samples = self._utterances.get()
                if samples is None or not self._running.is_set():
                    return
                try:
                    response = client.post(
                        self.asr_url,
                        content=encode_wav(samples),
                        headers={"Content-Type": "audio/wav"},
                        params={"language": self.language},
                    )
                    response.raise_for_status()
                    text = response.json()["text"]
                    if not isinstance(text, str):
                        raise ValueError("ASR text must be a string")
                    if text.strip() and self._running.is_set() and self._loop is not None:
                        self._loop.call_soon_threadsafe(self._deliver, text.strip(), generation)
                except Exception:
                    logger.exception("Microphone ASR request failed")
        finally:
            if self._client is None:
                client.close()

    def _deliver(self, text: str, generation: int) -> None:
        if self._running.is_set() and generation == self._generation:
            self.on_text(text)
