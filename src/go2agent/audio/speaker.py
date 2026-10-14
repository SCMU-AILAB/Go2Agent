"""Bounded speech queue and host playback through an external sound device."""

from __future__ import annotations

import asyncio
import io
import logging
import math
import threading
import wave
from collections.abc import Callable
from contextlib import suppress
from typing import Any

import httpx

logger = logging.getLogger(__name__)
AudioDevice = str | int | None
Player = Callable[[bytes, AudioDevice, threading.Event], None]


def select_device(sounddevice: Any, device: AudioDevice, kind: str) -> AudioDevice:
    """Accept PortAudio names/indices; prefer USB over onboard HDMI by default."""
    if isinstance(device, str) and device.isdecimal():
        return int(device)
    if device is not None:
        return device
    for index, info in enumerate(sounddevice.query_devices()):
        if info[f"max_{kind}_channels"] > 0 and "usb" in info["name"].lower():
            return index
    return None


def play_wav(audio: bytes, device: AudioDevice, stopped: threading.Event) -> None:
    """Write short PCM blocks so close can interrupt playback without a global sd.stop()."""
    import sounddevice as sd

    with wave.open(io.BytesIO(audio), "rb") as wav:
        if wav.getsampwidth() != 2 or wav.getcomptype() != "NONE":
            raise ValueError("TTS must return a 16-bit PCM WAV")
        with sd.RawOutputStream(
            samplerate=wav.getframerate(),
            channels=wav.getnchannels(),
            dtype="int16",
            device=select_device(sd, device, "output"),
            blocksize=1024,
        ) as stream:
            while not stopped.is_set():
                block = wav.readframes(1024)
                if not block:
                    break
                stream.write(block)


class Speaker:
    """say() waits for playback; at most two pending utterances, dropping the oldest.

    A dropped utterance completes normally. Pass say to Runtime(speaker=...), and
    pass lambda: speaker.playing to Microphone(muted=...). close() is async.
    """

    def __init__(
        self,
        tts_url: str,
        *,
        device: AudioDevice = None,
        player: Player = play_wav,
        client: httpx.AsyncClient | None = None,
        timeout: float = 20.0,
    ) -> None:
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be positive and finite")
        self.tts_url = tts_url
        self.device = device
        self._player = player
        self._client = client
        self._owns_client = client is None
        self._timeout = timeout
        self._queue: asyncio.Queue[tuple[str, asyncio.Future[None]]] = asyncio.Queue(maxsize=2)
        self._task: asyncio.Task[None] | None = None
        self._playing = threading.Event()
        self._stop_playback = threading.Event()
        self._closed = False

    @property
    def playing(self) -> bool:
        return self._playing.is_set()

    async def say(self, text: str) -> None:
        if self._closed:
            raise RuntimeError("Speaker is closed")
        text = text.strip()
        if not text:
            return
        future = asyncio.get_running_loop().create_future()
        if self._queue.full():
            _, dropped = self._queue.get_nowait()
            self._queue.task_done()
            if not dropped.done():
                dropped.set_result(None)
            logger.debug("Dropped oldest pending speech")
        self._queue.put_nowait((text, future))
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="speaker")
        await future

    async def close(self) -> None:
        self._closed = True
        self._stop_playback.set()
        while not self._queue.empty():
            _, future = self._queue.get_nowait()
            self._queue.task_done()
            future.cancel()
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    async def _run(self) -> None:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        while True:
            text, future = await self._queue.get()
            try:
                if future.cancelled():
                    continue
                response = await self._client.post(self.tts_url, json={"text": text})
                response.raise_for_status()
                self._stop_playback.clear()
                self._playing.set()
                playback = asyncio.create_task(
                    asyncio.to_thread(self._player, response.content, self.device, self._stop_playback)
                )
                try:
                    await asyncio.shield(playback)
                except asyncio.CancelledError:
                    self._stop_playback.set()
                    # Cancelling to_thread does not stop the player. Drain it before unmuting.
                    with suppress(Exception):
                        await playback
                    raise
                finally:
                    self._playing.clear()
                if not future.done():
                    future.set_result(None)
            except asyncio.CancelledError:
                future.cancel()
                raise
            except Exception as exc:
                logger.exception("Speech failed")
                if not future.done():
                    future.set_exception(exc)
            finally:
                self._queue.task_done()
