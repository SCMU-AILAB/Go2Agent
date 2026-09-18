from __future__ import annotations

import asyncio
import unittest
from collections.abc import Iterable
from pathlib import Path
from unittest.mock import patch

from adapters import FasterWhisperASR, HostSpeechOutput


class _Segment:
    def __init__(self, text: str) -> None:
        self.text = text


class _Model:
    def transcribe(
        self, audio: str, **kwargs: object
    ) -> tuple[Iterable[_Segment], object]:
        del audio, kwargs
        return iter([_Segment(" 给我"), _Segment("比个心 ")]), object()


class LocalVoiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_faster_whisper_model_is_loaded_once_and_reused(self) -> None:
        loads = 0

        def factory(*args: object, **kwargs: object) -> _Model:
            nonlocal loads
            del args, kwargs
            loads += 1
            return _Model()

        recognizer = FasterWhisperASR(
            record_seconds=0.5,
            model_factory=factory,
        )

        def fake_run(command: list[str], **kwargs: object) -> object:
            del kwargs
            Path(command[-1]).write_bytes(b"RIFF")
            return object()

        with (
            patch("adapters.voice_input.shutil.which", return_value="/usr/bin/arecord"),
            patch("adapters.voice_input.subprocess.run", side_effect=fake_run),
        ):
            first = await recognizer.transcribe_once()
            second = await recognizer.transcribe_once()

        self.assertEqual(first, "给我比个心")
        self.assertEqual(second, "给我比个心")
        self.assertEqual(loads, 1)

    async def test_host_speech_uses_shared_half_duplex_lock(self) -> None:
        lock = asyncio.Lock()
        output = HostSpeechOutput(audio_lock=lock)
        calls: list[list[str]] = []

        def fake_run(command: list[str], **kwargs: object) -> object:
            del kwargs
            calls.append(command)
            if "-w" in command:
                Path(command[command.index("-w") + 1]).write_bytes(b"RIFF")
            return object()

        with (
            patch("adapters.host_audio.shutil.which", side_effect=lambda name: name),
            patch("adapters.host_audio.subprocess.run", side_effect=fake_run),
        ):
            await output.connect()
            await lock.acquire()
            task = asyncio.create_task(output.speak("你好"))
            await asyncio.sleep(0)
            self.assertFalse(task.done())
            lock.release()
            await task

        self.assertTrue(any(command[0] == "espeak-ng" for command in calls))
        self.assertTrue(any(command[0] == "aplay" for command in calls))


if __name__ == "__main__":
    unittest.main()
