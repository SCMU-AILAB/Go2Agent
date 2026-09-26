from __future__ import annotations

import asyncio
import sys
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
    async def test_cancelled_speech_terminates_playback_process(self) -> None:
        output = HostSpeechOutput()
        started = asyncio.Event()
        original = asyncio.create_subprocess_exec
        processes: list[asyncio.subprocess.Process] = []

        async def launch(*command: str, **kwargs: object) -> asyncio.subprocess.Process:
            del command
            process = await original(
                sys.executable, "-c", "import time; time.sleep(60)", **kwargs
            )
            processes.append(process)
            started.set()
            return process

        with patch("adapters.host_audio.asyncio.create_subprocess_exec", side_effect=launch):
            task = asyncio.create_task(output.speak("hello"))
            await asyncio.wait_for(started.wait(), timeout=2.0)
            task.cancel()
            await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 2.0)
            self.assertTrue(output._audio_lock.locked() is False)
            self.assertIsNotNone(processes[0].returncode)

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

        class FakeProcess:
            returncode = 0

            async def communicate(self, input: bytes | None = None) -> tuple[bytes, bytes]:
                del input
                return b"", b""

        async def fake_exec(*command: str, **kwargs: object) -> FakeProcess:
            del kwargs
            calls.append(list(command))
            if "-w" in command:
                Path(command[command.index("-w") + 1]).write_bytes(b"RIFF")
            return FakeProcess()

        with (
            patch("adapters.host_audio.shutil.which", side_effect=lambda name: name),
            patch("adapters.host_audio.asyncio.create_subprocess_exec", side_effect=fake_exec),
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
