"""Speech output through the robot computer's external loudspeaker."""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Protocol


class AudioOutputError(RuntimeError):
    """Raised when host TTS initialization or playback fails."""


class SpeechOutput(Protocol):
    async def speak(self, text: str) -> None: ...


class HostSpeechOutput:
    """Synthesize locally with Piper, falling back to ``espeak-ng``.

    The optional shared lock also guards microphone recording, providing a
    simple half-duplex boundary that prevents the robot from transcribing its
    own speech.
    """

    def __init__(
        self,
        *,
        audio_device: str = "pulse",
        piper_model: str | None = None,
        piper_config: str | None = None,
        fallback_voice: str = "cmn",
        audio_lock: asyncio.Lock | None = None,
    ) -> None:
        self.audio_device = audio_device
        self.piper_model = piper_model
        self.piper_config = piper_config
        self.fallback_voice = fallback_voice
        self._audio_lock = audio_lock or asyncio.Lock()
        venv_piper = Path(sys.executable).with_name("piper")
        self._piper_bin = (
            str(venv_piper)
            if venv_piper.is_file()
            else shutil.which("piper")
        )

    @property
    def engine(self) -> str:
        if self.piper_model and self._piper_bin:
            return "piper"
        return "espeak-ng"

    async def connect(self) -> None:
        if shutil.which("aplay") is None:
            raise AudioOutputError("找不到 aplay，无法使用外接扬声器")
        if self.engine == "espeak-ng" and shutil.which("espeak-ng") is None:
            raise AudioOutputError(
                "找不到 Piper 模型或 espeak-ng，无法启用本地 TTS"
            )

    async def close(self) -> None:
        return None

    async def speak(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        async with self._audio_lock:
            with tempfile.TemporaryDirectory(prefix="go2-tts-") as temp_dir:
                wav_path = Path(temp_dir) / "speech.wav"
                if self.engine == "piper":
                    command = [
                        self._piper_bin or "piper",
                        "--model",
                        self.piper_model or "",
                        "--output_file",
                        str(wav_path),
                    ]
                    if self.piper_config:
                        command.extend(("--config", self.piper_config))
                    await self._run_command(command, timeout_s=120, input_text=text)
                else:
                    await self._run_command(
                        [
                            "espeak-ng", "-v", self.fallback_voice, "-s", "165",
                            "-w", str(wav_path), text,
                        ],
                        timeout_s=60,
                    )
                await self._run_command(
                    ["aplay", "-q", "-D", self.audio_device, str(wav_path)],
                    timeout_s=120,
                )

    @staticmethod
    async def _run_command(
        command: list[str], *, timeout_s: float, input_text: str | None = None
    ) -> None:
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdin=subprocess.PIPE if input_text is not None else subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except OSError as exc:
            raise AudioOutputError(f"本地 TTS 启动失败：{exc}") from exc
        try:
            _, stderr = await asyncio.wait_for(
                process.communicate(
                    input_text.encode("utf-8") if input_text is not None else None
                ),
                timeout=timeout_s,
            )
        except (asyncio.CancelledError, TimeoutError) as exc:
            if process.returncode is None:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=2.0)
                except TimeoutError:
                    process.kill()
                    await process.wait()
            if isinstance(exc, TimeoutError):
                raise AudioOutputError(
                    f"本地 TTS 命令超时：{command[0]}"
                ) from exc
            raise
        if process.returncode != 0:
            detail = stderr.decode("utf-8", errors="replace")[:300]
            raise AudioOutputError(
                f"本地 TTS 命令失败 ({process.returncode}): {detail}"
            )
