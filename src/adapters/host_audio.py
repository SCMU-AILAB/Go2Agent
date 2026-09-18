"""Speech output through the robot computer's external loudspeaker."""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from .unitree_audio import AudioOutputError


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
            await asyncio.to_thread(self._speak_sync, text)

    def _speak_sync(self, text: str) -> None:
        with tempfile.TemporaryDirectory(prefix="go2-tts-") as temp_dir:
            wav_path = Path(temp_dir) / "speech.wav"
            try:
                if self.engine == "piper":
                    self._run_piper(text, wav_path)
                else:
                    self._run_espeak(text, wav_path)
                subprocess.run(
                    ["aplay", "-q", "-D", self.audio_device, str(wav_path)],
                    check=True,
                    timeout=120,
                )
            except (OSError, subprocess.SubprocessError) as exc:
                raise AudioOutputError(f"本地 TTS 播放失败：{exc}") from exc

    def _run_piper(self, text: str, wav_path: Path) -> None:
        command = [
            self._piper_bin or "piper",
            "--model",
            self.piper_model or "",
            "--output_file",
            str(wav_path),
        ]
        if self.piper_config:
            command.extend(["--config", self.piper_config])
        subprocess.run(
            command,
            input=text,
            text=True,
            check=True,
            capture_output=True,
            timeout=120,
        )

    def _run_espeak(self, text: str, wav_path: Path) -> None:
        subprocess.run(
            [
                "espeak-ng",
                "-v",
                self.fallback_voice,
                "-s",
                "165",
                "-w",
                str(wav_path),
                text,
            ],
            check=True,
            capture_output=True,
            timeout=60,
        )
