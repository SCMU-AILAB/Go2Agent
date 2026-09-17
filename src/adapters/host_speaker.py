"""Speak agent replies on the host audio output (external speaker).

Go2 has no Unitree AudioClient TTS. This adapter uses system TTS + ALSA/Pulse
so an external speaker plugged into the Jetson (or a USB sound card) can play
replies without using the robot body speaker APIs.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import tempfile
from pathlib import Path


class HostSpeakerError(RuntimeError):
    """Raised when host TTS or playback fails."""


class HostSpeakerOutput:
    """Play Chinese/English text via espeak-ng (or espeak) on the host."""

    def __init__(
        self,
        *,
        voice: str = "cmn",
        speed_wpm: int = 160,
        audio_device: str | None = None,
        tts_bin: str | None = None,
    ) -> None:
        self.voice = voice
        self.speed_wpm = max(80, min(300, speed_wpm))
        self.audio_device = audio_device
        self._tts_bin = tts_bin

    @property
    def available(self) -> bool:
        return self._find_tts() is not None

    def _find_tts(self) -> str | None:
        if self._tts_bin:
            return self._tts_bin if shutil.which(self._tts_bin) else None
        for name in ("espeak-ng", "espeak"):
            if shutil.which(name):
                return name
        return None

    def _find_player(self) -> list[str] | None:
        if shutil.which("paplay"):
            return ["paplay"]
        if shutil.which("aplay"):
            command = ["aplay", "-q"]
            if self.audio_device:
                command.extend(["-D", self.audio_device])
            return command
        return None

    async def speak(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        await asyncio.to_thread(self._speak_sync, text)

    def _speak_sync(self, text: str) -> None:
        tts = self._find_tts()
        if not tts:
            raise HostSpeakerError(
                "找不到 espeak-ng/espeak。请安装：sudo apt install espeak-ng"
            )
        player = self._find_player()
        if not player:
            raise HostSpeakerError("找不到 paplay 或 aplay，无法播放到外接扬声器。")

        with tempfile.TemporaryDirectory(prefix="go2-tts-") as temp_dir:
            wav_path = Path(temp_dir) / "reply.wav"
            # espeak-ng writes WAV; Chinese voice name is cmn on recent packages.
            synth = [
                tts,
                "-v",
                self.voice,
                "-s",
                str(self.speed_wpm),
                "-w",
                str(wav_path),
                text,
            ]
            try:
                subprocess.run(
                    synth,
                    check=True,
                    capture_output=True,
                    timeout=max(15.0, len(text) * 0.15),
                )
            except (OSError, subprocess.SubprocessError) as exc:
                detail = ""
                if isinstance(exc, subprocess.CalledProcessError):
                    detail = (exc.stderr or b"").decode("utf-8", "replace")[:200]
                raise HostSpeakerError(f"主机 TTS 合成失败：{exc} {detail}") from exc
            if not wav_path.exists() or wav_path.stat().st_size == 0:
                raise HostSpeakerError("主机 TTS 未生成音频文件。")
            try:
                subprocess.run(
                    [*player, str(wav_path)],
                    check=True,
                    capture_output=True,
                    timeout=max(20.0, len(text) * 0.2),
                )
            except (OSError, subprocess.SubprocessError) as exc:
                raise HostSpeakerError(f"外接扬声器播放失败：{exc}") from exc

    async def close(self) -> None:
        return None


__all__ = ["HostSpeakerError", "HostSpeakerOutput"]
