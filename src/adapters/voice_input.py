"""Local microphone input adapters.

``FasterWhisperASR`` is the production path for the robot computer.  It keeps
the model resident between turns; ``MicrophoneASR`` remains as a small CLI
fallback for existing deployments.
"""

from __future__ import annotations

import asyncio
import importlib
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Protocol, cast


class ASRError(RuntimeError):
    """Raised when local audio recording or transcription is unavailable."""


class SpeechRecognizer(Protocol):
    async def warmup(self) -> None: ...

    async def transcribe_once(self) -> str: ...

    async def close(self) -> None: ...


class FasterWhisperSegment(Protocol):
    text: str


class FasterWhisperModel(Protocol):
    def transcribe(
        self, audio: str, **kwargs: object
    ) -> tuple[Iterable[FasterWhisperSegment], object]: ...


class FasterWhisperASR:
    """Record through ALSA/PulseAudio and transcribe with one resident model."""

    def __init__(
        self,
        *,
        record_seconds: float = 3.0,
        model: str = "small",
        language: str | None = "zh",
        audio_device: str = "pulse",
        device: str = "auto",
        compute_type: str = "default",
        beam_size: int = 1,
        model_factory: Callable[..., FasterWhisperModel] | None = None,
    ) -> None:
        if record_seconds < 0.5:
            raise ValueError("record_seconds must be at least 0.5")
        self.record_seconds = record_seconds
        self.model_name = model
        self.language = language
        self.audio_device = audio_device
        self.device = device
        self.compute_type = compute_type
        self.beam_size = max(1, beam_size)
        self._model_factory = model_factory
        self._model: FasterWhisperModel | None = None
        self._model_lock = asyncio.Lock()

    async def warmup(self) -> None:
        async with self._model_lock:
            if self._model is not None:
                return
            self._model = await asyncio.to_thread(self._load_model)

    async def transcribe_once(self) -> str:
        await self.warmup()
        return await asyncio.to_thread(self._record_and_transcribe)

    async def close(self) -> None:
        async with self._model_lock:
            self._model = None

    def _load_model(self) -> FasterWhisperModel:
        try:
            factory = self._model_factory
            if factory is None:
                module = importlib.import_module("faster_whisper")
                factory = cast(
                    Callable[..., FasterWhisperModel],
                    cast(object, module.WhisperModel),
                )
            return factory(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
            )
        except Exception as exc:
            raise ASRError(f"Faster Whisper 模型加载失败：{exc}") from exc

    def _record_and_transcribe(self) -> str:
        model = self._model
        if model is None:
            raise ASRError("Faster Whisper 尚未加载")
        recorder = shutil.which("arecord")
        if recorder is None:
            raise ASRError("找不到 arecord，无法从外接麦克风录音")
        with tempfile.TemporaryDirectory(prefix="go2-asr-") as temp_dir:
            wav_path = Path(temp_dir) / "input.wav"
            command = [
                recorder,
                "-q",
                "-D",
                self.audio_device,
                "-f",
                "S16_LE",
                "-r",
                "16000",
                "-c",
                "1",
                "-d",
                str(max(1, round(self.record_seconds))),
                str(wav_path),
            ]
            try:
                subprocess.run(
                    command,
                    check=True,
                    timeout=self.record_seconds + 10,
                )
            except (OSError, subprocess.SubprocessError) as exc:
                raise ASRError(f"录音失败：{exc}") from exc
            try:
                segments, _ = model.transcribe(
                    str(wav_path),
                    language=self.language,
                    beam_size=self.beam_size,
                    vad_filter=True,
                    condition_on_previous_text=False,
                )
                return "".join(
                    segment.text for segment in segments
                ).strip()
            except Exception as exc:
                raise ASRError(f"Faster Whisper 转写失败：{exc}") from exc


class MicrophoneASR:
    def __init__(
        self,
        record_seconds: float = 5.0,
        whisper_bin: str | None = None,
        model: str | None = None,
        language: str | None = None,
        audio_device: str | None = None,
    ) -> None:
        self.record_seconds = max(0.5, record_seconds)
        self.whisper_bin = whisper_bin or self._find_whisper()
        self.model = model
        self.language = language
        self.audio_device = audio_device

    @staticmethod
    def _find_whisper() -> str | None:
        for name in ("whisper-cli", "whisper", "whisper-cpp"):
            command = shutil.which(name)
            if command:
                return command
        return None

    def transcribe_once(self) -> str:
        if not self.whisper_bin:
            raise ASRError(
                "找不到 whisper 命令，请安装 Whisper 或改用 --input text。"
            )
        if self._uses_cpp() and not self.model:
            raise ASRError("whisper.cpp 需要 --whisper-model 指向 ggml 模型文件。")
        recorder = shutil.which("arecord") or shutil.which("ffmpeg")
        if not recorder:
            raise ASRError("找不到 arecord 或 ffmpeg，无法录音。")

        with tempfile.TemporaryDirectory(prefix="go2-asr-") as temp_dir:
            wav_path = Path(temp_dir) / "input.wav"
            self._record(recorder, wav_path)
            return self._transcribe(wav_path, Path(temp_dir)).strip()

    def _record(self, recorder: str, wav_path: Path) -> None:
        if Path(recorder).name == "arecord":
            command = [
                recorder,
                "-q",
                "-f",
                "S16_LE",
                "-r",
                "16000",
                "-c",
                "1",
                "-d",
                str(max(1, round(self.record_seconds))),
                str(wav_path),
            ]
            if self.audio_device:
                command[1:1] = ["-D", self.audio_device]
        else:
            command = [
                recorder,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "alsa",
                "-i",
                self.audio_device or "default",
                "-t",
                str(self.record_seconds),
                str(wav_path),
            ]
        try:
            subprocess.run(command, check=True, timeout=self.record_seconds + 10)
        except (OSError, subprocess.SubprocessError) as exc:
            raise ASRError(f"录音失败：{exc}") from exc

    def _transcribe(self, wav_path: Path, output_dir: Path) -> str:
        if self._uses_cpp():
            return self._transcribe_cpp(wav_path, output_dir)

        command = [
            self.whisper_bin or "whisper",
            str(wav_path),
            "--output_dir",
            str(output_dir),
            "--output_format",
            "txt",
            "--fp16",
            "False",
        ]
        if self.model:
            command.extend(["--model", self.model])
        if self.language:
            command.extend(["--language", self.language])
        try:
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=max(60.0, self.record_seconds * 20),
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise ASRError(f"Whisper 转写失败：{exc}") from exc

        text_path = output_dir / f"{wav_path.stem}.txt"
        if text_path.exists():
            return text_path.read_text(encoding="utf-8")
        return result.stdout.strip()

    def _transcribe_cpp(self, wav_path: Path, output_dir: Path) -> str:
        if not self.model:
            raise ASRError("whisper.cpp 需要 --whisper-model 指向 ggml 模型文件。")
        output_base = output_dir / "transcript"
        command = [
            self.whisper_bin or "whisper-cli",
            "-m",
            self.model,
            "-f",
            str(wav_path),
            "-otxt",
            "-of",
            str(output_base),
        ]
        if self.language:
            command.extend(["-l", self.language])
        try:
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=max(60.0, self.record_seconds * 20),
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise ASRError(f"whisper.cpp 转写失败：{exc}") from exc

        text_path = output_base.with_suffix(".txt")
        if text_path.exists():
            return text_path.read_text(encoding="utf-8")
        return result.stdout.strip()

    def _uses_cpp(self) -> bool:
        return Path(self.whisper_bin or "").name in {"whisper-cli", "whisper-cpp"}
