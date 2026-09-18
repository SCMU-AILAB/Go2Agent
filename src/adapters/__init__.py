"""External Agent and voice adapters."""

from .host_audio import HostSpeechOutput
from .langchain import build_langchain_tools
from .unitree_audio import (
    AudioClientApi,
    AudioOutputError,
    SpeechOutput,
    UnitreeAudioBindings,
    UnitreeAudioOutput,
)
from .voice_input import ASRError, FasterWhisperASR, MicrophoneASR, SpeechRecognizer

__all__ = [
    "ASRError",
    "AudioClientApi",
    "AudioOutputError",
    "FasterWhisperASR",
    "HostSpeechOutput",
    "MicrophoneASR",
    "SpeechOutput",
    "SpeechRecognizer",
    "UnitreeAudioBindings",
    "UnitreeAudioOutput",
    "build_langchain_tools",
]
