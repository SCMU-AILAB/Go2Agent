"""External Agent and voice adapters."""

from .host_audio import HostSpeechOutput
from .host_speaker import HostSpeakerError, HostSpeakerOutput
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
    "HostSpeakerError",
    "HostSpeakerOutput",
    "HostSpeechOutput",
    "MicrophoneASR",
    "SpeechOutput",
    "SpeechRecognizer",
    "UnitreeAudioBindings",
    "UnitreeAudioOutput",
    "build_langchain_tools",
]
