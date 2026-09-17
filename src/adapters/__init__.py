"""External Agent and voice adapters."""

from .host_speaker import HostSpeakerError, HostSpeakerOutput
from .langchain import build_langchain_tools
from .unitree_audio import (
    AudioClientApi,
    AudioOutputError,
    SpeechOutput,
    UnitreeAudioBindings,
    UnitreeAudioOutput,
)
from .voice_input import ASRError, MicrophoneASR

__all__ = [
    "ASRError",
    "AudioClientApi",
    "AudioOutputError",
    "HostSpeakerError",
    "HostSpeakerOutput",
    "MicrophoneASR",
    "SpeechOutput",
    "UnitreeAudioBindings",
    "UnitreeAudioOutput",
    "build_langchain_tools",
]
