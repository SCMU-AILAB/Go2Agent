"""External Agent and voice adapters."""

from .host_audio import AudioOutputError, HostSpeechOutput, SpeechOutput
from .host_speaker import HostSpeakerError, HostSpeakerOutput
from .langchain import build_langchain_tools
from .voice_input import ASRError, FasterWhisperASR, MicrophoneASR, SpeechRecognizer

__all__ = [
    "ASRError",
    "AudioOutputError",
    "FasterWhisperASR",
    "HostSpeakerError",
    "HostSpeakerOutput",
    "HostSpeechOutput",
    "MicrophoneASR",
    "SpeechOutput",
    "SpeechRecognizer",
    "build_langchain_tools",
]
