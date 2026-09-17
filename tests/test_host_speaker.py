import asyncio
import unittest

from adapters.host_speaker import HostSpeakerError, HostSpeakerOutput


class HostSpeakerTests(unittest.TestCase):
    def test_available_false_without_tts(self):
        speaker = HostSpeakerOutput(tts_bin="definitely-not-a-tts-binary")
        self.assertFalse(speaker.available)

    def test_empty_text_is_noop(self):
        speaker = HostSpeakerOutput(tts_bin="definitely-not-a-tts-binary")
        asyncio.run(speaker.speak("   "))

    def test_speak_raises_when_tts_missing(self):
        speaker = HostSpeakerOutput(tts_bin="definitely-not-a-tts-binary")
        with self.assertRaises(HostSpeakerError):
            asyncio.run(speaker.speak("你好"))


if __name__ == "__main__":
    unittest.main()
