"""Text or microphone input -> Agent -> SkillRuntime -> Go2."""

from __future__ import annotations

import argparse
import asyncio
import os

from adapters import (
    ASRError,
    AudioOutputError,
    FasterWhisperASR,
    HostSpeechOutput,
    MicrophoneASR,
    SpeechOutput,
)
from agent import AgentError, RobotAgent
from agent.service import GO2_SYSTEM_PROMPT
from core.runtime import SkillRuntime
from robot import (
    HardwareRobot,
    RobotAdapter,
    create_hardware_robot,
    create_simulated_robot,
)
from skills import register_go2_skills


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        choices=("text", "microphone"),
        default=os.getenv("GO2_INPUT_MODE", "text"),
    )
    parser.add_argument("--model", default=os.getenv("OLLAMA_MODEL", "qwen2.5:3b"))
    parser.add_argument("--ollama-url", default=os.getenv("OLLAMA_HOST"))
    parser.add_argument(
        "--network", default="", help="Unitree DDS interface, e.g. eth0"
    )
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument("--hardware", action="store_true")
    parser.add_argument("--no-audio", action="store_true", help="disable speech output")
    parser.add_argument(
        "--host-audio-device",
        default=os.getenv("GO2_AUDIO_DEVICE"),
        help="ALSA/Pulse device for Go2 external speaker TTS",
    )
    parser.add_argument("--host-tts-voice", default="cmn")
    parser.add_argument("--record-seconds", type=float, default=5.0)
    parser.add_argument("--whisper-bin", default=None)
    parser.add_argument("--whisper-model", default=None)
    parser.add_argument("--language", default=None)
    parser.add_argument("--whisper-device", default="auto")
    parser.add_argument("--whisper-compute-type", default="default")
    parser.add_argument("--audio-device", default=os.getenv("GO2_AUDIO_DEVICE", "pulse"))
    parser.add_argument("--audio-output-device", default="pulse")
    parser.add_argument("--piper-model")
    parser.add_argument("--piper-config")
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


async def _read_text() -> str:
    try:
        return (await asyncio.to_thread(input, "你: ")).strip()
    except EOFError:
        return ""


async def _run_turn(
    text: str,
    agent: RobotAgent,
    audio: SpeechOutput | None,
) -> None:
    reply = await agent.chat(text)
    print(f"机器人: {reply}")
    if audio is not None:
        await audio.speak(reply)


async def run(args: argparse.Namespace) -> None:
    hardware_robot: HardwareRobot | None = None
    audio: SpeechOutput | None = None
    robot: RobotAdapter
    robot_model = "go2"

    if args.hardware:
        hardware_robot = create_hardware_robot(
            robot_model,
            network_interface=args.network,
            domain_id=args.domain_id,
        )
        robot = hardware_robot
    else:
        robot = create_simulated_robot(robot_model)

    runtime = SkillRuntime(robot)
    register_go2_skills(runtime)
    agent = RobotAgent(
        runtime,
        model_name=args.model,
        base_url=args.ollama_url,
        system_prompt=GO2_SYSTEM_PROMPT,
    )
    microphone = None
    if args.input == "microphone":
        if args.whisper_bin:
            microphone = MicrophoneASR(
                record_seconds=args.record_seconds,
                whisper_bin=args.whisper_bin,
                model=args.whisper_model,
                language=args.language,
                audio_device=args.audio_device,
            )
        else:
            microphone = FasterWhisperASR(
                record_seconds=args.record_seconds,
                model=args.whisper_model or "small",
                language=args.language or "zh",
                audio_device=args.audio_device,
                device=args.whisper_device,
                compute_type=args.whisper_compute_type,
            )

    try:
        if hardware_robot is not None:
            await hardware_robot.connect()
        if not args.no_audio:
            audio = HostSpeechOutput(
                audio_device=args.host_audio_device or args.audio_output_device,
                piper_model=args.piper_model,
                piper_config=args.piper_config,
                fallback_voice=args.host_tts_voice,
            )
            await audio.connect()

        if microphone is None:
            print("文本模式：输入消息，输入 quit/exit 退出。")
        else:
            print("麦克风模式：每轮录音后把识别文本发送给 Agent，按 Ctrl-C 退出。")

        while True:
            if microphone is None:
                text = await _read_text()
            else:
                print("请说话...")
                if isinstance(microphone, FasterWhisperASR):
                    text = await microphone.transcribe_once()
                else:
                    text = await asyncio.to_thread(microphone.transcribe_once)
                print(f"用户: {text}")

            if not text:
                if args.once:
                    break
                continue
            if text.lower() in {"quit", "exit", "q", "退出"}:
                break

            try:
                await _run_turn(text, agent, audio)
            except (AgentError, AudioOutputError) as exc:
                print(f"错误: {exc}")
            if args.once:
                break
    except ASRError as exc:
        print(f"[asr] {exc}")
    except KeyboardInterrupt:
        print("\n已退出。")
    finally:
        if audio is not None:
            close = getattr(audio, "close", None)
            if callable(close):
                await close()
        if isinstance(microphone, FasterWhisperASR):
            await microphone.close()
        if hardware_robot is not None:
            await hardware_robot.close()


def main() -> int:
    try:
        asyncio.run(run(parse_args()))
    except (RuntimeError, AudioOutputError) as exc:
        print(f"错误: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
