"""Text or microphone input -> Agent -> SkillRuntime -> G1, then AudioClient TTS."""

from __future__ import annotations

import argparse
import asyncio
import os

from adapters import (
    ASRError,
    AudioOutputError,
    MicrophoneASR,
    SpeechOutput,
    UnitreeAudioOutput,
)
from agent import AgentError, RobotAgent
from agent.service import system_prompt_for
from core.runtime import SkillRuntime
from robot import (
    ROBOT_MODELS,
    HardwareRobot,
    RobotAdapter,
    RobotModel,
    create_hardware_robot,
    create_simulated_robot,
)
from skills import register_g1_skills, register_go2_skills


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        choices=("text", "microphone"),
        default=os.getenv("G1_INPUT_MODE", "text"),
    )
    parser.add_argument("--model", default=os.getenv("OLLAMA_MODEL", "qwen2.5:3b"))
    parser.add_argument("--ollama-url", default=os.getenv("OLLAMA_HOST"))
    parser.add_argument("--network", default="", help="Unitree DDS interface, e.g. eth0")
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument("--hardware", action="store_true")
    parser.add_argument(
        "--robot",
        choices=ROBOT_MODELS,
        default=os.getenv("G1_ROBOT_MODEL", "g1"),
        help="robot model to assemble; go2 uses SportClient and a reduced skill catalog",
    )
    parser.add_argument(
        "--include-operator-only-skills",
        action="store_true",
        help="register low-level and dangerous SDK controls for explicit operator use",
    )
    parser.add_argument("--no-audio", action="store_true", help="disable G1 AudioClient TTS")
    parser.add_argument("--speaker-id", type=int, default=0)
    parser.add_argument("--record-seconds", type=float, default=5.0)
    parser.add_argument("--whisper-bin", default=None)
    parser.add_argument("--whisper-model", default=None)
    parser.add_argument("--language", default=None)
    parser.add_argument("--audio-device", default=os.getenv("G1_AUDIO_DEVICE"))
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
    audio: UnitreeAudioOutput | None = None
    robot: RobotAdapter
    robot_model: RobotModel = args.robot

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
    include_operator_only = getattr(args, "include_operator_only_skills", False)
    if robot_model == "go2":
        register_go2_skills(runtime, include_operator_only=include_operator_only)
    else:
        register_g1_skills(runtime, include_operator_only=include_operator_only)
    agent = RobotAgent(
        runtime,
        model_name=args.model,
        base_url=args.ollama_url,
        system_prompt=system_prompt_for(robot_model),
    )
    microphone = (
        MicrophoneASR(
            record_seconds=args.record_seconds,
            whisper_bin=args.whisper_bin,
            model=args.whisper_model,
            language=args.language,
            audio_device=args.audio_device,
        )
        if args.input == "microphone"
        else None
    )

    try:
        if hardware_robot is not None:
            await hardware_robot.connect()
            if robot_model == "go2":
                if not args.no_audio:
                    print("[audio] Go2 does not use G1 AudioClient TTS; audio disabled.")
            elif not args.no_audio:
                audio = UnitreeAudioOutput(
                    hardware_robot,
                    speaker_id=args.speaker_id,
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
            await audio.close()
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
