"""Deterministic offline voice command Agent for field operation.

This Agent intentionally does not provide open-ended conversation.  It maps a
small set of explicit Mandarin/English commands to safe registered skills, so
Go2 voice control remains usable when the visual GPU is reserved for UnifoLM.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from pydantic import ValidationError

from adapters.langchain import SkillToolObserver
from core.runtime import SkillRuntime

from .service import AgentError


@dataclass(frozen=True, slots=True)
class _VoiceCommand:
    skill: str
    arguments: dict[str, object]
    reply: str


class LocalVoiceCommandAgent:
    """Execute explicit safe Go2 commands without a remote text model."""

    def __init__(
        self,
        runtime: SkillRuntime,
        *,
        tool_observer: SkillToolObserver | None = None,
    ) -> None:
        self.runtime = runtime
        self.tool_observer = tool_observer

    async def chat(self, text: str) -> str:
        text = text.strip()
        if not text:
            raise ValueError("agent input must not be empty")
        command = self._match(text)
        if command is None:
            return (
                "我没有识别到明确动作。你可以说比心、打招呼、坐下、站起来、"
                "跳舞、前进、后退、左转、右转或停止。"
            )
        try:
            skill = self.runtime.registry.get(command.skill)
            validated = skill.args_model.model_validate(command.arguments)
        except KeyError as exc:
            raise AgentError(f"语音动作当前不可用：{command.skill}") from exc
        except ValidationError as exc:
            raise AgentError(f"语音动作参数无效：{exc}") from exc
        arguments = validated.model_dump()
        if self.tool_observer is not None:
            await self.tool_observer.before_skill(command.skill, arguments)
        result = await self.runtime.execute(command.skill, **arguments)
        payload = result.to_dict()
        if self.tool_observer is not None:
            await self.tool_observer.after_skill(command.skill, arguments, payload)
        if not result.success:
            return f"动作执行失败：{result.message or '未知错误'}"
        return command.reply

    def reset(self) -> None:
        return None

    @classmethod
    def _match(cls, text: str) -> _VoiceCommand | None:
        normalized = cls._normalize(text)

        # Safety commands always win over any other word in the utterance.
        if cls._has(normalized, "急停", "停止", "停下", "别动", "stop"):
            return _VoiceCommand("stop", {}, "好的，已经停止。")
        if cls._has(normalized, "比心", "比个心", "爱心", "送心", "heart"):
            return _VoiceCommand("heart", {}, "好的，比心指令已发送。")
        if cls._has(normalized, "伸懒腰", "伸个懒腰", "拉伸", "stretch"):
            return _VoiceCommand("stretch", {}, "好的，伸懒腰指令已发送。")
        if cls._has(normalized, "趴下", "躺下", "休息", "liedown"):
            return _VoiceCommand("stand_down", {}, "好的，趴下指令已发送。")
        if cls._has(normalized, "坐下", "坐下来", "坐好", "sit"):
            return _VoiceCommand("sit", {}, "好的，坐下指令已发送。")
        if cls._has(normalized, "站起来", "起身", "起立", "standup"):
            return _VoiceCommand("stand_up", {}, "好的，站立指令已发送。")
        if cls._has(normalized, "随机跳舞", "随机舞蹈", "randomdance"):
            return _VoiceCommand("random_dance", {}, "好的，随机舞蹈指令已发送。")
        if cls._has(normalized, "舞蹈二", "第二支舞", "dance2"):
            return _VoiceCommand("dance2", {}, "好的，第二支舞指令已发送。")
        if cls._has(normalized, "跳舞", "跳个舞", "舞蹈", "dance"):
            return _VoiceCommand("dance1", {}, "好的，舞蹈指令已发送。")
        if cls._has(normalized, "向左走", "往左走", "左移", "moveleft"):
            return _VoiceCommand(
                "move_left", {"distance_m": 0.2}, "好的，已向左移动。"
            )
        if cls._has(normalized, "向右走", "往右走", "右移", "moveright"):
            return _VoiceCommand(
                "move_right", {"distance_m": 0.2}, "好的，已向右移动。"
            )
        if cls._has(normalized, "左转", "向左转", "turnleft"):
            return _VoiceCommand(
                "turn_left", {"angle_deg": 15.0}, "好的，已向左转。"
            )
        if cls._has(normalized, "右转", "向右转", "turnright"):
            return _VoiceCommand(
                "turn_right", {"angle_deg": 15.0}, "好的，已向右转。"
            )
        if cls._has(normalized, "后退", "向后走", "往后走", "backward"):
            return _VoiceCommand(
                "move_backward", {"distance_m": 0.2}, "好的，已后退。"
            )
        if cls._has(normalized, "前进", "向前走", "往前走", "forward"):
            return _VoiceCommand(
                "move_forward", {"distance_m": 0.2}, "好的，已向前移动。"
            )
        if cls._has(normalized, "打招呼", "打个招呼", "问好", "挥手", "hello"):
            return _VoiceCommand("hello", {}, "你好，很高兴见到你。")
        return None

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"[\s，。！？、,.!?;；:：]+", "", text).lower()

    @staticmethod
    def _has(text: str, *markers: str) -> bool:
        return any(marker in text for marker in markers)


__all__ = ["LocalVoiceCommandAgent"]
