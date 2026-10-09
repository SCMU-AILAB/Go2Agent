"""Stable request, result, and metadata contracts for robot skills."""

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict

from .types import FailureCode, SkillStatus


class SkillArgs(BaseModel):
    """Base schema for every skill's validated arguments."""

    model_config = ConfigDict(extra="forbid")


@dataclass(frozen=True, slots=True)
class SkillMetadata:
    name: str
    description: str

    version: str = "1.0"

    tags: tuple[str, ...] = ()

    required_resources: tuple[str, ...] = ()

    behavior: Literal["bounded", "persistent"] = "bounded"

    timeout_s: float = 30.0

    max_retries: int = 0

    interruptible: bool = True

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("skill name must not be empty")
        if self.timeout_s <= 0:
            raise ValueError("skill timeout must be greater than zero")
        if self.max_retries < 0:
            raise ValueError("skill max_retries must not be negative")


@dataclass(slots=True)
class SkillInvocation:
    skill_name: str
    arguments: dict[str, object] = field(default_factory=dict)

    request_id: str | None = None
    source: str | None = None

    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class SkillResult:
    success: bool
    status: SkillStatus

    message: str = ""

    failure_code: FailureCode | None = None

    data: dict[str, object] = field(default_factory=dict)

    verification: dict[str, object] = field(default_factory=dict)

    recoverable: bool = False

    duration_s: float | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "success": self.success,
            "status": self.status.value,
            "message": self.message,
            "data": self.data,
            "verification": self.verification,
            "recoverable": self.recoverable,
            "duration_s": self.duration_s,
        }
        if self.failure_code is not None:
            payload["failure_code"] = self.failure_code.value
        return payload

    @classmethod
    def ok(
        cls,
        message: str = "completed",
        **data: object,
    ) -> "SkillResult":
        return cls(
            success=True,
            status=SkillStatus.SUCCEEDED,
            message=message,
            data=data,
        )

    @classmethod
    def fail(
        cls,
        status: SkillStatus,
        message: str,
        *,
        failure_code: FailureCode = FailureCode.UNKNOWN,
        recoverable: bool = False,
        **data: object,
    ) -> "SkillResult":
        return cls(
            success=False,
            status=status,
            message=message,
            failure_code=failure_code,
            recoverable=recoverable,
            data=data,
        )
