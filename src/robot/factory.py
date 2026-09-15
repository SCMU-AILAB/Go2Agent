"""Build hardware robot adapters for the configured robot model."""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from .go2_adapter import UnitreeGo2Adapter, UnitreeGo2Config
from .simulated_adapter import SimulatedRobotAdapter
from .unitree_adapter import UnitreeG1Adapter, UnitreeG1Config

RobotModel = Literal["g1", "go2"]
ROBOT_MODELS: tuple[RobotModel, ...] = ("g1", "go2")


@runtime_checkable
class HardwareRobot(Protocol):
    """Shared lifecycle surface used by CLI, API, and perception entry points."""

    @property
    def connected(self) -> bool: ...

    async def connect(self) -> None: ...

    async def close(self) -> None: ...


def create_hardware_robot(
    model: RobotModel,
    *,
    network_interface: str = "",
    domain_id: int = 0,
) -> HardwareRobot:
    if model == "go2":
        return UnitreeGo2Adapter(
            UnitreeGo2Config(
                network_interface=network_interface,
                domain_id=domain_id,
            )
        )
    if model == "g1":
        return UnitreeG1Adapter(
            UnitreeG1Config(
                network_interface=network_interface,
                domain_id=domain_id,
            )
        )
    raise ValueError(f"unsupported robot model: {model}")


def create_simulated_robot(model: RobotModel) -> SimulatedRobotAdapter:
    if model not in ROBOT_MODELS:
        raise ValueError(f"unsupported robot model: {model}")
    return SimulatedRobotAdapter()


__all__ = [
    "ROBOT_MODELS",
    "HardwareRobot",
    "RobotModel",
    "create_hardware_robot",
    "create_simulated_robot",
]
