"""后端接口.

一个后端负责"游戏"这件事本身: 怎么开始新一局、怎么推进帧、怎么读状态、
怎么截图。奖励/统计/工具协议都在 :mod:`touhou_rl_mcp.env` 与
:mod:`touhou_rl_mcp.server` 里, 与后端无关。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from PIL import Image

from ..games.registry import GameSpec
from ..inputs.actions import Action
from ..inputs.controller import Controller
from ..types import GameState


class Backend(ABC):
    """后端基类."""

    name: str = "base"

    def __init__(self, spec: GameSpec, config: Any) -> None:
        self.spec = spec
        self.config = config
        self.controller: Controller | None = None
        self._closed = False

    # ------------------------------------------------------------------
    @abstractmethod
    def start(self) -> None:
        """连接/启动游戏 (幂等)."""

    @abstractmethod
    def reset(self, *, options: dict[str, Any] | None = None) -> GameState:
        """开始新一局, 返回初始状态."""

    @abstractmethod
    def read_state(self) -> GameState:
        """读数 (不推进时间)."""

    @abstractmethod
    def advance(self, frames: int) -> int:
        """推进 ``frames`` 帧 (按键状态由 controller 决定), 返回实际推进帧数."""

    @abstractmethod
    def screenshot(self) -> tuple[Image.Image, str]:
        """返回 ``(画面, 抓取方式)``."""

    def close(self) -> None:
        self._closed = True

    # ------------------------------------------------------------------
    @property
    def alive(self) -> bool:
        return not self._closed

    def status(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "game": self.spec.key,
            "alive": self.alive,
            "controller_ready": self.controller is not None,
        }

    def apply(self, action: Action) -> dict[str, Any]:
        if self.controller is None:
            raise RuntimeError("后端尚未 start()")
        report = self.controller.run(action)
        return report.to_dict()
