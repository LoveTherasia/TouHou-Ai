"""缺口老师: 把几何标签接进 :class:`~touhou_rl_mcp.training.trainer.Trainer`。

训练器原本的老师只有两种来源 —— 手写战术规则 (``guard``) 和规则瞄准 (``rule``)。
它们的共同问题是"带着一套人工代价函数", 而缺口任务上真正需要的只是**几何测量**:
"哪条车道还有余量的地方"。这里把 :func:`gap_label` 包成同一个协议, 于是
BC / DAgger / 纯 RL 都能用同一套训练循环, 不再需要 ``tools/research`` 里那套平行实现。

老师的输出是一串动作名 (如 ``"left+shoot"``), 由训练器用 ``_teacher_index`` 映射回
自己的动作表下标 —— 与 ``TacticalGuard`` 完全一致。
"""

from __future__ import annotations

from typing import Any

from ..types import GameState
from .gap_label import MACRO_ACTIONS, gap_label
from .occupancy import GapSpec


class GapTeacher:
    """按缺口几何打标签的老师 (无参数, 无代价函数, 无状态)。"""

    #: 名字 -> 下标 (与 DIRECTIONS 同序), 供训练器反查
    actions: tuple[str, ...] = MACRO_ACTIONS

    def __init__(self, spec: GapSpec | None = None, mode: str = "reach") -> None:
        self.spec = spec or GapSpec()
        self.mode = mode

    def choose(self, state: GameState, frame: int | None,
               learned_action: str = "") -> tuple[str, dict[str, Any]]:
        """返回 ``(动作名, 诊断信息)``; 与 ``TacticalGuard.choose`` 同签名。"""
        if state.player_x is None or state.player_y is None:
            raise ValueError("缺口标签需要有效的自机坐标")
        index = gap_label(state, self.spec, self.mode)
        return MACRO_ACTIONS[index], {
            "gap_label": index,
            "gap_mode": self.mode,
            "frame": frame,
            "learned_action": learned_action,
        }

    def describe(self) -> dict[str, Any]:
        return {"teacher": "gap", "mode": self.mode, **self.spec.__dict__}


__all__ = ["GapTeacher"]
