"""缺口老师: 把几何标签接进 :class:`~touhou_rl_mcp.training.trainer.Trainer`。

训练器原本的老师只有两种来源 —— 手写战术规则 (``guard``) 和规则瞄准 (``rule``)。
它们的共同问题是"带着一套人工代价函数", 而缺口任务上真正需要的只是**几何测量**:
"哪条车道还有余量的地方"。这里把 :func:`gap_label` 包成同一个协议, 于是
BC / DAgger / 纯 RL 都能用同一套训练循环, 不再需要 ``tools/research`` 里那套平行实现。

老师的输出是一串动作名 (如 ``"left+shoot"``), 由训练器用 ``_teacher_index`` 映射回
自己的动作表下标 —— 与 ``TacticalGuard`` 完全一致。

有状态口径
----------
``reach_hold`` / ``reach_hyst`` 需要一个 :class:`GapCommitment` 记住当前目标列。
老师对象在 ``Trainer._run_episode`` 里**每局新建**, 所以承诺状态天然按局重置。
评估脚本也必须每局新建一个老师, 不能跨局复用。
"""
from __future__ import annotations

from typing import Any

from ..types import GameState
from .gap_label import (
    HOLD_MODES, MACRO_ACTIONS, GapCommitment, gap_label,
)
from .occupancy import GapSpec


class GapTeacher:
    """按缺口几何打标签的老师 (无代价函数; 有状态口径带目标承诺)。"""

    #: 名字 -> 下标 (与 DIRECTIONS 同序), 供训练器反查
    actions: tuple[str, ...] = MACRO_ACTIONS

    def __init__(self, spec: GapSpec | None = None, mode: str = "reach",
                 commit_frames: int = 0, hysteresis: float = 0.0,
                 safety_margin: float = 0.0) -> None:
        self.spec = spec or GapSpec()
        self.mode = mode
        self.commit_frames = int(commit_frames)
        self.hysteresis = float(hysteresis)
        self.commit = (
            GapCommitment(self.spec, mode=mode, commit_frames=self.commit_frames,
                          hysteresis=self.hysteresis, safety_margin=safety_margin)
            if mode in HOLD_MODES else None
        )

    def reset(self) -> None:
        if self.commit is not None:
            self.commit.reset()

    def choose(self, state: GameState, frame: int | None,
               learned_action: str = "") -> tuple[str, dict[str, Any]]:
        """返回 ``(动作名, 诊断信息)``; 与 ``TacticalGuard.choose`` 同签名。"""
        if state.player_x is None or state.player_y is None:
            raise ValueError("缺口标签需要有效的自机坐标")
        if self.commit is not None:
            index, info = self.commit.decide(state, frame)
        else:
            index = gap_label(state, self.spec, self.mode)
            info = {"switched": False, "target": None}
        return MACRO_ACTIONS[index], {
            **info,
            "gap_label": index,
            "gap_mode": self.mode,
            "frame": frame,
            "learned_action": learned_action,
        }

    def describe(self) -> dict[str, Any]:
        out = {
            "teacher": "gap",
            "mode": self.mode,
            "commit_frames": self.commit_frames,
            "hysteresis": self.hysteresis,
            **self.spec.__dict__,
        }
        if self.commit is not None:
            out.update({
                "target": self.commit.target,
                "switches": self.commit.switches,
                "decisions": self.commit.decisions,
            })
        return out


__all__ = ["GapTeacher"]
