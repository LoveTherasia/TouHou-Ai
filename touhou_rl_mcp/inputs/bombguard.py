"""紧急放弹保护 (safety rule, 不是策略动作)。

目的: **只在迫不得已时放弹** —— 最近弹幕距离小于 ``radius`` 且灵力足够且过了冷却。
它像 DialogueAssist 一样是规则; 与策略动作区别在于:
  * 它不会替换策略动作 (策略动作照常执行), 只是在动作前补一次放弹;
  * 日志里用 ``bomb_guard`` 标出, 便于统计"规则放弹"次数。

th11 放弹消耗 1.0 灵力, 所以默认 min_power=1.0, 且半径很小(默认 14px, 判定点约 2px)。
"""
from __future__ import annotations

from typing import Any


class BombGuard:
    def __init__(
        self,
        *,
        enabled: bool = True,
        radius: float = 14.0,
        min_power: float = 1.0,
        cooldown_frames: int = 45,
        max_per_episode: int = 6,
    ) -> None:
        self.enabled = bool(enabled)
        self.radius = float(radius)
        self.min_power = float(min_power)
        self.cooldown_frames = int(cooldown_frames)
        self.max_per_episode = int(max_per_episode)
        self._cooldown = 0
        self.count = 0

    def reset(self) -> None:
        self._cooldown = 0
        self.count = 0

    def update(self, state: Any, frames: int) -> bool:
        """返回本步是否要放弹; 内部维护冷却与每局上限。"""
        if self._cooldown > 0:
            self._cooldown = max(0, self._cooldown - max(1, int(frames)))
            return False
        if not self.enabled or state is None or not state.in_game:
            return False
        if self.count >= self.max_per_episode:
            return False
        if state.lives is None or state.lives < 0:
            return False
        if state.player_x is None or state.player_y is None:
            return False
        if (state.power or 0.0) < self.min_power:
            return False
        entities = (state.raw or {}).get("entities", {})
        if entities.get("bullets", {}).get("status") != "ok":
            return False
        if not state.bullets:
            return False
        px, py = float(state.player_x), float(state.player_y)
        nearest = min(
            (((bx - px) ** 2 + (by - py) ** 2) ** 0.5 for bx, by in state.bullets),
            default=float("inf"),
        )
        if nearest <= self.radius:
            self._cooldown = self.cooldown_frames
            self.count += 1
            return True
        return False
