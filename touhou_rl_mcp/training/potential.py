"""势能场规划器：把弹幕危险度当成一个标量场，沿负梯度的方向走。

这是 ``docs/PLANNER_V2.md`` §2.6 的"替代路线": 不训练策略、也不做多步搜索，
直接把"危险度"写成一个可微的空间场, 每一步选 9 邻域里势能最低的方向。

优点: 零训练、可解释、单步成本极低 (一次 9×M 的距离计算)。
缺点: 短视 —— 只看"下一步站哪儿最安全", 会走进死胡同。它的作用是给 MPC 提供
      一个**廉价对照**，也能在真机上当兜底 (MPC 超时就用它)。
"""
from __future__ import annotations

import numpy as np

from ..types import PLAYFIELD_HEIGHT, PLAYFIELD_WIDTH, GameState
from .mpc import ACTION_NAMES, ACTION_NAMES_FOCUS
from .occupancy import DIRECTIONS, PLAYER_SPEED


class PotentialFieldPlanner:
    """``φ(p) = Σ exp(-d²/2σ²) + 边界项 + 站位项``，取下降最多的方向。

    ``sigma`` 越小越"只躲近弹", 越大越"提前避让"; 30px 左右在密弹里比较稳。
    """

    def __init__(self, *, sigma: float = 30.0, edge: float = 26.0,
                 y_target: float = 0.85, y_weight: float = 0.6,
                 speed: float = PLAYER_SPEED, lookahead: int = 3,
                 use_focus: bool = False, focus_distance: float = 45.0,
                 hysteresis: float = 0.05) -> None:
        self.sigma = float(sigma)
        self.edge = float(edge)
        self.y_target = float(y_target)
        self.y_weight = float(y_weight)
        self.speed = float(speed)
        self.lookahead = max(1, int(lookahead))
        self.use_focus = bool(use_focus)
        self.focus_distance = float(focus_distance)
        self.hysteresis = float(hysteresis)
        self._last_dir = 0
        self.last_potential: float | None = None

    # ------------------------------------------------------------------
    def potential(self, x: float, y: float, bullets: np.ndarray) -> float:
        phi = 0.0
        if bullets.size:
            d2 = (bullets[:, 0] - x) ** 2 + (bullets[:, 1] - y) ** 2
            phi += float(np.exp(-d2 / (2.0 * self.sigma ** 2)).sum())
            # 极近弹: 再叠一个 1/d 的核, 让"擦身而过"变得明确不可接受
            d = np.sqrt(d2)
            phi += float((np.maximum(1.0, 12.0 - d) / 12.0).clip(0).sum()) * 4.0
        # 边界: 越贴边越危险 (角落是"弹最少但必死"的陷阱)
        if x < self.edge:
            phi += (self.edge - x) / self.edge * 2.0
        elif x > PLAYFIELD_WIDTH - self.edge:
            phi += (x - (PLAYFIELD_WIDTH - self.edge)) / self.edge * 2.0
        if y < self.edge:
            phi += (self.edge - y) / self.edge * 2.0
        elif y > PLAYFIELD_HEIGHT - self.edge * 0.6:
            phi += (y - (PLAYFIELD_HEIGHT - self.edge * 0.6)) / (self.edge * 0.6)
        # 站位: 待在"能输出"的高度 (太靠上会被撞, 太靠下打不到)
        phi += self.y_weight * abs(y / PLAYFIELD_HEIGHT - self.y_target)
        return phi

    # ------------------------------------------------------------------
    def choose(self, state: GameState) -> str:
        if state.player_x is None or state.player_y is None:
            return ACTION_NAMES[0]
        px, py = float(state.player_x), float(state.player_y)
        bullets = np.asarray(state.bullets or [], dtype=np.float64).reshape(-1, 2)

        values = []
        for dx, dy in DIRECTIONS:
            nx = min(PLAYFIELD_WIDTH - 8.0, max(8.0, px + dx * self.speed * self.lookahead))
            ny = min(PLAYFIELD_HEIGHT - 8.0, max(8.0, py + dy * self.speed * self.lookahead))
            values.append(self.potential(nx, ny, bullets))
        values = np.asarray(values)

        # 滞回: 只有明显更优才换方向, 否则在等价缺口之间会左右抖
        best = int(np.argmin(values))
        if (self._last_dir != best
                and values[best] > values[self._last_dir] - self.hysteresis):
            best = self._last_dir
        self._last_dir = best
        self.last_potential = float(values[best])

        if not self.use_focus:
            return ACTION_NAMES[best]
        near = 1e9
        if bullets.size:
            near = float(np.hypot(bullets[:, 0] - px, bullets[:, 1] - py).min())
        return (ACTION_NAMES_FOCUS if near < self.focus_distance else ACTION_NAMES)[best]

    def describe(self) -> dict:
        return {"name": "potential", "sigma": self.sigma, "y_target": self.y_target,
                "lookahead": self.lookahead}
