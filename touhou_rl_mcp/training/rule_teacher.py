"""规则老师 (mock 与真机通用).

两种决策方式:
  * ``planner="force"``   —— 与 ``dodge`` 宏同源的弹幕斥力 + 归位;
  * ``planner="horizon"`` —— 对 9 个方向做**短程轨迹规划** (用弹速预测未来若干帧),
    评估最小间隙并选最安全方向 (比纯斥力更能提前躲开"朝自机飞来"的弹)。

它只用 ``state.bullets`` (+ 可选 ``bullet_velocity`` / Boss 位置), 因此在 mock 上
也能稳定躲弹, 适合做行为克隆 / DAgger 的标签来源, 给 RL 一个"会躲"的起点。
"""

from __future__ import annotations

import math

import numpy as np

from ..types import PLAYFIELD_HEIGHT, PLAYFIELD_WIDTH, GameState

#: (dx, dy) -> 方向索引 (与 trainer.DIRECTIONS 顺序一致)
_DIR_MAP = {(0, 0): 0, (0, -1): 1, (0, 1): 2, (-1, 0): 3, (1, 0): 4,
            (-1, -1): 5, (1, -1): 6, (-1, 1): 7, (1, 1): 8}
_NAMES = ["", "up", "down", "left", "right", "up_left", "up_right", "down_left", "down_right"]
_DIRS = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (1, -1), (-1, 1), (1, 1)]
_HORIZON = np.array([0.0, 3.0, 6.0, 10.0, 15.0, 22.0])


class RuleTeacher:
    """基于弹幕斥力 / 短程规划的离散动作老师."""

    def __init__(
        self,
        *,
        planner: str = "force",
        radius: float = 30.0,
        home_x: float = 192.0,
        home_y: float = 370.0,
        home_weight: float = 0.80,
        focus_radius: float = 0.0,
        boss_home: bool = True,
        aim_safe: float = 0.0,
        speed: float = 4.4,
        predict_frames: float = 0.0,
        strafe_amp: float = 100.0,
        strafe_period: float = 150.0,
        edge_margin: float = 45.0,
        edge_weight: float = 2.0,
    ) -> None:
        self.planner = planner
        self.radius = radius
        self.home_x = home_x
        self.home_y = home_y
        self.home_weight = home_weight
        self.focus_radius = focus_radius
        self.boss_home = boss_home
        self.aim_safe = aim_safe
        self.speed = speed
        #: >0 时用弹速预测未来位置再算斥力 (提前躲开"朝自机飞来"的弹)
        self.predict_frames = float(predict_frames)
        #: 横向游走: home_x 随时间左右摆动, 避免被"自机狙"锁定
        self.strafe_amp = float(strafe_amp)
        self.strafe_period = max(1.0, float(strafe_period))
        #: 边界回避: 离屏幕边缘 <edge_margin 像素时产生"回推力", 权重 edge_weight。
        #: 实测策略死于"被斥力顶到屏幕底部(y=440)贴脸", 不是死于环弹本身。
        self.edge_margin = max(0.0, float(edge_margin))
        self.edge_weight = float(edge_weight)
        self.last_direction = 0

    # ------------------------------------------------------------------
    def _home(self, state: GameState, nearest: float) -> tuple[float, float]:
        hx, hy = self.home_x, self.home_y
        if self.strafe_amp:
            hx += self.strafe_amp * math.sin((state.frame or 0) * 2.0 * math.pi / self.strafe_period)
        if self.boss_home and nearest >= self.aim_safe:
            pos = (state.raw or {}).get("boss_position")
            if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                try:
                    hx = float(pos[0])
                except (TypeError, ValueError):
                    pass
        return min(PLAYFIELD_WIDTH - 40.0, max(40.0, hx)), hy

    # ------------------------------------------------------------------
    def choose(self, state: GameState, frame: int, learned_action: str):
        px = float(state.player_x) if state.player_x is not None else PLAYFIELD_WIDTH / 2
        py = float(state.player_y) if state.player_y is not None else PLAYFIELD_HEIGHT - 32
        if self.planner == "horizon":
            direction, nearest = self._choose_horizon(state, px, py)
        else:
            direction, nearest = self._choose_force(state, px, py)

        self.last_direction = direction
        name = _NAMES[direction]
        parts = [name] if name else []
        parts.append("shoot")
        if nearest < self.focus_radius:
            parts.append("focus")
        action = "+".join(parts)
        return action, {"nearest": round(nearest, 2), "direction": direction,
                        "planner": self.planner}

    # ------------------------------------------------------------------
    def force_vector(self, state: GameState, px: float, py: float) -> tuple[float, float, float]:
        """弹幕斥力 + 归位 + 边界回避 的合力 ``(fx, fy, nearest)``.

        ``HybridTeacher`` 直接拿它当"安全先验", 保证两个老师的建议永远同源:
        改这里的力场, 两边的标签会一起变。
        """
        fx = fy = 0.0
        nearest = 9999.0
        if state.bullets:
            arr = np.asarray(state.bullets, dtype=np.float64).reshape(-1, 2)
            dx0 = px - arr[:, 0]
            dy0 = py - arr[:, 1]
            nearest = float(np.sqrt(dx0 * dx0 + dy0 * dy0).min())
            if self.predict_frames > 0:
                v = (state.raw or {}).get("bullet_velocity") or []
                if len(v) == len(state.bullets):
                    arr = arr + np.asarray(v, dtype=np.float64).reshape(-1, 2) * self.predict_frames
                    dx0 = px - arr[:, 0]
                    dy0 = py - arr[:, 1]
            d2 = dx0 * dx0 + dy0 * dy0
            mask = d2 <= self.radius * self.radius
            if mask.any():
                d = np.sqrt(d2[mask])
                d = np.where(d > 0.0, d, 1.0)
                w = (self.radius - d) / self.radius / d
                fx = float((dx0[mask] * w).sum())
                fy = float((dy0[mask] * w).sum())
        hx, hy = self._home(state, nearest)
        fx += (hx - px) * self.home_weight / 96.0
        fy += (hy - py) * self.home_weight / 96.0
        if self.edge_margin > 0.0 and self.edge_weight:
            # 边界回避: 贴墙时加"回推力", 免得被弹幕顶到角落送死
            m = self.edge_margin
            if px < m:
                fx += (m - px) / m * self.edge_weight
            elif px > PLAYFIELD_WIDTH - m:
                fx -= (px - (PLAYFIELD_WIDTH - m)) / m * self.edge_weight
            if py < m:
                fy += (m - py) / m * self.edge_weight
            elif py > PLAYFIELD_HEIGHT - m:
                fy -= (py - (PLAYFIELD_HEIGHT - m)) / m * self.edge_weight

        return fx, fy, nearest

    # ------------------------------------------------------------------
    @staticmethod
    def direction_from_force(fx: float, fy: float) -> int:
        """合力 -> 9 个方向之一 (带 12% 死区, 免得微弱合力导致左右互搏)."""
        mag = math.hypot(fx, fy)
        dx = dy = 0
        if mag > 1e-3:
            if fx > 0.12 * mag:
                dx = 1
            elif fx < -0.12 * mag:
                dx = -1
            if fy > 0.12 * mag:
                dy = 1
            elif fy < -0.12 * mag:
                dy = -1
        return _DIR_MAP.get((dx, dy), 0)

    def _choose_force(self, state: GameState, px: float, py: float) -> tuple[int, float]:
        fx, fy, nearest = self.force_vector(state, px, py)
        return self.direction_from_force(fx, fy), nearest

    # ------------------------------------------------------------------
    def _choose_horizon(self, state: GameState, px: float, py: float) -> tuple[int, float]:
        bullets = np.asarray(state.bullets, dtype=float).reshape(-1, 2) if state.bullets else np.zeros((0, 2))
        vel = (state.raw or {}).get("bullet_velocity") or []
        if len(bullets) and len(vel) == len(bullets):
            vel = np.asarray(vel, dtype=float).reshape(-1, 2)
        else:
            vel = np.zeros_like(bullets)
        nearest = 9999.0
        if len(bullets):
            nearest = float(np.linalg.norm(bullets - [px, py], axis=1).min())

        home = self._home(state, nearest)
        best_i = 0
        best_cost = 1e18
        best_clear = 9999.0
        for i, (ux, uy) in enumerate(_DIRS):
            n = math.hypot(ux, uy)
            d = np.array([ux, uy], dtype=float) / n if n else np.zeros(2)
            path = np.array([px, py]) + _HORIZON[:, None] * d * self.speed
            path = np.clip(path, [8.0, 32.0], [376.0, 432.0])
            cost = 0.0
            clearance = 9999.0
            if len(bullets):
                pred = bullets[None, :, :] + _HORIZON[:, None, None] * vel[None, :, :]
                dist = np.linalg.norm(pred - path[:, None, :], axis=2)  # (T, B)
                clearance = float(dist.min())
                cost += float(np.exp(-(dist ** 2) / (2 * 20.0 ** 2)).sum(axis=1).max()) * 200.0
                cost += max(0.0, 12.0 - clearance) * 120.0
            cost += float(np.linalg.norm(path[-1] - np.array(home))) * 0.12
            cost += max(0.0, 55.0 - path[-1, 0]) * 0.4 + max(0.0, path[-1, 0] - 329.0) * 0.4
            if i != self.last_direction:
                cost += 0.5
            if cost < best_cost:
                best_cost = cost
                best_i = i
                best_clear = clearance
        return best_i, (best_clear if len(bullets) else nearest)
