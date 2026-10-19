"""时空搜索老师 (model-based, 实验性).

状态: **实验性, 尚未达到可用老师的水平**。2026-10-19 的基准结论见
``docs/STATUS.md`` 的 P2 小节: 它能穿出"静态单缺口环", 但在"多层闭合环/密集环"
上仍会贴底或送死; 手写代价在"龟缩 / 贴边 / 冒进"之间反复横跳, 无法稳定产生
"找缝上穿"。因此它**不应**直接拿来当 BC 老师, 下一步应引入学习到的价值函数。

为什么需要它
------------
斥力/势场是**局部贪心**: 环形弹幕下各方向斥力抵消, 再叠加"回家"先验, 策略会被
按在底边; 墙角在"离弹距离"意义下反而最安全。要穿过环的缺口, 必须能回答三个问题:

1. 缺口在哪个角度、多宽;
2. 它什么时候到达自机所在的高度;
3. 现在往哪走, 才能在到达前对齐缺口并穿过去。

这三个问题都需要**未来**, 而不是"当前最近的几颗弹"。本模块用已知的弹道模型
(位置 + 速度) 做前向推演, 在"宏动作"(方向 × 固定帧数) 上做 beam search。

代价函数刻意**不用"离弹越远越好"的求和惩罚** (那正是把自机推向墙角的根因),
而是 **maximin**: 先保证不撞, 再最大化整条路径上的"最坏间隙", 最后用终局
"还有几条活路"打破平局。``escape_weight>0`` 时再加一项"远离弹幕质心", 用于
打破"待在包围圈里不动"的平局 —— 但这是一把双刃剑, 会诱发贴边/冒进。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..types import PLAYFIELD_HEIGHT, PLAYFIELD_WIDTH, GameState

#: 与 ``backends/mock.py`` / 真机判定一致
HITBOX_RADIUS = 2.0
BULLET_RADIUS = 3.0

#: 9 个方向 (顺序与 ``trainer.DIRECTIONS`` 一致)
DIRECTIONS: tuple[tuple[int, int], ...] = (
    (0, 0), (0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (1, -1), (-1, 1), (1, 1),
)
NAMES: tuple[str, ...] = (
    "", "up", "down", "left", "right", "up_left", "up_right", "down_left", "down_right",
)

#: 场地可活动范围 (留出一点边距, 避免贴边判定)
_X_MIN, _X_MAX = 8.0, PLAYFIELD_WIDTH - 8.0
_Y_MIN, _Y_MAX = 32.0, PLAYFIELD_HEIGHT - 16.0

#: 间隙超过它就不再区分 (避免"越空旷越好"的偏好; 20px 已足够安全)
_CLEAR_CAP = 20.0


@dataclass
class PlanResult:
    action: str
    direction: int
    worst_clearance: float
    horizon_frames: int
    beam: int


class MacroPlanner:
    """在宏动作上做时空 beam search 的规划器.

    Args:
        macro_frames: 每个宏动作持续的帧数 (承诺长度).
        horizon: 规划总帧数 (建议 >= 120, 覆盖一轮弹幕的到达时间).
        beam: 每层保留的候选数.
        margin: 期望的安全间隙 (像素); 小于它记一次"违规".
        speed: 自机移动速度 (px/帧).
        probe_frames: 终局评估"还有几条活路"时, 每个方向试走的帧数.
        bullet_radius: 弹幕半径 (真机接入后从状态读取, 见 TODO).
    """

    def __init__(
        self,
        *,
        macro_frames: int = 6,
        horizon: int = 120,
        beam: int = 16,
        margin: float = 9.0,
        speed: float = 4.4,
        probe_frames: int = 30,
        bullet_radius: float = BULLET_RADIUS,
        near_radius: float = 340.0,
        escape_weight: float = 0.0,
    ) -> None:
        self.macro_frames = max(1, int(macro_frames))
        self.horizon = max(self.macro_frames, int(horizon))
        self.beam = max(1, int(beam))
        self.margin = max(HITBOX_RADIUS + bullet_radius + 1.0, float(margin))
        self.speed = float(speed)
        self.probe_frames = max(1, int(probe_frames))
        self.hit_radius = HITBOX_RADIUS + float(bullet_radius)
        self.near_radius = float(near_radius)
        #: >0 时偏好"远离附近弹幕的质心" —— 打破"待在包围圈里不动"的平局。
        #: 注意: 这只是启发式; 真正的"提前脱离"应由学习到的 V(s) 提供。
        self.escape_weight = float(escape_weight)

    # ------------------------------------------------------------------
    def plan(self, state: GameState) -> PlanResult:
        px = float(state.player_x) if state.player_x is not None else PLAYFIELD_WIDTH / 2
        py = float(state.player_y) if state.player_y is not None else PLAYFIELD_HEIGHT - 32
        bullets, velocities = self._bullets(state, px, py)
        if not len(bullets):
            return PlanResult("shoot", 0, float("inf"), 0, 1)

        depth = max(1, self.horizon // self.macro_frames)
        center = self._centroid(bullets)
        # beam 节点: (违规帧数, 全路径最坏间隙, x, y, 首个方向, 上次方向)
        beam: list[tuple[int, float, float, float, int, int]] = [
            (0, float("inf"), px, py, -1, -1)
        ]
        for level in range(depth):
            t0 = level * self.macro_frames
            candidates: list[tuple[int, float, float, float, int, int]] = []
            for bad, worst, x, y, first, last in beam:
                for index, (ux, uy) in enumerate(DIRECTIONS):
                    seg_bad, seg_min, nx, ny = self._segment(
                        x, y, ux, uy, bullets, velocities, t0
                    )
                    penalty = 0.15 if last not in (-1, index) else 0.0
                    candidates.append(
                        (bad + seg_bad, min(worst, seg_min) - penalty,
                         nx, ny, index if first < 0 else first, index)
                    )
            # 先少违规, 再最大化"最坏间隙 + 脱离包围"; 后者同时用来打破"不动"的平局
            candidates.sort(
                key=lambda item: (
                    item[0],
                    -min(item[1], _CLEAR_CAP) - self.escape_weight * math.hypot(
                        item[2] - center[0], item[3] - center[1]
                    ),
                )
            )
            beam = candidates[: self.beam]

        # 终局: 谁的"活路"多选谁 (墙角的方向选择最少, 因此被自然惩罚)
        best = None
        for bad, worst, x, y, first, _last in beam:
            free = self._free_directions(x, y, bullets, velocities, depth * self.macro_frames)
            escape = 0.0
            if self.escape_weight:
                escape = float(np.hypot(x - center[0], y - center[1]))
            # 先保证不违规, 再最大化"最坏间隙 + 脱离包围", 最后才看终局活路。
            # (活路放在最后: 穿出环后可能贴边, 但"还在包围圈里"更危险。)
            score = (bad, -min(worst, _CLEAR_CAP) - self.escape_weight * escape, -free)
            if best is None or score < best[0]:
                best = (score, first, worst)
        assert best is not None
        direction = best[1] if best[1] >= 0 else 0
        name = NAMES[direction]
        action = "+".join(part for part in (name, "shoot") if part)
        return PlanResult(action, direction, best[2], depth * self.macro_frames, len(beam))

    @staticmethod
    def _centroid(bullets: np.ndarray) -> tuple[float, float]:
        if not len(bullets):
            return PLAYFIELD_WIDTH / 2, PLAYFIELD_HEIGHT / 2
        return float(bullets[:, 0].mean()), float(bullets[:, 1].mean())

    # ------------------------------------------------------------------
    def _bullets(self, state: GameState, px: float, py: float) -> tuple[np.ndarray, np.ndarray]:
        if not state.bullets:
            return np.zeros((0, 2)), np.zeros((0, 2))
        pos = np.asarray(state.bullets, dtype=np.float64).reshape(-1, 2)
        vel = (state.raw or {}).get("bullet_velocity") or []
        try:
            arr = np.asarray(vel, dtype=np.float64).reshape(-1, 2)
            if arr.shape[0] != pos.shape[0]:
                raise ValueError
        except (TypeError, ValueError):
            arr = np.zeros_like(pos)
        near = np.hypot(pos[:, 0] - px, pos[:, 1] - py) <= self.near_radius
        return pos[near], arr[near]

    def _segment(
        self,
        px: float,
        py: float,
        ux: int,
        uy: int,
        bullets: np.ndarray,
        velocities: np.ndarray,
        t0: int,
    ) -> tuple[int, float, float, float]:
        """走一个宏动作; 返回 (违规帧数, 最坏间隙, 终点 x, 终点 y).

        只统计"低于安全间隙"的帧作为硬约束, 并用整条路径的最坏间隙做 maximin;
        不做"越远越好"的求和奖励 —— 后者会把自机推向墙角。
        """
        length = math.hypot(ux, uy)
        dx = ux / length if length else 0.0
        dy = uy / length if length else 0.0
        ts = np.arange(1, self.macro_frames + 1, dtype=np.float64)
        xs = np.clip(px + dx * self.speed * ts, _X_MIN, _X_MAX)
        ys = np.clip(py + dy * self.speed * ts, _Y_MIN, _Y_MAX)
        dist = self._distances(bullets, velocities, t0 + ts, xs, ys)
        nearest = dist.min(axis=1)
        bad = int((nearest < self.margin).sum())
        return bad, float(nearest.min()), float(xs[-1]), float(ys[-1])

    def _free_directions(
        self,
        px: float,
        py: float,
        bullets: np.ndarray,
        velocities: np.ndarray,
        t0: int,
    ) -> int:
        """终局评估: 从 (px, py) 出发, 还有几个方向能安全走 probe_frames 帧."""
        free = 0
        ts = np.arange(1, self.probe_frames + 1, dtype=np.float64)
        for ux, uy in DIRECTIONS:
            length = math.hypot(ux, uy)
            dx = ux / length if length else 0.0
            dy = uy / length if length else 0.0
            xs = np.clip(px + dx * self.speed * ts, _X_MIN, _X_MAX)
            ys = np.clip(py + dy * self.speed * ts, _Y_MIN, _Y_MAX)
            dist = self._distances(bullets, velocities, t0 + ts, xs, ys)
            if float(dist.min()) >= self.margin:
                free += 1
        return free

    @staticmethod
    def _distances(
        bullets: np.ndarray,
        velocities: np.ndarray,
        offsets: np.ndarray,
        xs: np.ndarray,
        ys: np.ndarray,
    ) -> np.ndarray:
        """(T, N) 的距离矩阵: 每帧自机位置到每颗弹的预测位置."""
        if not len(bullets):
            return np.full((len(offsets), 1), 1e6)
        pred = bullets[None, :, :] + velocities[None, :, :] * offsets[:, None, None]
        return np.hypot(pred[:, :, 0] - xs[:, None], pred[:, :, 1] - ys[:, None])


def plan_action(planner: MacroPlanner, state: GameState) -> str:
    """便捷入口: 返回可直接传给 ``env.step`` 的动作字符串."""
    return planner.plan(state).action
