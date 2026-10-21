"""时间展开的可行域规划器 —— 真正回答"现在必须站在哪".

为什么重写
----------
旧的 ``MacroPlanner``（仍在 ``training/planner.py``）在 120 帧上做 beam search，
但排序键是"违规帧数 + 最大化最坏间隙（20px 封顶）"。这等价于**往最空的地方跑**：
密集弹幕里"最空"永远是角落，而它分不清"站在缺口里"和"站在场地另一端"。所以它在
``wall_gate`` 上只有 667 步。

它缺的是**全局可行性**：不是"哪条路看起来最松", 而是"从现在这个位置出发, 还有没有
一条能活到 T 帧之后的路"。这个问题只能靠整张网格上的反向可达性来回答。

做法
----
1. 场地离散成 ``cell`` 网格（默认 4.4px，与自机速度对齐）: 
   ``cell = speed * dt / 2``, 于是 **轴向走 2 格、斜向走 (1,1) 格** 都精确等于真实位移；
2. 每一层（``dt=2`` 帧）把弹幕预测位置栅格化成**危险场** ``risk[k]``（弹速 × dt 可能
   跨过一格，所以层内**逐帧**取最大值，避免"穿格"）；
3. 反向 DP 求**代价到终点**:
   ``C[K] = risk[K]``, ``C[k] = risk[k] + min_{9 个方向} C[k+1][p + delta]``；
4. 执行第一步: 选 ``argmin_delta C[1][p0 + delta]``（再叠一个很小的转向惩罚）。

因为 ``C`` 已经把"未来所有层的危险"累加进来，**提前站位会自然出现**: 一堵 150 帧后
才压到自机高度的墙，只要它和自机同列，代价立刻在里面体现 —— 不需要任何"墙"的特判。

代价: 每次决策约 60~80 层 × 9 方向的小矩阵运算, 远快于逐帧 beam search。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..types import PLAYFIELD_HEIGHT, PLAYFIELD_WIDTH, GameState

#: 9 个方向 (顺序与 ``trainer.DIRECTIONS`` / 动作表一致)
DIRECTIONS: tuple[tuple[int, int], ...] = (
    (0, 0), (0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (1, -1), (-1, 1), (1, 1),
)
NAMES: tuple[str, ...] = (
    "", "up", "down", "left", "right", "up_left", "up_right", "down_left", "down_right",
)

#: 自机判定 + 弹幕半径 (与 mock 一致) -> 5px; 再加 0.5px 余量
DANGER_RADIUS = 5.5
#: 自机可活动的格心范围（留边, 免得贴死屏幕边）
_X_LO, _X_HI = 8.0, PLAYFIELD_WIDTH - 8.0
_Y_LO, _Y_HI = 32.0, PLAYFIELD_HEIGHT - 8.0

_INF = 1e9
#: 撞进硬危险区的一次性代价 (远大于任何缓冲带累积, 所以"先保命"是硬约束)
_DEATH = 1e4


@dataclass
class PlanInfo:
    direction: int
    cost: float
    cost_by_dir: np.ndarray
    horizon: int
    cells: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "planner": "viability",
            "cost": round(float(self.cost), 4),
            "horizon": self.horizon,
            "cells": self.cells,
        }


class ViabilityPlanner:
    """网格化的时间展开规划器 (反向可行域 + 危险代价).

    Args:
        speed: 自机速度 (px/帧); mock/真机都是 4.4。
        dt: 每层几帧。
        cell: 网格边长 (px)。默认由 ``speed*dt/2`` 推出, 保证轴向 2 格 = 真实位移。
        horizon: 规划总帧数。``wall_gate`` 的墙从 y=108 落到自机高度约 150 帧,
            所以这个值要 >= 150 才看得见墙。
        danger_radius: 危险半径 (px)。
        turn_penalty: 换方向的固定代价 (抑制左右互搏, 很小即可)。
        aim: 可选 (aim_x, weight): 每层加 ``weight * |x - aim_x| / 宽度`` 的代价,
            让它在同样安全的位置里偏向能开火的那一列。0 = 纯躲避。
    """

    def __init__(
        self,
        *,
        speed: float = 4.4,
        dt: int = 2,
        cell: float | None = None,
        horizon: int = 180,
        kill_radius: float = DANGER_RADIUS,
        hard_buffer: float = 2.0,
        lead_frames: int = 20,
        soft_weight: float = 0.05,
        turn_penalty: float = 0.0,
        aim_weight: float = 0.0,
        max_bullets: int = 400,
    ) -> None:
        self.speed = float(speed)
        self.dt = max(1, int(dt))
        self.cell = float(cell) if cell else self.speed * self.dt / 2.0
        self.horizon = max(self.dt * 2, int(horizon))
        self.layers = self.horizon // self.dt
        self.kill_radius = float(kill_radius)
        #: 时间影子(帧): 弹幕**到达前**这么多帧就把该格标成危险。
        #: 没有它就是"最后一刻再闪"永远和"现在就走"同价 —— 逐帧重规划会一直拖延,
        #: 直到真的来不及。这是"提前站位"能出现的根本原因。
        self.lead_frames = max(0, int(lead_frames))
        self.k_lead = int(np.ceil(self.lead_frames / self.dt))
        self.soft_weight = float(soft_weight)
        #: 硬危险区的格子数: 判定半径 + 半格对角(吃量化误差) + hard_buffer
        self.k_hard = max(1, int(np.ceil(
            (self.kill_radius + 0.5 * np.hypot(self.cell, self.cell) + hard_buffer) / self.cell
        )))
        #: 缓冲带只再往外 1 格: 再宽就会把 48px 的缺口糊成 1 格, 规划器分不出好坏
        self.k_soft = self.k_hard + 1
        self.turn_penalty = float(turn_penalty)
        self.aim_weight = float(aim_weight)
        self.max_bullets = int(max_bullets)

        self.nx = int(np.floor(PLAYFIELD_WIDTH / self.cell))
        self.ny = int(np.floor(PLAYFIELD_HEIGHT / self.cell))
        #: 轴向 2 格、斜向 1 格 —— 由 cell = speed*dt/2 保证两者都精确
        self._moves = self._build_moves()
        self._x_centers = (np.arange(self.nx) + 0.5) * self.cell
        self._y_centers = (np.arange(self.ny) + 0.5) * self.cell
        #: 允许站立的格心 (贴边几格直接判死, 避免"缩到角落最安全")
        self._valid = (
            (self._x_centers[None, :] >= _X_LO) & (self._x_centers[None, :] <= _X_HI)
            & (self._y_centers[:, None] >= _Y_LO) & (self._y_centers[:, None] <= _Y_HI)
        )
        self._pad = 3

    # ------------------------------------------------------------------
    def _build_moves(self) -> list[tuple[int, int]]:
        """方向 -> (d_col, d_row) 的格数；row 向下为正.

        ``cell = speed*dt/2`` 保证: 轴向 = 2 格、斜向 = (1,1) 格，与真实位移都精确相等。
        """
        moves = []
        for ux, uy in DIRECTIONS:
            if ux == 0 and uy == 0:
                moves.append((0, 0))
            elif ux == 0 or uy == 0:
                moves.append((2 * ux, 2 * uy))
            else:
                moves.append((ux, uy))
        return moves

    # ------------------------------------------------------------------
    def _to_cell(self, x: float, y: float) -> tuple[int, int]:
        col = int(np.clip(np.rint(x / self.cell - 0.5), 0, self.nx - 1))
        row = int(np.clip(np.rint(y / self.cell - 0.5), 0, self.ny - 1))
        return col, row

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def _risk_layers(
        self, bullets: np.ndarray, vel: np.ndarray, aim_x: float | None
    ) -> tuple[np.ndarray, np.ndarray]:
        """``(hard, soft)`` 两个 (layers, ny, nx) 布尔场.

        关键设计（第一版就是死在这里）：

        * 只用 ``(1-d/r)²`` 这种"软"场时，格子边长 6.6px、判定半径 5px，
          一颗**正落在自机身上**的弹只算 0.26 代价 —— 等于"被打中几乎免费"，
          于是规划器永远原地不动。必须是**硬**危险区（命中 = 1.0）。
        * 硬区半径要把量化误差吃掉: ``5px + 半格对角``, 否则"格子中心安全"不代表
          "自机安全"。
        * 外面再套一圈缓冲带（默认只多 1 格、权重 0.2），作用是**在同为 0 的多条活路
          里挑最舒服的那条**；太宽会把 48px 的缺口糊住（实测 3 格以上缺口就只剩 1 格）。
        * **时间影子** ``lead_frames``: 弹幕到达前就标危险。没有它,"最后一刻再闪"和
          "现在就走"永远同价, 逐帧重规划会一路拖延到死。

        顺序: 逐帧栅格化 -> 按 dt 取最大 -> 时间影子 -> 空间膨胀。
        (膨胀是 max 滤波, 与"按 dt 取最大"可交换, 所以可以先降时间维再膨胀。)
        """
        frames = self.layers * self.dt
        t = np.arange(frames, dtype=np.float64)
        pos = bullets[None, :, :] + vel[None, :, :] * t[:, None, None]  # (F, B, 2)
        col = np.rint(pos[..., 0] / self.cell - 0.5)
        row = np.rint(pos[..., 1] / self.cell - 0.5)
        ok = ((col >= 0) & (col < self.nx) & (row >= 0) & (row < self.ny))
        fidx = np.broadcast_to(np.arange(frames, dtype=np.int64)[:, None], col.shape)

        flat = np.zeros(frames * self.ny * self.nx, dtype=bool)
        if ok.any():
            idx = (fidx[ok] * self.ny + row[ok].astype(np.int64)) * self.nx \
                + col[ok].astype(np.int64)
            flat[idx] = True
        base = flat.reshape(self.layers, self.dt, self.ny, self.nx).any(axis=1)

        shadow = base
        if self.k_lead:
            # 时间影子: 未来 k_lead 层内会被占的格子 —— 只算"软代价"(提前站位的压力),
            # **不能**算死亡: 否则一堵还在 30px 外的墙会让整行都变成"必死",
            # 规划器就只能选一个等价的"死法"(实测会挑着往墙上撞)。
            shadow = base.copy()
            for k in range(1, self.k_lead + 1):
                shadow[:-k] |= base[k:]

        hard = self._dilate(base, self.k_hard)
        soft = self._dilate(shadow, self.k_soft) | hard
        return hard, soft

    # ------------------------------------------------------------------
    @staticmethod
    def _dilate(mask: np.ndarray, k: int) -> np.ndarray:
        """按 k 格做可分离的方形膨胀 (先沿行, 再沿列); k=0 原样返回."""
        if k <= 0:
            return mask
        out = mask
        for _ in range(k):
            m = out
            o = m.copy()
            o[:, 1:, :] |= m[:, :-1, :]
            o[:, :-1, :] |= m[:, 1:, :]
            out = o
        for _ in range(k):
            m = out
            o = m.copy()
            o[:, :, 1:] |= m[:, :, :-1]
            o[:, :, :-1] |= m[:, :, 1:]
            out = o
        return out
    # ------------------------------------------------------------------
    def plan(self, state: GameState, aim_x: float | None = None) -> PlanInfo:
        px = float(state.player_x) if state.player_x is not None else PLAYFIELD_WIDTH / 2
        py = float(state.player_y) if state.player_y is not None else PLAYFIELD_HEIGHT - 32
        bullets, vel = self._bullets(state, px, py)
        col, row = self._to_cell(px, py)

        if not len(bullets):
            # 没有威胁: 保持 (或朝目标列微调)
            return PlanInfo(0, 0.0, np.zeros(len(DIRECTIONS)), 0, self.nx * self.ny)

        if aim_x is None and self.aim_weight > 0.0:
            pos = (state.raw or {}).get("boss_position")
            if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                aim_x = float(pos[0])
        hard, soft = self._risk_layers(bullets, vel, None)
        pad = self._pad
        big = np.float32(_INF)
        death = np.float32(_DEATH)
        layers = self.layers

        aim_cost = None
        if aim_x is not None and self.aim_weight > 0.0:
            row_cost = (self.aim_weight
                        * np.abs(self._x_centers - aim_x) / PLAYFIELD_WIDTH).astype(np.float32)
            aim_cost = np.broadcast_to(row_cost[None, :], (self.ny, self.nx))

        def build_cost1() -> np.ndarray:
            """反向 DP: 撞进硬危险区 = 一次性终止代价 (晚死更优), 否则累加缓冲带代价."""
            nxt = np.where(hard[layers - 1], death, np.float32(0.0)).astype(np.float32)
            pad_arr = np.full((self.ny + 2 * pad, self.nx + 2 * pad), big, dtype=np.float32)
            pad_arr[pad:pad + self.ny, pad:pad + self.nx] = nxt
            pad_arr[pad:pad + self.ny, pad:pad + self.nx][~self._valid] = death
            for k in range(layers - 2, -1, -1):
                best = None
                for dc, dr in self._moves:
                    view = pad_arr[pad + dr:pad + dr + self.ny,
                                   pad + dc:pad + dc + self.nx]
                    if best is None:
                        best = view.copy()
                    else:
                        np.minimum(best, view, out=best)
                cur = best + self.soft_weight * soft[k].astype(np.float32)
                if aim_cost is not None:
                    cur += aim_cost
                # 硬危险区: 死在这里, 代价 = DEATH + 剩余层数 (越晚死越便宜)
                cur[hard[k]] = death + np.float32(layers - k)
                cur[~self._valid] = death
                pad_arr[pad:pad + self.ny, pad:pad + self.nx] = cur
                if k == 1:
                    return cur.copy()
            return pad_arr[pad:pad + self.ny, pad:pad + self.nx].copy()

        cost1 = build_cost1() if layers > 2 else np.zeros((self.ny, self.nx), np.float32)

        # 掷出第一步: 在 k=1 的代价表上取最小
        cost_by_dir = np.full(len(DIRECTIONS), _INF, dtype=np.float64)
        for mi, (dc, dr) in enumerate(self._moves):
            c, r = col + dc, row + dr
            if 0 <= c < self.nx and 0 <= r < self.ny and self._valid[r, c]:
                cost_by_dir[mi] = float(cost1[r, c]) + (0.0 if mi == 0 else self.turn_penalty)
        if cost_by_dir.min() >= _INF * 0.5:
            mi = self._best_escape(bullets, vel, px, py)
            cost_to_go = _INF
        else:
            mi = int(np.argmin(cost_by_dir))
            cost_to_go = float(cost_by_dir[mi])
        return PlanInfo(mi, cost_to_go, cost_by_dir, self.horizon, self.nx * self.ny)

    # ------------------------------------------------------------------
    def _best_escape(self, bullets: np.ndarray, vel: np.ndarray,
                     px: float, py: float) -> int:
        """所有方向都被判死时的兜底: 选 dt 之后离弹幕最远的那个方向."""
        p = bullets + vel * float(self.dt)
        best_mi, best_d = 0, -1.0
        for mi, (dc, dr) in enumerate(self._moves):
            x, y = px + dc * self.cell, py + dr * self.cell
            d = float(np.hypot(p[:, 0] - x, p[:, 1] - y).min()) if len(p) else 1e6
            if d > best_d:
                best_mi, best_d = mi, d
        return best_mi

    def _bullets(self, state: GameState, px: float, py: float) -> tuple[np.ndarray, np.ndarray]:
        if not state.bullets:
            return np.zeros((0, 2)), np.zeros((0, 2))
        pos = np.asarray(state.bullets, dtype=np.float64).reshape(-1, 2)
        raw = (state.raw or {}).get("bullet_velocity") or []
        try:
            vel = np.asarray(raw, dtype=np.float64).reshape(-1, 2)
            if vel.shape[0] != pos.shape[0]:
                raise ValueError
        except (TypeError, ValueError):
            vel = np.zeros_like(pos)
        if len(pos) > self.max_bullets:
            d = np.hypot(pos[:, 0] - px, pos[:, 1] - py)
            keep = np.argsort(d)[: self.max_bullets]
            pos, vel = pos[keep], vel[keep]
        return pos, vel

    # ------------------------------------------------------------------
    def choose(self, state: GameState, frame: int = 0, learned_action: str = "") -> tuple[str, dict]:
        """与其它老师同签名, 便于直接接进 Trainer / 对比脚本."""
        info = self.plan(state)
        name = NAMES[info.direction]
        action = "+".join(p for p in (name, "shoot") if p)
        return action, info.as_dict()


def plan_action(planner: ViabilityPlanner, state: GameState) -> str:
    return planner.choose(state)[0]
