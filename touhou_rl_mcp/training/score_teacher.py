"""按**奖励权重**给候选动作排序的老师 —— 让 ``score_scale`` 真正进入监督标签。

为什么需要它
------------
``RuleTeacher`` 只会躲 + 一条硬编码的"朝 Boss 归位"，它并不知道"打中敌人值多少分"。
于是 BC/DAgger 的标签里**根本没有分数**：把 ``score_scale`` 从 1e-5 调到 3e-2，
蒸馏出来的策略一模一样 —— 分数只在 PPO 段才生效，而 PPO 段本身不稳。

这个老师把决策拆成两层，正好对应"生存第一、其次最大化分数"：

1. **安全是硬约束**：每个方向先按未来 H 步的最小弹幕间距 ``clearance`` 过滤，
   低于 ``safety_margin`` 的方向直接出局（和规则老师同源的时间展开）；
2. **分数是排序目标**：在所有安全方向里，按
   ``score_scale * 期望得分 - danger_scale * 危险度`` 选最大的。

期望得分用"未来 H 步里有多少帧站在目标（Boss/敌机）的火力线上"估计：
mock 与正作都是"对准才掉血/加分"，所以这个量正比于 ``score_rate`` 分/帧。
这样 ``score_scale`` 一变，标签就会变 —— 可以在测试里直接断言。
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..types import PLAYFIELD_HEIGHT, PLAYFIELD_WIDTH, GameState

#: 9 个方向 (与 ``trainer.DIRECTIONS`` / 动作表同序; tests 里有断言)
DIRS: tuple[tuple[int, int], ...] = (
    (0, 0), (0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (1, -1), (-1, 1), (1, 1),
)
NAMES: tuple[str, ...] = (
    "", "up", "down", "left", "right", "up_left", "up_right", "down_left", "down_right",
)
#: 时间展开的采样点 (帧)
HORIZON = np.array([0.0, 3.0, 6.0, 10.0, 15.0, 22.0])
#: 对准估计的采样点数 (窗口长度由 ``align_horizon`` 决定)
ALIGN_SAMPLES = 21
#: 长安全窗口(>22 帧)时的采样点数
SAFETY_SAMPLES = 13


class ScoreTeacher:
    """安全优先、其次按分数排序的离散动作老师."""

    def __init__(
        self,
        *,
        weights: dict[str, Any] | None = None,
        safety_margin: float = 12.0,
        align_tol: float = 20.0,
        speed: float = 4.4,
        score_rate: float = 4.0,
        commit_margin: float = 0.0,
        barrier_weight: float = 0.0,
        align_horizon: float = 22.0,
        safety_horizon: float = 22.0,
        wall_horizon: float = 0.0,
        wall_dash_frames: float = 26.0,
        edge_margin: float = 24.0,
        edge_weight: float = 0.0,
        focus_radius: float = 0.0,
    ) -> None:
        w = dict(weights or {})
        self.score_scale = float(w.get("score_scale", 0.03) or 0.0)
        self.boss_scale = float(w.get("boss_damage_scale", 0.1) or 0.0)
        self.graze_scale = float(w.get("graze_scale", 0.0) or 0.0)
        self.danger_scale = float(w.get("danger_scale", 0.8) or 0.0)
        self.danger_radius = float(w.get("danger_radius", 32.0) or 32.0)
        #: 安全阈值(px): 未来 H 步内与最近弹幕的最小距离低于它, 该方向直接出局
        self.safety_margin = float(safety_margin)
        #: 对齐窗口(px): |自机 x - 目标 x| 小于它才算"在火力线上"
        self.align_tol = max(1.0, float(align_tol))
        self.speed = float(speed)
        #: 对准时每帧的期望得分 (mock 是 4 分/帧; 正机可粗估为一次判定的得分/帧)
        self.score_rate = float(score_rate)
        #: 滞回: 新方向要比"保持当前方向"好这么多才切换 (减少左右互搏)
        self.commit_margin = float(commit_margin)
        #: 软安全屏障权重 (0 = 关闭): clearance < safety_margin 时按平方惩罚,
        #: ``HybridTeacher`` 用它替代硬过滤, 让标签在阈值附近保持连续。
        self.barrier_weight = float(barrier_weight)
        #: 对准收益的估计窗口(帧), 默认与安全窗口同长。
        #: 实测拉长到 60 帧会更爱对准、但**看不见远处弹幕**, 存活从 4000 掉到 1350,
        #: 所以这里保持短窗口; 想试长窗口请同时把 barrier 调大并重新验收存活。
        self.align_horizon = float(align_horizon)
        #: 安全窗口(帧): 22 帧看不见正在形成的弹幕墙; 45~60 帧才能提前横向占位
        self.safety_horizon = float(safety_horizon)
        #: 直线弹幕墙的前瞻帧数 (0=关闭)。只对 wall_mask 挑出的墙生效,
        #: 这样不会像"全体长时预测"那样把弧线弹的虚假危险算进来。
        self.wall_horizon = float(wall_horizon)
        #: 墙前瞻里"一次冲刺"的帧数: 超过它就不再假设还在同一方向移动
        self.wall_dash_frames = float(wall_dash_frames)
        #: 贴边惩罚: 离四面墙 < edge_margin 时按平方扣分。**默认关闭**:
        #: 实测开启(weight=2.0/margin=24) 后 stage1 存活 4770 -> 2650 (对整齐升到 95%,
        #: 即"为了不贴边而站在弹道里")。保留实现是因为它在"被墙挤到角落"的场景仍有用,
        #: 但不要默认开 —— 这是已经量过的负面结果, 不要重复踩。
        self.edge_margin = float(edge_margin)
        self.edge_weight = float(edge_weight)
        self.focus_radius = float(focus_radius)
        self.last_direction = 0

    # ------------------------------------------------------------------
    def aim_x(self, state: GameState, px: float, py: float) -> float | None:
        """攻击目标 x: 优先 Boss, 否则取正上方最近的敌机."""
        pos = (state.raw or {}).get("boss_position")
        if isinstance(pos, (list, tuple)) and len(pos) >= 2:
            try:
                return float(pos[0])
            except (TypeError, ValueError):
                pass
        above = [(float(x), float(y)) for x, y in state.enemy_positions if float(y) < py - 20]
        if not above:
            return None
        return min(above, key=lambda e: abs(e[0] - px))[0]

    # ------------------------------------------------------------------
    def wall_mask(self, state: GameState, py: float) -> np.ndarray:
        """挑出"正在压下来的直线弹幕墙"的弹幕下标.

        为什么单独挑: 长时(45~90 帧)预测只对**匀速直线**的弹幕成立。对全部弹幕
        做长时预测会把弧线/反弹弹的虚假危险算进来 —— 实测 stage1 存活从 4770 掉到
        2971; 而完全不做长时预测又看不见"墙", wall_gate 只能撑 546 步。

        判据: 在自机上方、正在下落、x 方向密集 (出现弹幕墙特有的等距排列)。
        """
        if not state.bullets or self.wall_horizon <= 0.0:
            return np.zeros(0, dtype=np.int64)
        vel_raw = (state.raw or {}).get("bullet_velocity") or []
        if len(vel_raw) != len(state.bullets):
            return np.zeros(0, dtype=np.int64)
        arr = np.asarray(state.bullets, dtype=np.float64).reshape(-1, 2)
        vel = np.asarray(vel_raw, dtype=np.float64).reshape(-1, 2)
        above = (arr[:, 1] < py - 30.0) & (vel[:, 1] > 0.3) & (np.abs(vel[:, 0]) < 1.0)
        idx = np.flatnonzero(above)
        if idx.size < 8:
            return np.zeros(0, dtype=np.int64)
        xs = np.sort(arr[idx, 0])
        if float(np.median(np.diff(xs))) > 16.0:
            return np.zeros(0, dtype=np.int64)  # 稀疏 -> 不是墙, 是零散弹
        return idx

    # ------------------------------------------------------------------
    def candidates(self, state: GameState, px: float, py: float) -> list[dict[str, Any]]:
        """9 个方向的时间展开: 位置路径、最小间距、对准帧数."""
        bullets = (np.asarray(state.bullets, dtype=np.float64).reshape(-1, 2)
                   if state.bullets else np.zeros((0, 2)))
        vel_raw = (state.raw or {}).get("bullet_velocity") or []
        vel = (np.asarray(vel_raw, dtype=np.float64).reshape(-1, 2)
               if len(vel_raw) == len(bullets) else np.zeros_like(bullets))
        aim = self.aim_x(state, px, py)
        # 安全窗口: 22 帧时看不见"正在形成的弹幕墙"(从画面上方压下来通常要 40+ 帧),
        # 拉长到 45~60 帧才能提前横向占位 —— 这是 wall_gate 型弹幕的唯一解法。
        if self.safety_horizon > float(HORIZON[-1]):
            t_safe = np.linspace(0.0, self.safety_horizon, SAFETY_SAMPLES)
        else:
            t_safe = HORIZON
        # 墙: 只用**直线下落**的弹做长时预测 (见 wall_mask 的说明)
        wall = self.wall_mask(state, py)
        if wall.size:
            t_wall = np.linspace(0.0, self.wall_horizon, SAFETY_SAMPLES)
            wpos = bullets[wall][None, :, :] + t_wall[:, None, None] * vel[wall][None, :, :]
        else:
            wpos = None

        out: list[dict[str, Any]] = []
        for i, (ux, uy) in enumerate(DIRS):
            norm = math.hypot(ux, uy)
            d = np.array([ux, uy], dtype=float) / norm if norm else np.zeros(2)
            path = np.array([px, py]) + t_safe[:, None] * d * self.speed
            path = np.clip(path, [8.0, 32.0], [PLAYFIELD_WIDTH - 8.0, PLAYFIELD_HEIGHT - 8.0])

            clearance = 9999.0
            if len(bullets):
                pred = bullets[None, :, :] + t_safe[:, None, None] * vel[None, :, :]
                clearance = float(np.linalg.norm(pred - path[:, None, :], axis=2).min())
            if wpos is not None:
                # 同一条"保持该方向"的假设下, 看墙压到自机高度时会不会撞上。
                # 但横向位移只算一次冲刺 (wall_dash_frames) 就饱和: 墙要 40+ 帧才落下来,
                # 真按 40 帧一直往左, 会先冲过缺口撞上另一侧的墙 —— 现实里会中途改判,
                # 所以把"我会一直走"换成"我先冲一段, 到位后重新决策"。
                dash = np.minimum(t_wall, self.wall_dash_frames)
                wpath = np.array([px, py]) + dash[:, None] * d * self.speed
                wpath = np.clip(wpath, [8.0, 32.0],
                                [PLAYFIELD_WIDTH - 8.0, PLAYFIELD_HEIGHT - 8.0])
                clearance = min(clearance, float(
                    np.linalg.norm(wpos - wpath[:, None, :], axis=2).min()))

            aligned = 0.0
            approach = 0.0
            if aim is not None:
                # 对准收益用**自己的**时间网格, 与安全窗口解耦 (两者的最优长度不同)
                if self.align_horizon > float(HORIZON[-1]):
                    t_align = np.linspace(0.0, self.align_horizon, ALIGN_SAMPLES)
                else:
                    t_align = HORIZON
                along = np.clip(
                    np.array([px, py]) + t_align[:, None] * d * self.speed,
                    [8.0, 32.0], [PLAYFIELD_WIDTH - 8.0, PLAYFIELD_HEIGHT - 8.0],
                )
                kernel = np.maximum(0.0, 1.0 - np.abs(along[:, 0] - aim) / self.align_tol)
                # 均值 × 窗口长度 = "等效完全对准帧数" (采样点等距时才成立, 这里两者都等距)
                aligned = float(kernel.mean() * float(t_align[-1]))
                # 远端目标够不到时 aligned 恒为 0 (例如 Boss 刚入场、离得很远),
                # 这时给一个很弱的"靠近"项, 免得所有方向同分 -> 原地不动。
                approach = float(1.0 - min(1.0, abs(float(along[-1, 0]) - aim) / PLAYFIELD_WIDTH))

            out.append({
                "index": i,
                "clearance": clearance,
                "aligned": aligned,
                "approach": approach,
                "path": path,
            })
        return out

    # ------------------------------------------------------------------
    def value(self, cand: dict[str, Any]) -> float:
        """候选的"奖励估计": 期望得分 − 危险度 − 软安全屏障 − 贴边惩罚."""
        attack = ((self.score_scale + self.boss_scale) * self.score_rate
                  * (cand["aligned"] + 0.25 * cand.get("approach", 0.0)))
        danger = 0.0
        if self.danger_scale > 0.0:
            closeness = max(0.0, 1.0 - cand["clearance"] / max(1e-6, self.danger_radius))
            danger = self.danger_scale * closeness
        barrier = 0.0
        if self.barrier_weight > 0.0:
            margin = max(1e-6, self.safety_margin)
            short = max(0.0, margin - float(cand["clearance"])) / margin
            barrier = self.barrier_weight * short * short
        edge = 0.0
        if self.edge_weight > 0.0:
            edge = self.edge_weight * self._edge_risk(cand.get("path"))
        return attack - danger - barrier - edge

    # ------------------------------------------------------------------
    def _edge_risk(self, path: Any) -> float:
        """贴边风险 ∈ [0, 1]: 终点离四面墙越近越大 (角落会被加倍计, 这是有意的).

        score/hybrid 老师最初没有这一项, 于是"往下躲"永远是最大间距的方向 ——
        实测策略被一路顶到底边送死。规则老师有边界斥力, 这里补齐。
        """
        if path is None or self.edge_margin <= 0.0:
            return 0.0
        p = np.asarray(path, dtype=np.float64)
        if p.ndim != 2 or p.shape[0] == 0:
            return 0.0
        x, y = float(p[-1, 0]), float(p[-1, 1])
        m = self.edge_margin
        risk = 0.0
        for d in (x - 8.0, (PLAYFIELD_WIDTH - 8.0) - x, y - 32.0, (PLAYFIELD_HEIGHT - 8.0) - y):
            if d < m:
                risk += ((m - d) / m) ** 2
        return risk

    # ------------------------------------------------------------------
    def choose(self, state: GameState, frame: int, learned_action: str):
        px = float(state.player_x) if state.player_x is not None else PLAYFIELD_WIDTH / 2
        py = float(state.player_y) if state.player_y is not None else PLAYFIELD_HEIGHT - 32
        cands = self.candidates(state, px, py)

        safe = [c for c in cands if c["clearance"] >= self.safety_margin]
        nearest = min(c["clearance"] for c in cands)
        if safe:
            for c in safe:
                c["value"] = self.value(c)
            best = max(safe, key=lambda c: (c["value"], c["clearance"], -c["index"]))
            # 滞回: 保持当前方向, 除非新方向明显更好 (避免每帧左右互搏)
            keep = next((c for c in safe if c["index"] == self.last_direction), None)
            if keep is not None and self.commit_margin > 0.0:
                if keep["value"] >= best["value"] - self.commit_margin:
                    best = keep
        else:
            # 全都不安全: 退化为"离弹幕最远" (与规则老师一致)
            best = max(cands, key=lambda c: (c["clearance"], -c["index"]))
            best["value"] = self.value(best)

        self.last_direction = int(best["index"])
        name = NAMES[best["index"]]
        parts = [name] if name else []
        parts.append("shoot")
        if nearest < self.focus_radius:
            parts.append("focus")
        action = "+".join(parts)
        return action, {
            "score": round(float(best.get("value", 0.0)), 4),
            "clearance": round(float(best["clearance"]), 2),
            "aligned": round(float(best["aligned"]), 2),
            "safe_dirs": len(safe),
            "planner": "score",
        }
