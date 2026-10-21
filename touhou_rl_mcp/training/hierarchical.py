"""分层切换: 常规局面交给策略, "墙 / 收网"局面交给可行域规划器.

为什么需要它
------------
`viability.ViabilityPlanner` 已经能在 8 种弹型上平均活到 1970/2000 步
(见 `docs/VIABILITY_PLANNER.md`), 但它有两条硬伤:

1. **慢**: 每次决策 25~40 ms (纯 numpy), 真机 60fps 闭环做不到;
2. **不打**: 纯躲避时对准率只有 0~20%, 而学习策略在常规弹型上有 90%+.

而学习策略的短板正好互补: 它的观测是"最近 16 颗弹", 天生是**局部**的 —— 看不见
150px 外正在压下来的墙, 所以在 `wall_gate` 上只有 814 步就死。

所以正确做法不是"用谁替谁", 而是**按局面切换**:

    默认: 策略 (快、会打、常规弹型够用)
    检测到"整幅横墙"或"多向收网": 规划器 (慢一点, 但那正是它唯一擅长的事)

检测器只读 `GameState` 的弹幕位置/速度, 是**符号**判断 (O(n) numpy), 不经过策略的
观测 —— 因为"墙封住了整幅"这件事本来就不可能塞进 16 颗最近弹里。

三个组件
--------
* :class:`WallDetector` —— 便宜的局面判别 (curtain / encircle);
* :class:`HierarchicalController` —— 推理期切换: 策略动作 or 规划器动作;
* :class:`HierarchicalTeacher` —— 训练期打标签: 只在被接管的步产生标签, 其余**弃权**
  (返回空动作)。弃权很重要: 若把"策略自己的动作"当标签, 交叉熵会变成
  "让它对自己更自信", 反而压掉熵、把坏习惯固化。

用法
----
    python tools/pattern_bench.py --policy .runtime/policies/attack_ppo_fixed.npz \
        --hier-switch            # 评测"分层切换"后的整体表现
    python tools/mock_full_clear.py --teacher hier_dagger --pattern-cycle wall_gate \
        --load .runtime/policies/attack_ppo_fixed.npz   # 只蒸墙段的标签
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from ..types import PLAYFIELD_HEIGHT, PLAYFIELD_WIDTH, GameState
from .viability import NAMES, ViabilityPlanner

__all__ = [
    "SwitchMetrics",
    "WallDetector",
    "HierarchicalController",
    "HierarchicalTeacher",
    "bullet_arrays",
]


def bullet_arrays(state: GameState) -> tuple[np.ndarray, np.ndarray]:
    """把 ``GameState.bullets`` / ``raw['bullet_velocity']`` 转成两个 (N,2) 数组.

    速度缺失时退回全 0 —— 检测器会把它们当成"静止弹", 不会误判成墙。
    """
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
    return pos, vel


@dataclass
class SwitchMetrics:
    """一次判别结果 (也用于诊断: 触发率、为什么触发)."""

    reason: str = ""          # ""=不切换 | "curtain" | "encircle"
    bullets: int = 0          # 场上弹数
    row_count: int = 0        # curtain: 命中该行带的弹数
    covered: float = 0.0      # curtain: 横向覆盖率 (0~1)
    eta: float = 0.0          # curtain: 该行带压到自机高度的帧数
    ring_bins: int = 0        # encircle: 命中的方向扇区数 (共 8)
    ring_count: int = 0       # encircle: 朝自机收缩的弹数

    @property
    def switched(self) -> bool:
        return bool(self.reason)

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": "planner" if self.reason else "policy",
            "reason": self.reason,
            "bullets": self.bullets,
            "covered": round(self.covered, 3),
            "eta": round(self.eta, 1),
            "ring_bins": self.ring_bins,
        }


class WallDetector:
    """判别"该交给规划器"的两类局面.

    ``curtain``: 一整排**匀速下落、横向密排**的弹, 且这一排封住了几乎整幅宽度
    (只留一个缺口)。判据是"覆盖率 + 最大连续空列", 所以散弹、扇形、环绕都不会命中。

    阈值是量出来的 (`.runtime/probe_curtain.py`, 用 ``attack_ppo_fixed`` 在 6 种局面
    上逐帧取"覆盖率最高的那一行"): ``wall_gate`` 弹数 36 / 覆盖 0.94 / y 标准差 0.0,
    其余弹型的覆盖率 p95 只有 0.62 (扇形 0.56、环 0.62、收网 0.31)。
    所以 ``min_coverage=0.78`` 能把两类局面干净地分开; 用 0.5 时
    ``dense_ring`` 会有 40% 的步被误判成墙 (把 90% 的对准率换成规划器的 79%)。

    ``encircle``: 以自机为中心、从 ≥7 个方向同时向内收缩的一圈弹 (``closing_ring``)。
    这类局面"站桩必死、往下躲也不行", 同样只有全局可行域能解。

    两个判据都只看弹幕的位置/速度, 不做时间展开, 所以每次判别约 0.1~0.3 ms。
    """

    def __init__(
        self,
        *,
        lookahead: float = 170.0,
        band: float = 26.0,
        column: float = 24.0,
        min_bullets: int = 14,
        min_coverage: float = 0.78,
        max_gate_columns: int = 4,
        down_vy: float = 0.7,
        max_abs_vx: float = 0.9,
        late_frames: float = 8.0,
        encircle_radius: float = 260.0,
        encircle_min_dist: float = 42.0,
        encircle_min_bullets: int = 8,
        encircle_min_bins: int = 7,
        encircle_max_spread: float = 120.0,
        encircle_min_approach: float = 0.5,
    ) -> None:
        self.lookahead = float(lookahead)
        self.band = float(band)
        self.column = float(column)
        self.min_bullets = int(min_bullets)
        self.min_coverage = float(min_coverage)
        self.max_gate_columns = int(max_gate_columns)
        self.down_vy = float(down_vy)
        self.max_abs_vx = float(max_abs_vx)
        self.late_frames = float(late_frames)
        self.encircle_radius = float(encircle_radius)
        self.encircle_min_dist = float(encircle_min_dist)
        self.encircle_min_bullets = int(encircle_min_bullets)
        self.encircle_min_bins = int(encircle_min_bins)
        self.encircle_max_spread = float(encircle_max_spread)
        self.encircle_min_approach = float(encircle_min_approach)
        self.ncols = max(1, int(np.ceil(PLAYFIELD_WIDTH / self.column)))

    # ------------------------------------------------------------------
    def detect(self, state: GameState) -> SwitchMetrics:
        px = float(state.player_x) if state.player_x is not None else PLAYFIELD_WIDTH / 2
        py = float(state.player_y) if state.player_y is not None else PLAYFIELD_HEIGHT - 32
        pos, vel = bullet_arrays(state)
        metrics = SwitchMetrics(bullets=int(len(pos)))
        if not len(pos):
            return metrics

        curtain = self._curtain(pos, vel, py)
        if curtain is not None:
            metrics.reason = "curtain"
            metrics.row_count, metrics.covered, metrics.eta = curtain
            return metrics

        ring = self._encircle(pos, vel, px, py)
        if ring is not None:
            metrics.reason = "encircle"
            metrics.ring_count, metrics.ring_bins = ring
        return metrics

    # ------------------------------------------------------------------
    def _curtain(
        self, pos: np.ndarray, vel: np.ndarray, py: float
    ) -> tuple[int, float, float] | None:
        """找"整幅横墙"; 返回 (弹数, 覆盖率, 到达帧数) 或 None."""
        vy = vel[:, 1]
        falling = (vy >= self.down_vy) & (np.abs(vel[:, 0]) <= self.max_abs_vx)
        if not falling.any():
            return None
        idx = np.flatnonzero(falling)
        # 只关心"还没越过自机、且在视野内"的那几排
        eta = (py - pos[idx, 1]) / np.maximum(vy[idx], 1e-6)
        near = (eta <= self.lookahead) & (eta >= -self.late_frames)
        idx, eta = idx[near], eta[near]
        if len(idx) < self.min_bullets:
            return None

        band = np.floor(pos[idx, 1] / self.band).astype(np.int64)
        best: tuple[int, float, float] | None = None
        best_eta = np.inf
        for b in np.unique(band):
            sel = np.flatnonzero(band == b)
            if len(sel) < self.min_bullets:
                continue
            cols = np.floor(pos[idx[sel], 0] / self.column).astype(np.int64)
            covered = np.unique(np.clip(cols, 0, self.ncols - 1))
            cov = len(covered) / self.ncols
            if cov < self.min_coverage:
                continue
            # "封住整幅" -> 不允许有太宽的连续空洞 (缺口之外不该有能钻的地方)
            if len(covered) > 1:
                widest = int(np.diff(covered).max() - 1)
                if widest > self.max_gate_columns:
                    continue
            eta_b = float(eta[sel].mean())
            if eta_b < best_eta:
                best_eta = eta_b
                best = (len(sel), float(cov), eta_b)
        return best

    # ------------------------------------------------------------------
    def _encircle(
        self, pos: np.ndarray, vel: np.ndarray, px: float, py: float
    ) -> tuple[int, int] | None:
        """找"多方向向内收缩的一圈弹"; 返回 (弹数, 命中扇区数) 或 None."""
        d = np.stack([px - pos[:, 0], py - pos[:, 1]], axis=1)
        dist = np.hypot(d[:, 0], d[:, 1])
        speed = np.hypot(vel[:, 0], vel[:, 1])
        near = (dist <= self.encircle_radius) & (dist >= self.encircle_min_dist) & (speed > 0.2)
        if int(near.sum()) < self.encircle_min_bullets:
            return None
        # 速度方向与"指向自机"的夹角余弦
        approach = (vel[near, 0] * d[near, 0] + vel[near, 1] * d[near, 1]) / (
            np.maximum(speed[near], 1e-6) * np.maximum(dist[near], 1e-6)
        )
        inward = near.copy()
        inward[near] = approach >= self.encircle_min_approach
        idx = np.flatnonzero(inward)
        if len(idx) < self.encircle_min_bullets:
            return None
        dd = dist[idx]
        if float(dd.max() - dd.min()) > self.encircle_max_spread:
            return None  # 距离太散 = 不是一圈, 更像路过
        angle = np.arctan2(d[idx, 1], d[idx, 0])
        bins = np.floor((angle + np.pi) / (2.0 * np.pi) * 8).astype(np.int64) % 8
        nbins = int(np.unique(bins).size)
        if nbins < self.encircle_min_bins:
            return None
        return int(len(idx)), nbins


class HierarchicalController:
    """推理期的分层切换: 策略动作为默认, 检测到墙/收网时改问规划器.

    ``policy`` 需提供 ``greedy(obs) -> int``; ``encode`` 是 ``state -> obs`` 的可调用
    对象 (通常是 ``ObservationEncoder(config)``)。
    """

    def __init__(
        self,
        policy: Any,
        encode: Callable[[GameState], np.ndarray],
        action_table: list[str],
        *,
        detector: WallDetector | None = None,
        planner: ViabilityPlanner | None = None,
        use_aim: bool = True,
        replan_every: int = 1,
    ) -> None:
        self.policy = policy
        self.encode = encode
        self.action_table = list(action_table)
        self.detector = detector or WallDetector()
        self.planner = planner or ViabilityPlanner()
        self.use_aim = use_aim
        #: 墙段内沿用上次方向的步数 (规划器 25~40ms/次, 逐帧算会拖慢演示/评测)
        self.replan_every = max(1, int(replan_every))
        self._dir = 0
        self._age = 0
        self._active = False
        self.last_metrics = SwitchMetrics()
        self.last_source = "policy"
        #: 累计交给规划器的步数 (演示/评估报告用)
        self.planner_steps = 0

    # ------------------------------------------------------------------
    def direction(self, state: GameState, obs: np.ndarray | None = None) -> int:
        """返回动作**下标** (对应 ``action_table``)."""
        metrics = self.detector.detect(state)
        self.last_metrics = metrics
        if not metrics.switched:
            self.last_source = "policy"
            self._active = False
            self._age = 0
            return int(self.policy.greedy(obs if obs is not None else self.encode(state)))
        self._age += 1
        if not self._active or self._age > self.replan_every:
            self._dir = int(self.planner.plan(state).direction)
            self._age = 1
        self._active = True
        self.last_source = "planner"
        self.planner_steps += 1
        return int(min(self._dir, len(self.action_table) - 1))

    def act(self, state: GameState, obs: np.ndarray | None = None) -> str:
        return self.action_table[self.direction(state, obs)]


class HierarchicalTeacher:
    """训练期的"只在需要的地方纠正"老师.

    非墙段**弃权** (返回空动作字符串), 墙段返回规划器动作。配合
    ``TrainConfig.teacher="hier_dagger"``: 策略驱动游戏, 墙段拿规划器的标签。

    ``replan_every`` 是"承诺期": 沿用上次方向这么多步再重算。规划器一次 25~40ms,
    墙段每步都算会让训练慢到不可用; 而相邻 2~4 帧的墙几乎没动, 沿用是安全的
    (顺带抑制左右互搏)。
    """

    def __init__(
        self,
        *,
        detector: WallDetector | None = None,
        planner: ViabilityPlanner | None = None,
        replan_every: int = 1,
        min_commit: int = 0,
    ) -> None:
        self.detector = detector or WallDetector()
        self.planner = planner or ViabilityPlanner()
        self.replan_every = max(1, int(replan_every))
        #: 进入墙段后至少承诺这么多步 (避免"刚切进来又走回去")
        self.min_commit = max(0, int(min_commit))
        self._dir = 0
        self._age = 0
        self._active = False
        self.stats: dict[str, int] = {"steps": 0, "curtain": 0, "encircle": 0, "replans": 0}

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self._age = 0
        self._active = False

    def choose(
        self, state: GameState, frame: int = 0, learned_action: str = ""
    ) -> tuple[str, dict]:
        metrics = self.detector.detect(state)
        self.stats["steps"] += 1
        if not metrics.switched:
            self._active = False
            self._age = 0
            return "", {"mode": "policy", "reason": ""}

        self.stats[metrics.reason] = self.stats.get(metrics.reason, 0) + 1
        self._age += 1
        need_plan = (not self._active) or self._age > self.replan_every
        if need_plan:
            info = self.planner.plan(state)
            self._dir = int(info.direction)
            self.stats["replans"] += 1
            self._age = 1
        self._active = True
        name = NAMES[self._dir]
        action = "+".join(p for p in (name, "shoot") if p)
        info = metrics.as_dict()
        info["planner_direction"] = self._dir
        return action, info
