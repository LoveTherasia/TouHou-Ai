"""把 MPC 当老师：在线规划 → 打标签 / 采样数据。

为什么需要这一层
----------------
``docs/PLANNER_V2.md`` 的路线是"规划器在线产生动作 → 收集 ``(state, planned_action)``
→ BC/DAgger 蒸馏到网络 → RL 微调"。但 ``MPCPlanner`` 的接口是
``choose(state) -> "up+left+shoot"``（bench 用的格式），而训练器的老师接口是
``choose(state, frame, prev_action) -> (动作名, info)``，且动作名必须用
``trainer.DIRECTIONS`` 的 ``up_left`` 下划线写法。本模块只做这层适配 + 两点工程处理：

1. **承诺期** ``plan_every``：MPC 每帧重规划既贵又会抖，规划一次执行 N 帧更稳；
2. **动作名转换**：``up+left+shoot`` → ``up_left+shoot``（``focus`` 后缀保留）。

实测（mock，6 弹型，2000 步上限，见 ``docs/PLANNER_V2.md`` §8）：

* 短视界 24 帧：反应式弹型全 2000，``wall_gate`` 只有 ~550；
* 长视界 72 帧 + 缺口锚点：``wall_gate`` ~1100，但反应式弹型退化；
* 自适应（按 ``WallDetector`` 切）：5/6 满分，``wall_gate`` ~1150。

## 老师确定化 + cost-to-go（2026-10-22 晚，见 PLANNER_V2 §10）

"MPC 当老师蒸馏"第一次失败的核心不是学生学不会，而是**老师自己每帧答案都在变**。
这里补了三件事：

1. ``commit_steps`` / ``switch_margin``：选定方向后 N 帧不换向（除非新方向明显更便宜）；
2. ``deterministic``：每帧用 ``seed + frame`` 重播种 CEM 噪声，同一帧不再受调用历史影响；
3. ``cost_vector()``：给蒸馏用的**逐动作 cost-to-go**（不再是"老师这一帧选了谁"）。
   口径是 ``MPCPlanner.action_costs(mode="plan")`` —— 先走这一步、再接着走 CEM 的
   剩余路径，所以 argmin 就等于老师会做的动作（无承诺时实测 100% 一致，
   而旧的"整段按住不动"口径只有 ~49%）。
"""
from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from ..types import GameState
from .mpc import ACTION_NAMES, ACTION_NAMES_FOCUS, AdaptiveMPCPlanner, MPCPlanner


def to_trainer_name(action: str) -> str:
    """``up+left+shoot`` → ``up_left+shoot``（``trainer.DIRECTIONS`` 的写法）。"""
    parts = action.split("+")
    head = [p for p in parts if p not in ("shoot", "focus", "bomb")]
    rest = [p for p in parts if p in ("shoot", "focus", "bomb")]
    if head:
        return "+".join(["_".join(head), *rest])
    return "+".join(rest)


class MPCTeacher:
    """``MPCPlanner`` / ``AdaptiveMPCPlanner`` 的老师适配层。"""

    def __init__(self, *, horizon: int = 24, dt: int = 3, n_samples: int = 48,
                 iters: int = 3, gap_weight: float = 0.0, column_weight: float = 0.0,
                 clearance_margin: float = 10.0, adaptive: bool = False,
                 long_horizon: int = 72, long_dt: int = 6,
                 use_focus: bool = False, plan_every: int = 1,
                 step_frames: int = 1, warm_start: bool = True,
                 seed: int = 0, commit_steps: int = 0, switch_margin: float = 0.0,
                 focus_margin: float = 0.0, deterministic: bool = True,
                 smooth_clear: float = 0.0, smooth_sigma: float = 20.0,
                 commit_frames: int = 0,
                 top_weight: float = 0.0, top_margin: float = 42.0,
                 bottom_weight: float = 0.0, bottom_margin: float = 30.0,
                 edge_weight: float = 0.0, edge_margin: float = 26.0,
                 dynamics: Any | None = None) -> None:
        #: 训练器的 step_frames。MPC 的运动模型是"每决策 1 帧 × PLAYER_SPEED",
        #: 所以 step_frames>1 时必须把**自机速度**与**弹幕速度**同时乘 k,
        #: 否则规划器以为自己每步走 4.4px、实际走了 17.6px —— 实测存活从 2000
        #: 掉到 140 步 (它算好的"安全位置"根本来不及到)。见 ``_scaled``。
        self.step_frames = max(1, int(step_frames))
        #: 参数按**游戏帧**给 (与 bench 口径一致), 内部换算成"决策步":
        #: k=4 时 horizon 24 帧 = 6 个决策, dt 3 帧 = 1 个决策。
        #: 不换算的话 8 个决策 × 3 决策 = 96 帧, 视界凭空变成 4 倍。
        self._k = self.step_frames
        horizon_steps = max(2, int(round(horizon / self._k)))
        dt_steps = max(1, int(round(dt / self._k)))
        #: 老师确定化的三个旋钮 (见 docs/PLANNER_V2.md §10)。承诺期是主项:
        #: 它把"代价平坦处每帧随机翻方向"从源头掐掉, 这样 student 学的才不是噪声。
        #:
        #: ``commit_frames`` 用**游戏帧**给（与 horizon/dt 同口径），内部换成决策步。
        #: 8~15 帧是实测的舒服区间：太短（<6 帧）等于没有承诺，换向率回到 50%+；
        #: 太长（>20 帧）会在弹幕变向时"锁死"一个已经错了的方向。承诺期一结束
        #: （``_commit_left`` 归零）下一次 ``choose`` 就会重新规划并自由换向。
        if commit_frames > 0:
            commit_steps = max(1, int(round(float(commit_frames) / self._k)))
        self.commit_steps = max(0, int(commit_steps))
        self.commit_frames = int(commit_frames)
        self.switch_margin = float(switch_margin)
        self.focus_margin = float(focus_margin)
        self.deterministic = bool(deterministic)
        common = dict(n_samples=n_samples, iters=iters, seed=seed, use_focus=use_focus,
                      commit_steps=commit_steps, switch_margin=switch_margin,
                      focus_margin=focus_margin, deterministic=deterministic,
                      smooth_clear=smooth_clear, smooth_sigma=smooth_sigma,
                      top_weight=top_weight, top_margin=top_margin,
                      bottom_weight=bottom_weight, bottom_margin=bottom_margin,
                      edge_weight=edge_weight, edge_margin=edge_margin,
                      dynamics=dynamics)
        if adaptive:
            self.planner: Any = AdaptiveMPCPlanner(
                short_horizon=horizon_steps, short_dt=dt_steps,
                long_horizon=max(2, int(round(long_horizon / self._k))),
                long_dt=max(1, int(round(long_dt / self._k))),
                long_column=column_weight, long_gap=gap_weight, **common,
            )
            for sub in (self.planner.short, self.planner.long):
                sub.speed = 4.4 * self._k
                sub.focus_speed = 2.0 * self._k
        else:
            self.planner = MPCPlanner(
                horizon=horizon_steps, dt=dt_steps,
                speed=4.4 * self._k,
                gap_weight=gap_weight, column_weight=column_weight,
                clearance_margin=clearance_margin, warm_start=warm_start, **common,
            )
        self.plan_every = max(1, int(plan_every))
        self._cached: str | None = None
        self._age = 0
        self._scaled_src: GameState | None = None
        self._scaled_cache: GameState | None = None
        #: 每次规划实际耗时的滑动均值（毫秒），打印进度时用
        self.last_ms = 0.0

    # ------------------------------------------------------------------
    def _scaled(self, state: GameState) -> GameState:
        """把状态换算到"每决策 = ``step_frames`` 帧"的单位（弹速 × k）。

        结果按**状态对象身份**缓存：``choose`` 和 ``cost_vector`` 对同一帧都会调用，
        缓存后规划器只会做一次 CEM（否则一帧两次搜索，训练直接慢一倍）。
        """
        k = self.step_frames
        if k == 1:
            return state
        if self._scaled_src is state:
            return self._scaled_cache
        raw = dict(state.raw or {})
        for key in ("bullet_velocity", "bullet_accel"):
            series = raw.get(key)
            if not series:
                continue
            try:
                raw[key] = [(float(v[0]) * k, float(v[1]) * k) for v in series]
            except Exception:
                raw.pop(key, None)
        out = dataclasses.replace(state, raw=raw)
        self._scaled_src = state
        self._scaled_cache = out
        return out

    def choose(self, state: GameState, frame: int = 0,
               prev_action: str | None = None) -> tuple[str, dict]:
        """返回 ``(动作名, info)``；承诺期内直接复用上一次的规划结果。"""
        import time

        if self._cached is not None and self._age < self.plan_every:
            self._age += 1
            return self._cached, {"reused": True}
        t0 = time.perf_counter()
        action = self.planner.choose(self._scaled(state))
        self.last_ms = (time.perf_counter() - t0) * 1000.0
        name = to_trainer_name(action)
        self._cached = name
        self._age = 1
        return name, {"ms": self.last_ms}

    def soft_labels(self, state: GameState, temperature: float = 20.0) -> np.ndarray | None:
        """逐动作目标分布 (软标签)。老师用 focus 时是 18 维。

        用 softmax(-cost/T) 代替 argmax: 并列动作共享概率质量, 学生的梯度不会
        被"这一帧老师随机选了左边"这种噪声带跑。
        """
        planner = self.planner
        if hasattr(planner, "_pick"):
            planner = planner._pick(self._scaled(state))
        return planner.soft_labels(self._scaled(state), temperature)

    def cost_vector(self, state: GameState) -> np.ndarray | None:
        """逐动作 cost-to-go ``(n_actions,)`` —— cost-to-go 蒸馏的监督信号。

        与 ``soft_labels`` 的区别: 这里**不**做 softmax。学生回归的是
        ``-相对代价 / T``，自己 argmax；并列动作天然免疫（代价接近 → logits 接近），
        而且比交叉熵多保留了"绝对差距"的信息（差距大时梯度不会饱和到 0）。
        """
        planner = self.planner
        scaled = self._scaled(state)
        if hasattr(planner, "cost_vector"):
            try:
                return np.asarray(planner.cost_vector(scaled), dtype=np.float64)
            except Exception:
                return None
        if hasattr(planner, "action_costs"):
            try:
                return np.asarray(planner.action_costs(scaled), dtype=np.float64)
            except Exception:
                return None
        return None

    def describe(self) -> dict:
        return {"name": "mpc-teacher", "plan_every": self.plan_every,
                "commit_steps": self.commit_steps,
                "switch_margin": self.switch_margin,
                "deterministic": self.deterministic,
                "planner": self.planner.describe()}


__all__ = ["MPCTeacher", "to_trainer_name", "ACTION_NAMES", "ACTION_NAMES_FOCUS"]
