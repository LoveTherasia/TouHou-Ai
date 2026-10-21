"""力场先验 + 分数重排的混合老师 —— 兼顾"可模仿"与"有攻击意图"。

为什么需要它
------------
前面两个老师各有一半:

* ``RuleTeacher`` 的标签**好模仿**(力场是局部平滑的, 只用身边几十像素的弹幕),
  但它只会"躲 + 朝 Boss 归位", 不知道"打中敌人值多少分";
* ``ScoreTeacher`` 的标签**看不懂**(9 路 argmax + 全局最小间距, 而策略只能看到
  K 个最近弹幕), 但它确实把 ``score_scale`` 用进了决策。

混合的思路: **用规则老师的力场当先验, 只在它的邻域里按分数重排。**

具体做法:

1. 先验 ``prior[k] = prior_weight * cos(方向 k, 力场方向)`` —— 力场说往左, 那么左
   得分最高, 左上/左下拿到 0.7 的分数, 右边是 0(cos<0 截断)。**同分时由力场决定**,
   所以标签默认落在规则老师的那一族里(空间上仍然平滑、可模仿); 只有分数收益足够
   大(例如"站到 Boss 下面"满对准约 11 分)时才会偏离力场 —— 这正是需要的攻击意图;
2. 分数 ``attack = (score_scale + boss_damage_scale) * score_rate * 对准帧数`` ——
   ``score_scale`` 真的进决策, 想让它更爱打就把权重调大;
3. 安全用**软屏障**而不是硬阈值:
   ``barrier = barrier_weight * (1 - clearance/safety_margin)^2`` (仅 clearance < margin)。
   硬阈值会让"间距 11.9 与 12.0"这两个几乎相同的局面产生相反的标签, 这正是
   argmax 老师难模仿的原因之一; 二次屏障既保持"贴脸必躲"的优先级, 又 C1 连续。

三者的量级 (默认 ``prior_weight=2.5`` / ``barrier_weight=100`` / ``align_horizon=60``):
满对准 60 帧的收益约 31, 所以"能对准就对准"; 间距掉到 6px 时屏障 25, 与一次完整
对准相当; 间距 0 时屏障 100, 而真实掉残是 ``miss_penalty=-200`` —— **存活仍是第一位**。

为什么收益窗口 (60 帧) 要比安全窗口 (22 帧) 长: 得分是"持续对准"的累积, 只看 22
帧时"走过去对准"几乎不产生价值(还没走到窗口就结束了), 于是先验永远赢 -> 退化成
规则老师。安全必须用短窗口(远期弹幕预测不可靠), 收益可以用中程窗口, 两者解耦。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..types import PLAYFIELD_HEIGHT, PLAYFIELD_WIDTH, GameState
from .rule_teacher import RuleTeacher
from .score_teacher import DIRS, NAMES, ScoreTeacher


class HybridTeacher(ScoreTeacher):
    """力场先验 + 分数重排; 安全用软屏障而非硬过滤."""

    def __init__(
        self,
        *,
        weights: dict[str, Any] | None = None,
        safety_margin: float = 12.0,
        align_tol: float = 20.0,
        prior_weight: float = 2.5,
        barrier_weight: float = 25.0,
        align_horizon: float = 22.0,
        safety_horizon: float = 22.0,
        wall_horizon: float = 0.0,
        edge_weight: float = 0.0,
        commit_margin: float = 0.0,
        focus_radius: float = 0.0,
        rule: RuleTeacher | None = None,
    ) -> None:
        super().__init__(
            weights=weights,
            safety_margin=safety_margin,
            align_tol=align_tol,
            commit_margin=commit_margin,
            align_horizon=align_horizon,
            safety_horizon=safety_horizon,
            wall_horizon=wall_horizon,
            edge_weight=edge_weight,
            focus_radius=focus_radius,
        )
        #: 力场先验的权重 (与"满对准收益约 11"同量级才起作用)
        self.prior_weight = float(prior_weight)
        #: 软屏障权重: clearance 归零时的惩罚 (= 40 时任何分数都不值得换命)
        self.barrier_weight = float(barrier_weight)
        #: 先验来源: 与 rule_dagger 完全同一个力场, 保证两个老师同源
        self.rule = rule or RuleTeacher()

    # ------------------------------------------------------------------
    def prior(self, state: GameState, px: float, py: float) -> np.ndarray:
        """力场方向上的余弦先验 (9 维, 与 ``DIRS`` 同序)."""
        fx, fy, _ = self.rule.force_vector(state, px, py)
        rule_dir = RuleTeacher.direction_from_force(fx, fy)
        w = np.zeros(len(DIRS), dtype=np.float64)
        if rule_dir == 0:
            # 力场认为"不用动": 原地不动拿先验, 只有分数明显更好才会移动
            w[0] = self.prior_weight
            return w
        u = np.asarray(DIRS[rule_dir], dtype=np.float64)
        u = u / np.linalg.norm(u)
        for i, d in enumerate(DIRS):
            if i == 0:
                continue
            v = np.asarray(d, dtype=np.float64)
            cos = float(v @ u) / float(np.linalg.norm(v))
            w[i] = max(0.0, cos) * self.prior_weight
        return w

    # ------------------------------------------------------------------
    def choose(self, state: GameState, frame: int, learned_action: str):
        px = float(state.player_x) if state.player_x is not None else PLAYFIELD_WIDTH / 2
        py = float(state.player_y) if state.player_y is not None else PLAYFIELD_HEIGHT - 32
        cands = self.candidates(state, px, py)
        prior = self.prior(state, px, py)

        for c in cands:
            # 重排: 分数 − 危险度 − 软屏障 + 力场先验。没有硬过滤, 全由量级决定。
            c["value"] = self.value(c) + float(prior[c["index"]])
        best = max(cands, key=lambda c: (c["value"], c["clearance"], -c["index"]))
        if self.commit_margin > 0.0:
            keep = next((c for c in cands if c["index"] == self.last_direction), None)
            if keep is not None and keep["value"] >= best["value"] - self.commit_margin:
                best = keep

        self.last_direction = int(best["index"])
        name = NAMES[best["index"]]
        parts = [name] if name else []
        parts.append("shoot")
        nearest = min(c["clearance"] for c in cands)
        if nearest < self.focus_radius:
            parts.append("focus")
        return "+".join(parts), {
            "score": round(float(best.get("value", 0.0)), 4),
            "clearance": round(float(best["clearance"]), 2),
            "aligned": round(float(best["aligned"]), 2),
            "safe_dirs": int(sum(1 for c in cands if c["clearance"] >= self.safety_margin)),
            "prior_dir": int(best["index"]),
            "planner": "hybrid",
        }
