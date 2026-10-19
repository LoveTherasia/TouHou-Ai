"""奖励函数.

奖励被刻意设计成"可在线调参"的形式 —— 每个分量都有独立的权重, Agent / 训练脚本
可以通过 MCP 的 ``configure`` 工具动态修改 ``reward`` 段, 这就是 agentic RL 里
"让 Agent 自己调整训练参数"的接口。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..types import GameState

#: 所有可调权重及其含义 (会出现在 get_config 的说明里)
WEIGHT_DOCS: dict[str, str] = {
    "score_scale": "Δ分数 * scale",
    "graze_scale": "Δ擦弹 * scale",
    "power_scale": "Δ灵力 * scale",
    "life_bonus": "每增加一个残机的奖励",
    "miss_penalty": "掉残(被弹)的惩罚",
    "death_penalty": "Game Over 的惩罚",
    "stage_clear_bonus": "通过一关的奖励",
    "game_clear_bonus": "通关的奖励",
    "boss_damage_scale": "对 Boss 造成伤害 * scale (需 boss_hp 偏移)",
    "spell_capture_bonus": "游戏确认获得符卡 Bonus 时的一次性额外奖励（不奖励超时/失去收取资格）",
    "spell_clear_bonus": "击破一张符卡的一次性奖励 (mock 由 spell_cleared 事件触发)",
    "invincible_output_scale": "无敌帧期间 分数/Boss伤害/存活奖励 的缩放; 0=被弹后无敌期刷输出不再得分 (封堵漏洞)",
    "invincible_frames": "回退推断用的无敌帧长度 (真机未标定 invincible 时, 掉残后多少帧内视为无敌)",
    "danger_scale": "危险度惩罚: 越靠近弹幕扣分越多 (需要 player_x/y + 弹幕)。这是教 RL「躲弹幕」最有效的密集信号",
    "danger_radius": "危险度感知半径(像素), 不是权重, 只是一个参数",
    "alive_bonus": "每个 step 的存活奖励 —— 把稀疏的死亡惩罚变成密集信号, 学习会快很多",
    "time_penalty": "每个 step 的固定惩罚, 鼓励速通",
    "clip": "单步奖励裁剪上限 (null 表示不裁剪)",
    "shaping_scale": "所有分量的整体缩放, Agent 在线调参入口",
}

#: 掉残后的默认无敌帧长度 (真机未标定 invincible 时的回退推断窗口)
DEFAULT_INVINCIBLE_FRAMES = 150


@dataclass
class RewardModel:
    weights: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.weights = {**{k: 0.0 for k in WEIGHT_DOCS}, **(self.weights or {})}
        #: 回退推断无敌帧用: 最近一次掉残的帧号 (真机未标定 invincible 时)
        self._hit_frame: int | None = None

    # ------------------------------------------------------------------
    def update_weights(self, patch: dict[str, Any]) -> dict[str, Any]:
        unknown = [k for k in patch if k not in WEIGHT_DOCS]
        if unknown:
            raise KeyError(f"未知的奖励权重: {', '.join(sorted(unknown))}")
        self.weights.update(patch)
        return dict(self.weights)

    def _w(self, name: str, default: float = 0.0) -> float:
        value = self.weights.get(name, default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    # ------------------------------------------------------------------
    def compute(
        self,
        prev: GameState | None,
        cur: GameState,
        *,
        done: bool = False,
        cleared: bool = False,
        game_over: bool = False,
    ) -> tuple[float, dict[str, float]]:
        parts: dict[str, float] = {}

        # 无敌帧封堵: 被弹后的无敌期内, 输出与存活奖励不再白给。
        # 否则策略会学成"故意被弹 -> 2.5 秒零风险输出"(把被弹变成收益)。
        invincible = self._invincible_active(prev, cur)
        output_scale = self._w("invincible_output_scale", 0.0) if invincible else 1.0

        if prev is not None:
            parts["score"] = self._delta(prev.score, cur.score) * self._w("score_scale") * output_scale
            parts["graze"] = self._delta(prev.graze, cur.graze) * self._w("graze_scale")
            parts["power"] = self._delta(prev.power, cur.power) * self._w("power_scale")

            d_lives = self._delta(prev.lives, cur.lives)
            if d_lives > 0:
                parts["life"] = d_lives * self._w("life_bonus")
            elif d_lives < 0:
                # 一帧内掉多次残机也各自计惩罚 (正常只会掉 1)
                parts["miss"] = self._w("miss_penalty") * (-d_lives)

            d_stage = self._delta(prev.stage, cur.stage)
            if d_stage and d_stage > 0:
                parts["stage_clear"] = d_stage * self._w("stage_clear_bonus")

            old = prev.raw.get("entities", {}).get("combat", {})
            new = cur.raw.get("entities", {}).get("combat", {})
            same_phase = (prev.stage, prev.spell_id) == (cur.stage, cur.spell_id)
            same_boss = old.get("boss_id") == new.get("boss_id")
            combat_valid = (not old and not new) or old.get("status") == new.get("status") == "ok"
            if (combat_valid and same_phase and same_boss and not new.get("timeout")
                    and prev.boss_hp is not None and cur.boss_hp is not None):
                dmg = prev.boss_hp - cur.boss_hp
                if dmg > 0:
                    parts["boss_damage"] = dmg * self._w("boss_damage_scale") * output_scale
            # 40CDC2 checks eligibility bit 2, then awards bonus/10 to raw score.
            bonus = new.get("spell_bonus", 0)
            if (old.get("status") == new.get("status") == "ok"
                    and old.get("spell_flags", 0) & 1
                    and not new.get("spell_flags", 0) & 1
                    and new.get("spell_flags", 0) & 2
                    and old.get("spell_id") == new.get("spell_id")
                    and prev.stage == cur.stage and bonus > 0
                    and self._delta(prev.score, cur.score) >= (bonus // 10) * 10):
                parts["spell_capture"] = self._w("spell_capture_bonus")

        if done:
            if game_over:
                parts["death"] = self._w("death_penalty")
            if cleared:
                parts["clear"] = self._w("game_clear_bonus")

        if cur.raw.get("spell_cleared"):
            parts["spell_clear"] = self._w("spell_clear_bonus")

        parts["alive"] = 0.0 if cur.raw.get("dialogue_assist") else self._w("alive_bonus") * output_scale
        parts["time"] = self._w("time_penalty")
        danger = self._danger(cur)
        if danger:
            parts["danger"] = danger

        total = sum(parts.values()) * self._w("shaping_scale", 1.0)
        clip = self.weights.get("clip")
        if clip is not None:
            try:
                limit = float(clip)
                total = max(-limit, min(limit, total))
            except (TypeError, ValueError):
                pass
        return total, {k: round(v, 6) for k, v in parts.items() if v}

    @staticmethod
    def _delta(before: Any, after: Any) -> float:
        if before is None or after is None:
            return 0.0
        return float(after) - float(before)

    # ------------------------------------------------------------------
    def _invincible_active(self, prev: GameState | None, cur: GameState) -> bool:
        """当前是否处于"被弹后的无敌帧".

        优先用状态里明确的无敌帧计数 (mock 直接提供, 真机需标定 ``invincible``);
        真机未标定时退化为"掉残后 N 帧内视为无敌", 同样能封堵刷输出。
        """
        inv = cur.invincible
        if inv is not None:
            try:
                return float(inv) > 0
            except (TypeError, ValueError):
                pass
        frame = int(cur.frame or 0)
        if prev is not None:
            pframe = int(prev.frame or 0)
            if frame < pframe:  # 帧计数重置 = 新的一局
                self._hit_frame = None
            d = self._delta(prev.lives, cur.lives)
            if d is not None and d < 0:
                self._hit_frame = frame
        if self._hit_frame is None:
            return False
        window = self._w("invincible_frames", float(DEFAULT_INVINCIBLE_FRAMES))
        return 0 <= (frame - self._hit_frame) < window

    # ------------------------------------------------------------------
    def _danger(self, state: GameState) -> float:
        """危险度惩罚: 离最近弹幕越近扣得越多.

        这是教 RL 躲弹幕最有效的**密集信号** —— 没有它, 策略只能从"什么时候死"
        这个稀疏事件里学, REINFORCE 基本学不动 (实测 600 局仍停在 -95)。
        """
        scale = self._w("danger_scale")
        if scale == 0.0:
            return 0.0  # 关掉时零开销
        if state.player_x is None or state.player_y is None or not state.bullets:
            return 0.0
        radius = self._w("danger_radius", 45.0) or 45.0
        px, py = float(state.player_x), float(state.player_y)
        try:
            import numpy as np

            arr = np.asarray(state.bullets, dtype=np.float64).reshape(-1, 2)
            distances = np.hypot(arr[:, 0] - px, arr[:, 1] - py)
            nearest = float(distances.min())
        except Exception:
            nearest = min(
                ((bx - px) ** 2 + (by - py) ** 2) ** 0.5 for bx, by in state.bullets
            )
        if nearest >= radius:
            return 0.0
        # 越近越接近 -scale
        return -scale * (1.0 - nearest / radius)

    # ------------------------------------------------------------------
    def describe(self) -> dict[str, Any]:
        return {
            "weights": dict(self.weights),
            "docs": WEIGHT_DOCS,
        }
