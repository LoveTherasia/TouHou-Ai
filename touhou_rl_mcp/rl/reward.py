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
    "gap_margin_scale": ("缺口余量的势函数塑形: 奖励 += scale * (φ' - φ), φ = 自机所在车道"
                         "的安全余量。势函数不改变最优策略, 只给'站着不动'一个密集负梯度"),
    "progress_scale": ("进度奖励: Δ(Boss 血量比) * scale —— 明确奖励「往前打」, "
                       "与存活项区分开, 避免策略把「活着」当成唯一目标"),
    "position_scale": ("位置偏好: 每步按「离理想站位多远」给分。理想站位 = 略低于中线 + "
                       "与 Boss 同列; 越贴边扣得越多。它治的是「贴底横移也能活满 2000 步」"),
    "position_y_target": "位置偏好的理想 y (归一化, 0=场地顶, 1=场地底; 默认 0.85)",
    "position_edge": "贴边判定的像素阈值 (默认 28px)",
    "clip": "单步奖励裁剪上限 (null 表示不裁剪)",
    "shaping_scale": "所有分量的整体缩放, Agent 在线调参入口",
}

#: 掉残后的默认无敌帧长度 (真机未标定 invincible 时的回退推断窗口)
DEFAULT_INVINCIBLE_FRAMES = 150

#: 实验用奖励覆盖 (具名 profile) —— 只写"偏离默认值"的项。
#: config.json 的 ``reward`` 段是唯一默认值来源; 工具做实验时引用这里的 profile,
#: 避免同一个实验在不同脚本里各写一套参数 (历史上 mock_clear / mock_train_clear
#: 就因为自带 REWARD 而与 config.json 冲突, 导致实验结论不可比)。
REWARD_PROFILES: dict[str, dict[str, float]] = {
    # 真机第一面测试: 削弱 Boss 伤害权重 (避免站桩对轰), 保留符卡收取奖励
    "real_stage1": {"boss_damage_scale": 0.01, "spell_capture_bonus": 20.0},
    # 缺口生存 (wall_gate 等"活下来就是目标"的基准): 过面进度 + 擦弹 - 碰撞。
    # **必须关掉 danger_scale**: 斥力项教的是"离弹越远越好", 而密集弹幕里"远"只有
    # 墙角 —— 这正是"被逼到角落再被弹"的成因。耗时/速通/输出项一并归零, 否则
    # 策略会拿"站着不动攒输出"换生存。实测依据见 docs/GAP_LEARNING.md。
    "gap_survive": {
        "alive_bonus": 1.0,          # 过面进度: 每多活一步 +1 (2400 帧上限 ≈ 400 步)
        "graze_scale": 0.05,         # 擦弹: 主动贴缝穿过才有, 是"找缝"的密集信号
        "miss_penalty": -40.0,       # 碰撞: 一次掉残 ≈ 40 步生存, 足够重但不致于瘫痪
        "death_penalty": -40.0,      # Game Over
        "danger_scale": 0.0,         # 关键: 关掉斥力 (它会把人赶进墙角)
        "time_penalty": 0.0,         # 不做速通压力
        "score_scale": 0.0,
        "power_scale": 0.0,
        "boss_damage_scale": 0.0,
        "spell_capture_bonus": 0.0,
        "spell_clear_bonus": 0.0,
        "life_bonus": 0.0,
        "stage_clear_bonus": 0.0,
        "game_clear_bonus": 0.0,
        "clip": 100.0,
        # profile 是"整份覆盖"的起点: 漏掉 shaping_scale 会让总奖励恒为 0
        # (RewardModel 里所有未列出的权重默认 0.0, 包括这个整体缩放)。
        "shaping_scale": 1.0,
    },
    # 攻击意识 (生存仍然优先): 让"打中敌人"从噪声变成有效信号。
    # **2026-10-21 起这份配平已是 config.json 的默认值**; 保留具名 profile 是为了
    # 在命令行/实验里显式引用 (tools/*.py --reward-profile attack)。
    #
    # 实测依据 (mock 第一面, 规则老师 3196 步, 旧权重):
    #   danger -3306 / stage_clear +3000 / alive +825 / boss_damage +460 / **score +0.2**
    # —— 危险度惩罚比其它所有项加起来还大 5 倍, 而分数项等于不存在。于是策略唯一的
    # 学习信号就是"离弹幕越远越好", 而密集弹幕下"远"只有角落: 不攻击是这套权重的
    # 必然结果, 不是模型学不会。
    #
    # 新配平的三条原则:
    #   1. **分数有分量**: score_scale 1e-5 -> 3e-2。对准 Boss 时 mock 每帧 +4 分,
    #      即 0.12 奖励/步, 与 alive_bonus (0.3) 同量级; 一局 2 万分 ≈ +600。
    #   2. **危险度降为"微调"而不是"主导"**: 4.0 -> 0.8、半径 50 -> 32。它仍会在
    #      贴脸时扣分, 但不再压过"站到 Boss 下面开火"的收益。
    #   3. **存活仍第一**: miss -200 / death -600 不动 —— 一次被弹约等于 1000 步的
    #      攻击收益, 所以"用命换分"依然不划算。
    "attack": {
        "score_scale": 3e-2,
        "boss_damage_scale": 0.1,
        "danger_scale": 0.8,
        "danger_radius": 32.0,
        "alive_bonus": 0.3,
        "time_penalty": -0.01,
        "graze_scale": 0.02,
        "miss_penalty": -200.0,
        "death_penalty": -600.0,
        "stage_clear_bonus": 1500.0,
        "game_clear_bonus": 6000.0,
        "spell_capture_bonus": 20.0,
        "spell_clear_bonus": 150.0,
        "life_bonus": 5.0,
        "power_scale": 0.1,
        "invincible_output_scale": 0.0,
        "clip": 8000.0,
        "shaping_scale": 1.0,
    },
    # 多目标: 存活 + 进度 + 分数 + 位置偏好
    #
    # 与 attack 档的区别只有两点:
    #   1. 有独立的"进度"项 (打掉 Boss 血量), 不再完全依赖 score 的间接信号;
    #   2. 有"位置偏好"项, 把"贴底横移/缩在角落"这种苟活行为变成负分。
    # 存活仍是第一优先级: miss -200 / death -600 不动。
    "multi": {
        "score_scale": 3e-2,
        "progress_scale": 6.0,
        "position_scale": 0.5,
        "position_y_target": 0.85,
        "position_edge": 28.0,
        "boss_damage_scale": 0.1,
        "danger_scale": 0.8,
        "danger_radius": 32.0,
        "alive_bonus": 0.3,
        "time_penalty": -0.01,
        "graze_scale": 0.02,
        "miss_penalty": -200.0,
        "death_penalty": -600.0,
        "stage_clear_bonus": 1500.0,
        "game_clear_bonus": 6000.0,
        "spell_capture_bonus": 20.0,
        "spell_clear_bonus": 150.0,
        "life_bonus": 5.0,
        "power_scale": 0.1,
        "invincible_output_scale": 0.0,
        "clip": 8000.0,
        "shaping_scale": 1.0,
    },
    # 攻击优先的多目标: 专门修 PPO 微调里的"苟化"漂移。
    #
    # 实测（PLANNER_V2 §15）: 用 multi 档跑 PPO，存活 463 -> 902 步（+95%），
    # 但训练日志里"伤害(均)"从 500 一路掉到 105、对准率 35% -> 7% ——
    # 策略学会了"贴着中场横移苟活"。根因是**量级**: multi 里
    #   progress_scale 6.0  × Δ血量比(0~1)  = 整条血最多 6 分
    #   alive_bonus    0.3  × 600 步        ≈ 180 分
    # 输出与存活是 1:30，PPO 当然选存活。这里把它们拉回同一量级：
    #   progress 60 (整条血 ≈ 200 步存活)、boss_damage 0.3、alive 0.1。
    # 注意 miss/death 惩罚一个字没动 —— 存活仍是硬约束，只是不再"躺着也赚"。
    "multi_attack": {
        "score_scale": 3e-2,
        "progress_scale": 60.0,
        "position_scale": 0.5,
        "position_y_target": 0.85,
        "position_edge": 28.0,
        "boss_damage_scale": 0.3,
        "danger_scale": 0.8,
        "danger_radius": 32.0,
        "alive_bonus": 0.1,
        "time_penalty": -0.01,
        "graze_scale": 0.02,
        "miss_penalty": -200.0,
        "death_penalty": -600.0,
        "stage_clear_bonus": 1500.0,
        "game_clear_bonus": 6000.0,
        "spell_capture_bonus": 20.0,
        "spell_clear_bonus": 150.0,
        "life_bonus": 5.0,
        "power_scale": 0.1,
        "invincible_output_scale": 0.0,
        "clip": 8000.0,
        "shaping_scale": 1.0,
    },
}


def reward_profile(name: str) -> dict[str, float]:
    """取一份具名奖励覆盖 (返回副本, 调用方可以安全修改)。"""
    try:
        return dict(REWARD_PROFILES[name])
    except KeyError:
        raise KeyError(
            f"未知的奖励 profile: {name}; 可用: {', '.join(sorted(REWARD_PROFILES))}"
        ) from None


@dataclass
class RewardModel:
    weights: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.weights = {**{k: 0.0 for k in WEIGHT_DOCS}, **(self.weights or {})}
        #: 回退推断无敌帧用: 最近一次掉残的帧号 (真机未标定 invincible 时)
        self._hit_frame: int | None = None
        #: 缺口势函数的上一帧 φ (None = 还没建立基线 / 刚换局)
        self._gap_phi: float | None = None

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

        # 进度项: Boss 血量比的下降 (与 boss_damage 分开, 归一化后与血量上限无关)
        progress = 0.0
        if (prev is not None and prev.boss_hp is not None and cur.boss_hp is not None
                and prev.boss_max_hp and cur.boss_max_hp):
            before = prev.boss_hp / max(1.0, prev.boss_max_hp)
            after = cur.boss_hp / max(1.0, cur.boss_max_hp)
            if before >= after:
                progress += (before - after) * self._w("progress_scale") * output_scale
        if progress:
            parts["progress"] = progress

        position = self._position_pref(cur)
        if position:
            parts["position"] = position

        parts["alive"] = 0.0 if cur.raw.get("dialogue_assist") else self._w("alive_bonus") * output_scale
        parts["time"] = self._w("time_penalty")
        danger = self._danger(cur)
        if danger:
            parts["danger"] = danger
        margin = self._gap_margin(prev, cur)
        if margin:
            parts["gap"] = margin

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
    def _gap_margin(self, prev: GameState | None, cur: GameState) -> float:
        """缺口势函数塑形: ``scale * (φ' − φ)``, φ = 自机所在车道的安全余量。

        为什么用势函数而不是直接奖励 φ: 势函数差分的累积回报只取决于首尾 (望远镜
        求和), **不改变最优策略**, 所以它不会像 ``danger_scale`` 那样把策略改造成
        "离弹越远越好" —— 那在密集弹幕里等价于"缩到墙角"。

        换局时 (帧号回退) 只重建基线, 不产生跨局的大额跳变。
        """
        scale = self._w("gap_margin_scale")
        if scale == 0.0:
            return 0.0
        if cur.player_x is None or cur.player_y is None or not cur.bullets:
            return 0.0
        try:
            from ..training.occupancy import lane_safety

            phi = lane_safety(cur)
        except Exception:  # pragma: no cover - 观测缺失时不影响其它奖励
            return 0.0
        previous = self._gap_phi
        frame = int(cur.frame or 0)
        prev_frame = int(prev.frame or 0) if prev is not None else -1
        self._gap_phi = phi
        if previous is None or prev is None or frame < prev_frame:
            return 0.0
        return scale * (phi - previous)

    # ------------------------------------------------------------------
    def _position_pref(self, state: GameState) -> float:
        """位置偏好: 惩罚贴边/贴底, 奖励"站在 Boss 下方、略低于中线"。

        密集弹幕里"离弹越远越好"会把策略推到角落 (那里弹最少), 于是出现
        "活满 2000 步但从不输出"的退化。位置偏好给这种退化解一个明确的负梯度。

        值域约 ``[-1.5, 0]``, 再乘 ``position_scale``。
        """
        scale = self._w("position_scale")
        if scale == 0.0:
            return 0.0
        if state.player_x is None or state.player_y is None:
            return 0.0
        px = float(state.player_x)
        py = float(state.player_y)
        target = self._w("position_y_target", 0.85)
        edge = self._w("position_edge", 28.0)

        # 纵向: 离理想高度的差距 (归一化)
        score = -abs(py / 448.0 - target)
        # 横向: 与 Boss 同列 (没有 Boss 位置时退化成"离中线近")
        boss = (state.raw or {}).get("boss_position")
        if boss:
            score -= abs(px - float(boss[0])) / 384.0
        else:
            score -= 0.5 * abs(px - 192.0) / 192.0
        # 贴边惩罚: 角落是"弹最少"的逃逸点, 必须让它的收益变成负的
        if px < edge:
            score -= (edge - px) / edge
        elif px > 384.0 - edge:
            score -= (px - (384.0 - edge)) / edge
        return scale * score

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
