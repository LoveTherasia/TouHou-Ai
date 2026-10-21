"""对话 / 静默期处理.

TH11 在**开局、道中与 Boss 前**都有剧情对话: 此时游戏会**屏蔽射击**, 玩家的攻击
被迫中断; 按 Z(射击键) 可以推进/快进对话。若只是偶尔轻点一下, 对话会拖很久,
策略在对话里"空转", 训练与通关都被拖慢。

本模块只做两件事:
  1. **检测** "局内但场上无弹幕、且没有会攻击的敌人" 的静默/对话窗口;
  2. 让上层控制器在该窗口内**把射击键按住整个 step** (连续快进), 尽快跳过对话、
     恢复攻击; 同时**保留策略本步的移动意图** (只补一个 shoot, 不覆盖方向)。

注意: 这只是启发式, 不是对话识别器 —— 波次间隙也可能被判为"对话", 但此时
按住射击本就无害 (正好在输出)。

真机实测 (2026-10-21): TH11 的对话窗口里 ``enemies_alive`` **不是 0** —— 剧本会
留 1~2 个不攻击的演员 (第一面 f=787 的 275 帧窗口就是这种)。原来要求"敌人必须为 0"
导致 ``assist`` 一次都没触发过, 对话只能靠策略自己偶尔按 Z 慢慢过。现在改成
"敌人 <= dialogue_max_enemies (默认 2) 且弹幕为 0", 并在配置里把触发阈值从 25 帧
降到 12 帧、单步点按上限从 12 次提到 30 次。
"""

from __future__ import annotations

from ..types import GameState

#: 连续多少帧"空场"才判定进入对话/静默期 (越小越激进)
DIALOGUE_THRESHOLD_FRAMES = 12
#: 空场时允许存在的敌人上限 (TH11 对话会留 1~2 个不攻击的演员)
DIALOGUE_MAX_ENEMIES = 2


class DialogueAssist:
    """检测对话/静默窗口, 供控制器连续快进。"""

    def __init__(self, threshold: int = DIALOGUE_THRESHOLD_FRAMES,
                 max_enemies: int = DIALOGUE_MAX_ENEMIES) -> None:
        self.threshold = max(1, int(threshold))
        self.max_enemies = max(0, int(max_enemies))
        self.empty_frames = 0
        self.active = False

    # ------------------------------------------------------------------
    def is_empty(self, state: GameState | None) -> bool:
        """局内 + 场上无弹幕 + 没有会攻击的敌人 -> 视为对话/静默窗口."""
        entities = (state.raw.get("entities", {}) if state else {}) or {}

        return bool(
            state is not None
            and state.in_game
            and (state.lives or 0) >= 0
            and all(entities.get(k, {}).get("status") == "ok" for k in ("bullets", "enemies"))
            and state.bullet_count == 0
            and (state.enemies_alive or 0) <= self.max_enemies
        )

    # ------------------------------------------------------------------
    def update(self, state: GameState | None, frames: int | None) -> bool:
        empty = self.is_empty(state)
        self.empty_frames = self.empty_frames + int(frames or 0) if empty else 0
        self.active = self.empty_frames >= self.threshold
        return self.active
