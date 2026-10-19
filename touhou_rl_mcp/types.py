"""环境观测 / 动作 / 结果的数据结构."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# 游戏场地坐标 (东方正作通用): 384 x 448, 位于 640x480 窗口内的 (32, 16)
PLAYFIELD_WIDTH = 384.0
PLAYFIELD_HEIGHT = 448.0


@dataclass
class GameState:
    """一帧的游戏状态.

    所有字段都可能为 ``None`` —— 取决于该游戏的偏移量标定到了什么程度。
    ``raw`` 里保留所有原始读数, 方便标定偏移量时对照。
    """

    frame: int = 0
    in_game: bool = False

    stage: int | None = None
    lives: int | None = None
    power: float | None = None
    bombs: int | None = None
    score: int | None = None
    graze: int | None = None

    spell_id: int | None = None
    spell_name: str | None = None

    char_main: int | None = None
    char_sub: int | None = None
    char_name: str | None = None

    player_x: float | None = None
    player_y: float | None = None
    focused: bool | None = None
    invincible: bool | None = None

    boss_hp: int | None = None
    boss_max_hp: int | None = None
    boss_name: str | None = None
    enemies_alive: int | None = None

    bullets: list[tuple[float, float]] = field(default_factory=list)
    enemy_positions: list[tuple[float, float]] = field(default_factory=list)
    bullet_count: int | None = None

    raw: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    def to_dict(self, *, include_bullets: bool = True, max_bullets: int = 256) -> dict[str, Any]:
        out: dict[str, Any] = {
            "frame": self.frame,
            "in_game": self.in_game,
            "stage": self.stage,
            "lives": self.lives,
            "power": self.power,
            "bombs": self.bombs,
            "score": self.score,
            "graze": self.graze,
            "spell_id": self.spell_id,
            "spell_name": self.spell_name,
            "char_main": self.char_main,
            "char_sub": self.char_sub,
            "char_name": self.char_name,
            "player": (
                None
                if self.player_x is None or self.player_y is None
                else {"x": round(self.player_x, 2), "y": round(self.player_y, 2)}
            ),
            "focused": self.focused,
            "invincible": self.invincible,
            "boss": {
                "hp": self.boss_hp,
                "max_hp": self.boss_max_hp,
                "name": self.boss_name,
            },
            "enemies_alive": self.enemies_alive,
            "bullet_count": (self.bullet_count if "entities" in self.raw else
                             self.bullet_count if self.bullet_count is not None else len(self.bullets)),
        }
        if include_bullets:
            out["bullets"] = [
                [round(x, 1), round(y, 1)] for x, y in self.bullets[:max_bullets]
            ]
            out["bullets_truncated"] = len(self.bullets) > max_bullets
            out["enemy_positions"] = [
                [round(x, 1), round(y, 1)] for x, y in self.enemy_positions[:max_bullets]
            ]
        out["raw"] = self.raw
        return out


@dataclass
class StepResult:
    """``step`` / ``reset`` 的返回结果 —— RL 里的 ``(obs, reward, done, info)``."""

    state: GameState
    reward: float
    done: bool
    truncated: bool = False
    info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "state": self.state.to_dict(**kwargs),
            "reward": round(self.reward, 6),
            "done": self.done,
            "truncated": self.truncated,
            "info": self.info,
        }
