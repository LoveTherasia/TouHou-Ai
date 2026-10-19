"""单局 / 跨局统计 —— 给训练循环和 Agent 观测用."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

from ..types import GameState, StepResult


@dataclass
class EpisodeStats:
    """一局之内的统计."""

    index: int = 0
    steps: int = 0
    frames: int = 0
    return_: float = 0.0
    reward_breakdown: dict[str, float] = field(default_factory=dict)
    max_stage: int | None = None
    max_score: int | None = None
    max_power: float | None = None
    total_graze: int | None = None
    misses: int = 0
    bombs_used: int = 0
    cleared: bool = False
    game_over: bool = False
    truncated: bool = False
    actions: dict[str, int] = field(default_factory=dict)

    def update(self, result: StepResult, prev: GameState | None, action: str) -> None:
        self.steps += 1
        self.frames += int((result.info or {}).get("frames") or 0)
        self.return_ += result.reward
        for key, value in (result.info.get("reward_parts") or {}).items():
            self.reward_breakdown[key] = round(self.reward_breakdown.get(key, 0.0) + value, 6)
        self.actions[action] = self.actions.get(action, 0) + 1

        state = result.state
        if state.stage is not None:
            self.max_stage = max(self.max_stage or 0, state.stage)
        if state.score is not None:
            self.max_score = max(self.max_score or 0, state.score)
        if state.power is not None:
            self.max_power = max(self.max_power or 0.0, state.power)
        if state.graze is not None:
            self.total_graze = max(self.total_graze or 0, state.graze)
        if prev is not None and prev.lives is not None and state.lives is not None:
            if state.lives < prev.lives:
                self.misses += prev.lives - state.lives

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode": self.index,
            "steps": self.steps,
            "frames": self.frames,
            "return": round(self.return_, 3),
            "reward_breakdown": self.reward_breakdown,
            "max_stage": self.max_stage,
            "max_score": self.max_score,
            "max_power": self.max_power,
            "total_graze": self.total_graze,
            "misses": self.misses,
            "cleared": self.cleared,
            "game_over": self.game_over,
            "truncated": self.truncated,
            "actions": self.actions,
        }


class StatsTracker:
    """跨局滚动统计."""

    def __init__(self, window: int = 50) -> None:
        self.window = window
        self.episodes: list[dict[str, Any]] = []
        self.recent: deque[dict[str, Any]] = deque(maxlen=window)

    def add(self, stats: EpisodeStats) -> None:
        record = stats.to_dict()
        self.episodes.append(record)
        self.recent.append(record)

    def summary(self) -> dict[str, Any]:
        if not self.recent:
            return {"episodes": 0, "window": self.window}
        recent = list(self.recent)
        n = len(recent)
        returns = [r["return"] for r in recent]
        stages = [r["max_stage"] or 0 for r in recent]
        clears = sum(1 for r in recent if r["cleared"])
        return {
            "episodes": len(self.episodes),
            "window": self.window,
            "window_size": n,
            "mean_return": round(sum(returns) / n, 3),
            "max_return": round(max(returns), 3),
            "min_return": round(min(returns), 3),
            "mean_stage": round(sum(stages) / n, 3),
            "best_stage": max(stages),
            "clear_rate": round(clears / n, 4),
            "mean_steps": round(sum(r["steps"] for r in recent) / n, 1),
            "mean_misses": round(sum(r["misses"] for r in recent) / n, 2),
            "best_score": max((r["max_score"] or 0) for r in recent),
        }

    def reset(self) -> None:
        self.episodes.clear()
        self.recent.clear()
