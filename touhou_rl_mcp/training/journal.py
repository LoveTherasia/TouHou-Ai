"""训练日志 (JSONL) —— 用来事后检查智能体的不足, 据此调参。

产出 (在训练 out_dir 下):
  train.jsonl   每局一行: 局号/机体/步数/回报/面数/掉残/奖励构成/更新统计 + 动作直方图
  steps.jsonl   每步一行 (需 log_steps=True): 动作/奖励/奖励构成/面数/残机/灵力/擦弹/
                弹幕数/敌机数/自机坐标/结束原因 —— 可以看清它在哪死、爱选什么动作、
                有没有放弹、奖励被哪一项主导。

线程安全: 训练线程写, 可并发读。
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

__all__ = ["Journal"]


class Journal:
    def __init__(self, out_dir: str | Path, *, log_steps: bool = False) -> None:
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.episodes_path = self.dir / "train.jsonl"
        self.steps_path = (self.dir / "steps.jsonl") if log_steps else None
        self._lock = threading.Lock()

    def _append(self, path: Path, payload: dict[str, Any]) -> None:
        line = json.dumps(payload, ensure_ascii=False, default=str)
        with self._lock:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    def episode(self, payload: dict[str, Any]) -> None:
        self._append(self.episodes_path, payload)

    def step(self, payload: dict[str, Any]) -> None:
        if self.steps_path is not None:
            self._append(self.steps_path, payload)

    @property
    def steps_enabled(self) -> bool:
        return self.steps_path is not None
