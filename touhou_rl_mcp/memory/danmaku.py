"""真机弹幕录制 —— 把地灵殿的真实弹幕变成"生成脚本", 供 mock 复刻。

为什么必须这么做: 东方 ``.rpy`` **只存随机种子 + 每帧输入, 不存弹幕**。
要"复刻地灵殿弹幕", 唯一办法是让游戏跑起来, 用 :mod:`memory.entities` 逐帧读弹幕,
记录每颗弹的**生成事件**(帧号 / 位置 / 初速), 再让 mock 按脚本生成 ——
这样 mock 训练用的就是**真实的东方弹幕分布**, 而不是简化模拟。

脚本格式 (JSON)::

    {"source": "th11", "fps": 60, "frames": 5400,
     "spawns": [[frame, x, y, vx, vy], ...]}
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..types import GameState


@dataclass
class SpawnEvent:
    frame: int
    x: float
    y: float
    vx: float
    vy: float


class DanmakuRecorder:
    """按 ``active_slots`` 跨帧差分检测"新弹生成", 并估计初速。

    注意: 必须以 **每帧一次** 的粒度喂状态 (``env.step(..., frames=1)``),
    否则一帧里生成又消失的弹会被漏掉。
    """

    def __init__(self, *, max_speed: float = 30.0, reset_gap: int = 30) -> None:
        self.max_speed = float(max_speed)
        #: 保留字段 (旧版用游戏帧号; 现改用观测计数 tick, 更鲁棒)
        self.reset_gap = int(reset_gap)
        self.prev: dict[int, tuple[float, float]] = {}
        self.pending: dict[int, tuple[int, float, float]] = {}
        self.spawns: list[SpawnEvent] = []
        self.frame = 0
        self.dropped = 0
        self.tick = 0
        self._offset = 0
        self._last_raw: int | None = None

    # ------------------------------------------------------------------
    def _monotonic_frame(self, raw: int) -> int:
        """[已弃用] 用游戏帧号续单调帧号。

        实测: 一次帧读数失败(返回 0)或轻微回退, 会被误判成"换面重置"而加偏移,
        导致后续所有帧错位。现在改用 :attr:`tick` (每帧观测一次, 天然单调)。
        """
        if raw <= 0 and self._last_raw is not None and self._last_raw > 100:
            return self.frame
        if self._last_raw is not None and raw < self._last_raw - self.reset_gap:
            self._offset += self._last_raw + 1
        self._last_raw = raw
        return raw + self._offset

    # ------------------------------------------------------------------
    def update(self, state: GameState) -> int:
        """喂一帧状态; 返回本次新确认的生成数。"""
        diag = (state.raw or {}).get("entities", {}).get("bullets", {})
        if diag.get("status") != "ok":
            return 0
        slots = diag.get("active_slots", []) or []
        points = state.bullets or []
        cur = {int(s): (float(p[0]), float(p[1])) for s, p in zip(slots, points)}
        # 脚本帧号用"观测计数": 每帧调用一次 update -> 天然单调, 不受游戏帧号重置影响
        self.tick += 1
        frame = self.tick
        new = 0

        # 1) 结算上一帧的 pending: 用这一帧的位移估计初速
        for slot, (f0, x0, y0) in list(self.pending.items()):
            if slot in cur:
                x1, y1 = cur[slot]
                dt = max(1, frame - f0)
                vx, vy = (x1 - x0) / dt, (y1 - y0) / dt
                if abs(vx) <= self.max_speed and abs(vy) <= self.max_speed:
                    self.spawns.append(SpawnEvent(f0, round(x0, 2), round(y0, 2),
                                                  round(vx, 3), round(vy, 3)))
                    new += 1
                else:
                    self.dropped += 1
            else:
                self.dropped += 1  # 生成后一帧就消失, 速度未知, 丢弃
            del self.pending[slot]

        # 2) 新出现的 slot -> pending
        for slot, (x, y) in cur.items():
            if slot not in self.prev:
                self.pending[slot] = (frame, x, y)

        self.prev = cur
        self.frame = frame
        return new

    # ------------------------------------------------------------------
    def to_script(self, *, game: str = "th11", fps: int = 60,
                  difficulty: int | None = None, character: int | None = None,
                  shot: int | None = None) -> dict[str, Any]:
        self.spawns.sort(key=lambda e: e.frame)
        return {
            "source": game,
            "fps": fps,
            # 弹幕与"难度 + 自机(角色/机体)"强相关, 必须区分
            "difficulty": difficulty,
            "character": character,
            "shot": shot,
            "frames": max(1, self.frame),
            "count": len(self.spawns),
            "dropped": self.dropped,
            "spawns": [[e.frame, e.x, e.y, e.vx, e.vy] for e in self.spawns],
        }

    def save(self, path: str | Path, *, game: str = "th11", fps: int = 60,
             difficulty: int | None = None, character: int | None = None,
             shot: int | None = None) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(
            self.to_script(game=game, fps=fps, difficulty=difficulty,
                           character=character, shot=shot), ensure_ascii=False),
            encoding="utf-8")
        return target


def load_script(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    spawns = data.get("spawns", [])
    base = min((int(r[0]) for r in spawns), default=0)
    by_frame: dict[int, list[tuple[float, float, float, float]]] = {}
    for row in spawns:
        f = int(row[0]) - base
        by_frame.setdefault(f, []).append((float(row[1]), float(row[2]), float(row[3]), float(row[4])))
    data["by_frame"] = by_frame
    data["frame_base"] = base
    #: 脚本可循环的帧长度
    data["length"] = max(1, max(by_frame) + 1 if by_frame else 1)
    return data
