"""东方常见弹幕类型库 (mock 用)。

每种弹型 = 一个类, 通过 ``config.danmaku_pattern`` 选择。用来**按弹型分别训练专属老师/策略**。
每种弹型的"发射位置 / 形状 / 速度 / 持续时间"都独立, 便于区分与对比。

参考正作:
  * dense_ring            密集环 (同一点放一圈)
  * expanding_homing_ring 扩张环 + 整体朝自机移动
  * aimed_trail           自机狙, 沿途留下静止弹 (一段时间后消失)
  * static_swarm          不断朝自机逼近的静止弹群 (收网)
  * spiral / fan / wall_gap / random_spray / flower / aimed_spread  其它常见弹型
"""

from __future__ import annotations

import math
from typing import Any

from ..types import PLAYFIELD_HEIGHT, PLAYFIELD_WIDTH

CX = PLAYFIELD_WIDTH / 2.0


def ring(game: Any, x: float, y: float, count: int, speed: float, *,
         phase: float = 0.0, life: int = 0, homing: float = 0.0,
         homing_frames: int = 0) -> None:
    for i in range(count):
        a = phase + 2.0 * math.pi * i / count
        game.spawn(x, y, math.cos(a) * speed, math.sin(a) * speed,
                   life=life, homing=homing, homing_frames=homing_frames)


def aimed(game: Any, x: float, y: float, count: int, spread: float, speed: float, *,
          life: int = 0, homing: float = 0.0) -> None:
    base = math.atan2(game.py - y, game.px - x)
    for i in range(-count, count + 1):
        a = base + i * spread
        game.spawn(x, y, math.cos(a) * speed, math.sin(a) * speed, life=life, homing=homing)


class Pattern:
    name = "base"

    def __init__(self, **kw: Any) -> None:
        for k, v in kw.items():
            setattr(self, k, v)

    def reset(self, game: Any) -> None:
        pass

    def update(self, game: Any) -> None:
        pass


# ----------------------------------------------------------------------
class DenseRing(Pattern):
    """密集环: 每 interval 帧从同一点放一圈 count 颗弹。"""

    name = "dense_ring"
    count = 64
    speed = 1.6
    interval = 30
    x = CX
    y = 96.0

    def update(self, game: Any) -> None:
        if game.frame % self.interval == 0:
            ring(game, self.x, self.y, self.count, self.speed, phase=game.frame * 0.07)


class ExpandingHomingRing(Pattern):
    """扩张环 + 朝自机移动: 环向外扩张的同时, 每颗弹持续朝自机加速。"""

    name = "expanding_homing_ring"
    count = 20
    speed = 1.0
    interval = 110
    homing = 0.010
    homing_frames = 30      # 只追踪一小段 (朝自机移动一段距离)
    x = CX
    y = 96.0

    def update(self, game: Any) -> None:
        if game.frame % self.interval == 0:
            ring(game, self.x, self.y, self.count, self.speed,
                 phase=game.frame * 0.05, homing=self.homing,
                 homing_frames=self.homing_frames)


class AimedTrail(Pattern):
    """自机狙 + 沿途留下静止弹幕 (静止弹存活 drop_life 帧后消失)。"""

    name = "aimed_trail"
    interval = 45
    speed = 2.2
    drop_every = 7          # 每多少帧落一颗静止弹
    drop_life = 150         # 静止弹存活帧数
    x = CX
    y = 88.0

    def __init__(self, **kw: Any) -> None:
        super().__init__(**kw)
        self.emitters: list[list[float]] = []

    def reset(self, game: Any) -> None:
        self.emitters = []

    def update(self, game: Any) -> None:
        alive = []
        for e in self.emitters:
            e[0] += e[2]
            e[1] += e[3]
            e[4] += 1
            if e[4] >= self.drop_every:
                game.spawn(e[0], e[1], 0.0, 0.0, life=self.drop_life)  # 静止弹
                e[4] = 0
            if -40 <= e[0] <= PLAYFIELD_WIDTH + 40 and -40 <= e[1] <= PLAYFIELD_HEIGHT + 40:
                alive.append(e)
        self.emitters = alive
        if game.frame % self.interval == 0:
            base = math.atan2(game.py - self.y, game.px - self.x)
            vx, vy = math.cos(base) * self.speed, math.sin(base) * self.speed
            game.spawn(self.x, self.y, vx, vy)
            self.emitters.append([self.x, self.y, vx, vy, 0.0])


class StaticSwarm(Pattern):
    """不断朝自机逼近的静止弹群: 在自机周围以逐渐缩小的半径生成静止弹 (收网)。"""

    name = "static_swarm"
    count = 16
    start_r = 250.0
    shrink_per_sec = 26.0
    interval = 20
    life = 260
    gap_keep = 2            # 每次留几个缺口 (给自机逃逸)

    def __init__(self, **kw: Any) -> None:
        super().__init__(**kw)
        self.r = self.start_r

    def reset(self, game: Any) -> None:
        self.r = self.start_r

    def update(self, game: Any) -> None:
        if game.frame % self.interval == 0:
            self.r = max(34.0, self.r - self.shrink_per_sec * self.interval / 60.0)
            cx, cy = game.px, game.py
            gap = game.frame // self.interval % self.count
            for i in range(self.count):
                if (i - gap) % self.count < self.gap_keep:
                    continue
                a = 2.0 * math.pi * i / self.count + game.frame * 0.03
                x = min(PLAYFIELD_WIDTH - 6, max(6, cx + math.cos(a) * self.r))
                y = min(PLAYFIELD_HEIGHT - 6, max(6, cy + math.sin(a) * self.r))
                game.spawn(x, y, 0.0, 0.0, life=self.life)


# ---------------- 其它常见弹型 ----------------
class Spiral(Pattern):
    """多臂螺旋 (持续旋转)。"""

    name = "spiral"
    arms = 3
    speed = 1.9
    interval = 6
    x = CX
    y = 96.0

    def update(self, game: Any) -> None:
        if game.frame % self.interval == 0:
            a0 = game.frame * 0.11
            for i in range(self.arms):
                a = a0 + 2.0 * math.pi * i / self.arms
                game.spawn(self.x, self.y, math.cos(a) * self.speed, math.sin(a) * self.speed)


class Fan(Pattern):
    """向下扇形弹幕。"""

    name = "fan"
    count = 13
    spread = 1.1
    speed = 2.0
    interval = 26
    x = CX
    y = 96.0

    def update(self, game: Any) -> None:
        if game.frame % self.interval == 0:
            base = math.pi / 2
            for i in range(self.count):
                frac = (i / (self.count - 1)) - 0.5 if self.count > 1 else 0.0
                a = base + frac * self.spread
                game.spawn(self.x, self.y, math.cos(a) * self.speed, math.sin(a) * self.speed)


class WallGap(Pattern):
    """带缺口的横向弹幕墙。"""

    name = "wall_gap"
    slots = 11
    speed = 1.6
    interval = 34
    y = 110.0

    def update(self, game: Any) -> None:
        if game.frame % self.interval == 0:
            gap = (game.frame // self.interval) % self.slots
            step = (PLAYFIELD_WIDTH - 48.0) / (self.slots - 1)
            for i in range(self.slots):
                if i == gap:
                    continue
                game.spawn(24.0 + i * step, self.y, 0.0, self.speed)


class RandomSpray(Pattern):
    """随机方向爆散。"""

    name = "random_spray"
    count = 12
    speed = 2.1
    interval = 18
    x = CX
    y = 96.0

    def update(self, game: Any) -> None:
        if game.frame % self.interval == 0:
            for _ in range(self.count):
                a = game.rng.random() * 2.0 * math.pi
                s = self.speed * (0.6 + 0.8 * game.rng.random())
                game.spawn(self.x, self.y, math.cos(a) * s, math.sin(a) * s)


class Flower(Pattern):
    """n 重对称"花弹": 多方向同时放扇形。"""

    name = "flower"
    petals = 6
    per_petal = 5
    speed = 1.9
    interval = 30
    x = CX
    y = 96.0

    def update(self, game: Any) -> None:
        if game.frame % self.interval == 0:
            for p in range(self.petals):
                base = 2.0 * math.pi * p / self.petals + game.frame * 0.02
                for j in range(-self.per_petal // 2, self.per_petal // 2 + 1):
                    a = base + j * 0.10
                    game.spawn(self.x, self.y, math.cos(a) * self.speed, math.sin(a) * self.speed)


class AimedSpread(Pattern):
    """自机狙 n 连 (经典追尾)。"""

    name = "aimed_spread"
    count = 4
    spread = 0.09
    speed = 2.4
    interval = 22
    x = CX
    y = 96.0

    def update(self, game: Any) -> None:
        if game.frame % self.interval == 0:
            aimed(game, self.x, self.y, self.count, self.spread, self.speed)


# ----------------------------------------------------------------------
_REGISTRY: dict[str, type[Pattern]] = {
    c.name: c for c in (
        DenseRing, ExpandingHomingRing, AimedTrail, StaticSwarm,
        Spiral, Fan, WallGap, RandomSpray, Flower, AimedSpread,
    )
}

#: 每个弹型的中文说明 (给工具/文档用)
PATTERN_DOCS: dict[str, str] = {
    "dense_ring": "密集环: 同一点每 30 帧放一圈 64 颗",
    "expanding_homing_ring": "扩张环 + 朝自机移动: 环扩张同时每颗弹持续朝自机加速",
    "aimed_trail": "自机狙 + 沿途留下静止弹 (存活 150 帧后消失)",
    "static_swarm": "静止弹收网: 自机周围半径逐渐缩小的静止弹群",
    "spiral": "多臂螺旋",
    "fan": "向下扇形",
    "wall_gap": "带缺口的横向弹幕墙",
    "random_spray": "随机爆散",
    "flower": "n 重对称花弹",
    "aimed_spread": "自机狙 n 连",
}


def pattern_names() -> list[str]:
    return sorted(_REGISTRY)


def make_pattern(name: str | None, **kw: Any) -> Pattern | None:
    if not name:
        return None
    cls = _REGISTRY.get(name)
    return cls(**kw) if cls else None
