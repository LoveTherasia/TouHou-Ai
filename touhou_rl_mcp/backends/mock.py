"""内置模拟器后端 (mock).

不依赖游戏本体, 用于:
  * 在没有游戏 / 非 Windows 环境时开发与自测 MCP
  * 给 agentic RL 提供一个**快速、可复现**的沙盒 (reset(seed=...) 完全确定)
  * 验证奖励函数、宏动作 (move_to / dodge)、可视化管线

它是一个简化但"有内味"的弹幕模拟: 波次敌人 + Boss + 弹幕 + 擦弹 + 被弹掉残,
难度随关卡上升, 打掉每关 Boss 就进下一面。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

from PIL import Image

from ..inputs.actions import Action
from ..inputs.controller import Controller, NullSink
from ..types import GameState, PLAYFIELD_HEIGHT, PLAYFIELD_WIDTH
from .base import Backend

PLAYER_SPEED = 4.4
PLAYER_SPEED_FOCUS = 2.0
HITBOX_RADIUS = 2.0
GRAZE_RADIUS = 14.0
MAX_BULLETS = 900
MAX_LIVES = 2
START_POWER = 1.0
MAX_POWER = 4.0
INVINCIBLE_FRAMES = 150

#: 每面 Boss 的基准血量 (随面数上升); 单张符卡血量 = 基准 * 该符卡倍率
BOSS_BASE_HP = 700
BOSS_HP_PER_STAGE = 300
BOSS_ENTER_SPEED = 1.2
#: Boss 可用的弹型 / 移动方式 (每张符卡各选一种, 参照正作多符卡)
SPELL_PATTERNS = ("ring", "aimed", "spiral", "fan", "wall", "burst")
BOSS_MOVES = ("static", "sine_x", "drift", "figure8")


@dataclass
class Bullet:
    x: float
    y: float
    vx: float
    vy: float
    r: float = 3.0
    grazed: bool = False
    #: 存活帧数 (0 = 无限); 到 0 就消失 (用于"留下静止弹幕, 一段时间后消失")
    life: int = 0
    #: 追踪加速度 (px/帧^2), 每帧朝自机方向加速 (用于"扩张同时朝自机移动")
    homing: float = 0.0
    #: 追踪剩余帧数 (只在开局一小段内追踪 = "朝自机移动一段距离")
    homing_left: int = 0


@dataclass
class Enemy:
    x: float
    y: float
    hp: int
    vx: float
    vy: float
    fire_timer: int


@dataclass
class SpellCard:
    """一张符卡: 独立血量 / 弹型 / 移动方式.

    参照正作: 同一个 Boss 有多张符卡, 每张符卡血量与弹幕都不同, 打空一张
    进入下一张, 全部打空才算击破该面的 Boss。
    """

    name: str
    hp: int
    pattern: str            # ring / aimed / spiral / fan / wall / burst
    move: str = "static"    # static / sine_x / drift / figure8
    fire_interval: int = 20
    timeout: int = 0


@dataclass
class Boss:
    spells: list[SpellCard]
    index: int = 0
    hp: int = 0
    max_hp: int = 0
    x: float = PLAYFIELD_WIDTH / 2
    y: float = -20.0
    home_x: float = PLAYFIELD_WIDTH / 2
    home_y: float = 96.0
    timer: int = 0
    fire_timer: int = 0
    entering: bool = True

    @property
    def spell(self) -> SpellCard:
        return self.spells[min(self.index, len(self.spells) - 1)]


class MockGame:
    """模拟器的状态机本体."""

    def __init__(
        self,
        *,
        seed: int = 0,
        max_stage: int = 6,
        difficulty: int = 1,
        start_stage: int = 1,
        danmaku_script: dict[str, Any] | None = None,
        danmaku_offset: int = 0,
        danmaku_pattern: str = "",
    ) -> None:
        self.seed = seed
        self.max_stage = max_stage
        self.difficulty = max(0, min(3, difficulty))
        self.start_stage = max(1, min(int(start_stage), int(max_stage)))
        #: 真机弹幕脚本 (由 memory.danmaku.load_script 载入); 有它就用真实弹幕代替内置弹型
        self.danmaku_script = danmaku_script
        self.scripted = bool(danmaku_script)
        #: 脚本起始偏移 (从回放的第几帧开始, 用来直接练某一面)
        self.script_offset = max(0, int(danmaku_offset or 0))
        #: 弹幕类型库 (backends/patterns.py): 按弹型分别训练
        from .patterns import make_pattern

        self.pattern = make_pattern(danmaku_pattern or None)
        self.patterned = bool(self.pattern)
        #: 脚本/弹型模式 = "弹幕生存": 不判面数、不通关, 打死 Boss 只刷新一个新 Boss
        self.endless = bool(danmaku_script) or self.patterned
        self.script_by_frame = (danmaku_script or {}).get("by_frame", {}) or {}
        self.script_len = int((danmaku_script or {}).get("length", 0) or 0)
        self.rng = random.Random(seed)
        self.reset(seed=seed)

    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None) -> None:
        if seed is not None:
            self.seed = seed
        self.rng = random.Random(self.seed)
        self.frame = 0
        self.px = PLAYFIELD_WIDTH / 2
        self.py = PLAYFIELD_HEIGHT - 32
        self.lives = MAX_LIVES
        self.power = START_POWER
        self.score = 0
        self.graze = 0
        self.stage = self.start_stage
        self.spell_id = 0
        self.bullets: list[Bullet] = []
        self.enemies: list[Enemy] = []
        self.boss: Boss | None = None
        self.invincible = INVINCIBLE_FRAMES
        self.stage_frame = 0
        self.game_over = False
        self.cleared = False
        self.deaths = 0
        self.wave_timer = 60
        self.wave_index = 0
        self.last_hit_frame = -999
        #: 本帧/本次推进内发生的事件 (state() 读取后清空) —— 供奖励使用
        self.events: set[str] = set()
        self.spells_cleared = 0
        if self.pattern is not None:
            self.pattern.reset(self)

    # ------------------------------------------------------------------
    def tick(self, buttons: set[str]) -> None:
        if self.game_over or self.cleared:
            self.frame += 1
            return

        self.frame += 1
        self.stage_frame += 1
        self._move_player(buttons)
        self._shoot(buttons)
        self._spawn()
        self._emit_scripted()
        if self.pattern is not None:
            self.pattern.update(self)
        self._update_enemies()
        self._update_boss()
        self._update_bullets()
        self._collisions()
        if self.invincible > 0:
            self.invincible -= 1

    # ------------------------------------------------------------------
    def _move_player(self, buttons: set[str]) -> None:
        dx = (1 if "right" in buttons else 0) - (1 if "left" in buttons else 0)
        dy = (1 if "down" in buttons else 0) - (1 if "up" in buttons else 0)
        if dx == 0 and dy == 0:
            return
        speed = PLAYER_SPEED_FOCUS if "focus" in buttons else PLAYER_SPEED
        if dx and dy:
            speed *= math.sqrt(0.5)
        margin = 8.0
        self.px = min(PLAYFIELD_WIDTH - margin, max(margin, self.px + dx * speed))
        self.py = min(PLAYFIELD_HEIGHT - margin, max(margin, self.py + dy * speed))

    def _shoot(self, buttons: set[str]) -> None:
        if "shoot" not in buttons:
            return
        # 自机射击: 对上方同列敌人 / Boss 造成伤害
        dps = 3.0 if self.power < 3.0 else 4.0
        for enemy in self.enemies:
            if abs(enemy.x - self.px) < 14 and enemy.y < self.py:
                enemy.hp -= dps
        if self.boss is not None and not self.boss.entering:
            if abs(self.boss.x - self.px) < 22:
                self.boss.hp -= dps
                self.score += int(dps)

    def _spawn(self) -> None:
        if self.boss is not None:
            return
        self.wave_timer -= 1
        if self.wave_timer > 0:
            return
        self.wave_timer = max(70, 130 - self.stage * 8 - self.difficulty * 6)
        self.wave_index += 1
        count = 2 + min(4, self.stage // 2) + self.difficulty
        for i in range(count):
            x = 48 + (PLAYFIELD_WIDTH - 96) * (i + 0.5) / count
            self.enemies.append(
                Enemy(x=x, y=-20 - i * 14, hp=6 + self.stage * 2, vx=0.0, vy=0.9 + 0.06 * self.stage, fire_timer=60 + i * 20)
            )
        if self.stage_frame > 260 and self.wave_index >= 3:
            self.boss = self._make_boss(self.stage)
            self.spell_id = 43

    def _update_enemies(self) -> None:
        alive: list[Enemy] = []
        for enemy in self.enemies:
            enemy.y += enemy.vy
            enemy.x += enemy.vx
            enemy.fire_timer -= 1
            if enemy.fire_timer <= 0 and enemy.y > 0:
                enemy.fire_timer = max(40, 90 - self.stage * 5)
                self._aimed_shot(enemy.x, enemy.y, spread=3, speed=1.7 + 0.12 * self.stage)
            if enemy.hp <= 0:
                self.score += 120
                self.power = min(MAX_POWER, self.power + 0.04)
                continue
            if enemy.y > PLAYFIELD_HEIGHT + 30:
                continue
            alive.append(enemy)
        self.enemies = alive

    def _make_boss(self, stage: int) -> Boss:
        """按面数生成一个多符卡 Boss; 位置/弹型/移动都不同, 不一定在中央."""
        plans = self._boss_plans(stage)
        base = BOSS_BASE_HP + stage * BOSS_HP_PER_STAGE
        spells = [
            SpellCard(
                name=f"{stage}-{i + 1} 符卡「{pattern}」",
                hp=max(1, int(base * mult)),
                pattern=pattern,
                move=move,
                fire_interval=interval,
            )
            for i, (pattern, move, mult, interval) in enumerate(plans)
        ]
        home_x = self.rng.choice([168.0, 192.0, 216.0])
        home_y = self.rng.choice([72.0, 96.0, 120.0])
        return Boss(
            spells=spells,
            index=0,
            hp=spells[0].hp,
            max_hp=spells[0].hp,
            x=home_x,
            y=-20.0,
            home_x=home_x,
            home_y=home_y,
            entering=True,
        )

    @staticmethod
    def _boss_plans(stage: int) -> list[tuple[str, str, float, int]]:
        """每面的符卡方案: ``(弹型, 移动, 血量倍率, 射击间隔)``.

        前两面 2 张符卡, 之后 3 张; 弹型/移动按面数轮换, 所以每一面都不同。
        """
        count = 2 if stage <= 2 else 3
        plans: list[tuple[str, str, float, int]] = []
        for i in range(count):
            pattern = SPELL_PATTERNS[(stage + i) % len(SPELL_PATTERNS)]
            move = BOSS_MOVES[(stage + 2 * i) % len(BOSS_MOVES)]
            mult = 0.85 + 0.20 * i
            interval = max(10, 22 - stage - i)
            plans.append((pattern, move, mult, interval))
        return plans

    def _boss_move(self, boss: Boss) -> None:
        """让 Boss 动起来 (不同符卡不同移动方式) —— 不再是站桩靶子."""
        move = boss.spell.move
        t = boss.timer
        if move == "sine_x":
            boss.x = boss.home_x + 60.0 * math.sin(t * 0.020)
        elif move == "drift":
            boss.x = boss.home_x + 70.0 * math.sin(t * 0.008)
        elif move == "figure8":
            boss.x = boss.home_x + 60.0 * math.sin(t * 0.030)
            boss.y = boss.home_y + 18.0 * math.sin(t * 0.060)
        else:
            boss.x = boss.home_x
        boss.x = min(PLAYFIELD_WIDTH - 40.0, max(40.0, boss.x))
        if move != "figure8":
            boss.y = boss.home_y

    def _update_boss(self) -> None:
        boss = self.boss
        if boss is None:
            return
        if boss.entering:
            boss.y += BOSS_ENTER_SPEED
            if boss.y >= boss.home_y:
                boss.y = boss.home_y
                boss.entering = False
            return
        boss.timer += 1
        boss.fire_timer -= 1
        self._boss_move(boss)

        if boss.hp <= 0:
            # 一张符卡被打空 -> 记事件并进入下一张; 全部打空才过面
            self.score += 2000 + self.stage * 500
            self.events.add("spell_cleared")
            self.spells_cleared += 1
            boss.index += 1
            if boss.index >= len(boss.spells):
                if self.endless:
                    # 脚本模式: 不判面数/不通关, 直接刷新一个新 Boss 继续生存
                    self.boss = self._make_boss(self.stage)
                    self.spell_id = 43
                    self.bullets.clear()
                    return
                self.stage += 1
                self.events.add("stage_cleared")
                if self.stage > self.max_stage:
                    self.cleared = True
                    self.boss = None
                    return
                self.boss = None
                self.spell_id = 0
                self.stage_frame = 0
                self.wave_index = 0
                self.wave_timer = 90
                self.bullets.clear()
                return
            nxt = boss.spells[boss.index]
            boss.hp = nxt.hp
            boss.max_hp = nxt.hp
            boss.timer = 0
            boss.fire_timer = 20
            self.spell_id = 43 + boss.index
            self.bullets.clear()  # 换符清屏 (与正作一致)
            return

        if boss.fire_timer > 0:
            return
        pattern = boss.spell.pattern
        boss.fire_timer = boss.spell.fire_interval
        if pattern == "ring":
            self._ring_shot(boss.x, boss.y, count=12 + self.stage * 2 + self.difficulty * 2,
                            speed=1.9, offset=boss.timer * 0.11)
        elif pattern == "aimed":
            self._aimed_shot(boss.x, boss.y, spread=4, speed=2.3 + 0.08 * self.stage)
        elif pattern == "spiral":
            self._spiral_shot(boss.x, boss.y, arms=3, speed=2.2, angle=boss.timer * 0.09)
        elif pattern == "fan":
            self._fan_shot(boss.x, boss.y, count=9, spread=1.0, speed=2.0)
        elif pattern == "wall":
            self._wall_shot(boss.x, boss.y, speed=1.7)
        else:
            self._burst_shot(boss.x, boss.y, count=10, speed=2.1)

    def _ring_shot(self, x: float, y: float, *, count: int, speed: float, offset: float) -> None:
        for i in range(count):
            angle = offset + 2 * math.pi * i / count
            self._add_bullet(x, y, math.cos(angle) * speed, math.sin(angle) * speed)

    def _aimed_shot(self, x: float, y: float, *, spread: int, speed: float) -> None:
        base = math.atan2(self.py - y, self.px - x)
        for i in range(-spread, spread + 1):
            angle = base + i * 0.10
            self._add_bullet(x, y, math.cos(angle) * speed, math.sin(angle) * speed)

    def _spiral_shot(self, x: float, y: float, *, arms: int, speed: float, angle: float) -> None:
        for i in range(arms):
            a = angle + 2 * math.pi * i / arms
            self._add_bullet(x, y, math.cos(a) * speed, math.sin(a) * speed)

    def _fan_shot(self, x: float, y: float, *, count: int, spread: float, speed: float) -> None:
        """向下扇形弹幕."""
        base = math.pi / 2
        for i in range(count):
            frac = (i / (count - 1)) - 0.5 if count > 1 else 0.0
            angle = base + frac * spread
            self._add_bullet(x, y, math.cos(angle) * speed, math.sin(angle) * speed)

    def _wall_shot(self, x: float, y: float, *, speed: float) -> None:
        """横向弹幕墙, 随机留一个缺口 (可用缺口穿过)."""
        slots = 9
        gap = self.rng.randrange(slots)
        step = (PLAYFIELD_WIDTH - 48.0) / (slots - 1)
        for i in range(slots):
            if i == gap:
                continue
            self._add_bullet(24.0 + i * step, y, 0.0, speed)

    def _burst_shot(self, x: float, y: float, *, count: int, speed: float) -> None:
        """随机方向爆散."""
        for _ in range(count):
            angle = self.rng.random() * 2 * math.pi
            scale = 0.6 + 0.8 * self.rng.random()
            self._add_bullet(x, y, math.cos(angle) * speed * scale, math.sin(angle) * speed * scale)

    def _add_bullet(self, x: float, y: float, vx: float, vy: float) -> None:
        if self.scripted or self.patterned:
            return  # 脚本/弹型模式: 只用真实/指定弹幕, 关掉内置弹型
        if len(self.bullets) >= MAX_BULLETS:
            return
        if self.difficulty == 0 and self.rng.random() < 0.25:
            return
        self.bullets.append(Bullet(x=x, y=y, vx=vx, vy=vy))

    def _emit_scripted(self) -> None:
        """按真机弹幕脚本生成子弹 (循环播放)。"""
        if not self.scripted or not self.script_len:
            return
        frame = (self.frame + self.script_offset) % self.script_len
        for row in self.script_by_frame.get(frame, ()):
            if len(self.bullets) >= MAX_BULLETS:
                break
            x, y, vx, vy = row[:4]
            life = int(row[4]) if len(row) > 4 else 0
            self.bullets.append(Bullet(x=x, y=y, vx=vx, vy=vy, life=life))

    def _update_bullets(self) -> None:
        alive: list[Bullet] = []
        for bullet in self.bullets:
            if bullet.homing and (not bullet.homing_left or bullet.homing_left > 0):
                dx, dy = self.px - bullet.x, self.py - bullet.y
                d = math.hypot(dx, dy) or 1.0
                bullet.vx += dx / d * bullet.homing
                bullet.vy += dy / d * bullet.homing
                if bullet.homing_left:
                    bullet.homing_left -= 1
            bullet.x += bullet.vx
            bullet.y += bullet.vy
            if bullet.life:
                bullet.life -= 1
                if bullet.life <= 0:
                    continue
            if -40 <= bullet.x <= PLAYFIELD_WIDTH + 40 and -40 <= bullet.y <= PLAYFIELD_HEIGHT + 40:
                alive.append(bullet)
        self.bullets = alive

    # ------------------------------------------------------------------
    def spawn(self, x: float, y: float, vx: float, vy: float, *,
              life: int = 0, homing: float = 0.0, homing_frames: int = 0) -> None:
        """直接生成一颗弹 (给"弹幕类型库"用; 不受 difficulty 稀释影响)。"""
        if len(self.bullets) >= MAX_BULLETS:
            return
        self.bullets.append(Bullet(x=x, y=y, vx=vx, vy=vy, life=int(life),
                                   homing=float(homing), homing_left=int(homing_frames)))

    def _collisions(self) -> None:
        if self.invincible > 0:
            return
        for bullet in self.bullets:
            dist = math.hypot(bullet.x - self.px, bullet.y - self.py)
            if dist < GRAZE_RADIUS and not bullet.grazed:
                bullet.grazed = True
                self.graze += 1
                self.score += 30
            if dist < HITBOX_RADIUS + bullet.r:
                self._hit()
                return

    def _hit(self) -> None:
        self.last_hit_frame = self.frame
        self.bullets = [b for b in self.bullets if math.hypot(b.x - self.px, b.y - self.py) > 60]
        self.py = PLAYFIELD_HEIGHT - 32
        self.px = PLAYFIELD_WIDTH / 2
        if self.lives <= 0:
            self.game_over = True
            self.deaths += 1
            return
        self.lives -= 1
        self.power = max(2.25, self.power - 1.0)
        self.invincible = INVINCIBLE_FRAMES

    def use_bomb(self) -> bool:
        if self.power < 1.0 or self.game_over or self.cleared:
            return False
        self.power -= 1.0
        self.bullets = [b for b in self.bullets if math.hypot(b.x - self.px, b.y - self.py) > 220]
        self.invincible = max(self.invincible, 90)
        self.score += 50
        return True

    # ------------------------------------------------------------------
    def state(self, frame: int = 0) -> GameState:
        boss = self.boss
        spell = boss.spell if boss else None
        state = GameState(
            frame=self.frame + frame,
            in_game=not self.game_over and not self.cleared,
            stage=min(self.stage, self.max_stage + 1),
            lives=self.lives,
            power=round(self.power, 2),
            bombs=max(0, int(self.power)),
            score=self.score,
            graze=self.graze,
            spell_id=self.spell_id,
            spell_name=spell.name if spell else None,
            char_main=0,
            char_sub=0,
            char_name="灵梦A",
            player_x=round(self.px, 2),
            player_y=round(self.py, 2),
            focused=None,
            invincible=self.invincible,
            boss_hp=int(boss.hp) if boss else None,
            boss_max_hp=int(boss.max_hp) if boss else None,
            boss_name=f"模拟Boss {self.stage}面" if boss else None,
            enemies_alive=len(self.enemies),
            bullets=[(round(b.x, 2), round(b.y, 2)) for b in self.bullets],
            enemy_positions=[(round(e.x, 2), round(e.y, 2)) for e in self.enemies],
            bullet_count=len(self.bullets),
            raw={
                "source": "mock",
                "difficulty": self.difficulty,
                "seed": self.seed,
                "start_stage": self.start_stage,
                "game_over": self.game_over,
                "cleared": self.cleared,
                "stage_frame": self.stage_frame,
                # 事件 (本次推进内发生): 供奖励函数识别符卡击破等
                "spell_cleared": "spell_cleared" in self.events,
                "stage_cleared": "stage_cleared" in self.events,
                "spells_cleared": self.spells_cleared,
                "boss_spell": boss.index if boss else None,
                "boss_spell_total": len(boss.spells) if boss else None,
                "boss_pattern": spell.pattern if spell else None,
                "boss_move": spell.move if spell else None,
                "boss_position": (round(boss.x, 2), round(boss.y, 2)) if boss else None,
                # 与真机一致: 提供弹幕速度 (mock 的 Bullet 自带 vx/vy)
                "bullet_velocity": [(round(b.vx, 3), round(b.vy, 3)) for b in self.bullets],
            },
        )
        # 事件只在被读取的那一次可见 (避免同一事件被重复计奖)
        self.events.clear()
        return state


class MockBackend(Backend):
    """模拟器后端."""

    name = "mock"

    def __init__(self, spec: Any, config: Any) -> None:
        super().__init__(spec, config)
        self.sink = NullSink()
        self._script = self._load_danmaku_script()
        self.game = MockGame(max_stage=int(config.get("final_stage", 6)),
                             danmaku_script=self._script,
                             danmaku_offset=int(config.get("danmaku_offset", 0) or 0),
                             danmaku_pattern=str(config.get("danmaku_pattern") or ""))
        self.controller = Controller(
            self.sink,
            self.advance,
            hz=int(config.get("control.tick_hz", 60)),
            state_fn=self.read_state,
            game=spec.key,
        )
        self._bomb_was_down = False
        self._started = False

    def _load_danmaku_script(self):
        """读取 config.danmaku_script 指向的真机弹幕脚本 (可选)。"""
        path = self.config.get("danmaku_script") or ""
        return self.set_danmaku_script(path)

    def set_danmaku_script(self, path):
        """运行时替换弹幕脚本 (下次 reset 生效)。"""
        if not path:
            self._script = None
            return None
        try:
            from ..memory.danmaku import load_script

            self._script = load_script(path)
        except Exception:
            self._script = None
        return self._script

    # ------------------------------------------------------------------
    def start(self) -> None:
        self._started = True

    def reset(self, *, options: dict[str, Any] | None = None) -> GameState:
        options = options or {}
        seed = options.get("seed")
        if seed is None:
            seed = random.randrange(1 << 30)
        self.game = MockGame(
            seed=int(seed),
            max_stage=int(options.get("max_stage") or self.config.get("final_stage", 6)),
            difficulty=int(options.get("difficulty") or 1),
            start_stage=int(options.get("start_stage") or self.config.get("start_stage", 1) or 1),
            danmaku_script=self._script,
            danmaku_offset=int(self.config.get("danmaku_offset", 0) or 0),
            danmaku_pattern=str(self.config.get("danmaku_pattern") or ""),
        )
        if self.controller is not None:
            self.controller.total_frames = 0
        self.sink.release_all()
        self._bomb_was_down = False
        return self.read_state()

    def read_state(self) -> GameState:
        state = self.game.state()
        # 让控制器里的"当前按键"体现在状态里
        return state

    def advance(self, frames: int) -> int:
        buttons = self.sink.held_names
        for _ in range(max(0, int(frames))):
            self.game.tick(buttons)
            bomb_down = "bomb" in buttons
            if bomb_down and not self._bomb_was_down:
                self.game.use_bomb()
            self._bomb_was_down = bomb_down
        return int(frames)

    def screenshot(self) -> tuple[Image.Image, str]:
        from ..capture.visualize import render_window

        image = render_window(
            self.read_state(),
            self.spec,
            options=self.config.section("visualize"),
            title=f"{self.spec.name_zh} [mock]",
        )
        return image, "mock-render"

    def status(self) -> dict[str, Any]:
        info = super().status()
        info.update(
            {
                "mock": True,
                "seed": self.game.seed,
                "stage": self.game.stage,
                "game_over": self.game.game_over,
                "cleared": self.game.cleared,
            }
        )
        return info

    def apply(self, action: Action) -> dict[str, Any]:
        report = super().apply(action)
        return report
