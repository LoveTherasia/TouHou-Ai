"""按 :class:`GameSpec` 的字段定义把内存读成 :class:`GameState`."""

from __future__ import annotations

from typing import Any

from ..errors import ProcessMemoryError
from ..games.registry import GameSpec
from ..types import GameState
from .process import ProcessMemory
from .entities import EntityReader

#: 简单变换
_TRANSFORMS = {
    "mul10": lambda v: v * 10,
    "div10": lambda v: v / 10,
    "div20": lambda v: v / 20.0,
    "mul20": lambda v: v * 20,
    "div100": lambda v: v / 100.0,
    "whole": lambda v: int(v),
}


class StateReader:
    """从进程内存读取游戏状态."""

    def __init__(self, mem: ProcessMemory, spec: GameSpec, *, auto_detect_module: bool = True) -> None:
        self.mem = mem
        self.spec = spec
        self.base_address = 0
        if auto_detect_module:
            self.base_address = self._detect_base()
        self.entities = EntityReader(mem, self.base_address, spec.entities) if spec.entities else None

    # ------------------------------------------------------------------
    def _detect_base(self) -> int:
        preferred = [self.spec.exes[0]] if self.spec.exes else []
        # 主模块名通常是 exe
        try:
            name, base, _size = self.mem.main_module
        except ProcessMemoryError:
            name, base = "", 0
        for exe in preferred:
            try:
                return self.mem.module_base(exe)
            except ProcessMemoryError:
                continue
        return base or 0

    # ------------------------------------------------------------------
    def address_of(self, field_name: str) -> int | None:
        spec = self.spec.field(field_name)
        if not spec.calibrated:
            return None
        if spec.pointer is not None:
            return self.resolve_pointer(spec.pointer, spec.offsets)
        if spec.address is not None:
            return spec.address
        assert spec.offset is not None
        return self.base_address + spec.offset

    # ------------------------------------------------------------------
    def resolve_pointer(self, pointer: int | None, offsets: list[int], *, absolute: bool = False) -> int | None:
        """解析指针链 (Cheat Engine 记法 ``[[base+ptr]+o0]+o1``).

        Args:
            pointer: 指针变量的位置 (默认相对模块基址)。
            offsets: 指针链偏移; 最后一项只相加不解引用。
            absolute: ``pointer`` 是否为绝对地址。
        """
        if pointer is None:
            return None
        address = pointer if absolute else self.base_address + pointer
        try:
            address = int(self.mem.read_value(address, "u32"))
            for off in offsets[:-1]:
                address = int(self.mem.read_value(address + off, "u32"))
        except ProcessMemoryError:
            return None
        if not address:
            return None
        return address + offsets[-1] if offsets else address

    # ------------------------------------------------------------------
    def read_raw(self, field_name: str) -> float | None:
        spec = self.spec.field(field_name)
        address = self.address_of(field_name)
        if address is None:
            return None
        try:
            return self.mem.read_value(address, spec.type)
        except ProcessMemoryError:
            return None

    def apply_transform(self, field_name: str, value: float | None) -> float | None:
        if value is None:
            return None
        spec = self.spec.field(field_name)
        if not spec.transform:
            return value
        func = _TRANSFORMS.get(spec.transform)
        return func(value) if func else value

    # ------------------------------------------------------------------
    def read_all(self) -> GameState:
        """读取所有字段 (只读已标定的), 每个字段只读一次."""
        raw: dict[str, Any] = {}
        for name, spec in self.spec.fields.items():
            if not spec.calibrated:
                raw[name] = None
                continue
            try:
                raw[name] = self.mem.read_value(self.address_of(name) or 0, spec.type)
            except ProcessMemoryError:
                raw[name] = None

        def value(name: str) -> float | None:
            return self.apply_transform(name, raw.get(name))

        def as_int(name: str) -> int | None:
            got = value(name)
            return None if got is None else int(got)

        def as_float(name: str) -> float | None:
            got = value(name)
            return None if got is None else float(got)

        state = GameState(
            stage=as_int("stage"),
            lives=as_int("lives"),
            power=as_float("power"),
            bombs=as_int("bombs"),
            score=as_int("score"),
            graze=as_int("graze"),
            spell_id=as_int("spell_id"),
            char_main=as_int("char_main"),
            char_sub=as_int("char_sub"),
            boss_hp=as_int("boss_hp"),
            boss_max_hp=as_int("boss_max_hp"),
            player_x=as_float("player_x"),
            player_y=as_float("player_y"),
            invincible=as_int("invincible"),
            raw={k: (None if v is None else (int(v) if float(v).is_integer() else v)) for k, v in raw.items()},
        )
        state.frame = as_int("frame") or 0
        state.char_name = self._char_name(state.char_main, state.char_sub)
        state.spell_name = self.spec.spell_name(state.stage, state.spell_id)
        state.in_game = self.detect_in_game(state)
        state.raw["backend"] = "memory"
        if self.entities is not None:
            if self.entities.base != self.base_address:
                self.entities = EntityReader(self.mem, self.base_address, self.spec.entities)
            collections, diagnostics = self.entities.read()
            state.raw["entities"] = diagnostics
            combat = diagnostics.get("combat", {})
            if combat.get("status") == "ok":
                state.boss_hp = combat.get("hp")
                state.boss_max_hp = combat.get("max_hp")
                if combat.get("x") is not None and combat.get("y") is not None:
                    state.raw["boss_position"] = (combat["x"], combat["y"])
            state.raw["coordinate_system"] = "playfield_top_left"
            # The old player fields in TH11 are animation/window coordinates.
            shift = self.spec.entities.get("player_shift", [0, 0])
            if state.player_x is not None:
                state.player_x += shift[0]
            if state.player_y is not None:
                state.player_y += shift[1]
            state.bullets = collections.get("bullets", [])
            state.bullet_count = len(state.bullets) if "bullets" in collections else None
            state.enemy_positions = collections.get("enemies", [])
            state.enemies_alive = len(state.enemy_positions) if "enemies" in collections else None
        return state

    # ------------------------------------------------------------------
    def _char_name(self, main: int | None, sub: int | None) -> str | None:
        if main is None:
            return None
        name = self.spec.chars.get(str(main))
        if name is None:
            return None
        return f"{name}{'ABC'[sub] if sub is not None and 0 <= sub < 3 else ''}"

    def detect_in_game(self, state: GameState) -> bool:
        """启发式判断"是否在游戏局内".

        东方正作在标题/菜单画面时, 全局结构里的数值通常是 0 或上一局的残留,
        因此用一组宽松但有效的区间约束。若某个版本误差大, 可以在 offsets json
        的 ``notes`` 里记录, 或改这里。
        """
        if state.stage is None or state.lives is None:
            return False
        if not (1 <= int(state.stage) <= 9):
            return False
        if not (0 <= int(state.lives) <= 9):
            return False
        if state.power is not None and not (0.0 <= float(state.power) <= 5.0):
            return False
        if state.score is not None and int(state.score) < 0:
            return False
        return True
