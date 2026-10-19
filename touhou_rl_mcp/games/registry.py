"""游戏注册表: 读取 ``offsets/*.json``, 自动查找游戏安装目录."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..errors import GameNotFoundError

OFFSETS_DIR = Path(__file__).resolve().parent / "offsets"

#: 内存字段类型 -> struct 格式
TYPE_FORMATS: dict[str, str] = {
    "i8": "<b",
    "u8": "<B",
    "i16": "<h",
    "u16": "<H",
    "i32": "<i",
    "u32": "<I",
    "f32": "<f",
    "f64": "<d",
}

#: 名字中必须包含的关键字 (用来在磁盘上找安装目录)
_GAME_FOLDER_ALIASES = {
    "th06": ["th06", "红魔乡", "紅魔郷", "koumakyou"],
    "th07": ["th07", "妖妖梦", "妖々夢", "youyoumu"],
    "th08": ["th08", "永夜抄", "eiyashou"],
    "th09": ["th09", "花映塚", "kaeidzuka"],
    "th10": ["th10", "风神录", "風神録", "fuujinroku"],
    "th11": ["th11", "地灵殿", "地霊殿", "chireiden"],
    "th12": ["th12", "星莲船", "星蓮船", "ufo"],
    "th13": ["th13", "神灵庙", "神霊廟", "ten desires"],
    "th14": ["th14", "辉针城", "輝針城", "double dealing"],
    "th15": ["th15", "绀珠传", "紺珠伝", "legacy of lunatic"],
    "th16": ["th16", "天空璋", "hidden star"],
    "th17": ["th17", "鬼形兽", "鬼形獣", "wily beast"],
    "th18": ["th18", "虹龙洞", "虹龍洞", "unconnected market"],
}


@dataclass
class FieldSpec:
    """单个内存字段的定义.

    三种定位方式 (按优先级):

    1. ``pointer`` + ``offsets`` —— 指针链 (th10+ 的自机/Boss 数据几乎都在堆上,
       只能靠指针链找到)。解析规则::

           addr = base + pointer
           addr = read_u32(addr)
           for off in offsets[:-1]:   addr = read_u32(addr + off)
           value_addr = addr + offsets[-1]

       即 Cheat Engine 记法 ``[[mod+pointer]+offsets[0]]+offsets[1]``。

    2. ``offset`` —— 相对模块基址的固定偏移 (th11 的全局游戏数据就是这种)。

    3. ``address`` —— 绝对地址 (东方正作是无 ASLR 的 32 位程序, 基址固定,
       所以绝对地址跨进程也稳定)。
    """

    name: str
    offset: int | None = None  # 相对模块基址
    address: int | None = None  # 绝对地址 (offset 为 None 时使用)
    pointer: int | None = None  # 相对模块基址的指针变量
    offsets: list[int] = field(default_factory=list)  # 指针链偏移
    type: str = "i32"
    transform: str | None = None
    label: str = ""

    @property
    def calibrated(self) -> bool:
        return self.offset is not None or self.address is not None or self.pointer is not None

    @classmethod
    def from_json(cls, name: str, data: Any) -> "FieldSpec":
        if data is None:
            return cls(name=name)
        if isinstance(data, int):
            return cls(name=name, offset=data)
        if isinstance(data, str):
            return cls(name=name, offset=int(data, 0))
        offsets = [_maybe_int(o) for o in (data.get("offsets") or [])]
        return cls(
            name=name,
            offset=_maybe_int(data.get("offset")),
            address=_maybe_int(data.get("address")),
            pointer=_maybe_int(data.get("pointer")),
            offsets=[o for o in offsets if o is not None],
            type=str(data.get("type", "i32")),
            transform=data.get("transform"),
            label=str(data.get("label", "")),
        )

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"type": self.type}
        if self.offset is not None:
            out["offset"] = f"0x{self.offset:X}"
        if self.address is not None:
            out["address"] = f"0x{self.address:X}"
        if self.pointer is not None:
            out["pointer"] = f"0x{self.pointer:X}"
            out["offsets"] = [f"0x{o:X}" for o in self.offsets]
        if self.transform:
            out["transform"] = self.transform
        if self.label:
            out["label"] = self.label
        return out


def _maybe_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, int):
        return value
    return int(str(value), 0)


@dataclass
class GameSpec:
    """一个游戏的定义."""

    key: str
    name_zh: str
    name_en: str = ""
    exes: list[str] = field(default_factory=list)
    folder_hints: list[str] = field(default_factory=list)
    fields: dict[str, FieldSpec] = field(default_factory=dict)
    playfield: list[int] = field(default_factory=lambda: [32, 16, 416, 464])
    window_size: list[int] = field(default_factory=lambda: [640, 480])
    menu_sequence: list[dict[str, Any]] = field(default_factory=list)
    spells: dict[str, dict[str, str]] = field(default_factory=dict)
    chars: dict[str, str] = field(default_factory=dict)
    calibrated: bool = False
    notes: str = ""
    # Collection layouts and coordinate transforms are separate from scalar fields.
    entities: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    def field(self, name: str) -> FieldSpec:
        spec = self.fields.get(name)
        return spec if spec is not None else FieldSpec(name=name)

    def uncalibrated(self) -> list[str]:
        return sorted(k for k, v in self.fields.items() if not v.calibrated)

    def spell_name(self, stage: int | None, spell_id: int | None) -> str | None:
        if stage is None or spell_id is None:
            return None
        table = self.spells.get(str(stage)) or self.spells.get(str(int(stage)))
        if not table:
            return None
        return table.get(str(spell_id)) or table.get(str(int(spell_id)))

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "GameSpec":
        fields = {
            name: FieldSpec.from_json(name, value)
            for name, value in (data.get("fields") or {}).items()
        }
        return cls(
            key=data["key"],
            name_zh=data.get("name_zh", data["key"]),
            name_en=data.get("name_en", ""),
            exes=list(data.get("exes") or []),
            folder_hints=list(
                data.get("folder_hints") or _GAME_FOLDER_ALIASES.get(data["key"], [])
            ),
            fields=fields,
            playfield=list(data.get("playfield") or [32, 16, 416, 464]),
            window_size=list(data.get("window_size") or [640, 480]),
            menu_sequence=list(data.get("menu_sequence") or []),
            spells=dict(data.get("spells") or {}),
            chars=dict(data.get("chars") or {}),
            calibrated=bool(data.get("calibrated", False)),
            notes=str(data.get("notes", "")),
            entities=dict(data.get("entities") or {}),
        )

    # ------------------------------------------------------------------
    def to_json(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "name_zh": self.name_zh,
            "name_en": self.name_en,
            "exes": self.exes,
            "folder_hints": self.folder_hints,
            "fields": {k: v.to_json() for k, v in self.fields.items()},
            "playfield": self.playfield,
            "window_size": self.window_size,
            "menu_sequence": self.menu_sequence,
            "spells": self.spells,
            "chars": self.chars,
            "calibrated": self.calibrated,
            "notes": self.notes,
            "entities": self.entities,
        }


class GameRegistry:
    """``offsets/*.json`` 的只读集合."""

    def __init__(self, offsets_dir: Path | None = None) -> None:
        self.dir = Path(offsets_dir or OFFSETS_DIR)
        self._games: dict[str, GameSpec] = {}
        self.reload()

    def reload(self) -> None:
        games: dict[str, GameSpec] = {}
        for path in sorted(self.dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            key = data.get("key") or path.stem
            data.setdefault("key", key)
            games[key] = GameSpec.from_json(data)
        self._games = games

    # ------------------------------------------------------------------
    def keys(self) -> list[str]:
        return sorted(self._games)

    def all(self) -> list[GameSpec]:
        return [self._games[k] for k in self.keys()]

    def get(self, key: str) -> GameSpec:
        if key not in self._games:
            raise GameNotFoundError(
                f"未知的游戏 {key!r}; 可用: {', '.join(self.keys())}"
            )
        return self._games[key]

    def has(self, key: str) -> bool:
        return key in self._games

    # ------------------------------------------------------------------
    def detect_dir(self, key: str, search_roots: list[str], depth: int = 2) -> str | None:
        """在 ``search_roots`` 的一级/二级子目录里找包含该游戏 exe 的目录."""
        spec = self.get(key)
        hints = [h.lower() for h in spec.folder_hints]
        exes = {e.lower() for e in spec.exes}
        candidates: list[Path] = []

        for root in search_roots:
            base = Path(root)
            if not base.is_dir():
                continue
            try:
                level1 = [p for p in base.iterdir() if p.is_dir()]
            except OSError:
                continue
            for d1 in level1:
                if self._dir_matches(d1, hints, exes):
                    candidates.append(d1)
                    continue
                if depth < 2:
                    continue
                try:
                    for d2 in d1.iterdir():
                        if d2.is_dir() and self._dir_matches(d2, hints, exes):
                            candidates.append(d2)
                except OSError:
                    continue

        if not candidates:
            return None
        # 优先中文目录（汉化版通常是窗口化 / 更好挂载）
        candidates.sort(key=lambda p: (0 if re.search(r"[\u4e00-\u9fff]", p.name) else 1, len(p.name)))
        return str(candidates[0])

    @staticmethod
    def _dir_matches(path: Path, hints: list[str], exes: set[str]) -> bool:
        name = path.name.lower()
        if any(e in name for e in exes):
            return True
        if not any(h in name for h in hints):
            return False
        try:
            children = {p.name.lower() for p in path.iterdir()}
        except OSError:
            return False
        return bool(children & exes)

    # ------------------------------------------------------------------
    def find_exe(self, key: str, game_dir: str | None) -> Path:
        """在 ``game_dir`` 中找到最合适的 exe (优先窗口化汉化版 ``thXXc.exe``)."""
        spec = self.get(key)
        if not game_dir:
            raise GameNotFoundError(
                f"未指定 {key} 的安装目录 (config.game_dir 为空且自动查找失败). "
                f"可用 `list_games` 查看检测结果, 或直接设置 config.game_dir。"
            )
        base = Path(game_dir)
        if not base.is_dir():
            raise GameNotFoundError(f"目录不存在: {base}")
        available = {p.name.lower(): p for p in base.iterdir() if p.is_file()}
        for exe in spec.exes:
            hit = available.get(exe.lower())
            if hit is not None:
                return hit
        raise GameNotFoundError(
            f"在 {base} 中找不到 {key} 的可执行文件 (期望: {', '.join(spec.exes)})"
        )
