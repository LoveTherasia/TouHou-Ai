"""把内存标定的结果写回 ``games/offsets/<game>.json``.

标定流程 (``find_player`` / ``find_frame`` / 手动 ``scan``) 产出的地址, 用这里写入
偏移表, 下次 attach 就能读到 —— 从而解锁 ``move_to`` / ``dodge`` / 伤害奖励 /
帧同步等能力。

写入是**原子**的: 先写临时文件再替换, 并默认留一份 ``.json.bak`` 备份, 避免把
一份好用的偏移表改坏。
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from ..errors import TouhouError
from .registry import OFFSETS_DIR, FieldSpec


def offsets_path(game_key: str, offsets_dir: str | Path | None = None) -> Path:
    return Path(offsets_dir or OFFSETS_DIR) / f"{game_key}.json"


def normalize_field(name: str, entry: Any) -> dict[str, Any]:
    """把各种写法 (int / hex 串 / dict) 规范化成 offsets json 里的字段写法."""
    return FieldSpec.from_json(name, entry).to_json()


def apply_calibration(
    game_key: str,
    fields: dict[str, Any],
    *,
    offsets_dir: str | Path | None = None,
    backup: bool = True,
) -> dict[str, Any]:
    """把 ``fields`` 深合并进 ``<game>.json`` 的 ``fields`` 段并落盘.

    Args:
        game_key: 作品 key, 例如 ``th11``。
        fields: ``{字段名: 定义}``; 定义可以是 int(偏移) / "0x1234" / 完整 dict。
        offsets_dir: 偏移表目录 (默认包内的 ``games/offsets``)。
        backup: 是否在覆盖前留一份 ``.json.bak``。

    Returns:
        ``{"path": ..., "written": {...}, "backup": ...}``
    """
    path = offsets_path(game_key, offsets_dir)
    if not path.exists():
        raise TouhouError(f"找不到偏移表: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TouhouError(f"偏移表 {path} 不是合法 JSON: {exc}") from exc

    existing = data.get("fields")
    if not isinstance(existing, dict):
        existing = {}
    written: dict[str, Any] = {}
    for name, entry in fields.items():
        normalized = normalize_field(name, entry)
        existing[name] = normalized
        written[name] = normalized
    data["fields"] = existing

    backup_path: Path | None = None
    if backup:
        backup_path = path.with_suffix(path.suffix + ".bak")
        shutil.copy2(path, backup_path)

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)

    return {
        "path": str(path),
        "written": written,
        "backup": str(backup_path) if backup_path else None,
    }


def snippet_for(candidate: Any, field: str, *, type_name: str = "f32") -> str:
    """把一个 :class:`~touhou_rl_mcp.memory.scanner.Candidate` 转成可粘贴的字段片段."""
    module = getattr(candidate, "module", None)
    module_offset = getattr(candidate, "module_offset", None)
    address = getattr(candidate, "address", None)
    if module and module_offset is not None:
        entry: Any = {"offset": f"0x{module_offset:X}", "type": type_name}
    elif address is not None:
        entry = {"address": f"0x{address:X}", "type": type_name}
    else:  # pragma: no cover - 理论到不了
        entry = {"type": type_name}
    return json.dumps({field: entry}, ensure_ascii=False)
