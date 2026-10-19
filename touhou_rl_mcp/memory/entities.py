"""Bounded, read-only collection extraction; unavailable is distinct from empty.

Layouts are versioned in offsets JSON. Never guess entity locations from floats:
check executable code signatures first, then follow the game's own containers.
All coordinates returned here use a 384 x 448 playfield, origin at top left.
"""
from __future__ import annotations

import math
import struct
from typing import Any

import numpy as np

from ..errors import ProcessMemoryError


def integer(value: Any) -> int:
    return int(value, 0) if isinstance(value, str) else int(value)


class EntityReader:
    def __init__(self, mem: Any, base: int, profile: dict[str, Any]) -> None:
        self.mem, self.base, self.profile = mem, base, profile
        self._verified = False

    def _verify(self) -> None:
        if self._verified:
            return
        signatures = self.profile.get("signatures", [])
        if not signatures:
            raise ValueError("Entity profile has no executable signatures")
        for signature in signatures:
            expected = bytes.fromhex(signature["bytes"])
            address = self.base + integer(signature["offset"])
            if self.mem.read_bytes(address, len(expected)) != expected:
                raise ValueError(f"Unsupported executable: entity signature mismatch at 0x{address:X}")
        self._verified = True

    @staticmethod
    def _point(x: float, y: float, shift: list[float]) -> tuple[float, float]:
        if not (math.isfinite(x) and math.isfinite(y)) or abs(x) > 8192 or abs(y) > 8192:
            raise ValueError("Invalid entity coordinates")
        return float(x + shift[0]), float(y + shift[1])

    def _manager(self, layout: dict[str, Any]) -> int:
        return int(self.mem.read_value(self.base + integer(layout["pointer"]), "u32"))

    def _array(self, layout: dict[str, Any], manager: int) -> tuple[list, dict]:
        count, stride = integer(layout["capacity"]), integer(layout["stride"])
        xy, status = integer(layout["position"]), integer(layout["state"])
        if not (0 < count <= 8192 and 0 < stride and count * stride <= 16 * 1024 * 1024
                and 0 <= xy <= stride - 8 and 0 <= status <= stride - 2):
            raise ValueError("Unsafe array layout")
        # One ReadProcessMemory call, rather than thousands of calls each frame.
        data = self.mem.read_bytes(manager + integer(layout["array"]), count * stride)
        states = np.ndarray((count,), dtype="<u2", buffer=data, offset=status, strides=(stride,))
        positions = np.ndarray((count, 2), dtype="<f4", buffer=data, offset=xy, strides=(stride, 4))
        allowed = layout.get("known_states", [0, 1, 2, 3])
        if not np.isin(states, allowed).all():
            raise ValueError("Unexpected bullet state: layout mismatch or torn snapshot")
        active = np.isin(states, layout["active_states"])
        shift = layout.get("shift", [0, 0])
        points = [self._point(float(x), float(y), shift) for x, y in positions[active]]
        return points, {"slots": count, "active": len(points),
                        "active_slots": np.flatnonzero(active).tolist(),
                        "nonzero_slots": int(np.count_nonzero(states))}

    def _linked(self, layout: dict[str, Any], manager: int) -> tuple[list, dict]:
        node = int(self.mem.read_value(manager + integer(layout["head"]), "u32"))
        limit = integer(layout.get("max_nodes", 1024))
        if not 0 < limit <= 8192:
            raise ValueError("Unsafe linked-list limit")
        position, flags = integer(layout["position"]), integer(layout["flags"])
        start, end = min(position, flags), max(position + 8, flags + 4)
        if start < 0 or end - start > 65536:
            raise ValueError("Unsafe enemy layout")
        seen, points = set(), []
        while node:
            if node in seen or len(seen) >= limit:
                raise ValueError("Cyclic or oversized enemy list")
            seen.add(node)
            obj, node = struct.unpack("<II", self.mem.read_bytes(node, 8))
            if not obj:
                raise ValueError("Null enemy object in live list")
            data = self.mem.read_bytes(obj + start, end - start)
            bits = struct.unpack_from("<I", data, flags - start)[0]
            if bits & integer(layout.get("exclude_flags", 0)):
                continue
            x, y = struct.unpack_from("<ff", data, position - start)
            points.append(self._point(x, y, layout.get("shift", [0, 0])))
        return points, {"nodes": len(seen), "active": len(points)}

    def read(self) -> tuple[dict[str, list], dict[str, dict]]:
        collections: dict[str, list] = {}
        diagnostics: dict[str, dict] = {}
        try:
            self._verify()
        except (ProcessMemoryError, ValueError) as exc:
            return {}, {name: {"status": "error", "error": str(exc)} for name in ("bullets", "enemies")}
        for name in ("bullets", "enemies"):
            layout = self.profile.get(name)
            if not layout:
                diagnostics[name] = {"status": "unconfigured"}
                continue
            try:
                manager = self._manager(layout)
                if not manager:
                    diagnostics[name] = {"status": "inactive"}
                    continue
                if layout["kind"] == "array":
                    points, detail = self._array(layout, manager)
                elif layout["kind"] == "linked":
                    points, detail = self._linked(layout, manager)
                else:
                    raise ValueError(f"Unknown entity container {layout['kind']}")
                # A stage transition can replace a manager while we read its data.
                if manager != self._manager(layout):
                    raise ValueError("Manager changed during snapshot")
                collections[name] = points
                diagnostics[name] = {"status": "ok", **detail}
            except (ProcessMemoryError, ValueError, struct.error) as exc:
                diagnostics[name] = {"status": "error", "error": str(exc)}
        if self.profile.get("combat"):
            try:
                layout = self.profile["combat"]
                for signature in layout.get("signatures", []):
                    expected = bytes.fromhex(signature["bytes"])
                    if self.mem.read_bytes(self.base+integer(signature["offset"]), len(expected)) != expected:
                        raise ValueError("Combat signature mismatch")
                manager = self._manager(self.profile["enemies"])
                boss = self.mem.read_value(manager + integer(layout["boss"]), "u32") if manager else 0
                detail = {"status": "ok", "boss_id": boss}
                if boss:
                    hp, maximum = struct.unpack("<ii", self.mem.read_bytes(boss + integer(layout["hp"]), 8))
                    if not (-100000 <= hp <= 100000000 and 0 <= maximum <= 100000000):
                        raise ValueError("Invalid boss HP")
                    detail.update(hp=hp, max_hp=maximum)
                    flags = self.mem.read_value(boss+integer(layout["flags"]), "u32")
                    detail["flags"] = flags
                    detail["timeout"] = bool(flags & 0x100000)
                    # Boss 位置 (与敌人同一布局): 有了它策略/老师才能对准输出
                    pos_off = integer(layout.get("position", "0x1070"))
                    bx, by = struct.unpack("<ff", self.mem.read_bytes(boss + pos_off, 8))
                    shift = self.profile.get("enemies", {}).get("shift", [0, 0])
                    if (math.isfinite(bx) and math.isfinite(by)
                            and abs(bx) < 8192 and abs(by) < 8192):
                        detail["x"] = bx + shift[0]
                        detail["y"] = by + shift[1]
                spell = self.mem.read_value(self.base + integer(layout["spell_manager"]), "u32")
                if spell:
                    sid, flags, bonus = struct.unpack("<iii", self.mem.read_bytes(spell+integer(layout["spell_id"]), 12))
                    detail.update(spell_id=sid, spell_flags=flags, spell_bonus=bonus)
                if manager != self._manager(self.profile["enemies"]):
                    raise ValueError("Combat manager changed during snapshot")
                diagnostics["combat"] = detail
            except (ProcessMemoryError, ValueError, struct.error) as exc:
                diagnostics["combat"] = {"status": "error", "error": str(exc)}
        return collections, diagnostics
