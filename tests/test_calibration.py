"""内存标定工具链测试: 写回 offsets json + 自动找帧计数器.

不需要真实游戏: 用假内存模型驱动 :class:`MemoryScanner.find_frame_counter`。
"""

from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path

from touhou_rl_mcp.errors import ProcessMemoryError, TouhouError
from touhou_rl_mcp.games.calibration import apply_calibration, normalize_field
from touhou_rl_mcp.games.registry import GameRegistry
from touhou_rl_mcp.memory.scanner import MemoryScanner

MODULE_BASE = 0x400000
MODULE_SIZE = 0x200000
REGION_BASE = 0x500000
REGION_SIZE = 0x1000

FRAME_ADDR = REGION_BASE + 0x100
STATIC_ADDR = REGION_BASE + 0x200
DOUBLE_ADDR = REGION_BASE + 0x300


def _write_i32(buf: bytearray, base: int, address: int, value: int) -> None:
    offset = address - base
    struct.pack_into("<i", buf, offset, value)


class FakeProcess:
    """极小的进程内存模型: 一个区域 + 一个模块."""

    pid = 4242

    def __init__(self, regions: dict[int, bytearray]) -> None:
        self.regions = regions

    def memory_regions(self) -> list[tuple[int, int, int, int]]:
        return [(base, len(buf), 0x04, 0x20000) for base, buf in self.regions.items()]

    def read_bytes(self, address: int, size: int) -> bytes:
        for base, buf in self.regions.items():
            if base <= address and address + size <= base + len(buf):
                offset = address - base
                return bytes(buf[offset : offset + size])
        raise ProcessMemoryError(f"未映射地址 0x{address:X}")

    def modules(self) -> dict[str, tuple[int, int]]:
        return {"fake.exe": (MODULE_BASE, MODULE_SIZE)}


def make_frame_scanner() -> tuple[MemoryScanner, bytearray, dict]:
    """构造一个含"帧计数器 + 静态值 + 2 倍速计数器"的假内存."""
    buf = bytearray(REGION_SIZE)
    _write_i32(buf, REGION_BASE, FRAME_ADDR, 1000)
    _write_i32(buf, REGION_BASE, STATIC_ADDR, 777)
    _write_i32(buf, REGION_BASE, DOUBLE_ADDR, 500)
    state = {"frame": 1000, "double": 500}

    def advance(frames: int) -> None:
        state["frame"] += frames
        state["double"] += 2 * frames
        _write_i32(buf, REGION_BASE, FRAME_ADDR, state["frame"])
        _write_i32(buf, REGION_BASE, DOUBLE_ADDR, state["double"])

    scanner = MemoryScanner(FakeProcess({REGION_BASE: buf}))  # type: ignore[arg-type]
    return scanner, buf, {"advance": advance}


class TestFindFrameCounter(unittest.TestCase):
    def test_finds_the_monotonic_counter(self) -> None:
        scanner, _buf, ctl = make_frame_scanner()
        candidates = scanner.find_frame_counter(ctl["advance"], frames=60, tolerance=0.25, repeats=2)
        addresses = {c.address for c in candidates}
        self.assertIn(FRAME_ADDR, addresses)
        # 静态值不动 -> 排除; 2 倍速计数器 -> 超出容差 -> 排除
        self.assertNotIn(STATIC_ADDR, addresses)
        self.assertNotIn(DOUBLE_ADDR, addresses)
        self.assertEqual(len(candidates), 1)

    def test_maps_address_to_module_offset(self) -> None:
        scanner, _buf, ctl = make_frame_scanner()
        candidates = scanner.find_frame_counter(ctl["advance"], frames=30, repeats=1)
        best = candidates[0]
        self.assertEqual(best.module, "fake.exe")
        self.assertEqual(best.module_offset, FRAME_ADDR - MODULE_BASE)


class TestApplyCalibration(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        (self.dir / "th11.json").write_text(
            json.dumps(
                {
                    "key": "th11",
                    "name_zh": "东方地灵殿",
                    "fields": {
                        "score": {"offset": "0xA56E4", "type": "i32"},
                        "player_x": {"type": "f32"},
                    },
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def _data(self) -> dict:
        return json.loads((self.dir / "th11.json").read_text(encoding="utf-8"))

    def test_write_merges_and_backs_up(self) -> None:
        result = apply_calibration(
            "th11",
            {"player_x": {"offset": "0x1234", "type": "f32"}, "graze": "0xA571C"},
            offsets_dir=self.dir,
        )
        data = self._data()
        self.assertEqual(data["fields"]["player_x"]["offset"], "0x1234")
        self.assertEqual(data["fields"]["graze"]["offset"], "0xA571C")
        self.assertEqual(data["fields"]["graze"]["type"], "i32")
        # 原有字段保留
        self.assertEqual(data["fields"]["score"]["offset"], "0xA56E4")
        self.assertTrue((self.dir / "th11.json.bak").exists())
        self.assertIn("written", result)

    def test_registry_reloads_new_field(self) -> None:
        apply_calibration("th11", {"player_y": {"offset": "0x2000", "type": "f32"}}, offsets_dir=self.dir)
        registry = GameRegistry(offsets_dir=self.dir)
        self.assertEqual(registry.get("th11").field("player_y").offset, 0x2000)

    def test_missing_file_raises(self) -> None:
        with self.assertRaises(TouhouError):
            apply_calibration("th99", {"lives": 0x10}, offsets_dir=self.dir)

    def test_normalize_accepts_forms(self) -> None:
        self.assertEqual(normalize_field("a", 0x10)["offset"], "0x10")
        self.assertEqual(normalize_field("b", "0x20")["offset"], "0x20")
        self.assertEqual(normalize_field("c", {"address": "0x30"})["address"], "0x30")


if __name__ == "__main__":
    unittest.main()
