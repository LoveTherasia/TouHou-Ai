"""内存字段定位 (偏移 / 指针链) 的单元测试.

用一个假的"内存"实现来验证 :class:`StateReader` 的地址解析, 不需要真的游戏进程。
"""

from __future__ import annotations

import struct
import unittest

from touhou_rl_mcp.games.registry import FieldSpec, GameRegistry, GameSpec
from touhou_rl_mcp.memory.reader import StateReader

BASE = 0x400000


class FakeMemory:
    """极小内存模型: ``{address: bytes}``."""

    def __init__(self, cells: dict[int, bytes]) -> None:
        self.cells = dict(cells)

    def read_value(self, address: int, type_name: str) -> float:
        fmt = {"i32": "<i", "u32": "<I", "f32": "<f", "i16": "<h", "u8": "<B"}[type_name]
        size = struct.calcsize(fmt)
        data = self.cells.get(address)
        if data is None or len(data) != size:
            from touhou_rl_mcp.errors import ProcessMemoryError

            raise ProcessMemoryError(f"未映射地址 0x{address:X}")
        return struct.unpack(fmt, data)[0]


def u32(value: int) -> bytes:
    return struct.pack("<I", value)


def i32(value: int) -> bytes:
    return struct.pack("<i", value)


def f32(value: float) -> bytes:
    return struct.pack("<f", value)


def make_spec(fields: dict[str, FieldSpec]) -> GameSpec:
    return GameSpec(key="fake", name_zh="假游戏", exes=["fake.exe"], fields=fields)


class TestStateReader(unittest.TestCase):
    def test_plain_offset_with_transform(self) -> None:
        spec = make_spec(
            {
                "score": FieldSpec("score", offset=0x100, transform="mul10"),
                "lives": FieldSpec("lives", offset=0x108),
            }
        )
        mem = FakeMemory({BASE + 0x100: i32(1234), BASE + 0x108: i32(2)})
        reader = StateReader(mem, spec, auto_detect_module=False)
        reader.base_address = BASE
        state = reader.read_all()
        self.assertEqual(state.score, 12340)
        self.assertEqual(state.lives, 2)

    def test_absolute_address(self) -> None:
        spec = make_spec({"lives": FieldSpec("lives", address=0x7FF000)})
        mem = FakeMemory({0x7FF000: i32(4)})
        reader = StateReader(mem, spec, auto_detect_module=False)
        state = reader.read_all()
        self.assertEqual(state.lives, 4)

    def test_pointer_chain_single_level(self) -> None:
        # ["fake.exe"+0x200]+0x10
        spec = make_spec(
            {"player_x": FieldSpec("player_x", pointer=0x200, offsets=[0x10], type="f32")}
        )
        mem = FakeMemory(
            {
                BASE + 0x200: u32(0x120000),  # 指针
                0x120000 + 0x10: f32(192.5),
            }
        )
        reader = StateReader(mem, spec, auto_detect_module=False)
        reader.base_address = BASE
        self.assertEqual(reader.address_of("player_x"), 0x120010)
        state = reader.read_all()
        self.assertAlmostEqual(state.player_x, 192.5, places=3)

    def test_pointer_chain_two_levels(self) -> None:
        # [[fake.exe"+0x200]+0x18]+0x24
        spec = make_spec(
            {"player_y": FieldSpec("player_y", pointer=0x200, offsets=[0x18, 0x24], type="f32")}
        )
        mem = FakeMemory(
            {
                BASE + 0x200: u32(0x120000),
                0x120000 + 0x18: u32(0x130000),
                0x130000 + 0x24: f32(400.0),
            }
        )
        reader = StateReader(mem, spec, auto_detect_module=False)
        reader.base_address = BASE
        self.assertEqual(reader.address_of("player_y"), 0x130024)
        self.assertAlmostEqual(reader.read_all().player_y, 400.0, places=3)

    def test_broken_pointer_returns_none(self) -> None:
        spec = make_spec({"player_x": FieldSpec("player_x", pointer=0x200, offsets=[0x10], type="f32")})
        mem = FakeMemory({BASE + 0x200: u32(0)})  # 空指针
        reader = StateReader(mem, spec, auto_detect_module=False)
        reader.base_address = BASE
        self.assertIsNone(reader.address_of("player_x"))
        self.assertIsNone(reader.read_all().player_x)

    def test_uncalibrated_is_none_and_in_game_false(self) -> None:
        spec = make_spec({"lives": FieldSpec("lives"), "stage": FieldSpec("stage")})
        reader = StateReader(FakeMemory({}), spec, auto_detect_module=False)
        state = reader.read_all()
        self.assertIsNone(state.lives)
        self.assertFalse(state.in_game)

    def test_in_game_heuristic(self) -> None:
        spec = make_spec(
            {"lives": FieldSpec("lives", offset=0), "stage": FieldSpec("stage", offset=4),
             "power": FieldSpec("power", offset=8, transform="div20")}
        )
        reader = StateReader(
            FakeMemory({BASE: i32(2), BASE + 4: i32(3), BASE + 8: i32(60)}),
            spec,
            auto_detect_module=False,
        )
        reader.base_address = BASE
        state = reader.read_all()
        self.assertTrue(state.in_game)
        self.assertAlmostEqual(state.power, 3.0, places=3)

    def test_spec_json_roundtrip(self) -> None:
        raw = {
            "key": "fake",
            "name_zh": "假",
            "fields": {
                "player_x": {"pointer": "0x200", "offsets": ["0x18", "0x24"], "type": "f32"},
                "lives": {"offset": "0x10", "type": "i32"},
            },
        }
        spec = GameSpec.from_json(raw)
        self.assertEqual(spec.field("player_x").pointer, 0x200)
        self.assertEqual(spec.field("player_x").offsets, [0x18, 0x24])
        self.assertEqual(spec.field("lives").offset, 0x10)
        again = GameSpec.from_json(spec.to_json())
        self.assertEqual(again.field("player_x").offsets, [0x18, 0x24])

    def test_registry_loads_all_shipped_games(self) -> None:
        registry = GameRegistry()
        keys = registry.keys()
        for expected in ("th06", "th07", "th08", "th10", "th11", "th12", "th13", "th14"):
            self.assertIn(expected, keys)
        th11 = registry.get("th11")
        self.assertTrue(th11.calibrated)
        self.assertEqual(th11.field("score").offset, 0xA56E4)
        self.assertEqual(th11.spell_name(1, 43), "罠符「捕捉之网」")
        self.assertEqual(th11.spell_name(6, 49), "「地狱的人工太阳」")


class BulletTrackerTest(unittest.TestCase):
    """速度追踪器: 不能被"差分出全 0"覆盖掉后端自带的权威弹速。

    mock 后端没有 ``active_slots``, 一旦被覆盖, 所有依赖弹道外推的观测
    (占用栅格 / 规划器 / 危险梯度) 都会把弹幕当成静止的 —— 曾经真实发生过。
    """

    def test_keeps_backend_velocity_when_no_slots(self) -> None:
        from touhou_rl_mcp.memory.bullettrack import BulletTracker
        from touhou_rl_mcp.types import GameState

        tracker = BulletTracker()
        first = GameState(frame=1, bullets=[(10.0, 20.0)],
                          raw={"bullet_velocity": [(0.0, 1.9)]})
        tracker.update(first)
        self.assertEqual(first.raw["bullet_velocity"], [(0.0, 1.9)])

        second = GameState(frame=2, bullets=[(10.0, 21.9)],
                           raw={"bullet_velocity": [(0.0, 1.9)]})
        tracker.update(second)
        self.assertEqual(second.raw["bullet_velocity"], [(0.0, 1.9)])

    def test_memory_slots_are_differenced(self) -> None:
        from touhou_rl_mcp.memory.bullettrack import BulletTracker
        from touhou_rl_mcp.types import GameState

        tracker = BulletTracker()
        raw = {"entities": {"bullets": {"active_slots": [7]}}}
        tracker.update(GameState(frame=10, bullets=[(100.0, 100.0)], raw=dict(raw)))
        state = GameState(frame=12, bullets=[(104.0, 100.0)],
                          raw={"entities": {"bullets": {"active_slots": [7]}}})
        tracker.update(state)
        vx, vy = state.raw["bullet_velocity"][0]
        self.assertAlmostEqual(vx, 2.0)      # (104-100)/(12-10)
        self.assertAlmostEqual(vy, 0.0)


if __name__ == "__main__":
    unittest.main()