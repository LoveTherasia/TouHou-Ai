"""TH11 production-layout regression tests, without a running game."""
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np

from touhou_rl_mcp.errors import BackendError, ProcessMemoryError
from touhou_rl_mcp.games.registry import GameRegistry, GameSpec
from touhou_rl_mcp.memory.entities import EntityReader
from touhou_rl_mcp.memory.reader import StateReader
from touhou_rl_mcp.training.features import encode_state
from touhou_rl_mcp.training.trainer import Trainer, TrainConfig
from touhou_rl_mcp.types import GameState, StepResult
from touhou_rl_mcp.rl.reward import RewardModel

BASE, BM, EM, PM = 0x400000, 0x1000000, 0x2000000, 0x3000000


class Memory:
    main_module = ("th11c.exe", BASE, 0xB0000)

    def __init__(self):
        self.blocks = {}

    def module_base(self, name):
        return BASE

    def put(self, address, fmt, *values):
        self.blocks[address] = struct.pack(fmt, *values)

    def read_bytes(self, address, size):
        for start, data in self.blocks.items():
            if start <= address and address + size <= start + len(data):
                return data[address - start:address - start + size]
        raise ProcessMemoryError(f"Unmapped {address:X}/{size}")

    def read_value(self, address, kind):
        fmt = {"u32": "<I", "i32": "<i", "f32": "<f"}[kind]
        return struct.unpack(fmt, self.read_bytes(address, 4))[0]


class TestEntities(unittest.TestCase):
    def setUp(self):
        self.spec = GameRegistry().get("th11")
        self.mem = Memory()
        for sig in self.spec.entities["signatures"]:
            self.mem.blocks[BASE + int(sig["offset"], 0)] = bytes.fromhex(sig["bytes"])
        self.mem.put(BASE + 0xA8D68, "<I", BM)
        self.mem.put(BASE + 0xA8D7C, "<I", EM)
        self.mem.put(EM + 0x68, "<I", 0)
        self.mem.blocks[BM + 0x64] = bytearray(2000 * 0x910)
        self.reader = EntityReader(self.mem, BASE, self.spec.entities)

    def bullet(self, slot, state, x, y):
        data = self.mem.blocks[BM + 0x64]
        struct.pack_into("<H", data, slot * 0x910 + 0x4B2, state)
        struct.pack_into("<ff", data, slot * 0x910 + 0x43C, x, y)

    def enemies(self, flags=(0, 0x20, 0x200000)):
        for i, bits in enumerate(flags):
            obj = 0x4000000 + i * 0x10000
            next_node = obj + 0x10000 + 0x11A4 if i + 1 < len(flags) else 0
            data = bytearray(0x2600)
            struct.pack_into("<ff", data, 0x1070, -100 + i, 80 + i)
            struct.pack_into("<II", data, 0x11A4, obj, next_node)
            struct.pack_into("<I", data, 0x25BC, bits)
            self.mem.blocks[obj] = data
        self.mem.put(EM + 0x68, "<I", 0x40011A4)

    def test_sparse_array_all_slots_and_filter_states(self):
        self.bullet(1999, 1, -20, 390)
        self.bullet(1571, 2, 10, 50)
        self.bullet(0, 3, 0, 0)
        self.enemies()
        points, detail = self.reader.read()
        self.assertEqual(points["bullets"], [(202, 50), (172, 390)])
        self.assertEqual(points["enemies"], [(92, 80)])
        self.assertEqual(detail["bullets"]["nonzero_slots"], 3)
        self.assertEqual(detail["enemies"]["nodes"], 3)

    def test_valid_empty_not_unavailable(self):
        points, detail = self.reader.read()
        self.assertEqual(points, {"bullets": [], "enemies": []})
        self.assertEqual(detail["bullets"]["status"], "ok")
        self.mem.put(BASE + 0xA8D68, "<I", 0)
        points, detail = self.reader.read()
        self.assertNotIn("bullets", points)
        self.assertEqual(detail["bullets"]["status"], "inactive")

    def test_signature_mismatch(self):
        self.mem.blocks[BASE + 0xAF9A] = bytes(10)
        points, detail = self.reader.read()
        self.assertFalse(points)
        self.assertIn("signature mismatch", detail["bullets"]["error"])

    def test_no_signatures_rejected(self):
        self.reader.profile = {**self.spec.entities, "signatures": []}
        self.assertEqual(self.reader.read()[1]["bullets"]["status"], "error")

    def test_bad_state_and_nan_rejected_no_stale_points(self):
        for status, x in ((4, 0), (1, float("nan"))):
            self.bullet(0, status, x, 1)
            points, detail = self.reader.read()
            self.assertNotIn("bullets", points)
            self.assertEqual(detail["bullets"]["status"], "error")

    def test_cycle_and_unreadable_enemy_bounded(self):
        self.enemies((0,))
        struct.pack_into("<I", self.mem.blocks[0x4000000], 0x11A8, 0x40011A4)
        points, detail = self.reader.read()
        self.assertNotIn("enemies", points)
        self.assertIn("Cyclic", detail["enemies"]["error"])
        del self.mem.blocks[0x4000000]
        self.assertEqual(self.reader.read()[1]["enemies"]["status"], "error")

    def test_manager_replacement_invalidates_snapshot(self):
        original = self.mem.read_value
        calls = 0
        def changing(address, kind):
            nonlocal calls
            if address == BASE + 0xA8D68:
                calls += 1
                return BM if calls == 1 else BM + 4
            return original(address, kind)
        self.mem.read_value = changing
        points, detail = self.reader.read()
        self.assertNotIn("bullets", points)
        self.assertIn("Manager changed", detail["bullets"]["error"])

    def test_state_reader_coordinates_and_nonzero_policy_features(self):
        self.mem.put(BASE + 0xA8EB4, "<I", PM)
        self.mem.put(PM + 0x3FC, "<ff", 224, 416)
        self.mem.put(BASE + 0xA5728, "<i", 1)
        self.mem.put(BASE + 0xA5718, "<i", 2)
        self.bullet(1999, 1, 5, 399)
        self.enemies((0,))
        reader = StateReader(self.mem, self.spec, auto_detect_module=False)
        reader.base_address = BASE
        state = reader.read_all()
        self.assertEqual((state.player_x, state.player_y), (192, 400))
        self.assertEqual(state.bullets, [(197, 399)])
        self.assertEqual(state.enemy_positions, [(92, 80)])
        vec = encode_state(state)
        self.assertEqual(vec.shape, (74,))
        np.testing.assert_allclose(vec[10:14], [5/384, -1/448, np.sqrt(26)/320, 1])
        self.assertLess(RewardModel({"danger_scale": 1, "danger_radius": 45})._danger(state), 0)
        self.assertEqual(encode_state(state, k_bullets=0).shape, (10,))
        self.mem.put(BASE + 0xA8D68, "<I", 0)
        failed = reader.read_all()
        self.assertIsNone(failed.to_dict()["bullet_count"])
        with self.assertRaises(BackendError):
            encode_state(failed)

    def test_profile_roundtrip(self):
        self.assertEqual(GameSpec.from_json(self.spec.to_json()).entities, self.spec.entities)

    def test_combat_reader_uses_verified_boss_and_spell_layout(self):
        for sig in self.spec.entities["combat"]["signatures"]:
            self.mem.blocks[BASE+int(sig["offset"],0)] = bytes.fromhex(sig["bytes"])
        self.mem.put(EM+0x1C,"<I",0x5000000)
        self.mem.put(0x500252C,"<ii",1234,5000)
        self.mem.put(0x50025BC,"<I",0)
        self.mem.put(0x5001070,"<ff",100.0,80.0)   # Boss 位置 (与敌人同布局)
        self.mem.put(BASE+0xA8D6C,"<I",0x6000000)
        self.mem.put(0x60008DC,"<iii",43,3,1000000)
        detail = self.reader.read()[1]["combat"]
        self.assertEqual(detail["hp"],1234)
        self.assertEqual(detail["max_hp"],5000)
        self.assertEqual(detail["spell_bonus"],1000000)
        self.assertEqual(detail["x"],292.0)   # 100 + shift(192,0)
        self.assertEqual(detail["y"],80.0)
        self.mem.blocks[BASE+0xCDC2] = bytes(7)
        self.assertEqual(self.reader.read()[1]["combat"]["status"],"error")

    def test_memory_without_entities_cannot_train(self):
        state = GameState(in_game=True, player_x=192, player_y=400, raw={"backend": "memory"})
        with self.assertRaisesRegex(BackendError, "unconfigured"):
            encode_state(state)

    def test_terminal_encoding_can_skip_action_validation(self):
        state = GameState(in_game=False, raw={"backend": "memory"})
        self.assertTrue(np.isfinite(encode_state(state, validate=False)).all())
        with self.assertRaises(BackendError):
            encode_state(state)

    def test_trainer_failure_releases_keys_and_respects_output_dir(self):
        env = Mock()
        env.reset.return_value = StepResult(GameState(raw={"backend": "memory"}), 0, False)
        with tempfile.TemporaryDirectory() as tmp:
            trainer = Trainer(env, TrainConfig(episodes=1, out_dir=tmp))
            self.assertEqual(trainer.out_dir, Path(tmp))
            trainer._loop()
            self.assertIn("BackendError", trainer.error)
            env.backend.controller.release_all.assert_called_once()
            env.step.assert_not_called()


if __name__ == "__main__":
    unittest.main()
