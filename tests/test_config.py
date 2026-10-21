"""配置加载 / 补丁校验的回归测试.

重点保护两件事:

1. ``patch`` 必须在**写之前**递归校验键名 —— 否则拼错的嵌套键会先落盘,
   下一帧才在 ``RewardModel.update_weights`` 抛错;
2. 仓库里发布的 ``config.json`` 与 ``config.DEFAULT_CONFIG`` 不许漂移
   (曾经 ``search_roots`` 被写成双重转义的 ``F:\\\\``)。
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from touhou_rl_mcp.config import DEFAULT_CONFIG, Config, _unknown_paths
from touhou_rl_mcp.errors import ConfigError
from touhou_rl_mcp.rl.reward import WEIGHT_DOCS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SHIPPED_CONFIG = PROJECT_ROOT / "config.json"


class TestPatchValidation(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "config.json"
        self.config = Config({}, path=self.path)

    # -- 拒绝 ---------------------------------------------------------
    def test_unknown_top_level_key(self) -> None:
        with self.assertRaises(ConfigError) as ctx:
            self.config.patch({"nope": 1})
        self.assertIn("nope", str(ctx.exception))

    def test_unknown_nested_reward_key(self) -> None:
        # 这是真实踩过的形态: 拼错的 key 会被 config.patch 接受并落盘
        with self.assertRaises(ConfigError) as ctx:
            self.config.patch({"reward": {"miss_penelty": -5.0}})
        self.assertIn("reward.miss_penelty", str(ctx.exception))

    def test_unknown_nested_visualize_key(self) -> None:
        with self.assertRaises(ConfigError) as ctx:
            self.config.patch({"visualize": {"trail_lenght": 10}})
        self.assertIn("visualize.trail_lenght", str(ctx.exception))

    def test_section_must_be_object(self) -> None:
        with self.assertRaises(ConfigError):
            self.config.patch({"reward": None})
        with self.assertRaises(ConfigError):
            self.config.patch({"reward": 3})

    def test_failed_patch_does_not_change_state(self) -> None:
        before = self.config.get("reward.miss_penalty")
        with self.assertRaises(ConfigError):
            self.config.patch({"reward": {"miss_penalty": -1.0, "bogus": 0}})
        self.assertEqual(self.config.get("reward.miss_penalty"), before)

    # -- 接受 ---------------------------------------------------------
    def test_nested_reward_patch(self) -> None:
        self.config.patch({"reward": {"miss_penalty": -42.0}})
        self.assertEqual(self.config.get("reward.miss_penalty"), -42.0)
        # 同段其它键不受影响 (深合并而不是整段替换)
        self.assertEqual(
            self.config.get("reward.death_penalty"), DEFAULT_CONFIG["reward"]["death_penalty"]
        )

    def test_freeform_keymap_allowed(self) -> None:
        self.config.patch({"control": {"keymap": {"bomb": [44], "custom_vk": [1, 2]}}})
        self.assertEqual(self.config.get("control.keymap")["bomb"], [44])

    def test_cross_section_patch(self) -> None:
        out = self.config.patch(
            {"backend": "memory", "step_frames": 1, "vision": {"enabled": True}}
        )
        self.assertEqual(out["backend"], "memory")
        self.assertEqual(out["step_frames"], 1)
        self.assertTrue(out["vision"]["enabled"])


class TestShippedConfig(unittest.TestCase):
    def setUp(self) -> None:
        self.raw = json.loads(SHIPPED_CONFIG.read_text(encoding="utf-8"))

    def test_top_level_keys_are_known(self) -> None:
        unknown = sorted(set(self.raw) - set(DEFAULT_CONFIG))
        self.assertEqual(unknown, [], f"config.json 有未知顶层键: {unknown}")

    def test_nested_keys_are_known(self) -> None:
        self.assertEqual(_unknown_paths(self.raw, DEFAULT_CONFIG), [])

    def test_search_roots_are_single_backslash(self) -> None:
        # 曾经的 bug: "F:\\\\" 解析成 F:\\ (两个反斜杠)
        for root in self.raw["search_roots"]:
            self.assertTrue(root.endswith("\\"), root)
            self.assertFalse(root.endswith("\\\\"), f"多余的转义: {root!r}")

    def test_reward_keys_cover_weight_docs(self) -> None:
        # WEIGHT_DOCS 是奖励权重的权威清单, config.json 必须一个不少
        missing = sorted(set(WEIGHT_DOCS) - set(self.raw["reward"]))
        self.assertEqual(missing, [], f"config.json 的 reward 缺: {missing}")

    def test_load_roundtrip(self) -> None:
        config = Config.load(SHIPPED_CONFIG)
        self.assertIn("vision", config.as_dict())
        self.assertFalse(config.get("vision.enabled"))


class TestVisionDefaults(unittest.TestCase):
    def test_vision_section_defaults(self) -> None:
        section = DEFAULT_CONFIG["vision"]
        self.assertFalse(section["enabled"], "视觉兜底默认必须关闭 (每次读数多一次截图)")
        for key in ("bright_thr", "sat_thr", "min_area", "max_area", "max_dim"):
            self.assertIsInstance(section[key], int, key)


if __name__ == "__main__":
    unittest.main()
