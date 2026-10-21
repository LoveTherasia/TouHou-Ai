"""`attack` 奖励 profile 的回归测试.

这些断言把"配平意图"写成可执行的形式 —— 以后谁再动权重, 至少不会
悄悄把分数项调回噪声级、或把死亡惩罚降到比攻击收益还小。
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from touhou_rl_mcp.config import DEFAULT_CONFIG, Config
from touhou_rl_mcp.rl.reward import WEIGHT_DOCS, reward_profile
from touhou_rl_mcp.training.trainer import EpisodeRecord

#: mock 里对准 Boss 时每帧的加分 (backends/mock.py::_shoot)
MOCK_SCORE_PER_ALIGNED_FRAME = 4.0
#: "100 步持续输出" 作为一次攻击权衡的参照时长
ATTACK_WINDOW_STEPS = 100
#: 配平前的历史值: 一局约 -3306 的危险度惩罚 (比其它所有项加起来大 5 倍)
LEGACY_DANGER_SCALE = 4.0
LEGACY_DANGER_RADIUS = 50.0


class AttackProfileTest(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = reward_profile("attack")
        self.default = DEFAULT_CONFIG["reward"]

    def test_all_keys_are_known_weights(self) -> None:
        unknown = sorted(set(self.profile) - set(WEIGHT_DOCS))
        self.assertEqual(unknown, [], f"profile 里有 RewardModel 不认识的键: {unknown}")

    def test_profile_is_the_shipped_default(self) -> None:
        """profile 与默认值必须一致, 否则"照着 profile 跑"和"直接跑"结果不同。"""
        for key in ("score_scale", "danger_scale", "danger_radius",
                    "miss_penalty", "death_penalty"):
            self.assertEqual(self.profile[key], self.default[key], key)

    def test_score_is_no_longer_noise(self) -> None:
        # 旧默认 1e-5 时, 一局 2 万分只值 +0.2
        per_step = MOCK_SCORE_PER_ALIGNED_FRAME * self.profile["score_scale"]
        self.assertGreaterEqual(per_step, 0.1, "对准输出每步应至少 0.1 奖励")

    def test_danger_is_demoted(self) -> None:
        self.assertLessEqual(self.profile["danger_scale"], 1.0)
        self.assertLessEqual(self.profile["danger_radius"], 40.0)
        self.assertLess(self.profile["danger_scale"], LEGACY_DANGER_SCALE / 2)
        self.assertLess(self.profile["danger_radius"], LEGACY_DANGER_RADIUS)

    def test_survival_still_dominates(self) -> None:
        per_step = MOCK_SCORE_PER_ALIGNED_FRAME * self.profile["score_scale"]
        attack_gain = per_step * ATTACK_WINDOW_STEPS
        self.assertLess(
            attack_gain, abs(self.profile["miss_penalty"]),
            "100 步输出的收益必须小于一次掉残惩罚, 否则会用命换分",
        )
        # 死亡/掉残惩罚沿用默认值 (没有为了让模型更"敢"而放松)
        self.assertEqual(self.profile["miss_penalty"], self.default["miss_penalty"])
        self.assertEqual(self.profile["death_penalty"], self.default["death_penalty"])

    def test_profile_applies_to_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config({"reward": self.profile}, path=Path(tmp) / "config.json")
        self.assertEqual(cfg.get("reward.score_scale"), self.profile["score_scale"])
        self.assertEqual(cfg.get("reward.danger_scale"), self.profile["danger_scale"])


class EpisodeAttackMetricsTest(unittest.TestCase):
    def test_score_gain_and_align_ratio(self) -> None:
        rec = EpisodeRecord(index=1, steps=100, frames=100, return_=0.0,
                            max_score=1250, start_score=50, boss_damage=480,
                            aligned_steps=30, boss_steps=60)
        self.assertEqual(rec.score_gain, 1200)
        self.assertAlmostEqual(rec.align_ratio, 0.5)
        out = rec.to_dict()
        self.assertEqual(out["boss_damage"], 480)
        self.assertEqual(out["align_ratio"], 0.5)
        self.assertEqual(out["score_gain"], 1200)

    def test_no_boss_means_no_align_ratio(self) -> None:
        rec = EpisodeRecord(index=2, steps=10, frames=10, return_=0.0)
        self.assertIsNone(rec.align_ratio)
        self.assertIsNone(rec.score_gain)


if __name__ == "__main__":
    unittest.main()
