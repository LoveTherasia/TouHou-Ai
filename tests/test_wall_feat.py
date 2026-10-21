"""墙特征 + checkpoint 零初始化扩维.

两条契约:

* **墙特征必须"看得见缺口"**: 143 维里只有最近 16 颗弹, 一堵 384px 宽的墙在观测里
  只是"身边几颗朝下飞的弹", 缺口位置不可见 —— 于是无论怎么监督, 策略学到的都是
  "别乱动" (实测 `wall_gate` 552~657 步 < 基线 814)。这 12 维就是补上这个缺口。
* **扩维必须"函数不变"**: 新特征追加在末尾、新列权重为 0, 所以 `widen_input` 之后
  对旧观测的输出必须**逐位相同**。否则"加特征"这一步本身就会破坏已有能力。
"""

from __future__ import annotations

import unittest

import numpy as np

from touhou_rl_mcp.training.features import (
    WALL_FEATURE_DIM,
    ObservationEncoder,
    encode_observation,
    feature_dim,
    wall_features,
)
from touhou_rl_mcp.training.policy import MLPPolicy
from touhou_rl_mcp.types import GameState

W, H = 384.0, 448.0


def state(px=192.0, py=400.0, bullets=(), vel=()):
    return GameState(
        frame=100, in_game=True, player_x=px, player_y=py, lives=3,
        bullets=list(bullets), enemy_positions=[],
        raw={"bullet_velocity": list(vel)} if vel else {},
    )


def wall_row(y=300.0, gap=(150.0, 246.0), speed=1.9, spacing=9.0):
    """整幅横墙, 只在 gap 处留口 (与 patterns.WallGate 同构)."""
    pos, vel = [], []
    x = 8.0
    while x < 380.0:
        if not (gap[0] < x < gap[1]):
            pos.append((x, y))
            vel.append((0.0, speed))
        x += spacing
    return pos, vel


class WallFeatureTest(unittest.TestCase):
    def test_no_bullets_is_all_zero(self) -> None:
        self.assertEqual(float(np.abs(wall_features(state())).max()), 0.0)

    def test_downward_sparse_bullets_are_not_a_wall(self) -> None:
        """只有两三颗下落的弹 -> 凑不满一排, 全 0 (不能让噪声喂给策略)."""
        st = state(bullets=[(100.0, 300.0), (200.0, 305.0)], vel=[(0.0, 1.9), (0.0, 1.9)])
        self.assertEqual(float(np.abs(wall_features(st)).max()), 0.0)

    def test_sealed_row_reports_high_coverage(self) -> None:
        pos, vel = wall_row(gap=(150.0, 246.0))
        feats = wall_features(state(bullets=pos, vel=vel))
        self.assertGreater(feats[0], 0.6, "整幅横墙的覆盖率必须高")
        # 缺口 96px = 2 个列带
        self.assertAlmostEqual(feats[2], 96.0 / W, places=6)

    def test_gap_offset_is_relative_to_the_player(self) -> None:
        """缺口在自机右边 -> 偏移为正; 左边 -> 为负. 这是"往哪边走"的唯一依据."""
        pos, vel = wall_row(gap=(260.0, 356.0))
        right = wall_features(state(px=192.0, bullets=pos, vel=vel))
        self.assertGreater(right[1], 0.05)
        pos, vel = wall_row(gap=(28.0, 124.0))
        left = wall_features(state(px=192.0, bullets=pos, vel=vel))
        self.assertLess(left[1], -0.05)

    def test_eta_shrinks_as_the_wall_approaches(self) -> None:
        pos_far, vel_far = wall_row(y=300.0)
        far = wall_features(state(bullets=pos_far, vel=vel_far))
        pos_near, vel_near = wall_row(y=390.0)
        near = wall_features(state(bullets=pos_near, vel=vel_near))
        self.assertGreater(far[3], near[3])
        self.assertLessEqual(near[3], 1.0)
        self.assertGreaterEqual(near[3], 0.0)

    def test_occupancy_marks_the_empty_bands(self) -> None:
        pos, vel = wall_row(gap=(150.0, 246.0))
        feats = wall_features(state(bullets=pos, vel=vel))
        occ = feats[4:]
        self.assertEqual(len(occ), 8)
        self.assertEqual(int(np.sum(occ)), 6, "8 个列带里应有 2 个被缺口腾空")
        self.assertEqual(list(occ[3:5]), [0.0, 0.0])

    def test_sideways_bullets_are_ignored(self) -> None:
        pos, vel = wall_row()
        vel = [(2.0, 1.9)] * len(vel)
        self.assertEqual(float(np.abs(wall_features(state(bullets=pos, vel=vel))).max()), 0.0)


class DimTest(unittest.TestCase):
    def test_feature_dim_grows_by_12(self) -> None:
        base = feature_dim(16, True, 8, True, True, True)
        wall = feature_dim(16, True, 8, True, True, True, True)
        self.assertEqual(wall - base, WALL_FEATURE_DIM)

    def test_encoder_dim_and_tail(self) -> None:
        enc = ObservationEncoder(k_bullets=16, bullet_velocity=True, k_enemies=8,
                                 use_invincible=True, use_boss=True, use_danger_grad=True,
                                 use_wall_feat=True)
        self.assertEqual(enc.dim, 155)
        pos, vel = wall_row(gap=(260.0, 356.0))
        obs = enc(state(bullets=pos, vel=vel))
        self.assertEqual(obs.shape, (155,))
        np.testing.assert_allclose(obs[-WALL_FEATURE_DIM:],
                                   wall_features(state(bullets=pos, vel=vel)), atol=1e-9)

    def test_encode_observation_honours_the_config(self) -> None:
        class Cfg:
            use_gap = False
            use_grid = False
            use_wall_feat = True
            k_bullets = 16
            bullet_velocity = True
            k_enemies = 8
            use_invincible = True
            use_boss = True
            use_danger_grad = True

        obs = encode_observation(state(), Cfg())
        self.assertEqual(obs.shape, (155,))
        self.assertEqual(float(np.abs(obs[-WALL_FEATURE_DIM:]).max()), 0.0)

    def test_wall_feat_is_rejected_by_grid_and_gap(self) -> None:
        class Cfg:
            use_gap = True
            use_grid = False
            use_wall_feat = True
            k_bullets = 16
            bullet_velocity = True
            k_enemies = 0
            use_invincible = True
            use_boss = True
            use_danger_grad = False
            gap_grid = True
            gap_geo = True
            gap_clearance = True
            gap_prev_action = True
            gap_action_age = False
            gap_action_age_cap = 12
            gap_layers = 1

        with self.assertRaises(ValueError):
            encode_observation(state(), Cfg())


class WidenInputTest(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = MLPPolicy(20, 9, hidden=(16, 16), seed=7)
        rng = np.random.default_rng(0)
        self.X = rng.standard_normal((64, 20))

    def test_function_is_unchanged_after_widening(self) -> None:
        before = self.policy.logits_batch(self.X)
        values_before = [self.policy.value(self.X[i]) for i in range(4)]
        self.policy.widen_input(32)
        self.assertEqual(self.policy.obs_dim, 32)
        X2 = np.concatenate([self.X, np.zeros((len(self.X), 12))], axis=1)
        np.testing.assert_allclose(self.policy.logits_batch(X2), before, atol=1e-12)
        # 价值头也必须一致
        np.testing.assert_allclose([self.policy.value(X2[i]) for i in range(4)],
                                   values_before, atol=1e-12)

    def test_new_columns_start_at_zero(self) -> None:
        self.policy.widen_input(32)
        self.assertTrue(np.all(self.policy.W[0][:, 20:] == 0.0))
        self.assertTrue(np.all(self.policy._m[0][:, 20:] == 0.0))
        self.assertTrue(np.all(self.policy._v[0][:, 20:] == 0.0))

    def test_reference_shape_kept(self) -> None:
        shapes = [p.shape for p in self.policy.snapshot_params()]
        self.policy.widen_input(32)
        after = [p.shape for p in self.policy.snapshot_params()]
        self.assertEqual(after[0], (16, 32))
        self.assertEqual(after[1:], shapes[1:], "除输入层外参数形状不该变")

    def test_shrink_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.policy.widen_input(10)

    def test_gradients_still_flow_after_widening(self) -> None:
        """扩维后训练必须真的能动: 新列要能获得非零梯度 (否则等于白加)."""
        self.policy.widen_input(32)
        X2 = np.concatenate([self.X, np.ones((len(self.X), 12)) * 0.5], axis=1)
        W_before = self.policy.W[0].copy()
        for _ in range(3):
            self.policy.imitate_batch(X2, np.full(len(X2), 3), weights=np.ones(len(X2)))
        self.assertGreater(float(np.abs(self.policy.W[0][:, 20:] - W_before[:, 20:]).max()),
                           0.0)


if __name__ == "__main__":
    unittest.main()
