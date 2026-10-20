"""占用栅格 / 几何缺口观测的契约测试。

这些断言固定的是**语义**, 不是实现细节: 缺口必须被识别成最安全的一列、
场地外的列不能伪装成安全区、弹速必须参与外推。改动编码器时如果这些挂了,
说明策略看到的世界已经变了。
"""

from __future__ import annotations

import unittest

import numpy as np

from touhou_rl_mcp.training.occupancy import (
    GapObservation, GapSpec, _bullets, _column_centers, _tta_profile, lane_safety,
)
from touhou_rl_mcp.types import GameState


def _wall_state(px: float, py: float, *, hole_x: float, hole_width: float,
                wall_y: float, vy: float = 1.8) -> GameState:
    bullets = []
    x = 12.0
    while x <= 372.0:
        if abs(x - hole_x) >= hole_width / 2.0:
            bullets.append((x, wall_y))
        x += 9.0
    return GameState(
        frame=100, in_game=True, player_x=px, player_y=py, lives=2,
        bullets=bullets,
        raw={"bullet_velocity": [(0.0, vy)] * len(bullets)},
    )


class GapSpecTest(unittest.TestCase):
    def test_dim_is_stable(self) -> None:
        for spec in (GapSpec(), GapSpec(use_grid=False), GapSpec(use_geo=False)):
            enc = GapObservation(spec)
            state = _wall_state(192.0, 416.0, hole_x=256.0, hole_width=60.0, wall_y=300.0)
            obs = enc(state)
            self.assertEqual(obs.shape[0], enc.dim)
            self.assertEqual(enc.dim, GapObservation(spec).dim)

    def test_velocity_is_read_from_state(self) -> None:
        state = _wall_state(192.0, 416.0, hole_x=256.0, hole_width=60.0, wall_y=300.0)
        _pos, vel = _bullets(state)
        self.assertTrue(np.allclose(vel[:, 1], 1.8))


class TtaProfileTest(unittest.TestCase):
    spec = GapSpec()

    def test_lanes_cover_the_whole_playfield(self) -> None:
        # 横向必须覆盖整个场地: 一堵全屏宽的弹墙, 缺口可能在任何一侧
        centers = _column_centers(self.spec)
        self.assertLess(centers[0], 12.0)
        self.assertGreater(centers[-1], 372.0)

    def test_gap_is_the_safest_column(self) -> None:
        px, py = 192.0, 416.0
        state = _wall_state(px, py, hole_x=px + 64.0, hole_width=60.0, wall_y=py - 200.0)
        pos, vel = _bullets(state)
        profile = _tta_profile(py, pos, vel, self.spec)
        col = int(np.argmax(profile))
        # 车道是**场地绝对**坐标 (覆盖整个 384px), 与自机位置无关
        expected_x = (col + 0.5) * 384.0 / self.spec.tta_cols
        # 识别到的缺口必须落在真实缺口内 (允许一个格子的量化误差)
        self.assertLessEqual(abs(expected_x - (px + 64.0)), 30.0 + 1e-6)

    def test_offscreen_columns_are_not_safe(self) -> None:
        px, py = 192.0, 416.0
        state = _wall_state(px, py, hole_x=px + 64.0, hole_width=60.0, wall_y=py - 200.0)
        pos, vel = _bullets(state)
        profile = _tta_profile(py, pos, vel, self.spec)
        # 场地外的列 (自机被夹在 [8, 376]) 必须判为危险, 否则策略会学成贴边
        self.assertEqual(float(profile[0]), 0.0)
        self.assertEqual(float(profile[-1]), 0.0)

    def test_faster_bullets_arrive_earlier(self) -> None:
        px, py = 192.0, 416.0
        slow, _ = _bullets(_wall_state(px, py, hole_x=px + 64.0, hole_width=60.0,
                                       wall_y=py - 200.0, vy=0.5))
        fast, _ = _bullets(_wall_state(px, py, hole_x=px + 64.0, hole_width=60.0,
                                       wall_y=py - 200.0, vy=2.5))
        prof = _tta_profile(py, slow, np.zeros_like(slow), self.spec)
        prof_slow = prof.copy()
        prof_fast = _tta_profile(py, fast, np.full_like(fast, 2.5), self.spec)
        # 快弹更早到达 -> 同一列的安全值更低
        self.assertLess(float(prof_fast.mean()), float(prof_slow.mean()))

    def test_no_bullets_is_all_clear(self) -> None:
        profile = _tta_profile(416.0, np.zeros((0, 2)), np.zeros((0, 2)), self.spec)
        reachable = slice(1, self.spec.tta_cols - 1)
        self.assertTrue(np.all(profile[reachable] == 1.0))


class ObservationTest(unittest.TestCase):
    def test_obs_is_deterministic(self) -> None:
        enc = GapObservation()
        state = _wall_state(192.0, 416.0, hole_x=240.0, hole_width=60.0, wall_y=300.0)
        a = enc(state).copy()
        b = enc(state, prev_action=3).copy()
        self.assertFalse(np.allclose(a, b))      # 上一动作影响了输入
        self.assertTrue(np.allclose(enc(state), a))

    def test_prev_action_one_hot(self) -> None:
        spec = GapSpec(use_grid=False, use_geo=False, use_clearance=False)
        enc = GapObservation(spec)
        state = _wall_state(192.0, 416.0, hole_x=240.0, hole_width=60.0, wall_y=300.0)
        obs = enc(state, prev_action=5)
        self.assertEqual(enc.dim, 7 + 9)               # 7 维全局 + 9 维 one-hot
        self.assertEqual(float(obs[7 + 5]), 1.0)
        self.assertAlmostEqual(float(obs[7:].sum()), 1.0, places=6)

    def test_each_call_returns_independent_buffer(self) -> None:
        """回归: 曾经复用内部缓冲区, 于是轨迹里每条观测都指向最后一帧。

        BC 因为显式 copy 躲过了, RL 却一直在"用终局状态重复 N 次"更新 —— 这足以
        单独解释"一微调就崩"。契约: 不传 out 就必须拿到独立数组。
        """
        enc = GapObservation(GapSpec(use_grid=False))
        first = enc(_wall_state(60.0, 416.0, hole_x=60.0, hole_width=60.0, wall_y=300.0))
        snapshot = first.copy()
        second = enc(_wall_state(340.0, 416.0, hole_x=340.0, hole_width=60.0, wall_y=300.0))
        self.assertIsNot(first, second)
        self.assertTrue(np.allclose(first, snapshot), "第二次编码改写了第一次的结果")
        self.assertFalse(np.allclose(first, second))


class LaneSafetyTest(unittest.TestCase):
    """塑形用的 φ 必须取**自机所在车道**, 不是场地正中间那条。"""

    def test_uses_player_column_not_field_centre(self) -> None:
        # 缺口在正中间 (192), 自机站在最左边 —— 那里马上就要被墙盖住
        state = _wall_state(24.0, 416.0, hole_x=192.0, hole_width=46.0, wall_y=340.0)
        phi = lane_safety(state, GapSpec())
        middle = _tta_profile(416.0, *_bullets(state), GapSpec())[16]
        self.assertLess(phi, 0.2, "贴在墙上却报告很安全 = 取错了车道")
        self.assertGreater(middle, 0.9, "缺口车道本身应该是安全的")

    def test_standing_in_the_gap_is_safe(self) -> None:
        state = _wall_state(192.0, 416.0, hole_x=192.0, hole_width=46.0, wall_y=340.0)
        self.assertGreater(lane_safety(state, GapSpec()), 0.9)


if __name__ == "__main__":
    unittest.main()
