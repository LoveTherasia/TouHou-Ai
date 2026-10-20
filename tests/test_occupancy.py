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
from touhou_rl_mcp.training.features import ObservationEncoder
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


def _two_wall_state(*, hole_front: float, hole_back: float,
                    front_y: float = 320.0, back_y: float = 130.0,
                    width: float = 44.0, vy: float = 2.05) -> GameState:
    """两堵墙同时在屏 (前墙近、后墙远), 缺口位置可以不同。"""
    bullets: list[tuple[float, float]] = []
    for hole, y in ((hole_front, front_y), (hole_back, back_y)):
        x = 12.0
        while x <= 372.0:
            if abs(x - hole) >= width / 2.0:
                bullets.append((x, y))
            x += 9.0
    return GameState(
        frame=100, in_game=True, player_x=192.0, player_y=416.0, lives=2,
        bullets=bullets, raw={"bullet_velocity": [(0.0, vy)] * len(bullets)},
    )


class TtaLayersTest(unittest.TestCase):
    """第 2 层剖面 = "下一堵墙什么时候到"。

    间隔短、缺口横移大时 (v3), 自机必须在**上一堵墙还没落下**时就开始为下一堵墙
    挪位。第 1 层是多颗弹取 min, 后一堵墙被前一堵完全遮住 —— 于是"现在整片都危险,
    但下一堵墙在右边留了口子"这件事在第 1 层里完全不可见。
    """

    def test_second_layer_costs_exactly_one_profile(self) -> None:
        base = GapSpec(use_grid=False, use_geo=True, use_clearance=False)
        two = GapSpec(use_grid=False, use_geo=True, use_clearance=False, tta_layers=2)
        self.assertEqual(GapObservation(two).dim - GapObservation(base).dim,
                         base.tta_cols)

    def test_second_layer_reveals_the_next_wall_gap(self) -> None:
        # 前墙缺口在 90, 后墙缺口在 300: 站在 300 这一列, 现在马上被前墙盖住,
        # 但后墙在这里是开的 —— 这正是"该往右准备"的信号。
        state = _two_wall_state(hole_front=90.0, hole_back=300.0)
        spec = GapSpec(use_grid=False, use_geo=True, use_clearance=False, tta_layers=2)
        obs = GapObservation(spec)(state)
        # 观测布局: [第1层 32][第2层 32][4 维几何]...
        layer1, layer2 = obs[:spec.tta_cols], obs[spec.tta_cols:2 * spec.tta_cols]
        col_next_gap = int(np.argmin(np.abs(_column_centers(spec) - 294.0)))
        self.assertLess(float(layer1[col_next_gap]), 0.3, "前墙很近, 第 1 层应该是危险")
        self.assertGreater(float(layer2[col_next_gap]), 0.9, "第 2 层看不到下一堵墙的缺口")

    def test_both_layers_report_the_near_wall_first(self) -> None:
        # 被两堵墙都盖住的车道: 第 1 层看前墙 (近), 第 2 层看后墙 (远)
        state = _two_wall_state(hole_front=90.0, hole_back=300.0)
        spec = GapSpec(use_grid=False, use_geo=True, use_clearance=False, tta_layers=2)
        obs = GapObservation(spec)(state)
        layer1, layer2 = obs[:spec.tta_cols], obs[spec.tta_cols:2 * spec.tta_cols]
        col_walled = int(np.argmin(np.abs(_column_centers(spec) - 198.0)))
        self.assertLess(float(layer1[col_walled]), 0.3)
        self.assertGreater(float(layer2[col_walled]), float(layer1[col_walled]) + 0.2)

    def test_single_layer_matches_the_old_profile(self) -> None:
        # 默认层数必须与旧行为逐位一致, 否则历史策略的编码会静默改变
        state = _wall_state(192.0, 416.0, hole_x=256.0, hole_width=60.0, wall_y=300.0)
        spec = GapSpec(use_grid=False, use_geo=True, use_clearance=False)
        obs = GapObservation(spec)(state)
        pos, vel = _bullets(state)
        self.assertTrue(np.allclose(obs[:spec.tta_cols],
                                    _tta_profile(416.0, pos, vel, spec)))


class ActionAgeTest(unittest.TestCase):
    """动作年龄是策略可部署的承诺记忆: 维度 +1, 数值按 cap 归一化。"""

    def test_dim_and_value(self) -> None:
        spec = GapSpec(use_grid=False, use_geo=False, use_clearance=False,
                       use_action_age=True, action_age_cap=12)
        enc = GapObservation(spec)
        self.assertEqual(enc.dim, 7 + 9 + 1)  # globals + prev_action + age
        state = GameState(player_x=192.0, player_y=400.0)
        self.assertAlmostEqual(float(enc(state, prev_action=4, action_age=6)[-1]), 0.5)
        self.assertAlmostEqual(float(enc(state, prev_action=-1, action_age=0)[-1]), 0.0)
        self.assertAlmostEqual(float(enc(state, prev_action=4, action_age=99)[-1]), 1.0)

    def test_default_dim_is_unchanged(self) -> None:
        # 旧策略的 policy.json 没有 gap_action_age, 必须仍然按 70 维编码
        spec = GapSpec(use_grid=False, use_geo=True, use_clearance=True)
        self.assertEqual(GapObservation(spec).dim, 70)

    def test_encoder_restores_age_from_meta(self) -> None:
        old = ObservationEncoder.from_meta({"use_gap": True, "gap_grid": False})
        self.assertEqual(old.dim, 70)
        new = ObservationEncoder.from_meta(
            {"use_gap": True, "gap_grid": False, "gap_action_age": True,
             "gap_action_age_cap": 12})
        self.assertEqual(new.dim, 71)
        self.assertTrue(new.gap_action_age)


class DiagonalAimTest(unittest.TestCase):
    """斜着朝自机飞来的单颗弹, 必须标记自机所在车道, 而不是它"刚进带"的车道。"""

    def test_player_lane_is_threatened(self) -> None:
        state = GameState(
            frame=100, in_game=True, player_x=192.0, player_y=416.0,
            bullets=[(100.0, 100.0)],
            raw={"bullet_velocity": [(0.69, 2.38)]},  # 从 (100,100) 指向 (192,416)
        )
        spec = GapSpec(use_grid=False, use_geo=True, use_clearance=False)
        col = int(192.0 / (384.0 / spec.tta_cols))
        profile = _tta_profile(416.0, *_bullets(state), spec)
        # 修复前: 取"刚进入 ±26px 高度带"的 x≈184 -> 自机所在列显示安全 (1.0)
        self.assertLess(float(profile[col]), 0.9)


if __name__ == "__main__":
    unittest.main()
