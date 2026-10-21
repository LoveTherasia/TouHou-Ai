"""ViabilityPlanner: 时间展开的可行域规划器.

可执行契约（都是踩过的坑，不许再退回去）：

* **几何精确**: ``cell = speed*dt/2`` 时，轴向 = 2 格、斜向 = (1,1) 格，与真实位移
  完全相等 —— 第一版把 (d_col, d_row) 写反了，导致"想往左却按了上"；
* **命中必须是硬代价**: 格子边长 6.6px、判定半径 5px 时，用 ``(1-d/r)²`` 的软场
  会让"正被弹打中"只值 0.26，规划器于是原地等死；
* **提前站位**: 一堵 100px 外、只留左侧缺口的墙，必须现在就开始往左走。
"""

from __future__ import annotations

import unittest

import numpy as np

from touhou_rl_mcp.training.trainer import DIRECTIONS as ACTION_NAMES
from touhou_rl_mcp.training.viability import DIRECTIONS, NAMES, ViabilityPlanner
from touhou_rl_mcp.types import GameState

W, H = 384.0, 448.0


def state(px=192.0, py=400.0, bullets=(), vel=()):
    return GameState(
        frame=100, in_game=True, player_x=px, player_y=py, lives=3,
        bullets=list(bullets), enemy_positions=[],
        raw={"bullet_velocity": list(vel)} if vel else {},
    )


def wall(y=300.0, gap=(60.0, 110.0), speed=1.9, spacing=9.0):
    """横跨全场、只在 gap 处留口的墙 (与 backends/patterns.py::WallGate 同构)."""
    pos, vel = [], []
    x = 8.0
    while x < 380.0:
        if not (gap[0] < x < gap[1]):
            pos.append((x, y))
            vel.append((0.0, speed))
        x += spacing
    return pos, vel


class GeometryTest(unittest.TestCase):
    def test_names_match_action_table(self) -> None:
        self.assertEqual(list(NAMES), list(ACTION_NAMES))

    def test_moves_are_exact_or_pessimistic(self) -> None:
        """轴向必须精确; 斜向只能**偏慢** (整数格放不下 1.41 格), 绝不允许偏快."""
        p = ViabilityPlanner(dt=2)
        self.assertAlmostEqual(p.cell, 4.4)  # speed*dt/2
        for (ux, uy), (dc, dr), name in zip(DIRECTIONS, p._moves, NAMES):
            dx = dc * p.cell / p.dt  # 真实 px/帧
            dy = dr * p.cell / p.dt
            norm = (ux * ux + uy * uy) ** 0.5
            if norm == 0:
                self.assertEqual((dx, dy), (0.0, 0.0), name)
                continue
            # 方向必须一致
            if ux:
                self.assertEqual(np.sign(dx), np.sign(ux), f"{name}: 横向方向反了")
            if uy:
                self.assertEqual(np.sign(dy), np.sign(uy), f"{name}: 纵向方向反了")
            speed = float(np.hypot(dx, dy))
            if norm == 1.0:
                self.assertAlmostEqual(speed, p.speed, places=6, msg=f"{name}: 轴向必须精确")
            else:
                self.assertLessEqual(speed, p.speed + 1e-9, f"{name}: 斜向不能比真实更快")
                self.assertGreaterEqual(speed, 0.6 * p.speed, f"{name}: 斜向不能慢得离谱")

    def test_grid_covers_playfield(self) -> None:
        p = ViabilityPlanner(dt=2)
        self.assertGreaterEqual(p.nx * p.cell, W - p.cell)
        self.assertGreaterEqual(p.ny * p.cell, H - p.cell)


class DangerFieldTest(unittest.TestCase):
    def test_direct_hit_costs_full(self) -> None:
        """一颗弹正压在自机格上: 该层的代价必须是 1.0 (不是 0.26)."""
        p = ViabilityPlanner(dt=4, horizon=40)
        col, row = p._to_cell(192.0, 400.0)
        bx = (col + 0.5) * p.cell
        by = (row + 0.5) * p.cell
        hard, _soft = p._risk_layers(np.array([[bx, by]]), np.zeros((1, 2)), None)
        self.assertTrue(bool(hard[0, row, col]))

    def test_quantisation_is_absorbed(self) -> None:
        """弹在格子内部偏移 2px 也必须算命中 (量化误差不能变成"安全")."""
        p = ViabilityPlanner(dt=4, horizon=40)
        col, row = p._to_cell(192.0, 400.0)
        bx = (col + 0.5) * p.cell + 2.0
        by = (row + 0.5) * p.cell - 2.0
        hard, _soft = p._risk_layers(np.array([[bx, by]]), np.zeros((1, 2)), None)
        self.assertTrue(bool(hard[0, row, col]))

    def test_empty_field_is_free(self) -> None:
        p = ViabilityPlanner(dt=4, horizon=40)
        hard, soft = p._risk_layers(np.zeros((0, 2)), np.zeros((0, 2)), None)
        self.assertFalse(bool(hard.any()))
        self.assertFalse(bool(soft.any()))


class DecisionTest(unittest.TestCase):
    def test_no_bullets_is_valid_action(self) -> None:
        p = ViabilityPlanner()
        action, info = p.choose(state())
        self.assertEqual(action, "shoot")
        self.assertEqual(info["planner"], "viability")

    def test_dodges_a_bullet_about_to_hit(self) -> None:
        """正上方 20px 处的弹 1.9px/帧 落下: 必须让开, 不能原地不动."""
        p = ViabilityPlanner(horizon=120, dt=4)
        st = state(py=400.0, bullets=[(192.0, 380.0)], vel=[(0.0, 1.9)])
        action, _ = p.choose(st)
        self.assertNotEqual(action, "shoot")

    def test_never_moves_into_the_descending_wall(self) -> None:
        """墙在正上方 30px 时, "up" 是往墙上撞 —— 死亡代价必须让它变成最差选项.

        这条来自一个真实 bug: 代价原本是"危险层数之和", 于是**早死比晚死便宜**,
        规划器真的会挑着往墙上撞。改成"死亡是一次性终止代价 + 越晚死越便宜"后才对。
        """
        p = ViabilityPlanner()
        pos, vel = wall(y=370.0)
        st = state(px=192.0, py=400.0, bullets=pos, vel=vel)
        info = p.plan(st)
        self.assertNotEqual(NAMES[info.direction], "up", "不能迎着下压的墙往上")
        self.assertNotEqual(NAMES[info.direction], "up_left")
        self.assertNotEqual(NAMES[info.direction], "up_right")

    def test_soft_shadow_does_not_declare_everything_dead(self) -> None:
        """墙在 100px 外时, 九个方向都必须是有限代价 (影子只能加压力, 不能判死)."""
        p = ViabilityPlanner()
        pos, vel = wall(y=300.0)
        st = state(px=192.0, py=400.0, bullets=pos, vel=vel)
        info = p.plan(st)
        self.assertLess(float(np.max(info.cost_by_dir)), 100.0)

    def test_does_not_leave_when_the_gap_is_under_it(self) -> None:
        """缺口正好在正下方时不该乱跑 (否则就是"贴角退化"的另一面)."""
        p = ViabilityPlanner(horizon=200, dt=4)
        pos, vel = wall(y=300.0, gap=(170.0, 218.0))
        st = state(px=192.0, py=400.0, bullets=pos, vel=vel)
        action, _ = p.choose(st)
        head = action.split("+")[0]
        self.assertFalse(head.endswith("left") or head.endswith("right"),
                         f"已经在缺口里, 不该横向跑: {action}")


if __name__ == "__main__":
    unittest.main()
