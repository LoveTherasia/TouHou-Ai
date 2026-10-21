"""分层切换: 墙/收网判别 + 只在需要的步产生标签.

可执行契约（每条都对应一个实测过的取舍）：

* **阈值要能把真墙和"环的平顶"分开**: 覆盖率 0.5 时 ``dense_ring`` 有 40% 的步被
  误判成墙, 于是把 90% 的对准率换成了规划器的 79%; 实测 ``wall_gate`` 覆盖 0.94、
  其余弹型 p95 只有 0.62, 所以卡在 0.78;
* **非墙段必须弃权**: 若把"策略自己的动作"当标签, 交叉熵会变成"让它对自己更自信",
  压掉熵并把坏习惯固化;
* **收网要认"向内收缩"**: 同样的一圈弹, 速度朝外就是 ``dense_ring`` (策略能处理),
  朝内才是 ``closing_ring`` (站桩必死)。
"""

from __future__ import annotations

import unittest

import numpy as np

from touhou_rl_mcp.training.hierarchical import (
    HierarchicalController,
    HierarchicalTeacher,
    WallDetector,
    bullet_arrays,
)
from touhou_rl_mcp.training.viability import NAMES, ViabilityPlanner
from touhou_rl_mcp.types import GameState

W, H = 384.0, 448.0


def state(px=192.0, py=400.0, bullets=(), vel=()):
    return GameState(
        frame=100, in_game=True, player_x=px, player_y=py, lives=3,
        bullets=list(bullets), enemy_positions=[],
        raw={"bullet_velocity": list(vel)} if vel else {},
    )


def curtain_row(y=300.0, gap=(60.0, 110.0), speed=1.9, spacing=9.0):
    """横跨全场、只在 gap 处留口的墙 (与 patterns.WallGate 同构)."""
    pos, vel = [], []
    x = 8.0
    while x < 380.0:
        if not (gap[0] < x < gap[1]):
            pos.append((x, y))
            vel.append((0.0, speed))
        x += spacing
    return pos, vel


def ring(radius=170.0, count=24, inward=True, gap_slots=2, center=(192.0, 224.0)):
    """围着圆心的一圈弹; inward=True 时速度指向圆心 (closing_ring).

    圆心放在场地中央而不是自机默认位置 (y=416): 自机贴着底边时, 圆环的下半圈
    会被场地裁掉, 只剩 4 个扇区 —— 那是**几何**问题, 不该混进单位测试里。
    """
    cx, cy = center
    pos, vel = [], []
    for i in range(count):
        if (i % count) < gap_slots:
            continue
        a = 2.0 * np.pi * i / count
        x = cx + np.cos(a) * radius
        y = cy + np.sin(a) * radius
        if not (8.0 <= x <= 376.0 and 32.0 <= y <= 440.0):
            continue
        sign = -1.0 if inward else 1.0
        pos.append((x, y))
        vel.append((sign * np.cos(a), sign * np.sin(a)))
    return pos, vel


class CurtainTest(unittest.TestCase):
    def setUp(self) -> None:
        self.detector = WallDetector()

    def test_sealed_row_is_a_wall(self) -> None:
        pos, vel = curtain_row()
        m = self.detector.detect(state(bullets=pos, vel=vel))
        self.assertEqual(m.reason, "curtain")
        self.assertGreater(m.covered, 0.8)
        self.assertGreater(m.row_count, 20)
        self.assertGreater(m.eta, 0.0)

    def test_wide_gate_is_not_a_wall(self) -> None:
        """缺口超过 4 格 (96px) 就不算"封死", 该继续交给策略.

        ``wall_gate`` 的缺口是 48px (2 格), 而 100px 的缺口在 5px 判定下已经能
        "挤过去", 交给规划器反而丢掉 90% 的对准率。
        """
        pos, vel = curtain_row(gap=(80.0, 220.0))   # 140px -> 6 格空洞
        m = self.detector.detect(state(bullets=pos, vel=vel))
        self.assertEqual(m.reason, "")
        # 边界: 96px (4 格) 仍然算墙
        pos, vel = curtain_row(gap=(100.0, 196.0))
        self.assertEqual(self.detector.detect(state(bullets=pos, vel=vel)).reason, "curtain")

    def test_sparse_row_is_not_a_wall(self) -> None:
        pos, vel = curtain_row(spacing=40.0)
        m = self.detector.detect(state(bullets=pos, vel=vel))
        self.assertEqual(m.reason, "")

    def test_sideways_row_is_not_a_wall(self) -> None:
        pos, vel = curtain_row()
        vel = [(2.0, 1.9)] * len(vel)   # 斜飞 = 不是下压的墙
        m = self.detector.detect(state(bullets=pos, vel=vel))
        self.assertEqual(m.reason, "")

    def test_row_below_the_player_is_ignored(self) -> None:
        pos, vel = curtain_row(y=430.0)
        m = self.detector.detect(state(py=400.0, bullets=pos, vel=vel))
        self.assertEqual(m.reason, "")

    def test_far_row_is_ignored(self) -> None:
        """还没进入规划视野 (170 帧) 的墙不该触发 —— 规划器也看不见它."""
        pos, vel = curtain_row(y=40.0, speed=1.0)
        m = self.detector.detect(state(py=430.0, bullets=pos, vel=vel))
        self.assertEqual(m.reason, "")


class EncircleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.detector = WallDetector()

    def test_inward_ring_is_encircle(self) -> None:
        pos, vel = ring(inward=True)
        m = self.detector.detect(state(px=192.0, py=224.0, bullets=pos, vel=vel))
        self.assertEqual(m.reason, "encircle")
        self.assertGreaterEqual(m.ring_bins, 7)

    def test_outward_ring_is_not_encircle(self) -> None:
        pos, vel = ring(inward=False)
        m = self.detector.detect(state(px=192.0, py=224.0, bullets=pos, vel=vel))
        self.assertEqual(m.reason, "")

    def test_far_ring_is_not_encircle(self) -> None:
        pos, vel = ring(radius=300.0, inward=True)
        m = self.detector.detect(state(px=192.0, py=224.0, bullets=pos, vel=vel))
        self.assertEqual(m.reason, "")


class TeacherAbstainTest(unittest.TestCase):
    def test_abstains_when_no_wall(self) -> None:
        teacher = HierarchicalTeacher()
        action, info = teacher.choose(state(bullets=[(100.0, 100.0)], vel=[(0.0, 1.0)]), 0, "right+shoot")
        self.assertEqual(action, "", "非墙段必须弃权 (空动作), 否则会把策略固化成'自信'")
        self.assertEqual(info["mode"], "policy")

    def test_labels_wall_steps(self) -> None:
        teacher = HierarchicalTeacher()
        pos, vel = curtain_row(y=360.0)
        action, info = teacher.choose(state(bullets=pos, vel=vel), 0, "up+shoot")
        self.assertTrue(action, "墙段必须给标签")
        self.assertEqual(info["mode"], "planner")
        self.assertIn(action.split("+")[0], set(NAMES) | {""})
        self.assertTrue(action.endswith("shoot"))

    def test_replan_every_holds_the_direction(self) -> None:
        """承诺期: 相邻 steps 复用上一方向, 规划器只被调用一次."""
        teacher = HierarchicalTeacher(replan_every=3)
        pos, vel = curtain_row(y=330.0)
        st = state(bullets=pos, vel=vel)
        acts = [teacher.choose(st, 0, "")[0] for _ in range(3)]
        self.assertEqual(len(set(acts)), 1)
        self.assertEqual(teacher.stats["replans"], 1)
        teacher.choose(st, 0, "")
        self.assertEqual(teacher.stats["replans"], 2)

    def test_counts_by_reason(self) -> None:
        teacher = HierarchicalTeacher()
        pos, vel = curtain_row(y=330.0)
        teacher.choose(state(bullets=pos, vel=vel), 0, "")
        pos, vel = ring(inward=True)
        teacher.choose(state(px=192.0, py=224.0, bullets=pos, vel=vel), 0, "")
        self.assertEqual(teacher.stats["curtain"], 1)
        self.assertEqual(teacher.stats["encircle"], 1)


class _StubPolicy:
    """永远返回同一个动作下标, 用来区分"走了策略还是规划器"."""

    def __init__(self, index: int = 4) -> None:
        self.index = index

    def greedy(self, obs) -> int:  # noqa: ANN001
        return self.index


class ControllerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.table = ["+".join(p for p in (n, "shoot") if p) for n in NAMES]

    def test_uses_policy_outside_walls(self) -> None:
        ctrl = HierarchicalController(_StubPolicy(4), lambda s: np.zeros(3), self.table)
        idx = ctrl.direction(state(bullets=[(100.0, 100.0)], vel=[(0.0, 1.0)]))
        self.assertEqual(idx, 4)
        self.assertEqual(ctrl.last_source, "policy")

    def test_switches_inside_walls(self) -> None:
        ctrl = HierarchicalController(_StubPolicy(4), lambda s: np.zeros(3), self.table)
        pos, vel = curtain_row(y=360.0)
        idx = ctrl.direction(state(bullets=pos, vel=vel))
        self.assertEqual(ctrl.last_source, "planner")
        self.assertNotEqual(idx, 4, "墙上不该继续按策略的'往右'")
        self.assertEqual(ctrl.last_metrics.reason, "curtain")

    def test_commit_window_reuses_direction(self) -> None:
        ctrl = HierarchicalController(_StubPolicy(4), lambda s: np.zeros(3), self.table,
                                     replan_every=5)
        pos, vel = curtain_row(y=360.0)
        st = state(bullets=pos, vel=vel)
        first = ctrl.direction(st)
        self.assertEqual({ctrl.direction(st) for _ in range(4)}, {first})

    def test_act_returns_action_string(self) -> None:
        ctrl = HierarchicalController(_StubPolicy(0), lambda s: np.zeros(3), self.table)
        self.assertEqual(ctrl.act(state()), "shoot")


class ArraysTest(unittest.TestCase):
    def test_missing_velocity_falls_back_to_zero(self) -> None:
        s = state(bullets=[(10.0, 20.0)])
        pos, vel = bullet_arrays(s)
        self.assertEqual(pos.shape, (1, 2))
        self.assertEqual(vel.tolist(), [[0.0, 0.0]])
        # 速度缺失 = 静止弹 -> 绝不能被当成下压的墙
        self.assertEqual(WallDetector().detect(s).reason, "")

    def test_empty_field(self) -> None:
        self.assertEqual(WallDetector().detect(state()).reason, "")


class PlannerIntegrationTest(unittest.TestCase):
    def test_teacher_direction_matches_planner(self) -> None:
        """分层老师的方向必须就是规划器的方向 (不允许有第二套逻辑)."""
        pos, vel = curtain_row(y=360.0)
        st = state(bullets=pos, vel=vel)
        teacher = HierarchicalTeacher(planner=ViabilityPlanner())
        action, _ = teacher.choose(st, 0, "")
        expect = NAMES[ViabilityPlanner(aim_weight=0.15).plan(st).direction]
        self.assertEqual(action, f"{expect}+shoot" if expect else "shoot")


if __name__ == "__main__":
    unittest.main()
