"""缺口标签的口径契约测试.

这些断言固定的是**标签语义**, 不是实现细节:

* 标签必须指向缺口方向 (而不是某个固定方向);
* 四种口径必须真的不同 —— 早期 ``now`` 在算 ``reach`` 之前就 return 了, 于是两者
  输出 100% 相同, 这个"假对照"骗了整整一轮实验;
* ``slack`` 里"永远没有第二次威胁"必须算成**最好**情况, 而不是 dwell=0 的最差情况。
"""

from __future__ import annotations

import unittest

import numpy as np

from touhou_rl_mcp.training.gap_label import (
    LABEL_MODES, MACRO_ACTIONS, SECOND_GAP_FRAMES, TEACHER_MODES,
    GapCommitment, gap_label, reach_label, reach_target, slack_label, tta_frames,
    update_action_age,
)
from touhou_rl_mcp.training.gap_teacher import GapTeacher
from touhou_rl_mcp.training.occupancy import GapSpec, _arrival_times
from touhou_rl_mcp.training.trainer import build_action_table
from touhou_rl_mcp.types import GameState, PLAYFIELD_WIDTH

RIGHT, LEFT = 4, 3


def wall(y: float, *, hole_x: float, hole_width: float = 46.0,
         vy: float = 1.8, spacing: float = 9.0) -> list[tuple[float, float]]:
    bullets: list[tuple[float, float]] = []
    x = 12.0
    while x <= 372.0:
        if abs(x - hole_x) >= hole_width / 2.0:
            bullets.append((x, y))
        x += spacing
    return bullets


def state_at(px: float, bullets: list[tuple[float, float]], *,
             py: float = 416.0, vy: float = 1.8) -> GameState:
    return GameState(
        frame=100, in_game=True, player_x=px, player_y=py, lives=2,
        bullets=bullets,
        raw={"bullet_velocity": [(0.0, vy)] * len(bullets)},
    )


class ActionSpaceTest(unittest.TestCase):
    def test_macro_actions_match_trainer_table(self) -> None:
        # 标签输出的是下标, 训练器按**同名方向**映射; 两套动作表必须逐项相同
        self.assertEqual(list(MACRO_ACTIONS), build_action_table(use_focus=False))


class LabelDirectionTest(unittest.TestCase):
    """缺口在右就往右, 在左就往左 —— 而且换一个相位要对得上。"""

    def test_points_at_the_gap(self) -> None:
        cases = (
            (192.0, 256.0, RIGHT),
            (192.0, 60.0, LEFT),
            (300.0, 40.0, LEFT),
            (20.0, 360.0, RIGHT),
        )
        for px, hole, expected in cases:
            state = state_at(px, wall(300.0, hole_x=hole))
            for mode in ("now", "reach", "slack"):
                with self.subTest(px=px, hole=hole, mode=mode):
                    self.assertEqual(gap_label(state, GapSpec(), mode), expected)

    def test_standing_in_the_gap_means_stay(self) -> None:
        state = state_at(256.0, wall(300.0, hole_x=256.0))
        for mode in ("now", "reach", "slack"):
            with self.subTest(mode=mode):
                self.assertEqual(gap_label(state, GapSpec(), mode), 0)

    def test_every_mode_returns_a_legal_action(self) -> None:
        rng = np.random.default_rng(0)
        for _ in range(20):
            px = float(rng.uniform(8, PLAYFIELD_WIDTH - 8))
            hole = float(rng.uniform(20, PLAYFIELD_WIDTH - 20))
            state = state_at(px, wall(float(rng.uniform(180, 380)), hole_x=hole))
            for mode in LABEL_MODES:
                action = gap_label(state, GapSpec(), mode)
                self.assertGreaterEqual(action, 0)
                self.assertLess(action, len(MACRO_ACTIONS))

    def test_unknown_mode_is_rejected(self) -> None:
        state = state_at(192.0, wall(300.0, hole_x=192.0))
        with self.assertRaises(KeyError):
            gap_label(state, GapSpec(), "vibes")


class NowIsNotReachTest(unittest.TestCase):
    """``now`` / ``reach`` 必须是两个真的能分开的对照。

    v2 上实测两者输出 100% 相同, 根因是 ``now`` 分支在 ``_reach_target`` 之后才判,
    永远走不到。这里构造一个"眼下最空的车道很远、但赶到那里已经来不及"的局面:
    ``now`` 会指过去, ``reach`` 应该选择留下 (目标就是脚下这条车道)。
    """

    def test_degenerate_case_separates_the_two_modes(self) -> None:
        # 墙 1 贴着自机 (56px, 31 帧后到), 缺口在 x=30 (160px 外, 36 帧才走到)
        far = wall(380.0, hole_x=30.0)
        # 墙 2 更远但盖住 x=30 附近 —— 保证"最空"和"能赶到"是两件事
        near = wall(360.0, hole_x=330.0)
        state = state_at(192.0, far + near)

        tta = tta_frames(state, GapSpec())
        emptiest = int(np.argmax(tta))
        target = reach_target(state, GapSpec())
        self.assertIsNotNone(target)
        self.assertLess(abs(target - 192.0), 60.0,
                        f"reach 不该指向 {target:.0f}; 最空车道 {emptiest}")
        # 最空的那条车道离自机至少 100px
        self.assertGreater(abs(emptiest * PLAYFIELD_WIDTH / GapSpec().tta_cols - 192.0), 60.0)


class SlackDwellTest(unittest.TestCase):
    """``slack`` 的 dwell 项: 没有第二堵墙 = 最好, 不是最差。"""

    def test_single_wall_dwell_is_not_zero(self) -> None:
        spec = GapSpec()
        hole = 256.0
        state = state_at(192.0, wall(300.0, hole_x=hole))
        from touhou_rl_mcp.training.occupancy import _bullets

        pos, vel = _bullets(state)
        t1, t2 = _arrival_times(state.player_y, pos, vel, spec)
        never = float(spec.tta_horizon + spec.tta_step)
        hole_col = int(hole / (PLAYFIELD_WIDTH / spec.tta_cols))
        self.assertGreaterEqual(t1[hole_col], never - 1e-9, "缺口车道不该被判成有威胁")
        self.assertGreaterEqual(t2[hole_col], never - 1e-9)
        # 若把 (t2 - t1) 直接当 dwell, 这一列会变成 slack=0 (最差) —— 那正是修掉的 bug
        self.assertEqual(slack_label(state, spec), RIGHT)

    def test_second_arrival_is_later_than_first(self) -> None:
        spec = GapSpec()
        # 两堵墙叠在一起, 同一车道有两次到达
        bullets = wall(300.0, hole_x=256.0) + wall(200.0, hole_x=256.0)
        state = state_at(192.0, bullets)
        from touhou_rl_mcp.training.occupancy import _bullets

        pos, vel = _bullets(state)
        t1, t2 = _arrival_times(state.player_y, pos, vel, spec)
        # 排除场地外那两条 (t1=0 的"永远到不了") 车道
        covered = np.flatnonzero((t1 > 0) & (t1 < float(spec.tta_horizon)))
        self.assertTrue(covered.size)
        col = int(covered[0])
        self.assertGreaterEqual(t2[col] - t1[col], SECOND_GAP_FRAMES - 1e-9)


class DiagonalAimLabelTest(unittest.TestCase):
    def test_diagonal_aimed_bullet_is_not_ignored(self) -> None:
        # 从 (100,100) 斜着飞向 (192,416): 自机所在列必须被标成受威胁
        state = GameState(
            frame=100, in_game=True, player_x=192.0, player_y=416.0,
            bullets=[(100.0, 100.0)],
            raw={"bullet_velocity": [(0.69, 2.38)]},
        )
        spec = GapSpec()
        tta = tta_frames(state, spec)
        col = int(192.0 / (PLAYFIELD_WIDTH / spec.tta_cols))
        self.assertLess(float(tta[col]), float(spec.tta_horizon))
        self.assertNotEqual(reach_label(state, spec), 0)


class CommitmentTest(unittest.TestCase):
    """目标承诺 / 滞回的语义: 承诺期内不换目标, 不安全时立刻换。"""

    class FakeCommitment(GapCommitment):
        def __init__(self, slack, *, px=150.0, **kwargs):
            super().__init__(GapSpec(), **kwargs)
            self._centers = np.array([0.0, 100.0, 200.0, 300.0])
            self._values = np.asarray(slack, dtype=np.float64)
            self._px = float(px)

        def _slack(self, state):
            return self._centers, self._values, self._px

    def test_hold_keeps_target_until_window_expires(self) -> None:
        c = self.FakeCommitment([0.0, 50.0, 100.0, 0.0], mode="reach_hold",
                                commit_frames=30)
        _a, first = c.decide(GameState(), 0)
        self.assertAlmostEqual(first["target"], 200.0)
        # 新目标只比当前好 6 帧; 承诺期未到 -> 不换
        c._values = np.array([0.0, 100.0, 95.0, 0.0])
        _a, held = c.decide(GameState(), 6)
        self.assertAlmostEqual(held["target"], 200.0)
        self.assertEqual(held["reason"], "hold")
        # 承诺期到期 -> 换到更好的目标
        _a, expired = c.decide(GameState(), 31)
        self.assertAlmostEqual(expired["target"], 100.0)
        self.assertEqual(expired["reason"], "expired")

    def test_hysteresis_needs_margin_to_switch(self) -> None:
        c = self.FakeCommitment([0.0, 100.0, 95.0, 0.0], mode="reach_hyst",
                                hysteresis=10.0)
        _a, _info = c.decide(GameState(), 0)
        # 初始就选最好的 100; 先强行把目标改成 200, 再验证 6 帧优势不够
        c.target = 200.0
        _a, kept = c.decide(GameState(), 6)
        self.assertAlmostEqual(kept["target"], 200.0)
        self.assertEqual(kept["reason"], "hold")

        c2 = self.FakeCommitment([0.0, 100.0, 95.0, 0.0], mode="reach_hyst",
                                 hysteresis=0.0)
        c2.decide(GameState(), 0)
        c2.target = 200.0
        _a, switched = c2.decide(GameState(), 6)
        self.assertAlmostEqual(switched["target"], 100.0)
        self.assertEqual(switched["reason"], "hysteresis")

    def test_unsafe_target_switches_even_during_hold(self) -> None:
        c = self.FakeCommitment([0.0, 50.0, 100.0, 0.0], mode="reach_hold",
                                commit_frames=30)
        c.decide(GameState(), 0)
        c._values = np.array([0.0, 100.0, -5.0, 0.0])
        _a, info = c.decide(GameState(), 6)
        self.assertEqual(info["reason"], "unsafe")
        self.assertAlmostEqual(info["target"], 100.0)

    def test_overshoot_does_not_reverse(self) -> None:
        # 承诺目标在右侧, 6 帧一步跨过目标后不能反向回正 (那会制造新的左右抖动)
        c = self.FakeCommitment([0.0, 50.0, 100.0, 0.0], mode="reach_hold",
                                commit_frames=30)
        c.decide(GameState(), 0)
        c._px = 210.0
        action, info = c.decide(GameState(), 6)
        self.assertEqual(action, 0)
        self.assertAlmostEqual(info["target"], 200.0)

    def test_stateful_modes_are_not_stateless_labels(self) -> None:
        self.assertIn("reach_hold", TEACHER_MODES)
        self.assertIn("reach_hyst", TEACHER_MODES)
        self.assertNotIn("reach_hold", LABEL_MODES)
        with self.assertRaises(KeyError):
            gap_label(GameState(player_x=192.0, player_y=400.0), GapSpec(),
                      "reach_hold")

    def test_action_age_counts_same_direction(self) -> None:
        self.assertEqual(update_action_age(-1, 0, RIGHT), 1)
        self.assertEqual(update_action_age(RIGHT, 1, RIGHT), 2)
        self.assertEqual(update_action_age(RIGHT, 5, LEFT), 1)
        self.assertEqual(update_action_age(RIGHT, 5, 0), 0)
        self.assertEqual(update_action_age(RIGHT, 11, RIGHT, cap=12), 12)
        self.assertEqual(update_action_age(RIGHT, 12, RIGHT, cap=12), 12)

    def test_teacher_resets_commitment_per_episode(self) -> None:
        teacher = GapTeacher(GapSpec(), mode="reach_hold", commit_frames=18)
        state = state_at(192.0, wall(300.0, hole_x=60.0))
        name, info = teacher.choose(state, 100)
        self.assertIn(name, MACRO_ACTIONS)
        self.assertIsNotNone(teacher.commit.target)
        teacher.reset()
        self.assertIsNone(teacher.commit.target)
        self.assertEqual(teacher.commit.switches, 0)


if __name__ == "__main__":
    unittest.main()
