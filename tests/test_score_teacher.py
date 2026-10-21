"""ScoreTeacher: 让 ``score_scale`` 真正决定标签.

这些用例是"分数进入监督"的可执行契约:

* 没有弹幕时, 高 ``score_scale`` 会把标签推向 Boss 所在列; 归零就变成原地不动;
* 有弹幕挡住那条路时, **安全优先** —— 即使那边的分数更高也不去;
* 方向顺序必须与动作表一致, 否则标签会整体错位一档。
"""

from __future__ import annotations

import unittest

from touhou_rl_mcp.training.score_teacher import NAMES, ScoreTeacher
from touhou_rl_mcp.training.trainer import DIRECTIONS
from touhou_rl_mcp.types import GameState

BASE_WEIGHTS = {
    "score_scale": 0.03,
    "boss_damage_scale": 0.1,
    "danger_scale": 0.8,
    "danger_radius": 32.0,
}


def state(px=192.0, py=400.0, boss=None, bullets=(), enemies=(), velocities=None):
    raw = {}
    if boss is not None:
        raw["boss_position"] = boss
    if velocities is not None:
        raw["bullet_velocity"] = list(velocities)
    return GameState(
        frame=100, in_game=True, player_x=px, player_y=py, lives=2,
        boss_hp=1000 if boss is not None else None,
        bullets=list(bullets), enemy_positions=list(enemies), raw=raw,
    )


class DirectionOrderTest(unittest.TestCase):
    def test_names_match_action_table(self) -> None:
        self.assertEqual(list(NAMES), list(DIRECTIONS))


class ScoreDrivesLabelTest(unittest.TestCase):
    def test_moves_toward_boss_when_score_matters(self) -> None:
        teacher = ScoreTeacher(weights=BASE_WEIGHTS)
        action, info = teacher.choose(state(boss=(300.0, 150.0)), 0, "shoot")
        self.assertIn("right", action)
        self.assertGreater(info["aligned"], 0.0)

    def test_zero_score_scale_falls_back_to_standing(self) -> None:
        """把 score_scale 归零, 同一个局面就不再有"去追 Boss"的理由."""
        weights = dict(BASE_WEIGHTS, score_scale=0.0, boss_damage_scale=0.0)
        teacher = ScoreTeacher(weights=weights)
        action, _ = teacher.choose(state(boss=(300.0, 150.0)), 0, "shoot")
        self.assertEqual(action, "shoot")

    def test_alignment_respects_tolerance(self) -> None:
        teacher = ScoreTeacher(weights=BASE_WEIGHTS, align_tol=20.0)
        close, _ = teacher.choose(state(boss=(192.0, 150.0)), 0, "shoot")
        far, _ = teacher.choose(state(boss=(370.0, 150.0)), 0, "shoot")
        self.assertEqual(close, "shoot", "目标就在正上方时不该乱动")
        self.assertNotEqual(far, "shoot", "目标在右边时应该过去")


class SafetyFirstTest(unittest.TestCase):
    def test_blocked_direction_is_rejected_even_if_it_scores(self) -> None:
        """右侧 38px 处有颗弹: 右边分数更高, 但路径会贴到 6px, 必须放弃."""
        teacher = ScoreTeacher(weights=BASE_WEIGHTS, safety_margin=12.0)
        st = state(boss=(300.0, 150.0), bullets=[(230.0, 400.0)])
        action, info = teacher.choose(st, 0, "shoot")
        self.assertNotIn("right", action)
        self.assertGreaterEqual(info["clearance"], 12.0)

    def test_all_unsafe_falls_back_to_largest_clearance(self) -> None:
        teacher = ScoreTeacher(weights=BASE_WEIGHTS, safety_margin=10_000.0)
        st = state(boss=(300.0, 150.0), bullets=[(200.0, 396.0), (184.0, 404.0)])
        action, info = teacher.choose(st, 0, "shoot")
        self.assertEqual(info["safe_dirs"], 0)
        head = action[: -len("+shoot")] if action.endswith("+shoot") else ""
        self.assertIn(head, NAMES)

    def test_bullet_velocity_is_used_for_clearance(self) -> None:
        """静止弹在右侧 60px 是安全的; 但同一颗弹以 6px/帧 向左飞就危险."""
        teacher = ScoreTeacher(weights=BASE_WEIGHTS, safety_margin=12.0)
        static = state(boss=(300.0, 150.0), bullets=[(252.0, 400.0)],
                       velocities=[(0.0, 0.0)])
        moving = state(boss=(300.0, 150.0), bullets=[(252.0, 400.0)],
                       velocities=[(-6.0, 0.0)])
        _, info_static = teacher.choose(static, 0, "shoot")
        _, info_moving = teacher.choose(moving, 0, "shoot")
        self.assertLess(info_moving["clearance"], info_static["clearance"])


class AimTargetTest(unittest.TestCase):
    def test_falls_back_to_enemy_when_no_boss(self) -> None:
        teacher = ScoreTeacher(weights=BASE_WEIGHTS)
        st = state(enemies=[(90.0, 200.0), (280.0, 220.0)])
        self.assertEqual(teacher.aim_x(st, 192.0, 400.0), 280.0)

    def test_enemies_below_player_are_ignored(self) -> None:
        teacher = ScoreTeacher(weights=BASE_WEIGHTS)
        st = state(enemies=[(300.0, 430.0)])
        self.assertIsNone(teacher.aim_x(st, 192.0, 400.0))


if __name__ == "__main__":
    unittest.main()
