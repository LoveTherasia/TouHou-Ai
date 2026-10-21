"""HybridTeacher: 力场先验 + 分数重排.

可执行契约 (这就是"混合"这个词的全部含义):

* **同分靠先验**: 两条路分数一样时, 选力场指的那条 —— 标签因此落在规则老师那一族;
* **分歧靠分数**: 对准 Boss 的收益足够大时, 允许偏离力场 (攻击意图的来源);
* **底线靠屏障**: ``clearance`` 归零的代价 (40) 必须远大于满对准收益 (约 11),
  这样"用命换分"永远不划算 —— 存活仍是第一要素;
* 屏障是**连续**的: 间距从 margin 降到 0, 惩罚单调不减, 没有硬阈值的跳变。
"""

from __future__ import annotations

import unittest

import numpy as np

from touhou_rl_mcp.training.hybrid_teacher import HybridTeacher
from touhou_rl_mcp.training.rule_teacher import RuleTeacher
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


def head(action: str) -> str:
    """动作串里的方向部分 (去掉 +shoot/+focus 这类尾巴)."""
    parts = [p for p in action.split("+") if p not in {"shoot", "focus"}]
    return parts[0] if parts else ""


def still_rule(**kw) -> RuleTeacher:
    """把归位/游走/边界全部关掉的规则老师: 只剩弹幕斥力, 便于断言方向."""
    cfg = dict(strafe_amp=0.0, boss_home=False, home_weight=0.0,
               edge_weight=0.0, edge_margin=0.0)
    cfg.update(kw)
    return RuleTeacher(**cfg)


class SharedForceFieldTest(unittest.TestCase):
    def test_prior_is_zero_sum_opposite(self) -> None:
        """力场指向左时: 左拿满先验, 左上/左下 0.707 倍, 右边为 0."""
        teacher = HybridTeacher(weights=BASE_WEIGHTS, prior_weight=2.0,
                                rule=still_rule())
        st = state(px=192.0, py=400.0, bullets=[(202.0, 400.0)])  # 右侧贴脸 -> 往左
        w = teacher.prior(st, 192.0, 400.0)
        self.assertEqual(len(w), 9)
        self.assertAlmostEqual(float(w[3]), 2.0, places=6)          # left
        self.assertAlmostEqual(float(w[5]), 2.0 * 0.7071, places=3)  # up_left
        self.assertAlmostEqual(float(w[7]), 2.0 * 0.7071, places=3)  # down_left
        self.assertEqual(float(w[4]), 0.0)                          # right
        self.assertEqual(float(w[0]), 0.0)                          # 力场让动, 原地不拿先验

    def test_prior_favours_staying_when_force_is_weak(self) -> None:
        teacher = HybridTeacher(weights=BASE_WEIGHTS, prior_weight=2.0,
                                rule=still_rule())
        st = state(px=192.0, py=400.0)  # 没有弹幕、没有归位 -> 合力为 0
        w = teacher.prior(st, 192.0, 400.0)
        self.assertAlmostEqual(float(w[0]), 2.0)
        self.assertEqual(float(np.abs(w[1:]).sum()), 0.0)

    def test_direction_matches_rule_teacher(self) -> None:
        """两个老师必须同源: 同样的局面, 力场方向与 RuleTeacher 的选择一致."""
        rule = still_rule()
        hybrid = HybridTeacher(weights=BASE_WEIGHTS, rule=still_rule())
        st = state(px=192.0, py=400.0, bullets=[(170.0, 395.0), (176.0, 402.0)])
        fx, fy, _ = rule.force_vector(st, 192.0, 400.0)
        expect = RuleTeacher.direction_from_force(fx, fy)
        self.assertEqual(hybrid.prior(st, 192.0, 400.0).argmax(), expect)


class ScoreStillEntersTest(unittest.TestCase):
    """score_scale 必须真的改变标签 —— 这是"混合"相对 rule 老师的唯一增益."""

    def _teacher(self, **weights) -> HybridTeacher:
        w = dict(BASE_WEIGHTS)
        w.update(weights)
        return HybridTeacher(weights=w, rule=still_rule())

    def test_moves_to_aim_when_score_matters(self) -> None:
        teacher = self._teacher(score_scale=0.1)
        action, info = teacher.choose(state(boss=(230.0, 150.0)), 0, "shoot")
        self.assertIn("right", head(action))
        self.assertGreater(info["aligned"], 0.0)

    def test_score_scale_flips_the_label(self) -> None:
        """同一局面, 只改 score_scale: 低权重听力场(原地), 高权重去对准.

        默认先验 2.5 与"22 帧内的对准收益(约 2.5)"同量级, 所以这条线落在
        score_scale≈0.03~0.1 之间 —— 这正是"分数进监督"的可调旋钮。
        """
        low, _ = self._teacher(score_scale=0.03).choose(state(boss=(230.0, 150.0)), 0, "shoot")
        high, _ = self._teacher(score_scale=0.1).choose(state(boss=(230.0, 150.0)), 0, "shoot")
        self.assertEqual(low, "shoot")
        self.assertNotEqual(high, "shoot")

    def test_zero_score_scale_falls_back_to_prior(self) -> None:
        teacher = self._teacher(score_scale=0.0, boss_damage_scale=0.0)
        action, _ = teacher.choose(state(boss=(300.0, 150.0)), 0, "shoot")
        self.assertEqual(action, "shoot", "没有分数收益时应该听力场的(原地不动)")

    def test_weak_prior_lets_alignment_win(self) -> None:
        """把先验调到 1.0 以下, 够得着的对准收益就能直接压过先验."""
        teacher = HybridTeacher(weights=BASE_WEIGHTS, prior_weight=1.0, rule=still_rule())
        action, _ = teacher.choose(state(boss=(230.0, 150.0)), 0, "shoot")
        self.assertIn("right", head(action))

    def test_prior_breaks_ties_between_candidates(self) -> None:
        """分数完全并列时, 由力场决定往哪边微调 (而不是原地/随机)."""
        # 没有弹幕 -> 九个方向的分数都是 0; 只有归位力场有主意 (它指向右边)
        teacher = HybridTeacher(weights=BASE_WEIGHTS,
                                rule=still_rule(home_weight=0.8, home_x=300.0))
        st = state(px=192.0, py=400.0)
        cands = {c["index"]: c for c in teacher.candidates(st, 192.0, 400.0)}
        prior = teacher.prior(st, 192.0, 400.0)
        self.assertGreater(prior[4], prior[3], "力场应当指向右边")
        left = teacher.value(cands[3]) + float(prior[3])
        right = teacher.value(cands[4]) + float(prior[4])
        self.assertAlmostEqual(teacher.value(cands[3]), teacher.value(cands[4]), places=6)
        self.assertGreater(right, left, "两边同分时应当选力场那一侧")


class SafetyBarrierTest(unittest.TestCase):
    def test_barrier_is_continuous_and_monotone(self) -> None:
        teacher = HybridTeacher(weights=BASE_WEIGHTS, safety_margin=12.0,
                                barrier_weight=40.0, rule=still_rule())
        cand = {"index": 0, "clearance": 0.0, "aligned": 20.0, "approach": 0.0}
        prev = -1e9
        for clearance in (0.0, 3.0, 6.0, 9.0, 12.0, 20.0):
            cand["clearance"] = clearance
            v = teacher.value(cand)
            self.assertGreater(v, prev, f"clearance={clearance} 时惩罚应递减(价值递增)")
            prev = v

    def test_barrier_outweighs_alignment(self) -> None:
        """满对准(22 帧)约 11.4 分, 贴脸屏障 25 分 —— 任何分数都不值得换命."""
        teacher = HybridTeacher(weights=BASE_WEIGHTS, safety_margin=12.0,
                                barrier_weight=25.0, rule=still_rule())
        aligned_max = ((BASE_WEIGHTS["score_scale"] + BASE_WEIGHTS["boss_damage_scale"])
                       * teacher.score_rate * teacher.align_horizon)
        zero = teacher.value({"index": 0, "clearance": 0.0, "aligned": teacher.align_horizon,
                              "approach": 1.0})
        self.assertLess(zero, 0.0)
        self.assertGreater(25.0, aligned_max, "屏障必须大于一次完整对准的收益")

    def test_align_horizon_defaults_to_safety_window(self) -> None:
        """收益窗口默认不拉长: 实测 60 帧会让老师看不见远处弹幕, 存活掉一半."""
        teacher = HybridTeacher(weights=BASE_WEIGHTS, rule=still_rule())
        self.assertAlmostEqual(teacher.align_horizon, 22.0)

    def test_dodges_instead_of_staying_aligned(self) -> None:
        """正对 Boss 且原地满对准, 但一颗弹正朝自机落下: 必须让开."""
        teacher = HybridTeacher(weights=BASE_WEIGHTS, safety_margin=12.0,
                                barrier_weight=40.0, rule=still_rule())
        st = state(px=192.0, py=400.0, boss=(192.0, 150.0),
                   bullets=[(192.0, 380.0)], velocities=[(0.0, 1.0)])
        action, info = teacher.choose(st, 0, "shoot")
        self.assertNotEqual(head(action), "", "不能原地不动")
        self.assertNotEqual(head(action), "up", "不能迎着弹幕往上")
        self.assertGreaterEqual(info["clearance"], 8.0)


class ScoreTeacherUnchangedTest(unittest.TestCase):
    def test_barrier_defaults_to_off(self) -> None:
        """ScoreTeacher 的既有行为不能被软屏障改掉 (barrier_weight 默认 0)."""
        self.assertEqual(ScoreTeacher().barrier_weight, 0.0)
        plain = ScoreTeacher(weights=BASE_WEIGHTS)
        cand = {"index": 0, "clearance": 0.0, "aligned": 22.0, "approach": 0.0}
        self.assertGreater(plain.value(cand), 0.0)


class DirectionOrderTest(unittest.TestCase):
    def test_dirs_match_action_table(self) -> None:
        self.assertEqual([NAMES[i] for i in range(len(DIRECTIONS))], list(DIRECTIONS))


if __name__ == "__main__":
    unittest.main()


class WallDetectionTest(unittest.TestCase):
    """长时预测只该作用在"匀速直线的弹幕墙"上.

    stage1 实测: 对**全部**弹幕做 45 帧预测, 存活 4770 -> 2971 (弧线弹的虚假危险);
    只对墙做长时预测, 才能既保住普通弹型又看见正在压下来的墙。
    """

    def _wall(self, spacing=9.0, vy=1.9, vx=0.0, y=300.0, count=24):
        bullets, vel = [], []
        for i in range(count):
            bullets.append((20.0 + i * spacing, y))
            vel.append((vx, vy))
        return bullets, vel

    def test_dense_falling_row_is_detected(self) -> None:
        teacher = HybridTeacher(weights=BASE_WEIGHTS, wall_horizon=90.0, rule=still_rule())
        b, v = self._wall()
        st = state(px=192.0, py=400.0, bullets=b, velocities=v)
        self.assertEqual(len(teacher.wall_mask(st, 400.0)), len(b))

    def test_sparse_row_is_not_a_wall(self) -> None:
        teacher = HybridTeacher(weights=BASE_WEIGHTS, wall_horizon=90.0, rule=still_rule())
        b, v = self._wall(spacing=40.0, count=8)
        st = state(px=192.0, py=400.0, bullets=b, velocities=v)
        self.assertEqual(len(teacher.wall_mask(st, 400.0)), 0)

    def test_curving_or_rising_bullets_are_excluded(self) -> None:
        teacher = HybridTeacher(weights=BASE_WEIGHTS, wall_horizon=90.0, rule=still_rule())
        b, v = self._wall(vx=2.0)  # 有横向速度 = 不是直线墙
        st = state(px=192.0, py=400.0, bullets=b, velocities=v)
        self.assertEqual(len(teacher.wall_mask(st, 400.0)), 0)
        b, v = self._wall(vy=-1.9)  # 上飞
        st = state(px=192.0, py=400.0, bullets=b, velocities=v)
        self.assertEqual(len(teacher.wall_mask(st, 400.0)), 0)

    def test_disabled_by_default(self) -> None:
        teacher = HybridTeacher(weights=BASE_WEIGHTS, rule=still_rule())
        b, v = self._wall()
        st = state(px=192.0, py=400.0, bullets=b, velocities=v)
        self.assertEqual(teacher.wall_horizon, 0.0)
        self.assertEqual(len(teacher.wall_mask(st, 400.0)), 0)

    def test_wall_changes_the_label_before_it_arrives(self) -> None:
        """墙还在 80px 上方、只留左侧一个缺口: 应当提前往左站, 而不是停在原地."""
        teacher = HybridTeacher(weights=BASE_WEIGHTS, wall_horizon=90.0, rule=still_rule())
        b, v = [], []
        x = 8.0
        while x < 380.0:                 # 横跨整个场地
            if not (60.0 < x < 110.0):   # 唯一缺口在左侧
                b.append((x, 320.0))
                v.append((0.0, 1.9))
            x += 9.0
        st = state(px=192.0, py=400.0, bullets=b, velocities=v)
        action, info = teacher.choose(st, 0, "shoot")
        self.assertIn("left", head(action))
        self.assertLess(info["clearance"], 40.0, "长时预测应当已经看到会撞上来的墙")
