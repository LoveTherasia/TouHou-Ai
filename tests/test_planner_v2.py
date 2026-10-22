"""规划器 v2 的单元测试: Deep Sets 观测 / 多目标奖励 / 动力学 / MPC / 势能场."""
from __future__ import annotations

import unittest

import numpy as np

from touhou_rl_mcp.types import PLAYFIELD_HEIGHT, PLAYFIELD_WIDTH, GameState


def state(**kw) -> GameState:
    base = dict(in_game=True, player_x=192.0, player_y=400.0, lives=3, score=0,
                boss_hp=1000, boss_max_hp=1000, stage=1, bullets=[])
    base.update(kw)
    return GameState(**base)


class TestDeepSetsObservation(unittest.TestCase):
    def test_dim_and_shape(self) -> None:
        from touhou_rl_mcp.training.features import (SET_EMBED_DIM, encode_set_block,
                                                     set_feature_dim)

        vec = encode_set_block(state(bullets=[(100.0, 100.0), (200.0, 200.0)]))
        self.assertEqual(vec.shape[0], set_feature_dim())
        self.assertEqual(vec.shape[0], 2 * SET_EMBED_DIM + 2)
        self.assertTrue(np.all(np.isfinite(vec)))

    def test_permutation_invariant(self) -> None:
        """Deep Sets 的核心性质: 打乱弹序, 观测逐位相等."""
        from touhou_rl_mcp.training.features import encode_set_block

        pts = [(100.0, 100.0), (200.0, 150.0), (300.0, 300.0), (50.0, 400.0)]
        vel = [(0.0, 2.0), (1.0, 1.0), (-1.0, 2.0), (0.3, 1.7)]
        a = state(bullets=pts, raw={"bullet_velocity": vel})
        order = [2, 0, 3, 1]
        b = state(bullets=[pts[i] for i in order],
                  raw={"bullet_velocity": [vel[i] for i in order]})
        np.testing.assert_allclose(encode_set_block(a), encode_set_block(b), atol=1e-12)

    def test_variable_length_and_empty(self) -> None:
        """弹数变化不改变维度 (mean/max 池化对 N 无要求)."""
        from touhou_rl_mcp.training.features import encode_set_block

        empty = encode_set_block(state(bullets=[]))
        one = encode_set_block(state(bullets=[(100.0, 100.0)]))
        many = encode_set_block(state(bullets=[(float(i), 100.0) for i in range(200)]))
        self.assertEqual(empty.shape, one.shape)
        self.assertEqual(empty.shape, many.shape)
        self.assertTrue(np.allclose(empty, 0.0))

    def test_observation_dim_and_encode_observation(self) -> None:
        from touhou_rl_mcp.training import TrainConfig
        from touhou_rl_mcp.training.features import observation_dim

        cfg = TrainConfig(use_set=True, use_boss=True, k_bullets=16)
        vec = observation_dim(cfg)
        from touhou_rl_mcp.training.features import encode_observation

        obs = encode_observation(state(bullets=[(100.0, 100.0)]), cfg)
        self.assertEqual(obs.shape[0], vec)
        self.assertTrue(np.all(np.isfinite(obs)))

    def test_set_and_gap_are_exclusive_by_config(self) -> None:
        """use_set 优先于 use_gap (同时开时按 set 编码, 不混出半套特征)."""
        from touhou_rl_mcp.training import TrainConfig
        from touhou_rl_mcp.training.features import encode_observation, observation_dim

        cfg = TrainConfig(use_set=True, use_gap=True)
        obs = encode_observation(state(bullets=[(100.0, 100.0)]), cfg)
        self.assertEqual(obs.shape[0], observation_dim(cfg))


class TestMultiObjectiveReward(unittest.TestCase):
    def test_progress_and_position_terms(self) -> None:
        from touhou_rl_mcp.rl.reward import RewardModel, reward_profile

        model = RewardModel(reward_profile("multi"))
        prev = state(boss_hp=1000, score=0)
        cur = state(boss_hp=900, score=100, frame=1,
                    raw={"boss_position": (192.0, 80.0)})
        total, parts = model.compute(prev, cur)
        self.assertIn("progress", parts)
        self.assertIn("position", parts)
        self.assertGreater(total, 0.0)

    def test_corner_is_penalised(self) -> None:
        """贴角必须比站中间分低 —— 这是"贴底横移活满 2000 步"的负梯度."""
        from touhou_rl_mcp.rl.reward import RewardModel, reward_profile

        model = RewardModel(reward_profile("multi"))
        center = state(player_x=192.0, player_y=380.0, frame=1,
                       raw={"boss_position": (192.0, 80.0)})
        corner = state(player_x=8.0, player_y=440.0, frame=1,
                       raw={"boss_position": (192.0, 80.0)})
        _, p_center = model.compute(center, center)
        _, p_corner = model.compute(corner, corner)
        self.assertGreater(p_center.get("position", 0.0), p_corner.get("position", 0.0))

    def test_default_profile_has_no_position_term(self) -> None:
        from touhou_rl_mcp.rl.reward import RewardModel, reward_profile

        model = RewardModel(reward_profile("attack"))
        _, parts = model.compute(state(), state(player_x=8.0, frame=1))
        self.assertNotIn("position", parts)


class TestDynamics(unittest.TestCase):
    def _tracks(self):
        rng = np.random.default_rng(0)
        tracks = []
        for _ in range(40):
            x, y = rng.uniform(20, 360), rng.uniform(0, 120)
            vx, vy = rng.uniform(-1, 1), rng.uniform(0.5, 2.5)
            ax, ay = rng.uniform(-0.03, 0.03), rng.uniform(-0.02, 0.05)
            pts = []
            for t in range(0, 120, 3):
                pts.append((t, x + vx * t + 0.5 * ax * t * t,
                            y + vy * t + 0.5 * ay * t * t, vx + ax * t, vy + ay * t, ax, ay))
            tracks.append(pts)
        return tracks

    def test_residual_model_beats_constant_velocity(self) -> None:
        from touhou_rl_mcp.training.dynamics import (ConstantVelocity, ResidualMLP,
                                                     evaluate, pairs_from_tracks)

        X, Y, _ = pairs_from_tracks(self._tracks())
        cv = evaluate(ConstantVelocity(), X, Y)["mae"]
        model = ResidualMLP(hidden=32, seed=0).fit(X, Y, epochs=200, batch=256, lr=5e-3)
        mae = evaluate(model, X, Y)["mae"]
        self.assertLess(mae, cv * 0.2)          # 加速弹上必须明显超过匀速外推
        self.assertLess(mae, 0.6)

    def test_constant_velocity_is_exact_on_linear_tracks(self) -> None:
        from touhou_rl_mcp.training.dynamics import ConstantVelocity

        cv = ConstantVelocity()
        px, py = cv.predict(100.0, 50.0, 1.5, 2.0, 12.0)
        self.assertAlmostEqual(px, 118.0)
        self.assertAlmostEqual(py, 74.0)


class TestMPCPlanner(unittest.TestCase):
    def test_avoids_bullet_column(self) -> None:
        """正上方一条弹柱朝自机压过来时, MPC 必须离开那一列 (不选原地/向上)."""
        from touhou_rl_mcp.training.mpc import MPCPlanner

        bullets = [(192.0, y) for y in (300.0, 330.0, 360.0, 390.0)]
        vel = [(0.0, 3.0)] * 4
        st = state(bullets=bullets, raw={"bullet_velocity": vel})
        planner = MPCPlanner(horizon=24, dt=3, n_samples=64, iters=4, seed=0)
        action = planner.choose(st)
        self.assertNotIn(action, ("shoot", "up+shoot"))

    def test_safe_state_keeps_position(self) -> None:
        from touhou_rl_mcp.training.mpc import MPCPlanner

        st = state(bullets=[(60.0, 60.0)], raw={"bullet_velocity": [(0.0, 1.0)]})
        planner = MPCPlanner(horizon=24, dt=3, seed=0)
        # 远处有弹 -> 不该乱跑; 允许任何动作, 只要求代价有限
        planner.choose(st)
        self.assertIsNotNone(planner.last_cost)
        self.assertLess(planner.last_cost, 1e6)

    def test_runaway_does_not_win(self) -> None:
        """没有弹的时候, 位置偏好把自机拉向目标高度 (不是继续往边上挤).

        自机在 y=440 (场地底边, 已被 clip 到 440), 目标高度 0.85*448≈381 ——
        所以正确动作是**往上**回到输出高度, 绝不能继续按下。
        """
        from touhou_rl_mcp.training.mpc import MPCPlanner

        st = state(player_y=440.0, bullets=[], raw={"boss_position": (192.0, 60.0)})
        planner = MPCPlanner(horizon=24, dt=3, y_weight=1.0, y_target=0.85, seed=0)
        action = planner.choose(st)
        self.assertNotIn(action.split("+")[0], ("down", "down+left", "down+right"))


class TestPotentialField(unittest.TestCase):
    def test_chooses_away_from_bullet(self) -> None:
        from touhou_rl_mcp.training.potential import PotentialFieldPlanner

        # 自机左侧 12px 处有一颗弹 -> 不该继续往左
        st = state(player_x=200.0, player_y=360.0, bullets=[(188.0, 360.0)],
                   raw={"bullet_velocity": [(0.0, 0.0)]})
        planner = PotentialFieldPlanner()
        action = planner.choose(st)
        self.assertFalse(action.startswith("left"))

    def test_potential_prefers_center(self) -> None:
        from touhou_rl_mcp.training.potential import PotentialFieldPlanner

        planner = PotentialFieldPlanner()
        bullets = np.array([[192.0, 100.0]])
        center = planner.potential(192.0, 380.0, bullets)
        corner = planner.potential(6.0, 442.0, bullets)
        self.assertLess(center, corner)


if __name__ == "__main__":
    unittest.main()


class TestMpcTeacherAdaptation(unittest.TestCase):
    """MPC 当老师时的两个坑: 动作名格式 + 时间单位换算."""

    def test_action_name_uses_trainer_direction_style(self) -> None:
        from touhou_rl_mcp.training.mpc_teacher import to_trainer_name
        from touhou_rl_mcp.training.trainer import DIRECTIONS

        for action, want in (("shoot", "shoot"), ("up+shoot", "up+shoot"),
                             ("up+left+shoot", "up_left+shoot"),
                             ("down+right+shoot", "down_right+shoot")):
            name = to_trainer_name(action)
            self.assertEqual(name, want)
            head = name.replace("+shoot", "").replace("+focus", "")
            if head == "shoot":
                head = ""
            self.assertIn(head, DIRECTIONS, f"{action} -> {name} 的方向不在动作表里")

    def test_step_frames_scales_speed_and_bullets(self) -> None:
        """step_frames=4 时自机速度与弹速都要 ×4, 否则规划器以为自己走 4.4px."""
        from touhou_rl_mcp.training.mpc_teacher import MPCTeacher
        from touhou_rl_mcp.types import GameState

        t1 = MPCTeacher(horizon=24, dt=3, step_frames=1)
        t4 = MPCTeacher(horizon=24, dt=3, step_frames=4)
        self.assertAlmostEqual(t1.planner.speed, 4.4)
        self.assertAlmostEqual(t4.planner.speed, 4.4 * 4)
        # 视界按**帧**给: k=4 时 24 帧 = 6 个决策步
        self.assertEqual(t4.planner.n_steps * t4.planner.dt, 6)

        state = GameState(player_x=192, player_y=400, bullets=[(100.0, 100.0)],
                          raw={"bullet_velocity": [(0.0, 2.0)]})
        scaled = t4._scaled(state)
        self.assertAlmostEqual(scaled.raw["bullet_velocity"][0][1], 8.0)
        self.assertAlmostEqual(state.raw["bullet_velocity"][0][1], 2.0)   # 不改原状态


class TestMpcGapAnchor(unittest.TestCase):
    """缺口锚点: 墙要能把终点拉到缺口列, 稀疏弹幕不能乱拉."""

    def _wall(self, gap_x: float) -> np.ndarray:
        rows = []
        for x in np.arange(12.0, 372.0, 9.0):
            if abs(x - gap_x) < 24.0:
                continue
            rows.append([x, 108.0, 0.0, 1.9, 0.0, 0.0])
        return np.array(rows, dtype=np.float64)

    def test_anchor_found_on_wall(self) -> None:
        from touhou_rl_mcp.training.mpc import MPCPlanner

        planner = MPCPlanner(horizon=72, dt=6, gap_weight=20.0)
        anchor = planner.gap_anchor(self._wall(120.0), 416.0)
        self.assertIsNotNone(anchor)
        self.assertLess(abs(anchor - 120.0), 16.0)

    def test_anchor_ignores_sparse_fire(self) -> None:
        from touhou_rl_mcp.training.mpc import MPCPlanner

        planner = MPCPlanner(horizon=72, dt=6, gap_weight=20.0)
        sparse = np.array([[100.0, 108.0, 0.0, 1.9, 0.0, 0.0],
                           [200.0, 108.0, 0.0, 1.9, 0.0, 0.0]], dtype=np.float64)
        self.assertIsNone(planner.gap_anchor(sparse, 416.0))

    def test_gap_cost_prefers_gap_column(self) -> None:
        from touhou_rl_mcp.training.mpc import MPCPlanner

        planner = MPCPlanner(horizon=72, dt=6, gap_weight=20.0)
        wall = self._wall(120.0)
        anchor = planner.gap_anchor(wall, 416.0)
        cost_gap = planner._gap_cost(np.array([anchor]), anchor)[0]
        cost_far = planner._gap_cost(np.array([anchor + 120.0]), anchor)[0]
        self.assertEqual(round(cost_gap, 6), 0.0)
        self.assertGreater(cost_far, 1.0)


class TestExactClearance(unittest.TestCase):
    """粗步长下的"穿透"必须被连续时间余量挡住."""

    def test_fast_bullet_tunneling_detected(self) -> None:
        from touhou_rl_mcp.training.mpc import MPCPlanner

        # 弹从自机正上方 30px 处、以 20px/帧直冲下来；dt=12 时采样点会跳过自机
        state = GameState(player_x=192.0, player_y=400.0, bullets=[(192.0, 370.0)],
                          bullet_count=1, in_game=True,
                          raw={"bullet_velocity": [(0.0, 20.0)]})
        seq = np.zeros((1, 3, 2))                     # 原地不动
        exact = MPCPlanner(horizon=36, dt=12, exact_clearance=True)._rollout(
            state, np.array([[192.0, 370.0, 0.0, 20.0]]), seq)[0][0]
        point = MPCPlanner(horizon=36, dt=12, exact_clearance=False)._rollout(
            state, np.array([[192.0, 370.0, 0.0, 20.0]]), seq)[0][0]
        self.assertGreater(exact, 100.0)     # 精确：撞了
        self.assertLess(point, 10.0)         # 点采样：以为没事（穿透）


class TestTrackPostprocess(unittest.TestCase):
    """真机轨迹采集的后处理：跳变断开 + 差分出加速度."""

    @staticmethod
    def _mod():
        import importlib.util
        from pathlib import Path

        path = Path(__file__).resolve().parent.parent / "tools" / "collect_tracks.py"
        spec = importlib.util.spec_from_file_location("collect_tracks", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_split_on_slot_reuse(self) -> None:
        mod = self._mod()
        pts = [(0, 100.0, 100.0), (1, 100.0, 101.9), (2, 300.0, 40.0), (3, 300.0, 41.9)]
        runs = mod.split_runs(pts)
        self.assertEqual(len(runs), 2)
        self.assertEqual(len(runs[0]), 2)

    def test_accel_recovered_from_parabola(self) -> None:
        mod = self._mod()
        ax, ay = 0.02, 0.05
        pts = []
        for f in range(0, 30):
            x = 100.0 + 0.5 * f + 0.5 * ax * f * f
            y = 50.0 + 2.0 * f + 0.5 * ay * f * f
            pts.append((f, x, y))
        rows = mod.track_to_rows(pts)
        self.assertGreater(len(rows), 20)
        mid = rows[len(rows) // 2]
        self.assertLess(abs(mid[4] - ax), 0.005)
        self.assertLess(abs(mid[5] - ay), 0.005)


class TestSoftLabels(unittest.TestCase):
    """软标签蒸馏: 规划器并列动作共享概率质量, 而不是硬 argmax 随机挑一个."""

    def _state(self):
        from touhou_rl_mcp.types import GameState

        # 一颗弹在正上方很远 -> 向左/右闪避都"同样好", 正是 argmax 抖动的场景
        return GameState(player_x=192.0, player_y=400.0, bullets=[(192.0, 120.0)],
                         raw={"bullet_velocity": [(0.0, 1.5)]})

    def test_soft_labels_form_distribution(self) -> None:
        from touhou_rl_mcp.training.mpc import MPCPlanner

        planner = MPCPlanner(horizon=36, dt=3, seed=0)
        p = planner.soft_labels(self._state(), temperature=20.0)
        self.assertEqual(p.shape, (9,))
        self.assertAlmostEqual(float(p.sum()), 1.0, places=6)
        self.assertGreaterEqual(float(p.min()), 0.0)

    def test_soft_labels_prefer_safe_side(self) -> None:
        from touhou_rl_mcp.training.mpc import MPCPlanner

        planner = MPCPlanner(horizon=36, dt=3, seed=0)
        p = planner.soft_labels(self._state(), temperature=5.0)
        # DIRECTIONS 顺序: 3=left, 4=right。站在弹的正下方, 左右都好于不动/继续往下
        self.assertGreater(p[3] + p[4], p[0] + p[2])

    def test_imitation_accepts_soft_targets(self) -> None:
        from touhou_rl_mcp.training.features import feature_dim
        from touhou_rl_mcp.training.policy import MLPPolicy

        dim = feature_dim(4, True, 2, False, True, False)
        policy = MLPPolicy(obs_dim=dim, hidden=(16,), n_actions=9, seed=0, lr=0.05)
        rng = np.random.default_rng(0)
        X = rng.standard_normal((32, dim))
        target = np.zeros((32, 9))
        target[:, 3] = 1.0                       # 目标: 一直向左
        first = policy.imitate_batch(X, target)
        for _ in range(20):
            last = policy.imitate_batch(X, target)
        self.assertLess(last["imitation_loss"], first["imitation_loss"])
        self.assertEqual(policy.greedy(X[0]), 3)


class TestDeterminisedTeacher(unittest.TestCase):
    """老师确定化: 承诺期 / focus 代价 / cost-to-go 蒸馏口径."""

    @staticmethod
    def _wall_state(frame: int = 0) -> GameState:
        """半边封死的贴脸弹幕: frame=0 封左 (只能往右), frame=1 封右 (只能往左)."""
        bullets, vel = [], []
        span = range(16, 184, 14) if frame % 2 == 0 else range(202, 370, 14)
        for y in (380.0, 392.0, 404.0):
            for x in span:
                bullets.append((float(x), y))
                vel.append((0.0, 0.0))
        return state(frame=frame, bullets=bullets, raw={"bullet_velocity": vel})

    def test_focus_costs_have_18_entries(self) -> None:
        """use_focus 时必须给出 18 个代价 (9 快 + 9 慢), 否则蒸馏维度对不上动作表."""
        from touhou_rl_mcp.training.mpc import MPCPlanner

        planner = MPCPlanner(horizon=24, dt=3, use_focus=True, seed=0)
        costs = planner.action_costs(self._wall_state())
        self.assertEqual(costs.shape[0], 18)
        # 慢速块用的是半速: 代价不能与全速块逐位相同
        self.assertFalse(np.allclose(costs[:9], costs[9:]))

    def test_choose_matches_cost_argmin(self) -> None:
        """老师动作 = 逐动作 cost-to-go 的 argmin（蒸馏教的就是这个）."""
        from touhou_rl_mcp.training.mpc import MPCPlanner

        planner = MPCPlanner(horizon=24, dt=3, seed=0, n_samples=32, iters=3)
        for frame in range(4):
            st = self._wall_state(frame)
            action = planner.choose(st)
            costs = planner.cost_vector(st)
            self.assertEqual(action, [n for n in (
                "shoot", "up+shoot", "down+shoot", "left+shoot", "right+shoot",
                "up+left+shoot", "up+right+shoot", "down+left+shoot",
                "down+right+shoot")][int(np.argmin(costs))])

    def test_commit_holds_direction(self) -> None:
        """承诺期内不允许因为一点点代价差就换向."""

        from touhou_rl_mcp.training.mpc import MPCPlanner

        # 对照: 不承诺时, 缺口从左边换到右边, 动作必须跟着换
        free = MPCPlanner(horizon=24, dt=3, seed=0, n_samples=32, iters=3)
        a0 = free.choose(self._wall_state(0))
        a1 = free.choose(self._wall_state(1))
        self.assertNotEqual(a0, a1)

        # 承诺期 4 帧: 同一批局面在第 2~4 帧里必须沿用第一个动作
        held = MPCPlanner(horizon=24, dt=3, seed=0, commit_steps=4,
                          switch_margin=1e9, n_samples=32, iters=3)
        first = held.choose(self._wall_state(0))
        self.assertEqual(held.choose(self._wall_state(1)), first)
        self.assertEqual(held.choose(self._wall_state(2)), first)

    def test_cost_to_go_regression_learns_argmin(self) -> None:
        """回归 -相对代价/T 之后, 学生自己的 argmax 应等于老师的 argmin."""
        from touhou_rl_mcp.training.policy import MLPPolicy

        rng = np.random.default_rng(0)
        X = rng.standard_normal((96, 12))
        C = rng.random((96, 9)) * 40.0
        policy = MLPPolicy(12, 9, hidden=(64, 64), lr=5e-3)
        for _ in range(200):
            stats = policy.fit_cost(X, C, temp=20.0, clip=8.0)
        self.assertGreater(stats["agree"], 0.9)
        self.assertLess(stats["cost_mae"], 0.05)

    def test_teacher_cost_vector_matches_action_table(self) -> None:
        """老师给的 cost 维数必须等于训练动作表长度 (focus 时 18)."""
        from touhou_rl_mcp.training.mpc_teacher import MPCTeacher
        from touhou_rl_mcp.training.trainer import build_action_table

        for use_focus in (False, True):
            teacher = MPCTeacher(horizon=24, dt=3, iters=2, use_focus=use_focus,
                                 step_frames=1, seed=0)
            vec = teacher.cost_vector(self._wall_state())
            self.assertEqual(vec.shape[0], len(build_action_table(use_focus=use_focus)))


class TestTieAwareRank(unittest.TestCase):
    """并列感知排序: 并列最优的整组共享目标质量, 不要求组内排序."""

    def test_all_tied_gives_uniform_target(self) -> None:
        """全部并列时目标就是均匀分布: 不该被推向任何一个动作."""
        from touhou_rl_mcp.training.policy import MLPPolicy

        policy = MLPPolicy(8, 9, hidden=(16, 16), lr=1e-2)
        X = np.random.default_rng(0).standard_normal((32, 8))
        C = np.zeros((32, 9))
        stats = policy.fit_rank(X, C, margin=20.0)
        self.assertAlmostEqual(stats["good_mean"], 9.0)
        # 均匀目标: 梯度把 logits 拉向"全部相等", 不会出现明显的单一偏好
        logits = policy.logits_batch(X)
        self.assertLess(float((logits.max(axis=1) - logits.min(axis=1)).max()), 3.0)

    def test_learns_order_when_gap_is_large(self) -> None:
        from touhou_rl_mcp.training.policy import MLPPolicy

        rng = np.random.default_rng(1)
        X = rng.standard_normal((128, 10))
        # 最优动作必须是**观测的函数**, 否则任何模型都学不出来
        W = rng.standard_normal((10, 9))
        best = np.argmax(X @ W, axis=1)
        C = rng.uniform(100.0, 200.0, size=(128, 9))
        C[np.arange(128), best] = 0.0
        policy = MLPPolicy(10, 9, hidden=(64, 64), lr=5e-3)
        for _ in range(300):
            stats = policy.fit_rank(X, C, margin=20.0)
        self.assertGreater(stats["agree"], 0.9)
        pred = np.argmax(policy.logits_batch(X), axis=1)
        self.assertGreater(float(np.mean(pred == best)), 0.9)

    def test_tied_group_shares_mass(self) -> None:
        """最优并列两个动作时, 两者的目标概率各一半, 不会被拉开."""
        from touhou_rl_mcp.training.policy import MLPPolicy

        rng = np.random.default_rng(2)
        X = rng.standard_normal((64, 8))
        C = np.full((64, 9), 300.0)
        C[:, [0, 4]] = 0.0             # 0 和 4 并列最优
        policy = MLPPolicy(8, 9, hidden=(32, 32), lr=5e-3)
        for _ in range(80):
            policy.fit_rank(X, C, margin=20.0)
        logits = policy.logits_batch(X)
        tied = np.abs(logits[:, 0] - logits[:, 4])
        gap_to_worse = logits[:, [0, 4]].max(axis=1) - logits[:, 1]
        self.assertLess(float(tied.max()), 0.5)
        self.assertGreater(float(gap_to_worse.min()), 1.0)
        stats = policy.fit_rank(X, C, margin=20.0)
        self.assertGreater(stats["rank_acc"], 0.99)


class TestTieAwareListMLE(unittest.TestCase):
    """tie-aware ListMLE: 按组连乘, 组内对称, 候选采样, 位置正则."""

    @staticmethod
    def _brute_loss(s: np.ndarray, costs: np.ndarray, margin: float) -> float:
        order = np.argsort(costs, kind="stable")
        groups, cur = [], [int(order[0])]
        for j in order[1:]:
            if costs[j] - costs[cur[-1]] > margin:
                groups.append(cur)
                cur = [int(j)]
            else:
                cur.append(int(j))
        groups.append(cur)
        loss = 0.0
        for gi, g in enumerate(groups):
            tail = [x for gg in groups[gi:] for x in gg]
            loss -= (np.log(np.exp(s[g]).sum()) - np.log(np.exp(s[tail]).sum()))
        return float(loss)

    def test_matches_bruteforce(self) -> None:
        """loss 必须与"按组连乘"的暴力实现逐位一致."""
        from touhou_rl_mcp.training.policy import MLPPolicy

        rng = np.random.default_rng(0)
        X = rng.standard_normal((3, 6))
        policy = MLPPolicy(6, 9, hidden=(16, 16), lr=0.0)
        logits = policy.logits_batch(X)
        C = np.array([0.0, 5.0, 300.0, 0.0, 5.0, 300.0, 150.0, 150.0, 150.0])
        stats = policy.fit_listmle(X, np.repeat(C[None, :], 3, axis=0), margin=20.0)
        ref = float(np.mean([self._brute_loss(logits[i], C, 20.0) for i in range(3)]))
        self.assertAlmostEqual(stats["listmle_loss"], ref, places=6)
        self.assertAlmostEqual(stats["groups"], 3.0, places=6)

    def test_all_tied_gives_zero_loss_and_no_update(self) -> None:
        """全部并列 → loss 0、梯度 0、参数一位不动."""
        from touhou_rl_mcp.training.policy import MLPPolicy

        policy = MLPPolicy(6, 9, hidden=(16, 16), lr=0.5)
        before = [p.copy() for p in policy._params]
        X = np.random.default_rng(1).standard_normal((16, 6))
        stats = policy.fit_listmle(X, np.zeros((16, 9)), margin=20.0)
        self.assertAlmostEqual(stats["listmle_loss"], 0.0, places=9)
        self.assertAlmostEqual(stats["grad_norm"], 0.0, places=9)
        for a, b in zip(before, policy._params):
            self.assertTrue(np.array_equal(a, b))

    def test_learns_full_order(self) -> None:
        """严格有序（无并列）时 rank_acc 应到 1."""
        from touhou_rl_mcp.training.policy import MLPPolicy

        rng = np.random.default_rng(2)
        X = rng.standard_normal((96, 10))
        W = rng.standard_normal((10, 9))
        best = np.argmax(X @ W, axis=1)
        C = rng.uniform(100.0, 400.0, size=(96, 9))
        C[np.arange(96), best] = 0.0
        # ListMLE 的稳定 lr 上限比 fit_rank 低: 实测 1e-3 收敛到 1.00,
        # 2e-3 会过冲 (200 步 1.00 -> 500 步 0.77)。这里按实测的安全值取。
        policy = MLPPolicy(10, 9, hidden=(64, 64), lr=1e-3)
        for _ in range(350):
            stats = policy.fit_listmle(X, C, margin=20.0)
        self.assertGreater(stats["rank_acc"], 0.9)
        pred = np.argmax(policy.logits_batch(X), axis=1)
        self.assertGreater(float(np.mean(pred == best)), 0.9)

    def test_tied_group_stays_tight(self) -> None:
        """并列最优的两个动作之间不该被拉开（RankNet 当年的坑是 1.1）."""
        from touhou_rl_mcp.training.policy import MLPPolicy

        rng = np.random.default_rng(3)
        X = rng.standard_normal((64, 8))
        C = np.full((64, 9), 300.0)
        C[:, [0, 4]] = 0.0
        C[:, 1] = 120.0
        policy = MLPPolicy(8, 9, hidden=(32, 32), lr=3e-3)
        for it in range(200):
            policy.fit_listmle(X, C, margin=20.0, seed=it)
        logits = policy.logits_batch(X)
        self.assertLess(float(np.abs(logits[:, 0] - logits[:, 4]).max()), 0.6)
        # 组间仍必须拉开
        self.assertGreater(float((logits[:, [0, 4]].max(axis=1) - logits[:, 1]).min()), 1.0)

    def test_candidate_sampling_runs_and_respects_k(self) -> None:
        """候选采样: 只取 K 个动作参与 loss, 组数不超过 K, 且不崩."""
        from touhou_rl_mcp.training.policy import MLPPolicy

        rng = np.random.default_rng(4)
        X = rng.standard_normal((32, 8))
        C = rng.uniform(0.0, 900.0, size=(32, 18))
        policy = MLPPolicy(8, 18, hidden=(32, 32), lr=1e-3)
        stats = policy.fit_listmle(X, C, margin=20.0, candidates=4, seed=1)
        self.assertLessEqual(stats["groups"], 4.0)
        self.assertEqual(stats["labeled"], 32)
        self.assertGreater(stats["grad_norm"], 0.0)

    def test_position_regularizer_pushes_away_from_top(self) -> None:
        """贴顶时对"继续向上"加正则 → 训练后向上动作概率必须下降."""
        from touhou_rl_mcp.training.policy import MLPPolicy
        from touhou_rl_mcp.training.mpc import DIRECTIONS

        up = [i for i, (_dx, dy) in enumerate(DIRECTIONS) if dy < 0]
        self.assertTrue(up)
        rng = np.random.default_rng(5)
        X = rng.standard_normal((64, 8))
        # 老师代价故意"平坦"（贴顶时它也不敢确定），正则才是决定性的那一项
        C = np.zeros((64, 9))
        reg = np.zeros((64, 9))
        reg[:, up] = 1.0
        policy = MLPPolicy(8, 9, hidden=(32, 32), lr=3e-3)
        base = policy._softmax_rows(policy.logits_batch(X))[:, up].sum(axis=1).mean()
        for _ in range(150):
            policy.fit_listmle(X, C, margin=20.0, posreg=reg, posreg_weight=3.0)
        after = policy._softmax_rows(policy.logits_batch(X))[:, up].sum(axis=1).mean()
        self.assertLess(float(after), float(base) * 0.5)
