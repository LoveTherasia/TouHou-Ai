"""本次改造的回归测试:

* 奖励: 无敌帧期间不再白给"分数/Boss伤害/存活" (封堵被弹刷输出)
* 奖励: 被弹惩罚显著加重 (防止被弹优先)
* mock: 多符卡 Boss (独立血量/弹型/位置/移动) + 击破事件 + 过面
* 课程学习: start_stage
* 动作空间: focus 提供"9 快 + 9 慢"
* 算法: PPO 更新数值有限
"""

from __future__ import annotations

import unittest

import numpy as np

from touhou_rl_mcp.backends.mock import MockGame
from touhou_rl_mcp.config import Config, DEFAULT_CONFIG
from touhou_rl_mcp.env import TouhouEnv
from touhou_rl_mcp.games.registry import GameRegistry
from touhou_rl_mcp.rl.reward import RewardModel
from touhou_rl_mcp.training.policy import MLPPolicy
from touhou_rl_mcp.training.trainer import build_action_table
from touhou_rl_mcp.types import GameState


def make_env() -> TouhouEnv:
    return TouhouEnv(Config({"backend": "mock", "game": "th11", "step_frames": 4}),
                     registry=GameRegistry())


class TestInvincibleExploit(unittest.TestCase):
    def setUp(self) -> None:
        self.reward = RewardModel({
            "boss_damage_scale": 1.0, "score_scale": 1.0,
            "alive_bonus": 1.0, "invincible_output_scale": 0.0,
        })

    def _state(self, *, score=0, hp=1000, invincible=0):
        return GameState(stage=1, in_game=True, lives=2, score=score,
                         boss_hp=hp, boss_max_hp=1000, invincible=invincible,
                         bullet_count=0, enemies_alive=0, raw={})

    def test_invincible_blocks_output_and_alive_reward(self) -> None:
        prev = self._state()
        cur = self._state(score=100, hp=900, invincible=100)
        parts = self.reward.compute(prev, cur)[1]
        self.assertNotIn("boss_damage", parts)
        self.assertNotIn("score", parts)
        self.assertNotIn("alive", parts)

    def test_normal_step_still_rewards_output(self) -> None:
        prev = self._state()
        cur = self._state(score=100, hp=900, invincible=0)
        parts = self.reward.compute(prev, cur)[1]
        self.assertEqual(parts["boss_damage"], 100.0)
        self.assertEqual(parts["alive"], 1.0)

    def test_fallback_window_after_miss(self) -> None:
        # 真机未标定 invincible (None) 时, 用"掉残后 N 帧"推断无敌窗口
        reward = RewardModel({"boss_damage_scale": 1.0, "invincible_output_scale": 0.0,
                              "invincible_frames": 150})
        a = GameState(lives=2, frame=10, boss_hp=1000, boss_max_hp=1000, in_game=True)
        b = GameState(lives=1, frame=14, boss_hp=950, boss_max_hp=1000, in_game=True)
        self.assertNotIn("boss_damage", reward.compute(a, b)[1])   # 掉残当步
        c = GameState(lives=1, frame=20, boss_hp=900, boss_max_hp=1000, in_game=True)
        self.assertNotIn("boss_damage", reward.compute(b, c)[1])   # 窗口内
        d = GameState(lives=1, frame=200, boss_hp=800, boss_max_hp=1000, in_game=True)
        self.assertIn("boss_damage", reward.compute(c, d)[1])      # 窗口外恢复

    def test_miss_penalty_is_prioritized(self) -> None:
        weights = DEFAULT_CONFIG["reward"]
        self.assertLessEqual(weights["miss_penalty"], -100.0)
        self.assertLess(weights["death_penalty"], weights["miss_penalty"])


class TestMockBossSpells(unittest.TestCase):
    def setUp(self) -> None:
        self.env = make_env()
        self.env.start()
        self.addCleanup(self.env.close)

    def test_boss_has_multiple_spells_and_stage_advances(self) -> None:
        self.env.reset(seed=3)
        game = self.env.backend.game
        for _ in range(600):
            self.env.step("shoot", frames=4)
            if game.boss is not None:
                break
        self.assertIsNotNone(game.boss, "应能刷出 Boss")
        boss = game.boss
        boss.entering = False  # 跳过入场动画, 直接测试符卡击破
        n_spells = len(boss.spells)
        self.assertGreaterEqual(n_spells, 2, "Boss 应有多张符卡")
        # 每张符卡独立血量
        self.assertGreater(boss.spells[1].hp, 0)

        stage_before = game.stage
        boss.hp = 0
        game.tick(set())
        st = game.state()
        self.assertTrue(st.raw["spell_cleared"])
        self.assertEqual(game.spells_cleared, 1)
        for _ in range(n_spells - 1):
            game.boss.hp = 0
            game.tick(set())
        self.assertEqual(game.stage, stage_before + 1, "打空全部符卡应过面")

    def test_boss_position_and_movement_vary(self) -> None:
        xs = set()
        for seed in range(20):
            g = MockGame(seed=seed, start_stage=3)
            xs.add(g._make_boss(3).home_x)
        self.assertGreater(len(xs), 1, "Boss 刷新位置不应固定在中央")

        g = MockGame(seed=1, start_stage=3)
        boss = g._make_boss(3)
        boss.spell.move = "sine_x"
        boss.entering = False
        before = boss.x
        for _ in range(30):
            boss.timer += 1
            g._boss_move(boss)
        self.assertNotAlmostEqual(before, boss.x, places=3, msg="Boss 应该会移动")

    def test_start_stage(self) -> None:
        st = self.env.reset(seed=1, start_stage=4).state
        self.assertEqual(st.stage, 4)
        self.assertTrue(st.in_game)


class TestActionAndAlgorithm(unittest.TestCase):
    def test_focus_gives_fast_and_slow_actions(self) -> None:
        fast = build_action_table(use_focus=False)
        focused = build_action_table(use_focus=True)
        self.assertEqual(len(fast), 9)
        self.assertEqual(len(focused), 18)
        self.assertTrue(all("focus" not in a for a in focused[:9]))
        self.assertTrue(all("focus" in a for a in focused[9:]))

    def test_ppo_update_is_finite(self) -> None:
        policy = MLPPolicy(8, 3, hidden=(16,), seed=0, lr=1e-3)
        rng = np.random.default_rng(0)
        X = rng.standard_normal((64, 8))
        actions = rng.integers(0, 3, 64)
        logp, _ent, _val = policy.evaluate(X, actions)
        stats = policy.update_ppo(X, actions, logp, np.ones(64), np.zeros(64),
                                  epochs=2, minibatch=16)
        self.assertTrue(all(np.isfinite(v) for v in stats.values()))
        self.assertIn("kl", stats)

    def test_default_config_uses_focus_and_ppo(self) -> None:
        from touhou_rl_mcp.training import TrainConfig
        cfg = TrainConfig()
        self.assertTrue(cfg.use_focus)
        self.assertEqual(cfg.algorithm, "ppo")


class TestBehaviorCloningAndTeacher(unittest.TestCase):
    def test_imitate_batch_learns_separable_labels(self) -> None:
        rng = np.random.default_rng(0)
        X = rng.standard_normal((256, 6))
        A = (X[:, 0] > 0).astype(int) * 2  # 0 或 2, 可由 x[0] 的符号分开
        policy = MLPPolicy(6, 3, hidden=(32,), seed=0, lr=5e-3, entropy_coef=0.0)
        first = policy.imitate_batch(X, A)["imitation_loss"]
        last = first
        for _ in range(300):
            last = policy.imitate_batch(X, A)["imitation_loss"]
        self.assertLess(last, first)
        accuracy = float((policy.logits_batch(X).argmax(1) == A).mean())
        self.assertGreater(accuracy, 0.9)

    def test_rule_teacher_returns_valid_action(self) -> None:
        from touhou_rl_mcp.training.rule_teacher import RuleTeacher

        teacher = RuleTeacher()
        state = GameState(in_game=True, player_x=192, player_y=400,
                          bullets=[(180, 380), (200, 360), (192, 350)],
                          raw={"boss_position": (240.0, 96.0)})
        action, info = teacher.choose(state, 0, "shoot")
        self.assertIn("shoot", action)
        self.assertIn("direction", info)

        empty = GameState(in_game=True, player_x=192, player_y=400, bullets=[], raw={})
        action2, _ = teacher.choose(empty, 0, "shoot")
        self.assertIn("shoot", action2)

    def test_weighted_imitate_batch_learns(self) -> None:
        rng = np.random.default_rng(0)
        X = rng.standard_normal((128, 4))
        A = (X[:, 0] > 0).astype(int) * 2
        w = np.where(np.abs(X[:, 0]) > 1.0, 5.0, 1.0)  # 强调 |x0|>1 的样本
        policy = MLPPolicy(4, 3, hidden=(32,), seed=0, lr=5e-3, entropy_coef=0.0)
        for _ in range(200):
            policy.imitate_batch(X, A, weights=w)
        accuracy = float((policy.logits_batch(X).argmax(1) == A).mean())
        self.assertGreater(accuracy, 0.9)

    def test_danger_state_weight(self) -> None:
        from touhou_rl_mcp.training import TrainConfig, Trainer

        trainer = Trainer.__new__(Trainer)  # 只测纯函数, 不做完整初始化
        trainer.config = TrainConfig(danger_weight=3.0, danger_weight_radius=60.0)
        near = GameState(player_x=192, player_y=400, bullets=[(192, 390)])  # 10px
        far = GameState(player_x=192, player_y=400, bullets=[(192, 200)])   # 200px
        self.assertGreater(trainer._state_weight(near), 1.0)
        self.assertEqual(trainer._state_weight(far), 1.0)

    def test_danger_grad_feature(self) -> None:
        from touhou_rl_mcp.training.features import encode_state, feature_dim

        self.assertEqual(feature_dim(0, False, 0, False, False, False), 10)
        self.assertEqual(feature_dim(0, False, 0, False, False, True), 12)
        state = GameState(in_game=True, player_x=192, player_y=400,
                          bullets=[(192, 390)], raw={"backend": "mock"})
        vec = encode_state(state, k_bullets=0, use_danger_grad=True, validate=False)
        self.assertEqual(vec.shape, (12,))
        self.assertAlmostEqual(vec[10], 0.0)   # fx: 弹在正上方, 无横向分量
        self.assertGreater(vec[11], 0.0)       # fy: 斥力朝下 (远离上方弹)

    def test_grid_feature(self) -> None:
        from touhou_rl_mcp.training.features import (
            ObservationEncoder, encode_state_grid, grid_feature_dim,
            grid_observation_dim,
        )

        self.assertEqual(grid_feature_dim(4, False), 16)
        self.assertEqual(grid_feature_dim(4, True), 48)
        self.assertEqual(grid_observation_dim(4, True, False, False, False), 48 + 10)

        # 弹幕在自机正上方 40px: 应落在中心列的"上方格子"里
        state = GameState(in_game=True, player_x=192, player_y=400,
                          bullets=[(192, 360)], raw={"backend": "mock"})
        vec = encode_state_grid(state, size=16, span=110.0, use_velocity=False,
                                validate=False)
        self.assertEqual(vec.shape, (grid_feature_dim(16, False) + 10,))
        self.assertTrue(np.isfinite(vec).all())
        grid = vec[:256].reshape(16, 16)
        self.assertGreater(grid.sum(), 0.0)            # 有弹幕被写进去
        cy, cx = np.unravel_index(int(np.argmax(grid)), grid.shape)
        self.assertLess(cy, 8)                         # 在上半 (y 更小)
        self.assertIn(cx, (7, 8))                      # 接近中心列
        self.assertGreater(vec[256 + 1], 0.0)          # 全局: 自机 y 归一化

        # 范围外的弹幕不参与
        far = GameState(in_game=True, player_x=192, player_y=400,
                        bullets=[(192, 100)], raw={"backend": "mock"})
        self.assertEqual(float(encode_state_grid(far, size=16, span=110.0,
                                                 use_velocity=False, validate=False)[:256].sum()), 0.0)

        # ObservationEncoder 从保存的 config 还原 (危险图 vs 最近K弹)
        enc = ObservationEncoder.from_meta({"use_grid": True, "grid_size": 16,
                                            "grid_span": 110.0, "bullet_velocity": True,
                                            "use_invincible": True, "use_boss": True})
        self.assertTrue(enc.use_grid)
        self.assertEqual(enc.dim, grid_observation_dim(16, True, True, True, False))
        self.assertEqual(enc(state).shape, (enc.dim,))
        plain = ObservationEncoder.from_meta({"k_bullets": 16, "bullet_velocity": True,
                                              "k_enemies": 8, "use_invincible": True,
                                              "use_boss": True})
        self.assertFalse(plain.use_grid)
        self.assertEqual(plain.dim, 10 + 1 + 2 + 16 * 6 + 8 * 4)

    def test_multiscale_grid(self) -> None:
        from touhou_rl_mcp.training.features import (
            ObservationEncoder, encode_state_grid, grid_feature_dim_scales,
            grid_observation_dim, grid_scales_from,
        )

        self.assertEqual(grid_scales_from(""), [(16, 110.0)])
        self.assertEqual(grid_scales_from("12:90,12:330"), [(12, 90.0), (12, 330.0)])
        scales = [(12, 90.0), (12, 330.0)]
        self.assertEqual(grid_feature_dim_scales(scales, True), (144 + 144) * 3)
        self.assertEqual(grid_observation_dim(16, True, True, True, False, scales=scales),
                         (144 + 144) * 3 + 13)

        state = GameState(in_game=True, player_x=192, player_y=400,
                          bullets=[(192, 360), (192, 120)], raw={"backend": "mock"})
        vec = encode_state_grid(state, scales=scales, use_velocity=False, validate=False)
        self.assertEqual(vec.shape, ((144 + 144) + 10,))
        near = vec[:144].reshape(12, 12)     # 近场: 只看到 40px 那颗
        far = vec[144:288].reshape(12, 12)   # 远场: 两颗都看得到
        self.assertGreater(near.sum(), 0.0)
        self.assertGreater(far.sum(), 0.0)
        self.assertGreater(far.sum(), near.sum())

        enc = ObservationEncoder.from_meta({"use_grid": True,
                                            "grid_scales": "12:90,12:330",
                                            "bullet_velocity": True,
                                            "use_invincible": True, "use_boss": True})
        self.assertEqual(enc.scales, scales)
        self.assertEqual(enc.dim, (144 + 144) * 3 + 13)
        self.assertEqual(enc(state).shape, (enc.dim,))


class TestDanmakuScript(unittest.TestCase):
    def test_load_script_normalizes_frames(self) -> None:
        import json
        import tempfile
        from pathlib import Path

        from touhou_rl_mcp.memory.danmaku import load_script

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "s.json"
            p.write_text(json.dumps({"frames": 10, "spawns": [
                [144, 1.0, 2.0, 0.5, 0.5], [147, 3.0, 4.0, 0.1, 0.2]]}), encoding="utf-8")
            data = load_script(p)
        self.assertEqual(data["frame_base"], 144)
        self.assertIn(0, data["by_frame"])
        self.assertIn(3, data["by_frame"])
        self.assertEqual(data["length"], 4)

    def test_mock_plays_back_scripted_danmaku(self) -> None:
        from touhou_rl_mcp.backends.mock import MockGame

        script = {"length": 4, "by_frame": {0: [(100.0, 50.0, 0.0, 2.0)],
                                            2: [(200.0, 60.0, 0.0, 1.0)]}}
        game = MockGame(seed=0, danmaku_script=script)
        self.assertTrue(game.scripted)
        for _ in range(4):
            game.tick(set())
        self.assertGreaterEqual(len(game.bullets), 2)  # by_frame[2] 与 [0] 各发一次

        plain = MockGame(seed=0)
        self.assertFalse(plain.scripted)

    def test_recorder_uses_tick_frames(self) -> None:
        from touhou_rl_mcp.memory.danmaku import DanmakuRecorder

        def st(frame, slots, pts):
            return GameState(frame=frame, in_game=True, bullets=pts,
                             raw={"entities": {"bullets": {"status": "ok", "active_slots": slots}}})

        rec = DanmakuRecorder()
        rec.update(st(100, [1], [(10.0, 10.0)]))
        rec.update(st(101, [1], [(10.0, 12.0)]))   # slot1 生成, v=(0,2)
        rec.update(st(5, [2], [(50.0, 50.0)]))     # 游戏帧号"重置" -> 不影响 tick 帧号
        rec.update(st(6, [2], [(52.0, 51.0)]))
        self.assertEqual(len(rec.spawns), 2)
        self.assertEqual(rec.spawns[0].frame, 1)   # 观测计数, 不是游戏帧号
        self.assertEqual(rec.spawns[0].vy, 2.0)
        self.assertEqual(rec.spawns[1].frame, 3)
        self.assertEqual(rec.spawns[1].vx, 2.0)
        self.assertEqual(rec.frame, 4)


class TestDialogueFastForward(unittest.TestCase):
    def test_is_empty_detection(self) -> None:
        from touhou_rl_mcp.inputs.dialogue import DialogueAssist

        assist = DialogueAssist(threshold=2)
        empty = GameState(in_game=True, lives=1, bullet_count=0, enemies_alive=0,
                          raw={"entities": {"bullets": {"status": "ok"},
                                            "enemies": {"status": "ok"}}})
        busy = GameState(in_game=True, lives=1, bullet_count=3, enemies_alive=2,
                         raw={"entities": {"bullets": {"status": "ok"},
                                           "enemies": {"status": "ok"}}})
        self.assertTrue(assist.is_empty(empty))
        self.assertFalse(assist.is_empty(busy))
        self.assertFalse(assist.update(busy, 60))   # 有敌人 -> 不算对话
        self.assertTrue(assist.update(empty, 2))    # 连续空场 -> 判定对话

    def test_blend_policies(self) -> None:
        from touhou_rl_mcp.training.policy import MLPPolicy, blend_policies

        a = MLPPolicy(4, 3, hidden=(8,), seed=1)
        b = MLPPolicy(4, 3, hidden=(8,), seed=2)
        m = blend_policies([a, b], [0.5, 0.5])
        self.assertEqual(m.obs_dim, 4)
        self.assertEqual(m.n_actions, 3)
        np.testing.assert_allclose(m.W[0], 0.5 * (a.W[0] + b.W[0]), rtol=1e-12)
        single = blend_policies([a], [1.0])
        np.testing.assert_allclose(single.W[0], a.W[0], rtol=1e-12)

    def test_tap_repeat_stops_early(self) -> None:
        from touhou_rl_mcp.inputs.controller import Controller, NullSink

        counter = {"frames": 0, "stops": 0}

        def advance(frames: int) -> int:
            counter["frames"] += frames
            return frames

        controller = Controller(NullSink(), advance, hz=60, game="th11")

        def stop() -> bool:
            counter["stops"] += 1
            return counter["stops"] >= 2   # 第 2 次点按后停止

        spent, taps = controller.tap_repeat(["confirm"], keep=["left"], hold=2, after=6,
                                            max_taps=10, stop=stop)
        self.assertEqual(taps, 2)
        self.assertEqual(spent, 16)             # 每次 hold2 + after6
        self.assertEqual(counter["frames"], 16)  # 敌人没出现前不会继续点


if __name__ == "__main__":
    unittest.main()
