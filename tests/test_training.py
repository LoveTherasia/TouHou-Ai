"""训练控制面测试.

覆盖两层:
  * 进程内: ``TouhouEnv`` 的锁 + ``Trainer`` 的启动/在线调参/奖励热更新/保存
  * 真实 stdio MCP 会话: ``th_train_*`` 工具能被 Agent 启动/观测/停止
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import unittest
from pathlib import Path

from touhou_rl_mcp.config import Config
from touhou_rl_mcp.env import TouhouEnv
from touhou_rl_mcp.games.registry import GameRegistry
from touhou_rl_mcp.training import TrainConfig, Trainer

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def make_env() -> TouhouEnv:
    return TouhouEnv(
        Config({"backend": "mock", "game": "th11", "step_frames": 8}),
        registry=GameRegistry(),
    )


def tiny_config(**overrides) -> TrainConfig:
    base = dict(
        episodes=2,
        max_steps=25,
        step_frames=8,
        batch_episodes=1,
        capture_every=0,
        eval_every=0,
        save_every=0,
        seed=7,
    )
    base.update(overrides)
    return TrainConfig(**base)


class TestEnvLocking(unittest.TestCase):
    def test_env_exposes_reentrant_lock(self) -> None:
        env = make_env()
        self.addCleanup(env.close)
        lock = env.lock
        self.assertTrue(hasattr(lock, "acquire") and hasattr(lock, "release"))
        self.assertIsInstance(lock, type(threading.RLock()))
        # 必须可重入: Trainer 会在 env_lock 里再调用加锁的 env.step
        with lock:
            with lock:
                pass

    def test_shared_env_is_safe_under_concurrent_access(self) -> None:
        env = make_env()
        env.start()
        self.addCleanup(env.close)
        env.reset(seed=1)

        errors: list[BaseException] = []

        def worker() -> None:
            try:
                for _ in range(50):
                    env.get_state()
                    env.configure({"reward": {"shaping_scale": 1.0}})
            except BaseException as exc:  # pragma: no cover - 失败时把异常带回去
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for thread in threads:
            thread.start()
        for _ in range(50):
            env.step("shoot", frames=4)
        for thread in threads:
            thread.join()
        self.assertFalse(errors, f"并发访问不应报错: {errors}")


class TestTrainer(unittest.TestCase):
    def test_trainer_runs_and_saves_policy(self) -> None:
        env = make_env()
        env.start()
        self.addCleanup(env.close)
        trainer = Trainer(env, tiny_config(), env_lock=env.lock)

        trainer.start()
        deadline = time.time() + 30
        while len(trainer.episodes) < 2 and trainer.running and time.time() < deadline:
            time.sleep(0.05)
        trainer.stop(save=True)

        self.assertGreaterEqual(len(trainer.episodes), 2)
        self.assertTrue(all(r.steps > 0 for r in trainer.episodes))
        self.assertGreater(trainer.total_steps, 0)
        self.assertTrue(trainer.policy_path().with_suffix(".npz").exists())

    def test_reconfigure_rebuilds_policy(self) -> None:
        env = make_env()
        env.start()
        self.addCleanup(env.close)
        trainer = Trainer(env, TrainConfig(episodes=0), env_lock=env.lock)

        out = trainer.reconfigure({"hidden": [32], "lr": 1e-3})
        self.assertTrue(out["rebuilt_policy"])
        self.assertEqual(trainer.config.hidden, (32,))
        self.assertAlmostEqual(trainer.policy.lr, 1e-3)

        with self.assertRaises(KeyError):
            trainer.reconfigure({"nope": 1})

    def test_reward_hot_reload(self) -> None:
        env = make_env()
        env.start()
        self.addCleanup(env.close)
        trainer = Trainer(env, TrainConfig(episodes=0), env_lock=env.lock)

        self.assertIs(trainer.env, env)
        env.configure({"reward": {"shaping_scale": 0.25, "danger_scale": 1.5}})
        self.assertAlmostEqual(env.reward_model.weights["shaping_scale"], 0.25)
        self.assertAlmostEqual(env.reward_model.weights["danger_scale"], 1.5)


class TestGapObservationInTrainer(unittest.TestCase):
    """缺口观测 / 缺口老师 / 基准场景接入 Trainer 的契约。"""

    def test_observation_dim_matches_the_encoder(self) -> None:
        from touhou_rl_mcp.training.features import (
            encode_observation, gap_spec_from, observation_dim,
        )
        from touhou_rl_mcp.training.occupancy import GapObservation
        from touhou_rl_mcp.types import GameState

        config = TrainConfig(use_gap=True, use_focus=False)
        spec = gap_spec_from(config)
        self.assertEqual(observation_dim(config), GapObservation(spec).dim)
        state = GameState(
            frame=100, in_game=True, player_x=192.0, player_y=416.0, lives=2,
            bullets=[(100.0, 300.0), (280.0, 320.0)],
            raw={"bullet_velocity": [(0.0, 2.0), (0.0, 2.0)]},
        )
        obs = encode_observation(state, config, prev_action=3)
        self.assertEqual(obs.shape, (observation_dim(config),))
        self.assertEqual(obs.dtype.name, "float64")
        self.assertAlmostEqual(float(obs[-9:].sum()), 1.0)   # 上一动作 one-hot
        self.assertEqual(float(obs[-9:][3]), 1.0)

    def test_gap_arms_change_the_dimension(self) -> None:
        from touhou_rl_mcp.training.features import observation_dim

        geo = observation_dim(TrainConfig(use_gap=True, gap_grid=False, use_focus=False))
        both = observation_dim(TrainConfig(use_gap=True, use_focus=False))
        self.assertGreater(both, geo)

    def test_apply_scenario_writes_pattern_fields(self) -> None:
        from touhou_rl_mcp.training.trainer import apply_scenario

        env = TouhouEnv(
            Config({"backend": "mock", "game": "th11", "danmaku_pattern": "wall_gate"}),
            registry=GameRegistry(),
        )
        env.start()
        self.addCleanup(env.close)
        env.reset(seed=3)
        apply_scenario(env, {"gap_width": 46.0, "interval": 95.0, "gap_phase": 0.27})
        pattern = env.backend.game.pattern
        self.assertAlmostEqual(pattern.gap_width, 46.0)
        self.assertEqual(pattern.interval, 95)      # 按声明类型转回 int
        self.assertAlmostEqual(pattern.gap_phase, 0.27)
        with self.assertRaises(KeyError):
            apply_scenario(env, {"nope": 1.0})

    def test_gap_trainer_reaches_an_episode(self) -> None:
        """教师 + 缺口观测 + 场景参数一起跑一局 (冒烟, 不验证学习效果)。"""
        from touhou_rl_mcp.training.trainer import apply_scenario

        env = TouhouEnv(
            Config({"backend": "mock", "game": "th11", "danmaku_pattern": "wall_gate",
                    "step_frames": 6, "max_episode_frames": 0}),
            registry=GameRegistry(),
        )
        env.start()
        self.addCleanup(env.close)
        config = tiny_config(
            episodes=1, max_steps=12, step_frames=6, use_focus=False,
            use_gap=True, gap_grid=False, teacher="gap", gap_label="slack",
            scenario={"interval": 95.0, "gap_width": 46.0},
            scenario_random={"gap_phase": [0.0, 1.0]},
        )
        trainer = Trainer(env, config, env_lock=env.lock)
        self.assertEqual(trainer.obs_dim, 70)
        self.assertEqual(len(trainer.action_table), 9)

        trainer.start()
        deadline = time.time() + 60
        while len(trainer.episodes) < 1 and trainer.running and time.time() < deadline:
            time.sleep(0.05)
        trainer.stop(save=False)

        self.assertIsNone(trainer.error, trainer.error)
        self.assertEqual(len(trainer.episodes), 1)
        self.assertEqual(trainer.episodes[0].steps, 12)
        self.assertAlmostEqual(env.backend.game.pattern.gap_width, 46.0)
        self.assertTrue(any(k.startswith("shoot") for k in trainer.episodes[0].actions))


class TestValueScale(unittest.TestCase):
    """价值回归必须按整批回报的量级缩放, 否则平方损失会冲垮共享隐层。"""

    def test_gae_is_linear_in_the_scale(self) -> None:
        import numpy as np

        from touhou_rl_mcp.training.trainer import Trajectory

        traj = Trajectory(
            observations=[], actions=[], logps=[], values=[10.0] * 20,
            rewards=[100.0] * 20, dones=[False] * 20, bootstrap_value=10.0,
        )
        adv_raw, ret_raw = Trainer._gae(traj, 0.99, 0.95)
        adv_scaled, ret_scaled = Trainer._gae(traj, 0.99, 0.95, scale=100.0)
        self.assertTrue(np.allclose(adv_scaled, adv_raw / 100.0))
        self.assertTrue(np.allclose(ret_scaled, ret_raw / 100.0))
        # 缩放后价值目标从 O(1e3) 降到 O(10), 平方损失从 1e6 降到 1e2 量级
        self.assertLess(float(np.abs(ret_scaled).max()),
                        0.05 * float(np.abs(ret_raw).max()))

    def test_value_scale_uses_the_trajectory_magnitude(self) -> None:
        from touhou_rl_mcp.training.trainer import Trajectory

        small = Trajectory(observations=[], actions=[], logps=[], values=[0.0],
                           rewards=[1.0] * 10, dones=[False] * 10)
        big = Trajectory(observations=[], actions=[], logps=[], values=[0.0],
                         rewards=[100.0] * 10, dones=[False] * 10)
        self.assertEqual(Trainer._value_scale([small]), 10.0)
        self.assertEqual(Trainer._value_scale([big]), 1000.0)


class TestRewardProfiles(unittest.TestCase):
    def test_gap_survive_turns_off_the_repulsion_term(self) -> None:
        from touhou_rl_mcp.rl.reward import reward_profile

        weights = reward_profile("gap_survive")
        # 斥力项教的是"离弹越远越好" = 缩墙角, 必须关掉
        self.assertEqual(weights["danger_scale"], 0.0)
        # profile 是整份覆盖: 少了整体缩放, 总奖励会恒为 0
        self.assertEqual(weights["shaping_scale"], 1.0)
        self.assertGreater(weights["alive_bonus"], 0.0)
        self.assertGreater(weights["graze_scale"], 0.0)
        self.assertLess(weights["miss_penalty"], 0.0)

    def test_gap_margin_shaping_is_a_potential(self) -> None:
        from touhou_rl_mcp.rl.reward import RewardModel, reward_profile
        from touhou_rl_mcp.types import GameState

        weights = {**reward_profile("gap_survive"), "gap_margin_scale": 2.0}
        model = RewardModel(weights)
        before = GameState(frame=10, in_game=True, player_x=192.0, player_y=416.0,
                           lives=2, bullets=[(192.0, 300.0)],
                           raw={"bullet_velocity": [(0.0, 2.0)]})
        after = GameState(frame=16, in_game=True, player_x=48.0, player_y=416.0,
                          lives=2, bullets=[(192.0, 312.0)],
                          raw={"bullet_velocity": [(0.0, 2.0)]})
        first = model.compute(before, before)[1]
        self.assertNotIn("gap", first)          # 第一帧只建立基线
        parts = model.compute(before, after)[1]
        # 从贴墙挪进缺口 -> φ 上升 -> 塑形项为正
        self.assertGreater(parts.get("gap", 0.0), 0.0)


def _params():
    from mcp.client.stdio import StdioServerParameters

    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "touhou_rl_mcp", "--backend", "mock", "--game", "th11"],
        cwd=str(PROJECT_ROOT),
        env=env,
    )


def _text(result) -> dict:
    for block in result.content:
        if getattr(block, "type", None) == "text":
            return json.loads(block.text)
    raise AssertionError(f"工具没有返回文本块: {result.content}")


def _kinds(result) -> list[str]:
    return [getattr(b, "type", None) for b in result.content]


class TestTrainingTools(unittest.TestCase):
    """像真实 Agent 一样, 通过 stdio MCP 驱动训练。"""

    def setUp(self) -> None:
        import anyio

        self.anyio = anyio

    def test_train_loop_over_mcp(self) -> None:
        from mcp import ClientSession
        from mcp.client.stdio import stdio_client

        async def scenario() -> None:
            async with stdio_client(_params()) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()

                    tools = {t.name for t in (await session.list_tools()).tools}
                    for name in (
                        "th_train_start",
                        "th_train_stop",
                        "th_train_pause",
                        "th_train_status",
                        "th_train_dashboard",
                        "th_train_configure",
                        "th_train_policy",
                    ):
                        self.assertIn(name, tools)

                    start = _text(
                        await session.call_tool(
                            "th_train_start",
                            {
                                "config": {
                                    "episodes": 1,
                                    "max_steps": 20,
                                    "step_frames": 8,
                                    "batch_episodes": 1,
                                    "capture_every": 0,
                                    "eval_every": 0,
                                    "save_every": 0,
                                    "difficulty": 1,
                                }
                            },
                        )
                    )
                    self.assertTrue(start["ok"])

                    done = False
                    for _ in range(400):
                        status = _text(await session.call_tool("th_train_status", {"tail": 2}))
                        if status.get("episodes_done", 0) >= 1:
                            done = True
                            break
                        await self.anyio.sleep(0.05)
                    self.assertTrue(done, "训练应在限时内完成至少 1 局")

                    # 在线调参 (含奖励热更新)
                    conf = _text(
                        await session.call_tool(
                            "th_train_configure",
                            {"patch": {"lr": 1e-3, "reward": {"danger_scale": 1.0}}},
                        )
                    )
                    self.assertAlmostEqual(conf["reward"]["danger_scale"], 1.0)

                    bad = await session.call_tool("th_train_configure", {"patch": {"nope": 1}})
                    self.assertTrue(bad.is_error)

                    paused = _text(await session.call_tool("th_train_pause", {"paused": True}))
                    self.assertTrue(paused["paused"])

                    dash = await session.call_tool("th_train_dashboard", {"quiet": True})
                    self.assertIn("text", _kinds(dash))

                    stop = _text(await session.call_tool("th_train_stop", {"save": True}))
                    self.assertTrue(stop["ok"])

                    described = _text(
                        await session.call_tool("th_train_policy", {"action": "describe"})
                    )
                    self.assertGreater(described["policy"]["params"], 0)

        self.anyio.run(scenario)


if __name__ == "__main__":
    unittest.main()
