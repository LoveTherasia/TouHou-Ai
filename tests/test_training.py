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
