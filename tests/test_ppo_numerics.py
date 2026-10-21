"""PPO 数值正确性的回归测试.

这里锁住两个曾经真实存在、且会让"PPO 微调必然退化"的 bug:

1. **GAE 的价值被重复缩放** —— ``values`` 已经是"缩放后的回报"量纲, 再除一次
   scale 会把基准压到 ~0, 优势退化成"逐步即时奖励", 策略只贪眼前一步;
2. **load 之后策略的 lr 仍是 checkpoint 里的值** —— 命令行 ``--lr 3e-5`` 完全不生效,
   实测所有 PPO 实验都以 10~30 倍学习率在跑。
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from touhou_rl_mcp.training.policy import MLPPolicy
from touhou_rl_mcp.training.trainer import Trajectory, Trainer


def traj(rewards, values, *, bootstrap=0.0, dones=None) -> Trajectory:
    n = len(rewards)
    return Trajectory(
        observations=[np.zeros(3) for _ in range(n)],
        actions=[0] * n,
        logps=[0.0] * n,
        values=list(values),
        rewards=list(rewards),
        dones=list(dones) if dones is not None else [False] * n,
        bootstrap_value=bootstrap,
    )


class GaeUnitsTest(unittest.TestCase):
    def test_discounted_sum_of_scaled_rewards(self) -> None:
        # gamma=lam=1, V=0, scale=1: adv[t] = 剩余奖励之和
        adv, ret = Trainer._gae(traj([1.0, 1.0, 1.0], [0.0, 0.0, 0.0]), 1.0, 1.0, scale=1.0)
        np.testing.assert_allclose(adv, [3.0, 2.0, 1.0])
        np.testing.assert_allclose(ret, adv)

    def test_bootstrap_is_not_divided_twice(self) -> None:
        adv, _ = Trainer._gae(
            traj([1.0, 1.0], [0.0, 0.0], bootstrap=5.0), 1.0, 1.0, scale=1.0)
        np.testing.assert_allclose(adv, [7.0, 6.0])

    def test_values_keep_their_own_scale(self) -> None:
        """scale=10 时奖励被除, 但 values 不能再除 (这里 V=2 就是 2, 不是 0.2)."""
        adv, ret = Trainer._gae(
            traj([10.0], [2.0]), gamma=0.0, lam=0.0, scale=10.0)
        np.testing.assert_allclose(adv, [-1.0])   # 1 - 2
        np.testing.assert_allclose(ret, [1.0])    # adv + values = 目标

    def test_terminal_step_has_no_bootstrap(self) -> None:
        adv, _ = Trainer._gae(
            traj([1.0, 1.0], [0.0, 0.0], bootstrap=100.0, dones=[False, True]),
            1.0, 1.0, scale=1.0)
        np.testing.assert_allclose(adv, [2.0, 1.0])


class LoadHyperparamsTest(unittest.TestCase):
    def test_load_applies_run_lr_instead_of_checkpoint_lr(self) -> None:
        from touhou_rl_mcp.config import Config
        from touhou_rl_mcp.env import TouhouEnv
        from touhou_rl_mcp.games.registry import GameRegistry
        from touhou_rl_mcp.training import TrainConfig

        with tempfile.TemporaryDirectory() as tmp:
            env = TouhouEnv(Config({"backend": "mock", "game": "th11", "step_frames": 1}),
                            registry=GameRegistry())
            env.start()
            self.addCleanup(env.close)
            config = TrainConfig(episodes=0, hidden=(8,), lr=3e-5, entropy_coef=0.002,
                                 step_frames=1)
            trainer = Trainer(env, config, env_lock=env.lock)
            path = Path(tmp) / "policy"
            # checkpoint 里自带 1e-3 (DAgger 阶段的学习率)
            MLPPolicy(trainer.obs_dim, len(trainer.action_table), hidden=(8,),
                      lr=1e-3, entropy_coef=0.01).save(path)

            trainer.load(path)
            self.assertAlmostEqual(trainer.policy.lr, 3e-5)
            self.assertAlmostEqual(trainer.policy.entropy_coef, 0.002)


if __name__ == "__main__":
    unittest.main()
