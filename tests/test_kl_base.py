"""监督学习里的 KL 锚: 有标签的步做 CE, 没标签的步只要求"别动".

这三条都是踩出来的:

* 分层老师只在墙段出标签 (空动作 = 弃权)。若把弃权步也当作"策略自己的动作"去拟合,
  等于让策略对自己更自信 —— 熵塌缩、坏习惯固化;
* 只在该状态子集上做 CE 会把通才打崩 (平均存活 1852 -> 901 步), 所以弃权步要留下来
  做 KL 锚;
* ``actions < 0`` 是"无标签"的标记, 必须被 CE 完全忽略。
"""

from __future__ import annotations

import unittest

import numpy as np

from touhou_rl_mcp.training.policy import MLPPolicy

DIM = 8
N_ACT = 9


def make_policy(seed: int = 0) -> MLPPolicy:
    return MLPPolicy(DIM, N_ACT, hidden=(12,), seed=seed, lr=1e-2, entropy_coef=0.0)


def batch(rows: int = 32, seed: int = 1) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((rows, DIM))


def distance_to(policy: MLPPolicy, ref: list[np.ndarray]) -> float:
    return float(np.sqrt(sum(float(np.sum((a - b) ** 2))
                             for a, b in zip(policy.snapshot_params(), ref))))


class UnlabeledRowsTest(unittest.TestCase):
    def test_all_unlabeled_is_a_noop(self) -> None:
        """整批无标签且没有参考分布 -> 梯度为零, 参数不动."""
        policy = make_policy()
        before = policy.snapshot_params()
        X = batch()
        stats = policy.imitate_batch(X, np.full(len(X), -1), weights=np.ones(len(X)))
        self.assertEqual(stats["labeled"], 0)
        self.assertAlmostEqual(stats["imitation_loss"], 0.0)
        self.assertLess(distance_to(policy, before), 1e-9)

    def test_only_labeled_rows_are_fitted(self) -> None:
        """同一批数据里混入无标签行, 损失必须等于"只用有标签行"的损失."""
        policy = make_policy()
        X = batch()
        a = np.array([-1, 3, -1, 7, -1, 0, -1, 5] + [-1] * 24)
        mixed = policy.imitate_batch(X, a, weights=np.ones(len(X)))

        policy2 = make_policy()
        keep = a >= 0
        pure = policy2.imitate_batch(X[keep], a[keep], weights=np.ones(int(keep.sum())))
        self.assertEqual(mixed["labeled"], int(keep.sum()))
        self.assertAlmostEqual(mixed["imitation_loss"], pure["imitation_loss"], places=9)


def kl_to_ref(policy: MLPPolicy, X: np.ndarray, ref: list[np.ndarray]) -> float:
    q = policy._softmax_rows(policy._logits_with_params(X, ref))
    p = policy._softmax_rows(policy.logits_batch(X))
    return float(np.mean(np.sum(q * np.log(np.maximum(q, 1e-12)
                                           / np.maximum(p, 1e-12)), axis=1)))


class KlAnchorTest(unittest.TestCase):
    def test_kl_pulls_back_toward_reference(self) -> None:
        """先让策略偏离基座, 再只喂无标签行 + KL 锚 -> KL 必须单调回落到接近 0.

        **注意 Adam 的动量**: 前 3~10 步 KL 会**先升后降** (实测 0.19 -> 0.35 -> 0.007),
        因为交叉熵阶段的动量还在推。所以这个锚是"慢刹车", 不是一步回位 ——
        验收要看若干次更新的趋势, 不能看单步。
        """
        policy = make_policy(seed=3)
        ref = policy.snapshot_params()
        X = batch(64, seed=5)
        for _ in range(8):
            policy.imitate_batch(X, np.full(len(X), 4), weights=np.ones(len(X)))
        after_ce = kl_to_ref(policy, X, ref)
        self.assertGreater(after_ce, 0.05, "前提: 用完标签行之后它确实偏了")

        for _ in range(30):
            policy.imitate_batch(X, np.full(len(X), -1), weights=np.ones(len(X)),
                                 ref_params=ref, kl_coef=1.0)
        self.assertLess(kl_to_ref(policy, X, ref), after_ce / 10.0)

    def test_kl_is_reported(self) -> None:
        policy = make_policy(seed=7)
        ref = make_policy(seed=9).snapshot_params()
        X = batch(16, seed=2)
        stats = policy.imitate_batch(X, np.full(len(X), -1), weights=np.ones(len(X)),
                                     ref_params=ref, kl_coef=1.0)
        self.assertGreater(stats["kl_base"], 0.0)

    def test_no_ref_means_no_kl(self) -> None:
        policy = make_policy(seed=11)
        before = policy.snapshot_params()
        X = batch(16, seed=3)
        stats = policy.imitate_batch(X, np.full(len(X), -1), weights=np.ones(len(X)),
                                     kl_coef=1.0)
        self.assertEqual(stats["kl_base"], 0.0)
        self.assertLess(distance_to(policy, before), 1e-9)


if __name__ == "__main__":
    unittest.main()
