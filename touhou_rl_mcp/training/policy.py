"""策略网络 (纯 numpy, 不依赖 torch).

两层 MLP + 线性价值头, 手写前向与反向。维度只有几十维, numpy 完全够快,
好处是**零重依赖** —— 训练链路立刻就能跑起来, 不用先下 2~3GB 的 torch。

想换 torch / Stable-Baselines3 时, 见 :mod:`.gym_env` 提供的 Gymnasium 适配。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def blend_policies(policies: list["MLPPolicy"], weights: list[float] | None = None) -> "MLPPolicy":
    """按权重平均多个**同结构**策略的权重 (model soup)。

    真机进程与 mock 进程从同一个初始策略出发, 各自训练后取平均, 常常比单条更好、更稳。
    """
    if not policies:
        raise ValueError("blend_policies: 需要至少一个策略")
    n = len(policies)
    weights = [1.0] * n if weights is None else list(weights)
    total = float(sum(weights)) or 1.0
    ref = policies[0]
    out = MLPPolicy(ref.obs_dim, ref.n_actions, hidden=ref.hidden, seed=0,
                    lr=ref.lr, entropy_coef=ref.entropy_coef)
    for j in range(len(ref.W)):
        out.W[j] = sum((weights[i] / total) * p.W[j] for i, p in enumerate(policies))
        out.b[j] = sum((weights[i] / total) * p.b[j] for i, p in enumerate(policies))
    out.Wv = sum((weights[i] / total) * p.Wv for i, p in enumerate(policies))
    out.bv = sum((weights[i] / total) * p.bv for i, p in enumerate(policies))
    out._params = [*out.W, *out.b, out.Wv, out.bv]
    out._m = [np.zeros_like(p) for p in out._params]
    out._v = [np.zeros_like(p) for p in out._params]
    return out


class MLPPolicy:
    """离散动作策略: ``softmax(W3 relu(W2 relu(W1 x + b1) + b2) + b3)`` + 价值头."""

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        *,
        hidden: tuple[int, ...] = (64, 64),
        seed: int = 0,
        lr: float = 3e-3,
        entropy_coef: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self.hidden = tuple(int(h) for h in hidden)
        self.lr = float(lr)
        self.entropy_coef = float(entropy_coef)
        self.beta1, self.beta2, self.eps = beta1, beta2, eps

        rng = np.random.default_rng(seed)
        dims = [self.obs_dim, *self.hidden]
        self.W: list[np.ndarray] = []
        self.b: list[np.ndarray] = []
        for i in range(len(dims) - 1):
            fan_in = dims[i]
            self.W.append(rng.standard_normal((dims[i + 1], fan_in)) * np.sqrt(2.0 / fan_in))
            self.b.append(np.zeros(dims[i + 1]))
        # 输出层用较小的初始化, 避免一开始就很自信
        self.W.append(rng.standard_normal((self.n_actions, dims[-1])) * 0.01)
        self.b.append(np.zeros(self.n_actions))
        # 价值头
        self.Wv = np.zeros((1, dims[-1]))
        self.bv = np.zeros(1)

        self._params = [*self.W, *self.b, self.Wv, self.bv]
        self._m = [np.zeros_like(p) for p in self._params]
        self._v = [np.zeros_like(p) for p in self._params]
        self._t = 0

    # ------------------------------------------------------------------
    # 前向
    # ------------------------------------------------------------------
    def _forward(self, x: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """返回 (每层的激活值, 每层的 z=激活前)."""
        activations = [x]
        zs: list[np.ndarray] = []
        h = x
        for i in range(len(self.W) - 1):
            z = self.W[i] @ h + self.b[i]
            zs.append(z)
            h = _relu(z)
            activations.append(h)
        logits = self.W[-1] @ h + self.b[-1]
        return activations, zs + [logits]

    def logits(self, obs: np.ndarray) -> np.ndarray:
        activations, _ = self._forward(np.asarray(obs, dtype=np.float64))
        h_last = activations[-1]
        return self.W[-1] @ h_last + self.b[-1]

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits)
        exp = np.exp(shifted)
        return exp / np.sum(exp)

    def action_probs(self, obs: np.ndarray) -> np.ndarray:
        return self._softmax(self.logits(obs))

    def sample(self, obs: np.ndarray, rng: np.random.Generator) -> tuple[int, float, float]:
        """采样动作, 返回 ``(动作, logπ(a|s), 价值)``."""
        activations, _ = self._forward(np.asarray(obs, dtype=np.float64))
        h_last = activations[-1]
        probs = self._softmax(self.W[-1] @ h_last + self.b[-1])
        action = int(rng.choice(self.n_actions, p=probs))
        logp = float(np.log(max(probs[action], 1e-12)))
        value = float((self.Wv @ h_last + self.bv).item())
        return action, logp, value

    def greedy(self, obs: np.ndarray) -> int:
        return int(np.argmax(self.logits(obs)))

    def value(self, obs: np.ndarray) -> float:
        activations, _ = self._forward(np.asarray(obs, dtype=np.float64))
        return float((self.Wv @ activations[-1] + self.bv).item())

    # ------------------------------------------------------------------
    # 批量前向 (PPO 用; 比逐样本循环快很多)
    # ------------------------------------------------------------------
    def _forward_batch(self, X: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
        """``X``: (N, obs_dim) -> (每层激活, 每层 z, logits(N, A))."""
        activations = [X]
        zs: list[np.ndarray] = []
        h = X
        for i in range(len(self.W) - 1):
            z = h @ self.W[i].T + self.b[i]
            zs.append(z)
            h = np.maximum(z, 0.0)
            activations.append(h)
        logits = h @ self.W[-1].T + self.b[-1]
        return activations, zs, logits

    def logits_batch(self, X: np.ndarray) -> np.ndarray:
        return self._forward_batch(np.asarray(X, dtype=np.float64))[2]

    def widen_input(self, new_dim: int) -> "MLPPolicy":
        """把输入层从 ``obs_dim`` 扩到 ``new_dim``: **新列权重为 0**。

        为什么是零: 记忆里旧策略的每一列权重都对应一个旧特征, 新特征追加在观测
        **末尾**, 所以只要把新列置 0, 扩展后的网络对旧观测的输出**逐位相同** ——
        等于"从原策略出发, 再学会用新的几列", 而不是从头训。

        Adam 的一阶/二阶矩同步扩维 (新列为 0): 否则新列的动量会从旧列的尺度起步,
        一上来就把原策略冲坏。
        """
        new_dim = int(new_dim)
        if new_dim < self.obs_dim:
            raise ValueError(f"只能扩维: new_dim={new_dim} < obs_dim={self.obs_dim}")
        if new_dim == self.obs_dim:
            return self
        old = self.obs_dim
        W0 = np.zeros((self.W[0].shape[0], new_dim), dtype=self.W[0].dtype)
        W0[:, :old] = self.W[0]
        m0 = np.zeros_like(W0)
        v0 = np.zeros_like(W0)
        m0[:, :old] = self._m[0]
        v0[:, :old] = self._v[0]
        self.W[0] = W0
        self._m[0] = m0
        self._v[0] = v0
        self.obs_dim = new_dim
        self._params = [*self.W, *self.b, self.Wv, self.bv]
        return self

    def snapshot_params(self) -> list[np.ndarray]:
        """冻结一份当前参数, 作为 RL 微调的参考策略 (KL 锚)。"""
        return [p.copy() for p in self._params]

    def _logits_with_params(self, X: np.ndarray, params: list[np.ndarray]) -> np.ndarray:
        """用另一套参数做前向 (参数布局与 ``_params`` 一致: W..., b..., Wv, bv)。"""
        n = len(self.W)
        h = X
        for i in range(n - 1):
            h = np.maximum(h @ params[i].T + params[n + i], 0.0)
        return h @ params[n - 1].T + params[2 * n - 1]

    @staticmethod
    def _softmax_rows(logits: np.ndarray) -> np.ndarray:
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)

    def evaluate(self, X: np.ndarray, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """批量: 返回 ``(logπ(a|s), 熵, 价值)``."""
        activations, _zs, logits = self._forward_batch(np.asarray(X, dtype=np.float64))
        probs = self._softmax_rows(logits)
        a = np.asarray(actions, dtype=np.int64)
        logp = np.log(np.maximum(probs[np.arange(len(a)), a], 1e-12))
        entropy = -np.sum(probs * np.log(np.maximum(probs, 1e-12)), axis=1)
        values = activations[-1] @ self.Wv[0] + self.bv[0]
        return logp, entropy, values

    # ------------------------------------------------------------------
    # PPO (clipped surrogate + GAE 优势; 纯 numpy)
    # ------------------------------------------------------------------
    def update_ppo(
        self,
        obs_batch: np.ndarray,
        actions: np.ndarray,
        old_logp: np.ndarray,
        advantages: np.ndarray,
        returns: np.ndarray,
        *,
        clip_ratio: float = 0.2,
        epochs: int = 4,
        minibatch: int = 256,
        value_coef: float = 0.5,
        entropy_coef: float | None = None,
        max_grad_norm: float = 5.0,
        target_kl: float = 0.0,
        seed: int | None = None,
        ref_params: list[np.ndarray] | None = None,
        kl_coef: float = 0.0,
    ) -> dict[str, float]:
        X = np.asarray(obs_batch, dtype=np.float64)
        a = np.asarray(actions, dtype=np.int64)
        old_logp = np.asarray(old_logp, dtype=np.float64)
        adv = np.asarray(advantages, dtype=np.float64)
        ret = np.asarray(returns, dtype=np.float64)
        n = X.shape[0]
        if n == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
                    "grad_norm": 0.0, "kl": 0.0, "kl_ref": 0.0, "clip_frac": 0.0}
        coef = self.entropy_coef if entropy_coef is None else float(entropy_coef)
        rng = np.random.default_rng(self._t if seed is None else seed)
        mb = max(1, min(int(minibatch), n))

        agg = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
               "grad_norm": 0.0, "kl": 0.0, "kl_ref": 0.0, "clip_frac": 0.0}
        updates = 0
        stop = False
        for _ in range(max(1, int(epochs))):
            if stop:
                break
            order = rng.permutation(n)
            for start in range(0, n, mb):
                idx = order[start:start + mb]
                stats = self._ppo_step(
                    X[idx], a[idx], old_logp[idx], adv[idx], ret[idx],
                    clip_ratio=clip_ratio, value_coef=value_coef,
                    entropy_coef=coef, max_grad_norm=max_grad_norm,
                    ref_params=ref_params, kl_coef=kl_coef,
                )
                for k in agg:
                    agg[k] += stats.get(k, 0.0)
                updates += 1
                # KL 早停: 策略偏离旧策略太远就停止本批更新 (防塌缩/退化)
                if target_kl and stats.get("kl", 0.0) > target_kl:
                    stop = True
                    break
        if updates:
            for k in agg:
                agg[k] /= updates
        return {k: round(v, 6) for k, v in agg.items()}

    def _ppo_step(
        self,
        X: np.ndarray,
        a: np.ndarray,
        old_logp: np.ndarray,
        adv: np.ndarray,
        ret: np.ndarray,
        *,
        clip_ratio: float,
        value_coef: float,
        entropy_coef: float,
        max_grad_norm: float,
        ref_params: list[np.ndarray] | None = None,
        kl_coef: float = 0.0,
    ) -> dict[str, float]:
        m = X.shape[0]
        activations, _zs, logits = self._forward_batch(X)
        h_last = activations[-1]
        probs = self._softmax_rows(logits)
        row = np.arange(m)
        logp = np.log(np.maximum(probs[row, a], 1e-12))
        entropy = -np.sum(probs * np.log(np.maximum(probs, 1e-12)), axis=1)
        values = h_last @ self.Wv[0] + self.bv[0]

        ratio = np.exp(logp - old_logp)
        lo, hi = 1.0 - clip_ratio, 1.0 + clip_ratio
        # min(r*A, clip(r)*A) 走的是哪一支 (只有未裁剪那支对 logp 有梯度)
        unclipped_is_min = np.where(adv >= 0, ratio <= hi, ratio >= lo)
        grad_logp = np.where(unclipped_is_min, -adv * ratio, 0.0)

        # dL/dlogits = grad_logp*(onehot - p) + entropy_coef*p*(log p + H)
        grad_logits = -grad_logp[:, None] * probs
        grad_logits[row, a] += grad_logp
        grad_logits += entropy_coef * probs * (np.log(np.maximum(probs, 1e-12)) + entropy[:, None])

        # KL 锚: 把策略拴在"开始 RL 之前"的那份参数上。
        #
        # 实测 RL 阶段最大的问题不是学不会, 而是**忘掉** —— DAgger 已经到 1700+,
        # 一段 PPO 之后回落到 1550。奖励只关心"多活几步", 而缺口感知是靠 60~160 局
        # 监督堆出来的, 两边的梯度强度差一个量级。加一项 ``β·KL(π_ref ‖ π_θ)`` 之后,
        # 策略只能在参考策略附近做局部修正, 监督学到的东西不会被一批样本抹掉。
        # KL 对 logits 的梯度是 ``π_θ - π_ref`` (逐样本), 与下面按批求和再取平均的
        # 缩放一致。
        kl_ref = 0.0
        if ref_params is not None and kl_coef > 0.0:
            ref_probs = self._softmax_rows(self._logits_with_params(X, ref_params))
            grad_logits += kl_coef * (probs - ref_probs)
            kl_ref = float(np.mean(np.sum(
                ref_probs * (np.log(np.maximum(ref_probs, 1e-12))
                             - np.log(np.maximum(probs, 1e-12))), axis=1)))

        policy_loss = float(-np.minimum(ratio * adv, np.clip(ratio, lo, hi) * adv).mean())
        value_err = values - ret
        value_loss = float(0.5 * np.mean(value_err ** 2))
        grad_value = value_coef * value_err

        grads: list[np.ndarray] = [np.zeros_like(p) for p in self._params]
        gW = grads[: len(self.W)]
        gb = grads[len(self.W) : 2 * len(self.W)]
        gWv, gbv = grads[-2], grads[-1]

        gW[-1] += grad_logits.T @ h_last
        gb[-1] += grad_logits.sum(axis=0)
        gWv += (grad_value @ h_last)[None, :]
        gbv += grad_value.sum()

        delta = grad_logits @ self.W[-1] + grad_value[:, None] * self.Wv[0]
        for layer in range(len(self.W) - 2, -1, -1):
            delta = delta * (activations[layer + 1] > 0)
            gW[layer] += delta.T @ activations[layer]
            gb[layer] += delta.sum(axis=0)
            if layer > 0:
                delta = delta @ self.W[layer]

        inv = 1.0 / m
        for g in grads:
            g *= inv
        total = float(np.sqrt(sum(float(np.sum(g * g)) for g in grads)))
        if max_grad_norm and total > max_grad_norm:
            scale = max_grad_norm / (total + 1e-8)
            for g in grads:
                g *= scale
        self._adam_step(grads)

        # KL 估计量必须**非负**, 否则 ``target_kl`` 早停永远不触发。
        #
        # 原来用的是 ``mean(old_logp - logp)`` —— 逐样本看它是无偏的, 但**每一条**
        # 都可以是负数 (策略在某几个动作上概率变大), 于是均值在 0 附近来回抖, 早停
        # 形同虚设; 而 RL 阶段恰恰需要这道闸 —— 没有它, 一批样本就能把 DAgger 学到
        # 的感知冲掉。改用 Schulman 的二阶近似 ``E[(r-1) - log r]``: 对每个样本都
        # ``>= 0``, 只在 r=1 时为 0。
        safe_ratio = np.maximum(ratio, 1e-12)
        kl = float(np.mean((safe_ratio - 1.0) - np.log(safe_ratio)))
        clip_frac = float(np.mean(np.abs(ratio - 1.0) > clip_ratio))
        return {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": float(entropy.mean()),
            "grad_norm": total,
            "kl": kl,
            "kl_ref": kl_ref,
            "clip_frac": clip_frac,
        }

    # ------------------------------------------------------------------
    # 反向 + Adam
    # ------------------------------------------------------------------
    def update(
        self,
        obs_batch: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
        returns: np.ndarray,
        *,
        value_coef: float = 0.5,
        grad_clip: float = 5.0,
    ) -> dict[str, float]:
        """一次 REINFORCE + 基线 的梯度更新 (对 batch 取平均)."""
        obs_batch = np.asarray(obs_batch, dtype=np.float64)
        n = obs_batch.shape[0]
        if n == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "grad_norm": 0.0}

        dims = [self.obs_dim, *self.hidden]
        grads: list[np.ndarray] = [np.zeros_like(p) for p in self._params]
        gW = grads[: len(self.W)]
        gb = grads[len(self.W) : 2 * len(self.W)]
        gWv, gbv = grads[-2], grads[-1]

        policy_loss = 0.0
        value_loss = 0.0
        entropy_sum = 0.0

        for i in range(n):
            x = obs_batch[i]
            activations, zs = self._forward(x)
            h_last = activations[-1]
            logits = zs[-1]
            probs = self._softmax(logits)
            value = float((self.Wv @ h_last + self.bv).item())

            # dL/dlogits, 其中 L = -adv*logπ - coef*H
            #   dlogπ/dlogits = onehot - p
            #   dH/dlogits    = -p*(log p + H)
            # 所以 dL/dlogits = adv*(p - onehot) + coef*p*(log p + H)
            grad_logits = probs.copy()
            grad_logits[actions[i]] -= 1.0  # = p - onehot
            grad_logits *= advantages[i]
            entropy = -float(np.sum(probs * np.log(np.maximum(probs, 1e-12))))
            entropy_sum += entropy
            grad_logits += self.entropy_coef * probs * (np.log(np.maximum(probs, 1e-12)) + entropy)

            # 梯度裁剪 (按单样本的 logits 梯度)
            norm = float(np.linalg.norm(grad_logits))
            if norm > grad_clip:
                grad_logits *= grad_clip / norm

            policy_loss += -advantages[i] * float(np.log(max(probs[actions[i]], 1e-12)))

            gW[-1] += np.outer(grad_logits, h_last)
            gb[-1] += grad_logits

            # 价值头: 0.5 * (V - G)^2 -> dL/dV = (V - G)
            value_error = value - float(returns[i])
            value_loss += 0.5 * value_error * value_error
            dv = value_coef * value_error
            gWv += dv * h_last
            gbv += dv

            # 反传到隐藏层
            # 注意 Wv 形状是 (1, H), 必须取 Wv[0] 得到 (H,), 否则会广播成 (H,H)
            delta = self.W[-1].T @ grad_logits + self.Wv[0] * dv
            for layer in range(len(self.W) - 2, -1, -1):
                delta = delta * (activations[layer + 1] > 0)  # relu 的导数
                gW[layer] += np.outer(delta, activations[layer])
                gb[layer] += delta
                if layer > 0:
                    delta = self.W[layer].T @ delta

        scale = 1.0 / n
        for g in grads:
            g *= scale

        self._adam_step(grads)
        grad_norm = float(np.sqrt(sum(float(np.sum(g * g)) for g in grads))) * n
        return {
            "policy_loss": round(policy_loss * scale, 6),
            "value_loss": round(value_loss * scale, 6),
            "entropy": round(entropy_sum * scale, 6),
            "grad_norm": round(grad_norm, 6),
        }

    def imitate(
        self,
        obs_batch: np.ndarray,
        actions: np.ndarray,
        *,
        entropy_coef: float | None = None,
        grad_clip: float = 5.0,
    ) -> dict[str, float]:
        """行为克隆: 交叉熵拟合 ``actions`` (留给"模仿老师"用, 不做价值头)。

        用于把已经能通关的规则(如 TacticalGuard)蒸馏进策略 —— 比纯 REINFORCE 快得多。
        """
        obs_batch = np.asarray(obs_batch, dtype=np.float64)
        n = obs_batch.shape[0]
        if n == 0:
            return {"imitation_loss": 0.0, "entropy": 0.0, "grad_norm": 0.0}
        coef = self.entropy_coef if entropy_coef is None else float(entropy_coef)

        grads: list[np.ndarray] = [np.zeros_like(p) for p in self._params]
        gW = grads[: len(self.W)]
        gb = grads[len(self.W) : 2 * len(self.W)]
        loss = 0.0
        entropy_sum = 0.0
        for i in range(n):
            x = obs_batch[i]
            activations, zs = self._forward(x)
            h_last = activations[-1]
            probs = self._softmax(zs[-1])
            a = int(actions[i])
            loss += -float(np.log(max(float(probs[a]), 1e-12)))

            grad_logits = probs.copy()
            grad_logits[a] -= 1.0  # d(CE)/dlogits = p - onehot
            entropy = -float(np.sum(probs * np.log(np.maximum(probs, 1e-12))))
            entropy_sum += entropy
            grad_logits += coef * probs * (np.log(np.maximum(probs, 1e-12)) + entropy)

            norm = float(np.linalg.norm(grad_logits))
            if norm > grad_clip:
                grad_logits *= grad_clip / norm
            gW[-1] += np.outer(grad_logits, h_last)
            gb[-1] += grad_logits

            delta = self.W[-1].T @ grad_logits
            for layer in range(len(self.W) - 2, -1, -1):
                delta = delta * (activations[layer + 1] > 0)
                gW[layer] += np.outer(delta, activations[layer])
                gb[layer] += delta
                if layer > 0:
                    delta = self.W[layer].T @ delta

        scale = 1.0 / n
        for g in grads:
            g *= scale
        self._adam_step(grads)
        grad_norm = float(np.sqrt(sum(float(np.sum(g * g)) for g in grads))) * n
        return {
            "imitation_loss": round(loss * scale, 6),
            "entropy": round(entropy_sum * scale, 6),
            "grad_norm": round(grad_norm, 6),
        }

    def imitate_batch(
        self,
        obs_batch: np.ndarray,
        actions: np.ndarray,
        *,
        weights: np.ndarray | None = None,
        entropy_coef: float | None = None,
        grad_clip: float = 5.0,
        ref_params: list[np.ndarray] | None = None,
        kl_coef: float = 0.0,
    ) -> dict[str, float]:
        """行为克隆 (向量化): 加权交叉熵拟合 ``actions``。

        ``weights`` 用于**对危险状态加权** (例如离弹幕很近、刚掉残的样本),
        让策略优先学会在这些"示范里少见但致命"的状态下怎么做。

        ``actions < 0`` 表示**该行没有标签**, 只参与 KL 锚 (不参与交叉熵)。这是给
        "分层老师"用的: 它只在墙段出标签, 其余步弃权; 此时若不加锚, 共享隐层会被
        墙段的标签整体拉走, 常规弹型的能力直接崩掉 (实测平均存活 1852 -> 901 步)。

        ``ref_params`` + ``kl_coef``: 对**无标签行**加 ``kl_coef * KL(p_ref || p)``,
        梯度是 ``kl_coef * (p - p_ref) / m`` —— 语义就是"没有监督的地方别动"。
        """
        X = np.asarray(obs_batch, dtype=np.float64)
        a = np.asarray(actions, dtype=np.int64)
        m = X.shape[0]
        if m == 0:
            return {"imitation_loss": 0.0, "entropy": 0.0, "grad_norm": 0.0}
        coef = self.entropy_coef if entropy_coef is None else float(entropy_coef)
        if weights is None:
            w = np.ones(m, dtype=np.float64)
        else:
            w = np.asarray(weights, dtype=np.float64).reshape(-1)
            if w.shape[0] != m:
                w = np.ones(m, dtype=np.float64)
            w = np.maximum(w, 0.0)
        valid = a >= 0
        if not bool(valid.any()):
            # 整批都没有标签: 只剩 KL 锚 (等价于把策略拉回参考分布)
            w_ce = np.zeros(m, dtype=np.float64)
        else:
            w_ce = w * valid.astype(np.float64)
        wsum = float(w_ce.sum()) or 1.0
        wscale = (w_ce / wsum)[:, None]
        a_ce = np.where(valid, a, 0)

        activations, _zs, logits = self._forward_batch(X)
        h_last = activations[-1]
        probs = self._softmax_rows(logits)
        row = np.arange(m)
        logp = np.log(np.maximum(probs[row, a_ce], 1e-12))
        entropy = -np.sum(probs * np.log(np.maximum(probs, 1e-12)), axis=1)
        loss = float((w_ce * (-logp)).sum() / wsum)

        # dL/dlogits = w*(p - onehot) + coef*w*p*(log p + H)  (按样本权重缩放)
        grad_logits = probs.copy()
        grad_logits[row, a_ce] -= 1.0
        grad_logits += coef * probs * (np.log(np.maximum(probs, 1e-12)) + entropy[:, None])
        grad_logits *= wscale

        kl_mean = 0.0
        if ref_params is not None and kl_coef > 0.0 and bool((~valid).any()):
            ref_logits = self._logits_with_params(X, ref_params)
            q = self._softmax_rows(ref_logits)
            kl_rows = np.sum(q * (np.log(np.maximum(q, 1e-12))
                                  - np.log(np.maximum(probs, 1e-12))), axis=1)
            mask = (~valid).astype(np.float64)
            n_kl = float(mask.sum()) or 1.0
            grad_logits += (kl_coef / n_kl) * mask[:, None] * (probs - q)
            kl_mean = float((mask * kl_rows).sum() / n_kl)

        grads: list[np.ndarray] = [np.zeros_like(p) for p in self._params]
        gW = grads[: len(self.W)]
        gb = grads[len(self.W) : 2 * len(self.W)]
        gW[-1] += grad_logits.T @ h_last
        gb[-1] += grad_logits.sum(axis=0)
        delta = grad_logits @ self.W[-1]
        for layer in range(len(self.W) - 2, -1, -1):
            delta = delta * (activations[layer + 1] > 0)
            gW[layer] += delta.T @ activations[layer]
            gb[layer] += delta.sum(axis=0)
            if layer > 0:
                delta = delta @ self.W[layer]

        total = float(np.sqrt(sum(float(np.sum(g * g)) for g in grads)))
        if grad_clip and total > grad_clip:
            scale = grad_clip / (total + 1e-8)
            for g in grads:
                g *= scale
        self._adam_step(grads)
        return {
            "imitation_loss": round(loss, 6),
            "entropy": round(float((w * entropy).sum() / wsum), 6),
            "grad_norm": round(total, 6),
            "labeled": int(valid.sum()),
            "kl_base": round(kl_mean, 6),
        }

    def _adam_step(self, grads: list[np.ndarray]) -> None:
        self._t += 1
        corr1 = 1.0 - self.beta1**self._t
        corr2 = 1.0 - self.beta2**self._t
        for i, (param, grad) in enumerate(zip(self._params, grads)):
            self._m[i] = self.beta1 * self._m[i] + (1 - self.beta1) * grad
            self._v[i] = self.beta2 * self._v[i] + (1 - self.beta2) * (grad * grad)
            m_hat = self._m[i] / corr1
            v_hat = self._v[i] / corr2
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        blob: dict[str, Any] = {
            "obs_dim": self.obs_dim,
            "n_actions": self.n_actions,
            "hidden": list(self.hidden),
            "lr": self.lr,
            "entropy_coef": self.entropy_coef,
            "adam_t": self._t,
        }
        for i, (w, b) in enumerate(zip(self.W, self.b)):
            blob[f"W{i}"] = w
            blob[f"b{i}"] = b
        blob["Wv"] = self.Wv
        blob["bv"] = self.bv
        return blob

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez(target.with_suffix(".npz"), **self.state_dict())
        return target.with_suffix(".npz")

    @classmethod
    def load(cls, path: str | Path) -> "MLPPolicy":
        data = np.load(Path(path), allow_pickle=False)
        policy = cls(
            int(data["obs_dim"]),
            int(data["n_actions"]),
            hidden=tuple(int(h) for h in data["hidden"]),
            lr=float(data["lr"]),
            entropy_coef=float(data["entropy_coef"]),
        )
        index = 0
        while f"W{index}" in data:
            policy.W[index] = data[f"W{index}"]
            policy.b[index] = data[f"b{index}"]
            index += 1
        policy.Wv = data["Wv"]
        policy.bv = data["bv"]
        policy._t = int(data["adam_t"]) if "adam_t" in data else 0
        policy._params = [*policy.W, *policy.b, policy.Wv, policy.bv]
        policy._m = [np.zeros_like(p) for p in policy._params]
        policy._v = [np.zeros_like(p) for p in policy._params]
        return policy

    def describe(self) -> dict[str, Any]:
        total = sum(int(p.size) for p in self._params)
        return {
            "obs_dim": self.obs_dim,
            "n_actions": self.n_actions,
            "hidden": list(self.hidden),
            "params": total,
            "lr": self.lr,
            "entropy_coef": self.entropy_coef,
            "updates": self._t,
        }

    def to_json(self) -> str:
        return json.dumps(self.describe(), ensure_ascii=False)
