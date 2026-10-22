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
        A = np.asarray(actions)
        # 软标签: ``actions`` 是 ``(N, n_actions)`` 的目标分布。
        # 为什么需要它: 规划器(MPC/CEM)在"多个动作同样安全"时 argmax 会随机挑,
        # 实测老师与自己的一致率只有 ~25%、相邻帧 50~77% 会换动作 —— 硬标签
        # 等于让学生拟合噪声。把"每个动作的代价"变成分布再蒸馏就没有这个问题
        # (代价接近的动作目标概率也接近, 梯度不冲突)。
        soft: np.ndarray | None = None
        if A.ndim == 2:
            soft = np.maximum(A.astype(np.float64), 0.0)
            row_sum = soft.sum(axis=1)
            good = row_sum > 1e-9
            soft[good] /= row_sum[good, None]
            a = np.argmax(soft, axis=1).astype(np.int64)
        else:
            a = A.astype(np.int64)
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
        valid = (soft.sum(axis=1) > 1e-9) if soft is not None else (a >= 0)
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
        if soft is not None:
            target = np.where(valid[:, None], soft, 0.0)
            logp = np.log(np.maximum(probs[row, a_ce], 1e-12))
            nll = -np.sum(target * np.log(np.maximum(probs, 1e-12)), axis=1)
            entropy = -np.sum(probs * np.log(np.maximum(probs, 1e-12)), axis=1)
            loss = float((w_ce * nll).sum() / wsum)
            grad_logits = probs - target
        else:
            logp = np.log(np.maximum(probs[row, a_ce], 1e-12))
            entropy = -np.sum(probs * np.log(np.maximum(probs, 1e-12)), axis=1)
            loss = float((w_ce * (-logp)).sum() / wsum)
            grad_logits = probs.copy()
            grad_logits[row, a_ce] -= 1.0
        grad_logits = grad_logits + coef * probs * (
            np.log(np.maximum(probs, 1e-12)) + entropy[:, None])
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

    def fit_cost(
        self,
        obs_batch: np.ndarray,
        cost_batch: np.ndarray,
        *,
        weights: np.ndarray | None = None,
        temp: float = 20.0,
        clip: float = 8.0,
        grad_clip: float = 5.0,
    ) -> dict[str, float]:
        """**cost-to-go 回归**蒸馏: 输出层直接回归 ``-相对代价 / temp``。

        为什么不是"再来一种软标签"
        ------------------------
        ``imitate_batch`` 的软目标是 ``softmax(-cost/T)``，损失是交叉熵，梯度是
        ``p − target``。代价差一旦超过几倍 T，target 就饱和成 one-hot，梯度随之
        归零 —— 学生只知道"这个动作最差"，不知道"差多少"，也学不到"现在整体有多
        危险"。这里改成对 **logits** 做 MSE：

            target_logit(a) = -clip((cost(a) − min cost) / T, 0, clip)

        梯度是线性的 ``(logits − target)``，不会饱和；同时因为 logits 与
        ``softmax`` 共享语义，学生 argmax 就等于自己 argmin cost-to-go ——
        这正是"蒸馏价值函数、让学生自己决策"，而不是"模仿老师这一帧选了谁"。

        ``cost_batch`` 里 ``NaN`` 的行（那一步老师弃权/算失败）直接跳过。
        返回里带 ``agree``：学生 argmax 与老师 argmin 的一致率，这是蒸馏是否
        真的把排序学过去的直接指标。
        """
        X = np.asarray(obs_batch, dtype=np.float64)
        C = np.asarray(cost_batch, dtype=np.float64)
        if X.ndim != 2 or C.ndim != 2 or X.shape[0] != C.shape[0]:
            return {"cost_loss": 0.0, "agree": 0.0, "labeled": 0}
        valid = np.isfinite(C).all(axis=1)
        m = int(valid.sum())
        if m == 0:
            return {"cost_loss": 0.0, "agree": 0.0, "labeled": 0}
        Xv, Cv = X[valid], C[valid]
        rel = Cv - Cv.min(axis=1, keepdims=True)
        target = -np.clip(rel / max(1e-6, float(temp)), 0.0, max(0.0, float(clip)))
        if weights is None:
            w = np.ones(m, dtype=np.float64)
        else:
            w = np.asarray(weights, dtype=np.float64).reshape(-1)
            w = np.ones(m, dtype=np.float64) if w.shape[0] != X.shape[0] else w[valid]
            w = np.maximum(w, 0.0)
        wsum = float(w.sum()) or 1.0

        activations, _zs, logits = self._forward_batch(Xv)
        h_last = activations[-1]
        diff = logits - target
        loss = float((w * np.sum(diff * diff, axis=1)).sum() / (2.0 * wsum))
        agree = float(np.mean(np.argmax(logits, axis=1) == np.argmin(Cv, axis=1)))
        mae = float((w * np.mean(np.abs(diff), axis=1)).sum() / wsum)

        grad_logits = (w[:, None] * diff) / wsum
        grads: list[np.ndarray] = [np.zeros_like(p) for p in self._params]
        gW = grads[: len(self.W)]
        gb = grads[len(self.W): 2 * len(self.W)]
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
            "cost_loss": round(loss, 6),
            "cost_mae": round(mae, 6),
            "agree": round(agree, 4),
            "labeled": m,
            "grad_norm": round(total, 6),
        }

    def fit_rank(
        self,
        obs_batch: np.ndarray,
        cost_batch: np.ndarray,
        *,
        weights: np.ndarray | None = None,
        margin: float = 20.0,
        grad_clip: float = 5.0,
    ) -> dict[str, float]:
        """**并列感知的排序蒸馏**：把"并列最优"的那一组动作当成同一个答案。

        为什么不是回归、也不是成对 RankNet
        ---------------------------------
        老师代价向量里 **74~78% 的帧"最优与次优完全并列"**（都是"足够安全"），
        其余动作的相对代价动辄 300+（除以温度后全部撞截断）。于是：

        * **回归**的目标实际只有 {0, −截断} 两个值 —— 没有连续信息，还会把
          "并列的几个动作"当成"必须有一个赢"，学生只能靠浮点误差决定 argmax；
        * **成对 RankNet**（只拿 argmin 当正例）会让并列组里没被选中的那个得不到
          强化，实测并列的两个 logits 被拉开 **1.1**，等于偷偷把"并列"改成了
          "必须挑第一个"。

        所以这里直接把并列组当**一个目标的全部质量**：

            good(s) = { a : cost(s,a) ≤ min_a cost + margin }        # 并列最优集合
            target  = 均匀分布在 good(s) 上，其余动作 0
            loss    = CE(softmax(logits), target)

        差在 ``margin`` 以内的一律视为同样好（这是"并列感知"的全部含义）；
        超过 ``margin`` 的动作只要求"别被选中"，不要求它们之间的相对顺序。
        对称性天然成立（并列组内共享同一份质量），不会出现"谁先谁占便宜"。

        ``margin`` 用代价单位：默认 20 ≈ 老师眼里"值得为此换方向"的程度。
        返回 ``rank_acc``（选中并列组内动作的比例）与 ``agree``（严格 argmin 一致率）。
        """
        X = np.asarray(obs_batch, dtype=np.float64)
        C = np.asarray(cost_batch, dtype=np.float64)
        if X.ndim != 2 or C.ndim != 2 or X.shape[0] != C.shape[0]:
            return {"rank_loss": 0.0, "rank_acc": 0.0, "agree": 0.0, "labeled": 0}
        valid = np.isfinite(C).all(axis=1)
        rows = np.nonzero(valid)[0]
        n = int(rows.size)
        if n == 0:
            return {"rank_loss": 0.0, "rank_acc": 0.0, "agree": 0.0, "labeled": 0}

        Cv = C[rows]
        best = Cv.min(axis=1, keepdims=True)
        good = (Cv <= best + max(0.0, float(margin))).astype(np.float64)
        target = good / np.maximum(1e-9, good.sum(axis=1, keepdims=True))

        if weights is None:
            w = np.ones(n, dtype=np.float64)
        else:
            w = np.asarray(weights, dtype=np.float64).reshape(-1)
            w = np.ones(X.shape[0], dtype=np.float64) if w.shape[0] != X.shape[0] else w[rows]
            w = np.maximum(w, 0.0)
        wsum = float(w.sum()) or 1.0

        activations, _zs, logits = self._forward_batch(X)
        h_last = activations[-1]
        probs = self._softmax_rows(logits)
        logp = np.log(np.maximum(probs[rows], 1e-12))
        loss = float(-(w * (target * logp).sum(axis=1)).sum() / wsum)
        picked = np.argmax(logits[rows], axis=1)
        rank_acc = float(np.mean(good[np.arange(n), picked] > 0.5))
        agree = float(np.mean(picked == np.argmin(Cv, axis=1)))

        grad_logits = np.zeros_like(logits)
        grad_logits[rows] = (probs[rows] - target) * (w / wsum)[:, None]

        grads: list[np.ndarray] = [np.zeros_like(p) for p in self._params]
        gW = grads[: len(self.W)]
        gb = grads[len(self.W): 2 * len(self.W)]
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
        good_mean = float(good.sum(axis=1).mean())
        return {
            "rank_loss": round(loss, 6),
            "rank_acc": round(rank_acc, 4),
            "agree": round(agree, 4),
            "good_mean": round(good_mean, 2),
            "labeled": n,
            "grad_norm": round(total, 6),
        }

    def fit_listmle(
        self,
        obs_batch: np.ndarray,
        cost_batch: np.ndarray,
        *,
        weights: np.ndarray | None = None,
        margin: float = 20.0,
        candidates: int = 0,
        seed: int = 0,
        posreg: np.ndarray | None = None,
        posreg_weight: float = 0.0,
        grad_clip: float = 5.0,
    ) -> dict[str, float]:
        """**tie-aware ListMLE**：在"同一状态的一组候选动作"上做 listwise 排序学习。

        为什么是 listwise，而不是 §11 的"并列组 CE"
        --------------------------------------------
        并列组 CE 只用了 **"哪些动作属于最优组"** 这一个 bit —— 组外动作之间的
        好坏次序全部丢掉了。可老师其实知道完整的次序（代价差 300 和差 30 不是一回事），
        listwise 把整条排序都用上，样本效率更高：同一个状态不只告诉你"该往左"，
        还告诉你"往左 > 往下 > 往右"。

        tie-aware 的关键：**组内不连乘**
        ---------------------------------
        标准 ListMLE 是 ``P = ∏_k e^{s_k} / Σ_{j≥k} e^{s_j}``（逐元素连乘）。直接
        套用会把并列组内的元素也排出先后 —— 这就是 §11.2 那个成对 RankNet 的坑
        （并列的两个 logits 被拉开 1.1）。这里改成**按组连乘**：

            P = ∏_g  (Σ_{i∈g} e^{s_i}) / (Σ_{j∈g 及其后} e^{s_j})
            loss = −Σ_g [ LSE(s_g) − LSE(s_{g 之后的所有组}) ]

        组 g 只通过**它内部的和**参与 loss，组内元素梯度完全相同 → 对称性天然成立。
        两个边界都对：全部并列时只剩一组，``LSE(全) − LSE(全) = 0``，loss 与梯度
        双双为 0（不会乱推）；严格无并列时退化成标准 ListMLE。

        候选动作采样
        ------------
        ``candidates=K>0`` 时每个状态只随机取 K 个动作（**必含 argmin**），其余
        不参与 loss。这既省算力，也让每轮覆盖不同的负例（负采样）；``K=0`` 表示
        用全部动作。

        ``posreg`` (``(N, n_actions)``) 是**位置正则**：每个动作一个非负权重，损失
        加 ``w·Σ_a p_a·reg_a``。贴顶时对"继续向上"给正值，直接压住"越贴越贴"的
        退化 —— 老师那边已经有位置偏好（``mpc.top_weight``），这一项是学生侧的
        兜底，因为学生只看到 32 颗最近的弹，未必能自己推出"上面没路"。
        """
        X = np.asarray(obs_batch, dtype=np.float64)
        C = np.asarray(cost_batch, dtype=np.float64)
        empty = {"listmle_loss": 0.0, "rank_acc": 0.0, "agree": 0.0,
                 "labeled": 0, "groups": 0.0}
        if X.ndim != 2 or C.ndim != 2 or X.shape[0] != C.shape[0]:
            return empty
        valid = np.isfinite(C).all(axis=1)
        rows = np.nonzero(valid)[0]
        n = int(rows.size)
        if n == 0:
            return empty

        Cv = C[rows]
        n_act = Cv.shape[1]
        rng = np.random.default_rng(int(seed))

        keep = np.ones((n, n_act), dtype=bool)
        if 0 < int(candidates) < n_act:
            keep[:] = False
            best0 = np.argmin(Cv, axis=1)
            keep[np.arange(n), best0] = True
            for r in range(n):
                others = np.nonzero(np.arange(n_act) != best0[r])[0]
                k = min(int(candidates) - 1, others.size)
                if k > 0:
                    keep[r, rng.choice(others, size=k, replace=False)] = True
        counts = keep.sum(axis=1).astype(np.int64)

        pad = float(Cv.max()) + 1e7
        order = np.argsort(np.where(keep, Cv, pad), axis=1, kind="stable")
        srt_cost = np.take_along_axis(np.where(keep, Cv, pad), order, axis=1)
        pos = np.arange(n_act, dtype=np.int64)[None, :]
        in_play = pos < counts[:, None]
        new_grp = np.ones((n, n_act), dtype=bool)
        if n_act > 1:
            gap = srt_cost[:, 1:] - srt_cost[:, :-1]
            new_grp[:, 1:] = (gap > float(margin)) | (pos[:, 1:] >= counts[:, None])
        gid = np.cumsum(new_grp & in_play, axis=1) - 1
        gid = np.where(in_play, gid, -1)
        n_grp = int(gid.max()) + 1

        activations, _zs, logits = self._forward_batch(X)
        srt_s = np.take_along_axis(logits[rows], order, axis=1)
        shift = np.where(in_play, srt_s, -1e9).max(axis=1, keepdims=True)
        e = np.exp(np.clip(srt_s - shift, -60.0, 0.0)) * in_play

        gmask = (gid[:, :, None] == np.arange(n_grp)[None, None, :]) * in_play[:, :, None]
        Zg = np.einsum("nag,na->ng", gmask.astype(np.float64), e)
        Zt = np.cumsum(Zg[:, ::-1], axis=1)[:, ::-1]
        present = Zg > 0.0
        log_zg = np.log(np.maximum(Zg, 1e-30))
        log_zt = np.log(np.maximum(Zt, 1e-30))
        loss_rows = -(np.where(present, log_zg - log_zt, 0.0)).sum(axis=1)

        if weights is None:
            w = np.ones(n, dtype=np.float64)
        else:
            w = np.asarray(weights, dtype=np.float64).reshape(-1)
            w = np.ones(n, dtype=np.float64) if w.shape[0] != X.shape[0] else w[rows]
            w = np.maximum(w, 0.0)
        wsum = float(w.sum()) or 1.0
        wn = w / wsum

        inv_g = 1.0 / np.maximum(Zg, 1e-30)
        pref = np.cumsum(1.0 / np.maximum(Zt, 1e-30), axis=1)
        gsel = np.where(gid >= 0, gid, 0)
        pref_p = np.take_along_axis(pref, gsel, axis=1) * in_play
        invg_p = np.take_along_axis(inv_g, gsel, axis=1) * in_play
        grad_s = e * (pref_p - invg_p) * wn[:, None]
        # 组内梯度均摊：这一步是这个损失能不能用的关键。
        # ``loss`` 本身对组内顺序**不变**（组只通过 Σe^s 参与），但它的梯度
        # ∝ e^{s_t} 仍带着每个元素自己的分数 —— 直接把 grad_naive 用下去，
        # 并列组里分数高的那个会被越推越高。实测并列的两个动作梯度差 35 倍，
        # 与 §11.2 的成对 RankNet 是同一个坑（loss 对称 ≠ 梯度对称）。
        # 在组内取均值 = 把整组当成**一个**决策单元：总推力守恒
        # （Σ_t 均值 = Σ_t grad_naive），而组内每个元素拿到的梯度完全相同。
        gm_f = gmask.astype(np.float64)
        gsum = np.einsum("nag,na->ng", gm_f, grad_s)
        gavg = gsum / np.maximum(gm_f.sum(axis=1), 1.0)
        grad_s = np.take_along_axis(gavg, gsel, axis=1) * in_play

        grad_rows = np.empty_like(srt_s)
        np.put_along_axis(grad_rows, order, grad_s, axis=1)

        loss = float((wn * loss_rows).sum())
        cost_terms = None
        if posreg is not None and float(posreg_weight) > 0.0:
            reg = np.asarray(posreg, dtype=np.float64)[rows]
            if reg.shape == logits[rows].shape:
                probs = self._softmax_rows(logits[rows])
                cost_terms = (probs * reg).sum(axis=1, keepdims=True)
                grad_rows = grad_rows + float(posreg_weight) * probs * (reg - cost_terms)
                loss += float(posreg_weight) * float(cost_terms.mean())

        grad_logits = np.zeros_like(logits)
        grad_logits[rows] = grad_rows
        total = self._backprop_logits(activations, grad_logits, grad_clip)

        picked = np.argmax(logits[rows], axis=1)
        best_grp = np.where(present, np.arange(n_grp)[None, :], n_grp).min(axis=1)
        # argmax 落在"排序后的第几个位置"→ 该位置所属组；等于最优组才算 rank_acc
        pick_pos = np.argmax(srt_s, axis=1)
        picked_gid = np.where(in_play, gid, -1)[np.arange(n), pick_pos]
        rank_acc = float(np.mean(picked_gid == best_grp))
        agree = float(np.mean(picked == np.argmin(Cv, axis=1)))
        return {
            "listmle_loss": round(loss, 6),
            "rank_acc": round(rank_acc, 4),
            "agree": round(agree, 4),
            "labeled": n,
            "groups": round(float(present.sum(axis=1).mean()), 3),
            "grad_norm": round(total, 6),
        }

    def _backprop_logits(self, activations: list[np.ndarray],
                         grad_logits: np.ndarray, grad_clip: float = 5.0) -> float:
        """从 ``dL/dlogits`` 反传到所有参数并更新一步 Adam，返回梯度范数。"""
        grads: list[np.ndarray] = [np.zeros_like(p) for p in self._params]
        gW = grads[: len(self.W)]
        gb = grads[len(self.W): 2 * len(self.W)]
        gW[-1] += grad_logits.T @ activations[-1]
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
        return total

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
