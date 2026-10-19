"""缺口任务上的最小学习实验: 策略能不能自己学会"找缝"。

它检验的正是那个关键问题 —— **把未来占用栅格 / 几何缺口放进观测, 策略是否能靠
试错 (而不是靠背下某个固定方向) 学会躲避**。

任务
----
``wall_gate`` 弹幕墙: 每 ``interval`` 帧落下**一堵真的封死**的弹幕墙 (弹间距 9px,
除缺口外无缝可钻), 缺口位置每堵墙横向移动。自机必须在墙到达前提前占位。

三组观测对照 (其余完全一致: 同样的动作空间、奖励、种子、更新次数)

* ``geo``  : 每列最早到达时间 (TTA, "几何缺口"剖面) + 8 方向安全余量
* ``grid`` : 时间分层占用栅格 (t = 0/12/26 帧)
* ``both`` : 两者拼接

泛化测试用**训练时没见过的**缺口相位、反向漂移与更高弹速 —— 背方向的做法会在这里
露馅, 真正学会"看缺口"的策略则不会。

用法::

    python tools/research/gap_rl.py --arm both --updates 60
    python tools/research/gap_rl.py --arm geo  --baseline      # 附 noop / 缺口跟随 基准
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from touhou_rl_mcp.config import Config, runtime_dir   # noqa: E402
from touhou_rl_mcp.env import TouhouEnv            # noqa: E402
from touhou_rl_mcp.training.journal import Journal  # noqa: E402
from touhou_rl_mcp.types import PLAYFIELD_WIDTH    # noqa: E402
from touhou_rl_mcp.training.occupancy import (     # noqa: E402
    DIRECTIONS, PLAYER_SPEED, GapObservation, GapSpec,
    _best_column, _bullets, _column_centers, _tta_profile,
)

#: 动作表: 8 方向 + 原地, 每个动作承诺 6 帧 (60Hz 下的 0.1 秒)
ACTIONS: tuple[str, ...] = (
    "shoot", "up+shoot", "down+shoot", "left+shoot", "right+shoot",
    "up_left+shoot", "up_right+shoot", "down_left+shoot", "down_right+shoot",
)
MACRO_FRAMES = 6

#: 训练场景 (v1: 初版缺口墙, 泛化上限 1800 帧已经被打满)
TRAIN_V1: dict[str, float] = dict(
    interval=130.0, gap_width=60.0, speed=1.8, gap_drift=0.13, gap_phase=0.0,
)
HELDOUT_V1: tuple[dict[str, float], ...] = (
    dict(interval=130.0, gap_width=60.0, speed=1.8, gap_drift=-0.17, gap_phase=0.31),
    dict(interval=130.0, gap_width=60.0, speed=1.9, gap_drift=0.29, gap_phase=0.62),
    dict(interval=120.0, gap_width=64.0, speed=1.8, gap_drift=-0.24, gap_phase=0.15),
)

#: v2: 更窄的缺口 (46px, 扣掉判定半径后只有 36px)、更快的墙、更大的漂移。
#: 难度按"缺口跟随能活满、站着不动很快死"标定 —— 两者都成立才是一个有区分度的基准。
TRAIN_V2: dict[str, float] = dict(
    interval=95.0, gap_width=46.0, speed=2.05, gap_drift=0.34, gap_phase=0.0,
)
#: 泛化: 反向漂移 / 更快弹速 / 更短间隔, 全部标定过"缺口跟随能活满 2400 帧"
HELDOUT_V2: tuple[dict[str, float], ...] = (
    dict(interval=95.0, gap_width=46.0, speed=2.05, gap_drift=-0.41, gap_phase=0.27),
    dict(interval=88.0, gap_width=48.0, speed=2.2, gap_drift=0.29, gap_phase=0.58),
    dict(interval=102.0, gap_width=44.0, speed=1.95, gap_drift=-0.36, gap_phase=0.13),
)

BENCHMARKS: dict[str, tuple[dict[str, float], tuple[dict[str, float], ...]]] = {
    "v1": (TRAIN_V1, HELDOUT_V1),
    "v2": (TRAIN_V2, HELDOUT_V2),
}

#: 缺口的判定半径 (自机 2 + 弹 3), 用来把"可选列"折成"真正站得住的列"
REACH_PAD = 6.0


# ----------------------------------------------------------------------
def make_env(**scenario: float) -> TouhouEnv:
    cfg = Config.load()
    cfg.patch({
        "backend": "mock", "game": "th11", "danmaku_pattern": "wall_gate",
        "step_frames": MACRO_FRAMES, "max_episode_frames": 0,
    })
    env = TouhouEnv(cfg)
    env.start()
    env.reset(seed=1)
    for key, value in scenario.items():
        setattr(env.backend.game.pattern, key, value)
    return env


def reset_env(env: TouhouEnv, seed: int, scenario: dict[str, float]):
    state = env.reset(seed=seed).state
    for key, value in scenario.items():
        setattr(env.backend.game.pattern, key, value)
    return state


# ----------------------------------------------------------------------
class Policy:
    """单隐层 MLP + softmax, 手写向量化 REINFORCE (含价值基线)。"""

    def __init__(self, obs_dim: int, n_actions: int, *, hidden: int = 64,
                 seed: int = 0, lr: float = 4e-3, entropy: float = 0.02,
                 value_coef: float = 0.5) -> None:
        rng = np.random.default_rng(seed)
        self.obs_dim, self.n_actions, self.lr = obs_dim, n_actions, lr
        self.hidden = int(hidden)           # 0 = 线性策略 (便于判断表示本身是否可分)
        self.entropy, self.value_coef = entropy, value_coef
        fan = self.hidden if self.hidden > 0 else obs_dim
        self.W1 = rng.standard_normal((self.hidden, obs_dim)) * np.sqrt(2.0 / obs_dim)
        self.b1 = np.zeros(self.hidden)
        self.W2 = rng.standard_normal((n_actions, fan)) * 0.01
        self.b2 = np.zeros(n_actions)
        self.Wv = np.zeros(fan)
        self.bv = 0.0
        self._adam = {k: [np.zeros_like(v), np.zeros_like(v)]
                      for k, v in self.params().items()}
        self._t = 0

    def params(self) -> dict[str, np.ndarray]:
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2,
                "Wv": self.Wv, "bv": np.asarray(self.bv)}

    # -- 前向 -----------------------------------------------------------
    def _forward(self, X: np.ndarray):
        if self.hidden == 0:
            z1 = X
            h = X
        else:
            z1 = X @ self.W1.T + self.b1
            h = np.maximum(z1, 0.0)
        logits = h @ self.W2.T + self.b2
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        probs = exp / exp.sum(axis=1, keepdims=True)
        value = h @ self.Wv + self.bv
        return z1, h, probs, value

    def act(self, obs: np.ndarray, rng: np.random.Generator, greedy: bool = False,
            epsilon: float = 0.0):
        probs = self._forward(obs[None, :])[2][0]
        if greedy:
            return int(np.argmax(probs)), 1.0
        # ε-探索: 纯 softmax 采样在"乱动比不动更容易死"的任务上会迅速坍缩成
        # "始终按同一个键", 之后再也采不到左右移动, 学习就永久停止了。
        if epsilon > 0.0 and rng.random() < epsilon:
            a = int(rng.integers(self.n_actions))
            return a, float(probs[a])
        a = int(rng.choice(self.n_actions, p=probs))
        return a, float(probs[a])

    # -- 更新 -----------------------------------------------------------
    def update(self, X: np.ndarray, A: np.ndarray, G: np.ndarray,
               anchor: np.ndarray | None = None, bc_coef: float = 0.0) -> float:
        n = X.shape[0]
        if n == 0:
            return 0.0
        z1, h, probs, value = self._forward(X)
        # 回报量级是"几百"(每步 +1, 一局几百步)。直接拿它做价值回归, 平方损失会把
        # 共享隐层冲垮 —— 所以先按批次尺度归一化, 再算优势。
        scale = float(np.abs(G).mean()) + 1e-8
        target = G / scale
        # 优势 = 回报 - 价值基线。**不能用"整局回报归一化"**代替基线:
        # 那样死亡前十几步的动作分不到梯度, 学到的只是"多活一点"的平均趋势。
        adv = target - value
        adv /= (adv.std() + 1e-8)
        logp = np.log(np.maximum(probs, 1e-12))
        ent = -np.sum(probs * logp, axis=1)

        dlogits = probs.copy()
        dlogits[np.arange(n), A] -= 1.0
        dlogits *= adv[:, None]
        dlogits += self.entropy * probs * (logp + ent[:, None])
        if anchor is not None and bc_coef > 0.0:
            # BC 锚: 把策略拉回"看得懂缺口"的分布, 防止 RL 微调把它忘掉。
            # 没有这个约束, 微调会退化成"站着不动"的局部最优 (实测 1136 -> 926 帧)。
            dlogits += bc_coef * probs.copy()
            dlogits[np.arange(n), anchor] -= bc_coef

        dvalue = (value - target) * self.value_coef
        self._apply(dlogits, dvalue, X, z1, h)
        return float(-np.mean(adv * np.log(np.maximum(probs[np.arange(n), A], 1e-12))))

    def imitate(self, X: np.ndarray, A: np.ndarray) -> float:
        """交叉熵: 让策略复现给定的"缺口方向"标签 (感知层面的监督, 不是控制器)。"""
        n = X.shape[0]
        if n == 0:
            return 0.0
        z1, h, probs, _value = self._forward(X)
        dlogits = probs.copy()
        dlogits[np.arange(n), A] -= 1.0
        dlogits /= n
        self._apply(dlogits, np.zeros(n), X, z1, h)
        return float(-np.mean(np.log(np.maximum(probs[np.arange(n), A], 1e-12))))

    def _apply(self, dlogits: np.ndarray, dvalue: np.ndarray,
               X: np.ndarray, z1: np.ndarray, h: np.ndarray) -> None:
        n = X.shape[0]
        grads = {
            "W2": dlogits.T @ h / n,
            "b2": dlogits.sum(axis=0) / n,
            "Wv": dvalue @ h / n,
            "bv": np.asarray(dvalue.mean()),
        }
        dh = dlogits @ self.W2 + dvalue[:, None] * self.Wv[None, :]
        if self.hidden == 0:
            grads["W1"] = np.zeros_like(self.W1)
            grads["b1"] = np.zeros_like(self.b1)
        else:
            dz1 = dh * (z1 > 0)
            grads["W1"] = dz1.T @ X / n
            grads["b1"] = dz1.sum(axis=0) / n

        self._t += 1
        for name, grad in grads.items():
            m, v = self._adam[name]
            m *= 0.9
            m += 0.1 * grad
            v *= 0.999
            v += 0.001 * grad * grad
            mh = m / (1 - 0.9 ** self._t)
            vh = v / (1 - 0.999 ** self._t)
            step = self.lr * mh / (np.sqrt(vh) + 1e-8)
            if name == "bv":
                self.bv = float(self.bv - step)
            else:
                getattr(self, name)[...] -= step


# ----------------------------------------------------------------------
def lane_safety(state, spec: GapSpec) -> float:
    """自机**当前所在列**的安全余量 (TTA 剖面的中点)。

    剖面是以自机为中心算的, 所以自机永远在中间那一列。这个量回答的是
    "我站的地方还安全吗", 是纯粹由占用栅格导出的测量, 不包含"该往哪走"。
    把它当**势函数**做奖励塑形 (r += β·(φ' - φ)), 可以在不改变最优策略的前提下
    给出密集梯度 —— 没有它, "站着不动"就是一个很难跳出的局部最优。
    """
    pos, vel = _bullets(state)
    profile = _tta_profile(state.player_y, pos, vel, spec)
    mid = spec.tta_cols // 2
    return float(profile[mid])


def rollout(env: TouhouEnv, enc: GapObservation, policy: Policy | None,
            rng: np.random.Generator, *, seed: int,
            scenario: dict[str, float], max_frames: int,
            greedy: bool = False, shaping: float = 0.0,
            epsilon: float = 0.0, label_mode: str | None = "future") -> dict:
    state = reset_env(env, seed, scenario)
    obs_list, act_list, rew_list = [], [], []
    prev_action = -1
    frames = 0
    deaths = 0
    lives = state.lives
    prev_phi = lane_safety(state, enc.spec) if shaping else 0.0
    labels: list[int] = []
    while frames < max_frames:
        obs = enc(state, prev_action=prev_action)
        if policy is None:
            action, _ = 0, 1.0            # noop 基准
        else:
            action, _ = policy.act(obs, rng, greedy=greedy, epsilon=epsilon)
        result = env.step(ACTIONS[action], frames=MACRO_FRAMES, auto_reset=False)
        state = result.state
        frames += MACRO_FRAMES
        death = (lives is not None and state.lives is not None and state.lives < lives)
        if death:
            deaths += 1
        lives = state.lives
        reward = 1.0 + (-30.0 if death else 0.0)
        if shaping:
            phi = lane_safety(state, enc.spec)
            reward += shaping * (phi - prev_phi)
            prev_phi = phi
        obs_list.append(obs)
        act_list.append(action)
        labels.append(gap_label(state, enc.spec, label_mode) if label_mode else 0)
        rew_list.append(reward)
        prev_action = action
        if result.done:
            break
    return {
        "obs": np.asarray(obs_list, dtype=np.float64),
        "act": np.asarray(act_list, dtype=np.intp),
        "rew": np.asarray(rew_list, dtype=np.float64),
        "frames": frames,
        "deaths": deaths,
        "label": np.asarray(labels, dtype=np.intp),
    }


def discounted(rew: np.ndarray, gamma: float) -> np.ndarray:
    out = np.empty_like(rew)
    running = 0.0
    for i in range(rew.shape[0] - 1, -1, -1):
        running = rew[i] + gamma * running
        out[i] = running
    return out


# ----------------------------------------------------------------------
def gap_follower(env: TouhouEnv, spec: GapSpec, *, seed: int,
                 scenario: dict[str, float], max_frames: int) -> dict:
    """几何缺口基准: 直接朝 TTA 最大的一列横移 (人工规则, 只做对照)。"""
    state = reset_env(env, seed, scenario)
    frames = 0
    while frames < max_frames:
        pos, vel = _bullets(state)
        if pos.shape[0] == 0:
            action = 0
        else:
            prof = _tta_profile(state.player_y, pos, vel, spec)
            # 并列最优时取离自机最近的一列: 否则整片安全时 argmax 恒为最左列,
            # 基准策略会一路贴左墙, 反而比站着不动死得更早。
            k = _best_column(prof, state.player_x, spec)
            centers = _column_centers(spec)
            delta = centers[k] - state.player_x
            action = 0 if abs(delta) < 8 else (4 if delta > 0 else 3)
        result = env.step(ACTIONS[action], frames=MACRO_FRAMES, auto_reset=False)
        state = result.state
        frames += MACRO_FRAMES
        if result.done:
            break
    return {"frames": frames}


def evaluate(env: TouhouEnv, enc: GapObservation, policy: Policy,
             rng: np.random.Generator, *, max_frames: int,
             heldout: tuple[dict[str, float], ...], label_mode: str = "future",
             seeds: tuple[int, ...] = (101, 202)) -> tuple[float, list[float]]:
    scores = []
    for scenario in heldout:
        for seed in seeds:
            out = rollout(env, enc, policy, rng, seed=seed, scenario=scenario,
                          max_frames=max_frames, greedy=True, label_mode=label_mode)
            scores.append(out["frames"])
    return float(np.mean(scores)), scores


def action_profile(env: TouhouEnv, enc: GapObservation, policy: Policy,
                   rng: np.random.Generator, *, max_frames: int,
                   heldout: tuple[dict[str, float], ...],
                   label_mode: str = "future") -> np.ndarray:
    """贪心策略在测试场景下的动作分布 —— 用来识别"学会不动"这种退化。"""
    counts = np.zeros(len(ACTIONS))
    for scenario in heldout:
        out = rollout(env, enc, policy, rng, seed=101, scenario=scenario,
                      max_frames=max_frames, greedy=True, label_mode=label_mode)
        for a in out["act"]:
            counts[a] += 1
    total = counts.sum()
    return counts / total if total else counts


def label_agreement(env: TouhouEnv, enc: GapObservation, policy: Policy,
                    rng: np.random.Generator, *, max_frames: int,
                    heldout: tuple[dict[str, float], ...],
                    label_mode: str) -> float:
    """策略动作与几何标签的一致率 (贪心, 只看未见过的场景)。

    这是比存活帧数更直接的"学会了没有": 存活可能靠苟, 一致率不行 —— 它要求策略在
    每个状态下都选对方向, 而且是在没见过的相位/漂移/弹速上。
    """
    hits = total = 0
    for scenario in heldout:
        out = rollout(env, enc, policy, rng, seed=101, scenario=scenario,
                      max_frames=max_frames, greedy=True, label_mode=label_mode)
        hits += int(np.sum(out["act"] == out["label"]))
        total += int(out["act"].shape[0])
    return hits / total if total else 0.0


def spec_for_arm(arm: str) -> GapSpec:
    if arm == "geo":
        return GapSpec(use_grid=False, use_geo=True, use_clearance=True)
    if arm == "grid":
        return GapSpec(use_grid=True, use_geo=False, use_clearance=False)
    return GapSpec(use_grid=True, use_geo=True, use_clearance=True)


def _tta_frames(state, spec: GapSpec) -> np.ndarray:
    """把归一化剖面换算回"每列还有多少帧被占" (帧)。"""
    pos, vel = _bullets(state)
    prof = _tta_profile(state.player_y, pos, vel, spec)
    return prof * float(spec.tta_horizon)


def _reach_target(state, spec: GapSpec) -> float | None:
    """考虑"能不能及时赶到"之后, 值得去的目标列中心 (绝对 x)。

    原来的标签只问"现在哪一列最空", 于是会指着一列 200px 外、但 30 帧后就要被封死
    的缝。这里改成 ``可用余量 - 赶路时间``: 站得住且来得及的列才值得去, 已经站对了
    就不动 (距离代价为 0, 天然占优)。
    """
    tta = _tta_frames(state, spec)
    if not np.any(tta > 0):
        return None
    centers = _column_centers(spec)
    travel = np.abs(centers - state.player_x) / PLAYER_SPEED
    slack = tta - travel                      # 赶到之后还剩多少帧安全余量
    # 判定半径之内的列差一个像素没有区别, 用 pad 把"站得住"的列并成一档
    padded = np.floor(slack / REACH_PAD) * REACH_PAD
    best = float(padded.max())
    tied = np.flatnonzero(padded >= best - 1e-9)
    k = int(tied[np.argmin(np.abs(centers[tied] - state.player_x))])
    return float(centers[k])


def _macro_x(x: float, action: int) -> float:
    """朝某方向承诺 ``MACRO_FRAMES`` 帧后的横坐标 (斜向按 45° 归一化)。"""
    ux, uy = DIRECTIONS[action]
    norm = float(np.hypot(ux, uy))
    dx = (ux / norm) if norm else 0.0
    return min(PLAYFIELD_WIDTH - 8.0, max(8.0, x + dx * PLAYER_SPEED * MACRO_FRAMES))


def gap_label(state, spec: GapSpec, mode: str = "future") -> int:
    """弱监督标签: 只回答"往哪个方向走", 不回答"该不该躲"。

    三种口径, 依次递进 (都只用几何测量, 不含手写代价函数):

    ``now``
        当前剖面里最空的那一列。**问题**: 它不管自机来不来得及, 而且整片安全时
        argmax 会退化成某一个固定列 (已修成"挑离自机最近的列")。
    ``reach``
        在 ``now`` 上加入可达性: 目标列必须"赶过去以后还有余量"。已经站对时标签是
        原地, 不会为了 3 个像素的偏差来回摆。
    ``future``
        承诺语义: 沿同一方向推 4 步 (24 帧) 取最好的时刻, 直接输出动作。详见
        :func:`_future_label`。
    """
    if mode == "future":
        return _future_label(state, spec)
    target = _reach_target(state, spec)
    if target is None:
        return 0
    delta = target - state.player_x
    if abs(delta) < 10.0:
        return 0
    if mode == "reach":
        return 4 if delta > 0 else 3
    # now: 只取当前最空的一列, 不考虑赶路
    tta = _tta_frames(state, spec)
    if not np.any(tta > 0):
        return 0
    k = _best_column(tta / float(spec.tta_horizon), state.player_x, spec)
    delta = float(_column_centers(spec)[k]) - state.player_x
    if abs(delta) < 10.0:
        return 0
    return 4 if delta > 0 else 3


def _future_label(state, spec: GapSpec, lookahead: int = 4) -> int:
    """前瞻标签: 朝某个方向连续走 ``lookahead`` 个宏动作, 取这条直线上最好的时刻。

    ``now`` / ``reach`` 都只看"此刻哪条车道最空", 于是缺口在 100px 外时它给不出
    "现在就该往那边挪"的信号 —— 而每步只能走 26px, 等到"此刻最空的那条"变成眼前
    的车道时往往已经来不及。

    这里换成**承诺语义**: 沿同一方向推 4 步 (24 帧, ~105px), 看看这条路上最好的时刻
    能有多安全。它和策略的动作空间同构, 也不会左右摇摆。

    两个实测过的坑 (都写在这里, 免得再踩):
    * 只算一步落点 -> 所有方向都不如原地 (还要付 6 帧赶路), 标签退化成"永远别动";
    * 把纵向位移按弹速折算成"少了几帧余量" -> 下移看起来永远划算, 标签退化成
      "一直按着下" (71% 的样本)。在只有存活奖励的任务里, 苟活永远是最省事的最优。
    """
    tta = _tta_frames(state, spec)
    if not np.any(tta > 0):
        return 0
    cols = spec.tta_cols
    cell = float(PLAYFIELD_WIDTH) / cols

    def column(x: float) -> int:
        return int(min(cols - 1, max(0, x / cell)))

    best: tuple[tuple[float, float], int] | None = None
    for action in range(len(DIRECTIONS)):
        x = state.player_x
        score = -1e18
        for step in range(1, lookahead + 1):
            x = _macro_x(x, action)
            score = max(score, float(tta[column(x)]) - step * MACRO_FRAMES)
        # 量化到 REACH_PAD, 避免"差 1 帧"就换动作的抖动; 并列时位移小的胜出
        quantised = math.floor(score / REACH_PAD) * REACH_PAD
        key = (-quantised, abs(_macro_x(state.player_x, action) - state.player_x))
        if best is None or key < best[0]:
            best = (key, action)
    return 0 if best is None else best[1]


def collect_labels(env: TouhouEnv, enc: GapObservation, rng: np.random.Generator,
                   *, episodes: int, max_frames: int,
                   scenario: dict[str, float],
                   label_mode: str = "future",
                   policy: Policy | None = None,
                   epsilon: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """走一遍并把**每个访问到的状态**用几何规则打上标签。

    ``policy=None`` 时用均匀随机动作 (离线 BC 的数据来源); 传入策略时用它自己的
    轨迹 (DAgger) —— 这时标签覆盖的是"策略真正会走到的状态", 而不是随机游走的
    状态分布, 这正是 BC 一步准确率 89% 却在闭环里只有 47% 的根因。
    """
    xs, ys = [], []
    for _ in range(episodes):
        state = reset_env(env, seed=int(rng.integers(1, 10 ** 6)), scenario=scenario)
        prev, frames = -1, 0
        while frames < max_frames:
            obs = enc(state, prev_action=prev).copy()
            xs.append(obs)
            ys.append(gap_label(state, enc.spec, label_mode))
            if policy is None:
                action = int(rng.integers(len(ACTIONS)))
            else:
                action, _ = policy.act(obs, rng, epsilon=epsilon)
            result = env.step(ACTIONS[action], frames=MACRO_FRAMES, auto_reset=False)
            state, frames, prev = result.state, frames + MACRO_FRAMES, action
            if result.done:
                break
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.intp)


def fit_imitation(policy: Policy, X: np.ndarray, Y: np.ndarray,
                  rng: np.random.Generator, *, epochs: int, batch: int = 256,
                  verbose: bool = True) -> dict[str, float]:
    """在给定 (状态, 标签) 上做交叉熵模仿, 返回训练/留出准确率。"""
    if X.shape[0] == 0:
        return {"samples": 0, "train_acc": 0.0, "holdout_acc": 0.0}
    split = int(X.shape[0] * 0.8)
    Xtr, Ytr, Xte, Yte = X[:split], Y[:split], X[split:], Y[split:]
    for _ in range(epochs):
        order = rng.permutation(Xtr.shape[0])
        for start in range(0, order.shape[0], batch):
            idx = order[start:start + batch]
            policy.imitate(Xtr[idx], Ytr[idx])
    acc = float(np.mean(np.argmax(policy._forward(Xte)[2], axis=1) == Yte)) if Xte.shape[0] else 0.0
    train_acc = float(np.mean(np.argmax(policy._forward(Xtr)[2], axis=1) == Ytr))
    if verbose:
        print(f"  模仿学习: {X.shape[0]} 样本  训练准确率 {train_acc:.0%}  "
              f"留出准确率 {acc:.0%}")
    return {"samples": int(X.shape[0]), "train_acc": round(train_acc, 4),
            "holdout_acc": round(acc, 4)}


def pretrain(env: TouhouEnv, enc: GapObservation, policy: Policy,
             rng: np.random.Generator, *, episodes: int, epochs: int,
             max_frames: int, scenario: dict[str, float],
             label_mode: str = "future"
             ) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """随机策略采样 + 模仿, 返回 (统计, X, Y) —— X/Y 供 DAgger 继续聚合。"""
    X, Y = collect_labels(env, enc, rng, episodes=episodes, max_frames=max_frames,
                          scenario=scenario, label_mode=label_mode)
    stats = fit_imitation(policy, X, Y, rng, epochs=epochs)
    stats["label"] = label_mode
    return stats, X, Y


def dagger_rounds(env: TouhouEnv, enc: GapObservation, policy: Policy,
                  rng: np.random.Generator, *, X: np.ndarray, Y: np.ndarray,
                  rounds: int, episodes: int, epochs: int, max_frames: int,
                  scenario: dict[str, float], label_mode: str,
                  epsilon: float, cap: int,
                  on_round=None) -> tuple[np.ndarray, np.ndarray]:
    """DAgger: 用**当前策略**跑, 在它自己的状态上重新打标签, 再聚合重训。

    BC 的问题就出在数据集分布上: 随机策略走两步就死, 收集到的大多是"已经没救"的
    状态; 真正需要学会的是"被自己带偏之后怎么救回来"的状态。DAgger 每轮都在这些
    状态上补标签, 所以一步准确率不用提高多少, 闭环表现就能上去。
    """
    for index in range(1, rounds + 1):
        Xn, Yn = collect_labels(env, enc, rng, episodes=episodes, max_frames=max_frames,
                                scenario=scenario, label_mode=label_mode,
                                policy=policy, epsilon=epsilon)
        X = np.concatenate([X, Xn])
        Y = np.concatenate([Y, Yn])
        if X.shape[0] > cap:                 # 只保留最近 cap 个样本, 控制内存与耗时
            X, Y = X[-cap:], Y[-cap:]
        stats = fit_imitation(policy, X, Y, rng, epochs=epochs, verbose=False)
        if on_round is not None:
            on_round(index, stats)
        print(f"  DAgger 第 {index} 轮: 样本 {X.shape[0]}  "
              f"留出准确率 {stats['holdout_acc']:.0%}")
    return X, Y


# ----------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="缺口任务上的最小 RL 实验")
    ap.add_argument("--arm", choices=("geo", "grid", "both"), default="both")
    ap.add_argument("--bench", choices=tuple(BENCHMARKS), default="v2",
                    help="基准版本: v1=初版缺口墙, v2=更窄更快、漂移更大的新基准")
    ap.add_argument("--label", choices=("now", "reach", "future"), default="reach",
                    help="弱监督标签口径: now=当前最空; reach=考虑赶路时间; future=两步 maximin")
    ap.add_argument("--updates", type=int, default=60)
    ap.add_argument("--episodes", type=int, default=16, help="每次更新采集多少局")
    ap.add_argument("--max-frames", type=int, default=2400, help="单局上限帧数")
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--lr", type=float, default=4e-3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--baseline", action="store_true", help="同时跑 noop / 缺口跟随 基准")
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--run-dir", default="",
                    help="训练记录目录 (默认 .runtime/runs/<arm>-<时间戳>, 可被看板直接读取)")
    ap.add_argument("--no-journal", action="store_true", help="不写训练记录")
    ap.add_argument("--shaping", type=float, default=0.0,
                    help="自机所在列安全余量的势函数奖励系数 (0=关)")
    ap.add_argument("--hidden", type=int, default=64, help="隐层宽度, 0=线性策略")
    ap.add_argument("--epsilon", type=float, default=0.0, help="训练时的 ε-探索 (greedy 评估不受影响)")
    ap.add_argument("--bc-episodes", type=int, default=0,
                    help="感知预训练的采样局数 (>0 才启用\"缺口方向\"监督)")
    ap.add_argument("--bc-epochs", type=int, default=6)
    ap.add_argument("--bc-coef", type=float, default=0.0,
                    help="RL 微调时锚定缺口感知的强度 (0=不做约束)")
    ap.add_argument("--dagger", type=int, default=0,
                    help="DAgger 轮数: 用策略自己的轨迹补标签 (>0 才启用)")
    ap.add_argument("--dagger-episodes", type=int, default=0,
                    help="每轮 DAgger 采多少局 (0=沿用 --bc-episodes)")
    ap.add_argument("--dagger-epsilon", type=float, default=0.15,
                    help="DAgger 采样时的 ε-探索, 用来覆盖被带偏后的状态")
    ap.add_argument("--dagger-cap", type=int, default=40000,
                    help="聚合数据集上限 (只保留最近 N 个样本)")
    ap.add_argument("--entropy", type=float, default=0.02)
    ap.add_argument("--train-override", default="",
                    help="覆盖训练场景, 逗号分隔 key=value (课程用), 如 gap_width=140,interval=180")
    ap.add_argument("--curriculum", action="store_true",
                    help="宽缺口 -> 窄缺口的课程: 前 1/3 用宽缺口, 中 1/3 过渡")
    args = ap.parse_args()

    train_scenario = dict(BENCHMARKS[args.bench][0])
    heldout = BENCHMARKS[args.bench][1]
    for part in args.train_override.split(","):
        if "=" in part:
            key, _, value = part.partition("=")
            train_scenario[key.strip()] = float(value)

    spec = spec_for_arm(args.arm)
    enc = GapObservation(spec)
    env = make_env(**train_scenario)
    rng = np.random.default_rng(args.seed)
    policy = Policy(enc.dim, len(ACTIONS), seed=args.seed, lr=args.lr,
                    hidden=args.hidden, entropy=args.entropy)

    print(f"bench={args.bench}  label={args.label}  arm={args.arm}  "
          f"obs_dim={enc.dim} {enc.describe()}")
    print(f"train scenario: {train_scenario}")

    def _bench_line(tag: str, scenarios: tuple[dict[str, float], ...]) -> None:
        noop = [rollout(env, enc, None, rng, seed=101 + i, scenario=sc,
                        max_frames=args.max_frames, label_mode=None)["frames"]
                for sc in scenarios for i in range(1)]
        follow = [gap_follower(env, spec, seed=101 + i, scenario=sc,
                               max_frames=args.max_frames)["frames"]
                  for sc in scenarios for i in range(1)]
        print(f"  {tag} noop 基准: {np.mean(noop):7.0f} 帧   {[int(x) for x in noop]}")
        print(f"  {tag} 缺口跟随: {np.mean(follow):7.0f} 帧   {[int(x) for x in follow]}")

    if args.baseline:
        _bench_line("训练场景", (train_scenario,))
        _bench_line("泛化场景", heldout)

    run_dir = None
    journal = None
    if not args.no_journal:
        run_dir = Path(args.run_dir) if args.run_dir else (
            runtime_dir() / "runs" / f"{args.bench}-{args.label}-{args.arm}-"
                                     f"{time.strftime('%Y%m%d-%H%M%S')}")
        journal = Journal(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"训练记录: {run_dir}")

    meta: dict[str, object] = {
        "script": "tools/research/gap_rl.py",
        "bench": args.bench,
        "label": args.label,
        "arm": args.arm,
        "obs": enc.describe(),
        "train_scenario": train_scenario,
        "heldout": list(heldout),
        "args": vars(args),
    }

    def _write_meta() -> None:
        if run_dir is not None:
            (run_dir / "meta.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8")

    def _eval_row(step: int, name: str, score: float, detail: list[int],
                  **extra: object) -> None:
        if run_dir is None:
            return
        row = {"step": step, "name": name, "score": round(score, 1),
               "detail": [int(x) for x in detail]}
        row.update(extra)
        with open(run_dir / "eval.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    super_X = super_Y = None
    if args.bc_episodes > 0:
        stats, super_X, super_Y = pretrain(
            env, enc, policy, rng, episodes=args.bc_episodes,
            epochs=args.bc_epochs, max_frames=args.max_frames,
            scenario=train_scenario, label_mode=args.label)
        meta["pretrain"] = stats
        score, detail = evaluate(env, enc, policy, rng, max_frames=args.max_frames,
                                 heldout=heldout, label_mode=args.label)
        agree = label_agreement(env, enc, policy, rng, max_frames=args.max_frames,
                                heldout=heldout, label_mode=args.label)
        print(f"  仅感知预训练 (无 RL): 泛化 {score:.0f} 帧  "
              f"一致率 {agree:.0%}  {[int(x) for x in detail]}")
        _eval_row(0, "pretrain", score, detail, acc=stats["holdout_acc"],
                  agree=round(agree, 4))
        # 纯模仿学习没有 RL 更新, 但看板需要一条曲线 —— 每一轮就是一条"episode"
        if journal is not None:
            journal.episode({"episode": 0, "return": round(score, 2),
                             "agree": round(agree, 4),
                             "acc": stats["holdout_acc"], "steps": stats["samples"]})
        dist = action_profile(env, enc, policy, rng, max_frames=args.max_frames,
                              heldout=heldout, label_mode=args.label)
        print("  动作分布: " + _dist_text(dist))

    if args.dagger > 0 and super_X is not None:
        def _on_round(index: int, st: dict[str, float]) -> None:
            score, detail = evaluate(env, enc, policy, rng, max_frames=args.max_frames,
                                     heldout=heldout, label_mode=args.label)
            agree = label_agreement(env, enc, policy, rng, max_frames=args.max_frames,
                                    heldout=heldout, label_mode=args.label)
            print(f"    泛化 {score:.0f} 帧  一致率 {agree:.0%}  {[int(x) for x in detail]}")
            _eval_row(index, f"dagger{index}", score, detail,
                      acc=st.get("holdout_acc"), agree=round(agree, 4))
            if journal is not None:
                dist = action_profile(env, enc, policy, rng, max_frames=args.max_frames,
                                      heldout=heldout, label_mode=args.label)
                journal.episode({
                    "episode": index, "return": round(score, 2),
                    "agree": round(agree, 4), "acc": st.get("holdout_acc"),
                    "steps": int(st.get("samples", 0)),
                    "actions": {ACTIONS[i]: int(round(float(p) * 1000))
                                for i, p in enumerate(dist) if p > 0.01},
                })

        super_X, super_Y = dagger_rounds(
            env, enc, policy, rng, X=super_X, Y=super_Y, rounds=args.dagger,
            episodes=args.dagger_episodes or args.bc_episodes, epochs=args.bc_epochs,
            max_frames=args.max_frames, scenario=train_scenario, label_mode=args.label,
            epsilon=args.dagger_epsilon, cap=args.dagger_cap, on_round=_on_round)
        score, detail = evaluate(env, enc, policy, rng, max_frames=args.max_frames,
                                 heldout=heldout, label_mode=args.label)
        agree = label_agreement(env, enc, policy, rng, max_frames=args.max_frames,
                                heldout=heldout, label_mode=args.label)
        meta["dagger"] = {"rounds": args.dagger, "score": round(score, 1),
                          "agree": round(agree, 4),
                          "detail": [int(x) for x in detail]}
        dist = action_profile(env, enc, policy, rng, max_frames=args.max_frames,
                              heldout=heldout, label_mode=args.label)
        print(f"  DAgger 之后: 泛化 {score:.0f} 帧  一致率 {agree:.0%}  "
              f"{[int(x) for x in detail]}")
        print("  动作分布: " + _dist_text(dist))

    _write_meta()

    best = 0.0
    t0 = time.time()
    for it in range(1, args.updates + 1):
        batch_obs, batch_act, batch_ret, batch_lab = [], [], [], []
        survived = []
        for _ in range(args.episodes):
            scenario = dict(train_scenario)
            scenario["gap_phase"] = float(rng.random())
            if args.curriculum:
                # 前 1/3: 宽缺口 (随机探索也能撞上); 中 1/3: 线性收窄; 后 1/3: 目标难度
                frac = it / args.updates
                lo, hi = 150.0, float(train_scenario["gap_width"])
                scenario["gap_width"] = hi if frac >= 2 / 3 else lo + (hi - lo) * (frac * 1.5)
            out = rollout(env, enc, policy, rng, seed=int(rng.integers(1, 10 ** 6)),
                          scenario=scenario, max_frames=args.max_frames,
                          shaping=args.shaping,
                          label_mode=args.label if args.bc_coef > 0 else None,
                          epsilon=(max(0.05, args.epsilon * (1.0 - it / args.updates))
                                   if args.epsilon > 0 else 0.0))
            if out["obs"].shape[0] == 0:
                continue
            batch_obs.append(out["obs"])
            batch_act.append(out["act"])
            batch_ret.append(discounted(out["rew"], args.gamma))
            batch_lab.append(out["label"])
            survived.append(out["frames"])
        if batch_obs:
            loss = policy.update(np.concatenate(batch_obs), np.concatenate(batch_act),
                                 np.concatenate(batch_ret),
                                 anchor=np.concatenate(batch_lab), bc_coef=args.bc_coef)
        else:
            loss = 0.0
        mean_frames = float(np.mean(survived)) if survived else 0.0
        if journal is not None:
            acts = np.bincount(np.concatenate(batch_act), minlength=len(ACTIONS)) \
                if batch_act else np.zeros(len(ACTIONS), dtype=int)
            journal.episode({
                "episode": it,
                "return": round(mean_frames, 2),
                "steps": int(acts.sum()),
                "loss": round(float(loss), 4),
                "actions": {ACTIONS[i]: int(acts[i]) for i in range(len(ACTIONS)) if acts[i]},
            })
        msg = (f"[{it:3d}/{args.updates}] 训练存活 {mean_frames:6.0f} 帧 "
               f"loss={loss:+.3f}  {time.time() - t0:5.1f}s")
        if it % args.eval_every == 0 or it == args.updates:
            score, detail = evaluate(env, enc, policy, rng, max_frames=args.max_frames,
                                     heldout=heldout, label_mode=args.label)
            agree = label_agreement(env, enc, policy, rng, max_frames=args.max_frames,
                                    heldout=heldout, label_mode=args.label)
            best = max(best, score)
            _eval_row(it, "heldout", score, detail, agree=round(agree, 4))
            msg += f" | 泛化 {score:6.0f} 帧 一致率 {agree:4.0%} {[int(x) for x in detail]}"
        print(msg, flush=True)

    score, detail = evaluate(env, enc, policy, rng, max_frames=args.max_frames,
                             heldout=heldout, label_mode=args.label)
    agree = label_agreement(env, enc, policy, rng, max_frames=args.max_frames,
                            heldout=heldout, label_mode=args.label)
    dist = action_profile(env, enc, policy, rng, max_frames=args.max_frames,
                          heldout=heldout, label_mode=args.label)
    meta["final"] = {"score": round(score, 1), "agree": round(agree, 4),
                     "detail": [int(x) for x in detail], "best": round(best, 1)}
    _write_meta()
    print(f"\n最终泛化 (未见过的相位/反向漂移/弹速): {score:.0f} 帧  "
          f"标签一致率 {agree:.0%}  {[int(x) for x in detail]}")
    print(f"最好一次泛化: {best:.0f} 帧")
    print("动作分布: " + _dist_text(dist))
    env.close()
    return 0


def _dist_text(dist: np.ndarray) -> str:
    return "  ".join(f"{ACTIONS[i].replace('+shoot', '') or 'stay'}={dist[i]:.0%}"
                    for i in np.argsort(-dist) if dist[i] > 0.02)


if __name__ == "__main__":
    raise SystemExit(main())
