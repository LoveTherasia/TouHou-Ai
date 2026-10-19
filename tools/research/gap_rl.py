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
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from touhou_rl_mcp.config import Config, runtime_dir   # noqa: E402
from touhou_rl_mcp.env import TouhouEnv            # noqa: E402
from touhou_rl_mcp.training.journal import Journal  # noqa: E402
from touhou_rl_mcp.training.occupancy import (     # noqa: E402
    GapObservation, GapSpec, _bullets, _tta_profile,
)

#: 动作表: 8 方向 + 原地, 每个动作承诺 6 帧 (60Hz 下的 0.1 秒)
ACTIONS: tuple[str, ...] = (
    "shoot", "up+shoot", "down+shoot", "left+shoot", "right+shoot",
    "up_left+shoot", "up_right+shoot", "down_left+shoot", "down_right+shoot",
)
MACRO_FRAMES = 6

#: 训练场景
TRAIN_SCENARIO: dict[str, float] = dict(
    interval=130.0, gap_width=60.0, speed=1.8, gap_drift=0.13, gap_phase=0.0,
)
#: 泛化测试: 训练时没出现过的相位 / 反向漂移 / 弹速
HELDOUT_SCENARIOS: tuple[dict[str, float], ...] = (
    dict(interval=130.0, gap_width=60.0, speed=1.8, gap_drift=-0.17, gap_phase=0.31),
    dict(interval=130.0, gap_width=60.0, speed=1.9, gap_drift=0.29, gap_phase=0.62),
    dict(interval=120.0, gap_width=64.0, speed=1.8, gap_drift=-0.24, gap_phase=0.15),
)


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
    profile = _tta_profile(state.player_x, state.player_y, pos, vel, spec)
    mid = spec.tta_cols // 2
    return float(profile[mid])


def rollout(env: TouhouEnv, enc: GapObservation, policy: Policy | None,
            rng: np.random.Generator, *, seed: int,
            scenario: dict[str, float], max_frames: int,
            greedy: bool = False, shaping: float = 0.0,
            epsilon: float = 0.0) -> dict:
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
        labels.append(gap_label(state, enc.spec))
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
            prof = _tta_profile(state.player_x, state.player_y, pos, vel, spec)
            k = int(np.argmax(prof))
            target = state.player_x - spec.span_x + (k + 0.5) * 2 * spec.span_x / spec.tta_cols
            delta = target - state.player_x
            action = 0 if abs(delta) < 8 else (4 if delta > 0 else 3)
        result = env.step(ACTIONS[action], frames=MACRO_FRAMES, auto_reset=False)
        state = result.state
        frames += MACRO_FRAMES
        if result.done:
            break
    return {"frames": frames}


def evaluate(env: TouhouEnv, enc: GapObservation, policy: Policy,
             rng: np.random.Generator, *, max_frames: int,
             seeds: tuple[int, ...] = (101, 202)) -> tuple[float, list[float]]:
    scores = []
    for scenario in HELDOUT_SCENARIOS:
        for seed in seeds:
            out = rollout(env, enc, policy, rng, seed=seed, scenario=scenario,
                          max_frames=max_frames, greedy=True)
            scores.append(out["frames"])
    return float(np.mean(scores)), scores


def action_profile(env: TouhouEnv, enc: GapObservation, policy: Policy,
                   rng: np.random.Generator, *, max_frames: int) -> np.ndarray:
    """贪心策略在测试场景下的动作分布 —— 用来识别"学会不动"这种退化。"""
    counts = np.zeros(len(ACTIONS))
    for scenario in HELDOUT_SCENARIOS:
        out = rollout(env, enc, policy, rng, seed=101, scenario=scenario,
                      max_frames=max_frames, greedy=True)
        for a in out["act"]:
            counts[a] += 1
    total = counts.sum()
    return counts / total if total else counts


def spec_for_arm(arm: str) -> GapSpec:
    if arm == "geo":
        return GapSpec(use_grid=False, use_geo=True, use_clearance=True)
    if arm == "grid":
        return GapSpec(use_grid=True, use_geo=False, use_clearance=False)
    return GapSpec(use_grid=True, use_geo=True, use_clearance=True)


def gap_label(state, spec: GapSpec) -> int:
    """单步几何标签: 缺口在自机左边 (3) / 右边 (4) / 已在脚下 (0)。

    这是**感知层面的弱监督** —— 只回答"缺口在哪", 不回答"该怎么躲"。它不依赖任何
    手写代价函数, 也不依赖搜索, 但足以把"看缺口"这件事教给网络; 真正怎么躲仍然由
    RL 在环境里优化。
    """
    pos, vel = _bullets(state)
    if pos.shape[0] == 0:
        return 0
    profile = _tta_profile(state.player_x, state.player_y, pos, vel, spec)
    k = int(np.argmax(profile))
    target = state.player_x - spec.span_x + (k + 0.5) * 2 * spec.span_x / spec.tta_cols
    delta = target - state.player_x
    if abs(delta) < 10.0:
        return 0
    return 4 if delta > 0 else 3


def collect_labels(env: TouhouEnv, enc: GapObservation, rng: np.random.Generator,
                   *, episodes: int, max_frames: int,
                   scenario: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    """用随机策略走一遍, 收集 (观测, 缺口方向) 对 —— 覆盖各种站位与相位。"""
    xs, ys = [], []
    for _ in range(episodes):
        state = reset_env(env, seed=int(rng.integers(1, 10 ** 6)), scenario=scenario)
        prev, frames = -1, 0
        while frames < max_frames:
            xs.append(enc(state, prev_action=prev).copy())
            ys.append(gap_label(state, enc.spec))
            action = int(rng.integers(len(ACTIONS)))
            result = env.step(ACTIONS[action], frames=MACRO_FRAMES, auto_reset=False)
            state, frames, prev = result.state, frames + MACRO_FRAMES, action
            if result.done:
                break
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.intp)


def pretrain(env: TouhouEnv, enc: GapObservation, policy: Policy,
             rng: np.random.Generator, *, episodes: int, epochs: int,
             max_frames: int, scenario: dict[str, float]) -> float:
    X, Y = collect_labels(env, enc, rng, episodes=episodes, max_frames=max_frames,
                          scenario=scenario)
    split = int(X.shape[0] * 0.8)
    Xtr, Ytr, Xte, Yte = X[:split], Y[:split], X[split:], Y[split:]
    for _ in range(epochs):
        order = rng.permutation(Xtr.shape[0])
        for start in range(0, order.shape[0], 256):
            idx = order[start:start + 256]
            policy.imitate(Xtr[idx], Ytr[idx])
    acc = float(np.mean(np.argmax(policy._forward(Xte)[2], axis=1) == Yte))
    train_acc = float(np.mean(np.argmax(policy._forward(Xtr)[2], axis=1) == Ytr))
    print(f"  感知预训练: {X.shape[0]} 样本  训练准确率 {train_acc:.0%}  "
          f"留出准确率 {acc:.0%}")
    return acc


# ----------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="缺口任务上的最小 RL 实验")
    ap.add_argument("--arm", choices=("geo", "grid", "both"), default="both")
    ap.add_argument("--updates", type=int, default=60)
    ap.add_argument("--episodes", type=int, default=16, help="每次更新采集多少局")
    ap.add_argument("--max-frames", type=int, default=1800, help="单局上限帧数")
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
    ap.add_argument("--entropy", type=float, default=0.02)
    ap.add_argument("--train-override", default="",
                    help="覆盖训练场景, 逗号分隔 key=value (课程用), 如 gap_width=140,interval=180")
    ap.add_argument("--curriculum", action="store_true",
                    help="宽缺口 -> 窄缺口的课程: 前 1/3 用宽缺口, 中 1/3 过渡")
    args = ap.parse_args()

    train_scenario = dict(TRAIN_SCENARIO)
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

    print(f"arm={args.arm} obs_dim={enc.dim} {enc.describe()}")
    print(f"train scenario: {train_scenario}")

    if args.baseline:
        noop = [rollout(env, enc, None, rng, seed=101 + i, scenario=train_scenario,
                        max_frames=args.max_frames)["frames"] for i in range(3)]
        follow = [gap_follower(env, spec, seed=101 + i, scenario=train_scenario,
                               max_frames=args.max_frames)["frames"] for i in range(3)]
        print(f"  noop 基准: {np.mean(noop):7.0f} 帧   {noop}")
        print(f"  缺口跟随: {np.mean(follow):7.0f} 帧   {follow}")

    if args.bc_episodes > 0:
        pretrain(env, enc, policy, rng, episodes=args.bc_episodes,
                 epochs=args.bc_epochs, max_frames=args.max_frames,
                 scenario=train_scenario)
        score, detail = evaluate(env, enc, policy, rng, max_frames=args.max_frames)
        print(f"  仅感知预训练 (无 RL): 泛化 {score:.0f} 帧  {[int(x) for x in detail]}")
        dist = action_profile(env, enc, policy, rng, max_frames=args.max_frames)
        print("  动作分布: " + "  ".join(
            f"{ACTIONS[i].replace('+shoot', '') or 'stay'}={dist[i]:.0%}"
            for i in np.argsort(-dist) if dist[i] > 0.02))

    journal = None
    run_dir = None
    if not args.no_journal:
        run_dir = Path(args.run_dir) if args.run_dir else (
            runtime_dir() / "runs" / f"{args.arm}-{time.strftime('%Y%m%d-%H%M%S')}")
        journal = Journal(run_dir)
        (run_dir / "meta.json").write_text(json.dumps({
            "script": "tools/research/gap_rl.py",
            "arm": args.arm,
            "obs": enc.describe(),
            "train_scenario": train_scenario,
            "heldout": HELDOUT_SCENARIOS,
            "args": vars(args),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"训练记录: {run_dir}")

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
            score, detail = evaluate(env, enc, policy, rng, max_frames=args.max_frames)
            best = max(best, score)
            if run_dir is not None:
                with open(run_dir / "eval.jsonl", "a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"step": it, "name": "heldout",
                                         "score": round(score, 1),
                                         "detail": [int(x) for x in detail]},
                                        ensure_ascii=False) + "\n")
            msg += f" | 泛化 {score:6.0f} 帧 {[int(x) for x in detail]}"
        print(msg, flush=True)

    score, detail = evaluate(env, enc, policy, rng, max_frames=args.max_frames)
    dist = action_profile(env, enc, policy, rng, max_frames=args.max_frames)
    print(f"\n最终泛化 (未见过的相位/反向漂移/弹速): {score:.0f} 帧  {[int(x) for x in detail]}")
    print(f"最好一次泛化: {best:.0f} 帧")
    print("动作分布: " + "  ".join(
        f"{ACTIONS[i].replace('+shoot', '') or 'stay'}={dist[i]:.0%}"
        for i in np.argsort(-dist) if dist[i] > 0.02))
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
