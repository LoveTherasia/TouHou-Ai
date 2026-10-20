"""学习 V(s) + 短视界规划器 + 蒸馏策略（实验性）。

三阶段：

1. ``--stage value``：用规则老师 / 随机策略在多种弹型上收集轨迹，标签是
   “距死亡还有多少帧”，训练一个 numpy MLP 价值网络；
2. ``--stage distill``：用价值网络做短视界 beam search，生成 ``(状态, 动作)``
   示范，再用 BC 蒸馏成策略；
3. ``--stage eval``：在固定种子的弹型上评估蒸馏策略。

用法::

    python tools/research/value_distill.py --stage value
    python tools/research/value_distill.py --stage distill
    python tools/research/value_distill.py --stage eval
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from touhou_rl_mcp.config import Config  # noqa: E402
from touhou_rl_mcp.env import TouhouEnv  # noqa: E402
from touhou_rl_mcp.training.features import encode_state  # noqa: E402
from touhou_rl_mcp.training.gap_label import MACRO_ACTIONS, update_action_age  # noqa: E402
from touhou_rl_mcp.training.planner import MacroPlanner  # noqa: E402
from touhou_rl_mcp.training.policy import MLPPolicy  # noqa: E402
from touhou_rl_mcp.training.rule_teacher import RuleTeacher  # noqa: E402
from touhou_rl_mcp.types import (  # noqa: E402
    GameState, PLAYFIELD_HEIGHT, PLAYFIELD_WIDTH,
)

ROOT = Path(".runtime/value_distill")
#: 价值网络 / 蒸馏策略共用的观测 (不含 boss/enemy, 叶子状态容易构造)
FEATURES: dict[str, Any] = dict(
    k_bullets=16, use_velocity=True, k_enemies=0,
    use_invincible=False, use_boss=False, use_danger_grad=True,
)
ACTIONS = tuple(MACRO_ACTIONS)
PATTERNS = ("aimed_single", "aimed_spread", "dense_ring", "ring_gate",
            "closing_ring", "wall_gate", "random_spray")


def policy_obs(state: GameState, prev_action: int, age: int,
               cap: int = 12) -> np.ndarray:
    """策略观测 = 弹幕特征 + 上一动作 one-hot + 同向连续步数。

    规划器是有状态的 (它记得上一步/承诺), 只给当前帧的 BC 会有**同状态不同动作**
    的歧义; 加上上一动作是最低成本的消歧。
    """
    base = encode_state(state, **FEATURES)
    extra = np.zeros(len(ACTIONS) + 1)
    if 0 <= prev_action < len(ACTIONS):
        extra[prev_action] = 1.0
    extra[-1] = min(1.0, max(0, age) / float(cap))
    return np.concatenate([base, extra])


def make_env(pattern: str, seed: int, step_frames: int = 1,
             randomize: bool = False) -> TouhouEnv:
    env = TouhouEnv(Config({
        "backend": "mock", "game": "th11", "danmaku_pattern": pattern,
        "step_frames": int(step_frames), "max_episode_frames": 0,
        "auto_reset": False,
    }))
    env.start()
    env.reset(seed=seed)
    if randomize:
        randomize_episode(env, seed)
    return env


def randomize_episode(env: TouhouEnv, seed: int) -> None:
    """域随机化: 自机起点 + 弹型参数。

    同一 seed 的 mock 弹型是确定性的; 不做随机化, 所谓"多个 seed"只是同一条轨迹
    重复, BC 会以为世界只有一条路线。这里只改"初始条件/节奏", 不改弹型结构。
    """
    game = env.backend.game
    rng = np.random.default_rng(seed)
    game.px = float(rng.uniform(72.0, 312.0))
    game.py = float(rng.uniform(250.0, 416.0))
    pattern = getattr(game, "pattern", None)
    if pattern is None:
        return
    for name, lo, hi in (("interval", 0.85, 1.15), ("speed", 0.90, 1.10),
                         ("radius", 0.92, 1.08)):
        value = getattr(pattern, name, None)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            setattr(pattern, name, type(value)(value * rng.uniform(lo, hi)))
    if hasattr(pattern, "x"):
        setattr(pattern, "x", float(rng.uniform(64.0, 320.0)))
    if hasattr(pattern, "gap_phase"):
        phase = getattr(pattern, "gap_phase")
        if isinstance(phase, int):
            count = int(getattr(pattern, "count", 1) or 1)
            setattr(pattern, "gap_phase", int(rng.integers(0, count)))
        else:
            setattr(pattern, "gap_phase", float(rng.uniform(0.0, 1.0)))
    if hasattr(pattern, "gap_drift"):
        setattr(pattern, "gap_drift", float(rng.uniform(-0.9, 0.9)))


def _edge_xy(px: float, py: float, margin: float = 24.0) -> float:
    dx = max(0.0, (margin - min(px, PLAYFIELD_WIDTH - px)) / margin)
    dy = max(0.0, (margin - min(py, PLAYFIELD_HEIGHT - py)) / margin)
    return float(min(1.0, dx + dy))


def edge_penalty(state: GameState, margin: float = 24.0) -> float:
    """自机离边界越近, 惩罚越大 (0~1)。

    没有这一项, V 很容易学成"角落最安全": 在回放里很多弹幕覆盖不到底角, 策略于是
    整局贴在 (376,440), 存活数字好看但不是真正的躲避。真实弹幕会用自机狙/环形弹
    惩罚这种站位。
    """
    px = float(state.player_x if state.player_x is not None else PLAYFIELD_WIDTH / 2)
    py = float(state.player_y if state.player_y is not None else PLAYFIELD_HEIGHT - 32)
    return _edge_xy(px, py, margin)


# ---------------------------------------------------------------------------
class ValueNet:
    """两层 ReLU MLP 回归器 (numpy Adam)。输出 0~1 的"未来生存价值"。"""

    def __init__(self, dim: int, hidden: int = 128, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.dim = int(dim)
        self.hidden = int(hidden)
        self.W1 = rng.normal(0.0, math.sqrt(2.0 / max(1, dim)), (dim, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0.0, math.sqrt(2.0 / max(1, hidden)), (hidden, 1))
        self.b2 = np.zeros(1)
        self.mu = np.zeros(dim)
        self.sigma = np.ones(dim)
        self._m = [np.zeros_like(p) for p in self.params()]
        self._v = [np.zeros_like(p) for p in self.params()]

    def params(self) -> list[np.ndarray]:
        return [self.W1, self.b1, self.W2, self.b2]

    def _norm(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=np.float64) - self.mu) / self.sigma

    def _forward(self, Xn: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = np.maximum(0.0, Xn @ self.W1 + self.b1)
        y = h @ self.W2 + self.b2
        return h, y[:, 0]

    def predict(self, X: np.ndarray) -> np.ndarray:
        if np.asarray(X).ndim == 1:
            X = np.asarray(X)[None, :]
        _h, y = self._forward(self._norm(X))
        return np.clip(y, 0.0, 1.0)

    def fit(self, X: np.ndarray, y: np.ndarray, *, epochs: int = 30,
            batch: int = 256, lr: float = 1e-3, val_frac: float = 0.1,
            seed: int = 0, verbose: bool = True) -> dict[str, float]:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        rng = np.random.default_rng(seed)
        order = rng.permutation(X.shape[0])
        n_val = max(1, int(X.shape[0] * val_frac))
        val_idx, train_idx = order[:n_val], order[n_val:]
        self.mu = X[train_idx].mean(axis=0)
        self.sigma = np.maximum(X[train_idx].std(axis=0), 1e-6)
        Xtr, ytr = self._norm(X[train_idx]), y[train_idx]
        Xva, yva = self._norm(X[val_idx]), y[val_idx]
        b1 = 0.9
        b2 = 0.999
        eps = 1e-8
        step = 0
        best = (float("inf"), None)
        for epoch in range(epochs):
            perm = rng.permutation(Xtr.shape[0])
            total = 0.0
            for start in range(0, len(perm), batch):
                idx = perm[start:start + batch]
                Xb, yb = Xtr[idx], ytr[idx]
                h = np.maximum(0.0, Xb @ self.W1 + self.b1)
                pred = h @ self.W2 + self.b2
                err = (pred[:, 0] - yb) / max(1, len(idx))
                total += float(np.sum((pred[:, 0] - yb) ** 2))
                gW2 = h.T @ err[:, None]
                gb2 = np.array([err.sum()])
                dh = (err[:, None] @ self.W2.T) * (h > 0)
                gW1 = Xb.T @ dh
                gb1 = dh.sum(axis=0)
                grads = [gW1, gb1, gW2, gb2]
                step += 1
                for i, (p, g) in enumerate(zip(self.params(), grads)):
                    self._m[i] = b1 * self._m[i] + (1 - b1) * g
                    self._v[i] = b2 * self._v[i] + (1 - b2) * (g * g)
                    mh = self._m[i] / (1 - b1 ** step)
                    vh = self._v[i] / (1 - b2 ** step)
                    p -= lr * mh / (np.sqrt(vh) + eps)
            _h, val_pred = self._forward(Xva)
            val_mse = float(np.mean((val_pred - yva) ** 2))
            if val_mse < best[0]:
                best = (val_mse, [p.copy() for p in self.params()])
            if verbose and (epoch == 0 or epoch == epochs - 1 or (epoch + 1) % 10 == 0):
                print(f"  epoch {epoch+1:3d}/{epochs}  train_mse={total/max(1,len(train_idx)):.5f}  "
                      f"val_mse={val_mse:.5f}", flush=True)
        if best[1] is not None:
            for p, saved in zip(self.params(), best[1]):
                p[...] = saved
        _h, val_pred = self._forward(Xva)
        mae = float(np.mean(np.abs(val_pred - yva)))
        corr = float(np.corrcoef(val_pred, yva)[0, 1]) if len(yva) > 1 else 0.0
        return {"val_mse": best[0], "val_mae": mae, "val_corr": corr,
                "train_samples": int(len(train_idx)), "val_samples": int(len(val_idx))}

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2,
                 mu=self.mu, sigma=self.sigma, hidden=self.hidden, dim=self.dim)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "ValueNet":
        data = np.load(path, allow_pickle=False)
        net = cls(int(data["dim"]), int(data["hidden"]))
        net.W1 = data["W1"]
        net.b1 = data["b1"]
        net.W2 = data["W2"]
        net.b2 = data["b2"]
        net.mu = data["mu"]
        net.sigma = data["sigma"]
        net._m = [np.zeros_like(p) for p in net.params()]
        net._v = [np.zeros_like(p) for p in net.params()]
        return net


# ---------------------------------------------------------------------------
def collect_value_data(*, max_frames: int = 2400, cap: int = 600,
                       seeds: tuple[int, ...] = (11, 17),
                       patterns: tuple[str, ...] = PATTERNS,
                       edge_weight: float = 0.5,
                       out: Path = ROOT / "value_data.npz") -> dict[str, Any]:
    """规则老师 + 随机策略的轨迹; 标签 = min(剩余帧, cap)/cap。"""
    X: list[np.ndarray] = []
    y: list[float] = []
    episodes: list[dict[str, Any]] = []
    for pattern in patterns:
        for seed in seeds:
            for mode in ("rule", "random"):
                env = make_env(pattern, seed, randomize=True)
                state = env.get_state()
                teacher = RuleTeacher(planner="horizon", focus_radius=0.0)
                rng = np.random.default_rng(seed)
                samples: list[tuple[np.ndarray, int, float]] = []
                frames = 0
                while frames < max_frames:
                    if frames % 2 == 0:
                        samples.append((encode_state(state, **FEATURES), frames,
                                        edge_penalty(state)))
                    if mode == "rule":
                        action, _info = teacher.choose(state, state.frame, "")
                    else:
                        action = ACTIONS[int(rng.integers(len(ACTIONS)))]
                    result = env.step(action, frames=1, auto_reset=False)
                    state = result.state
                    frames += 1
                    if result.done:
                        break
                end = frames
                for feats, f, edge in samples:
                    X.append(feats)
                    base = min(cap, max(0, end - f)) / float(cap)
                    y.append(max(0.0, base - edge_weight * edge))
                episodes.append({"pattern": pattern, "seed": seed, "mode": mode,
                                 "frames": end, "samples": len(samples)})
                print(f"  {pattern:14s} seed={seed:3d} {mode:6s} frames={end:4d} "
                      f"samples={len(samples)}", flush=True)
                env.close()
    Xa = np.asarray(X, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, X=Xa, y=ya)
    meta = {"episodes": episodes, "samples": int(len(ya)), "dim": int(Xa.shape[1])}
    (out.with_suffix(".json")).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


# ---------------------------------------------------------------------------
def _leaf_state(state: GameState, x: float, y: float, t: int,
                bullets: np.ndarray, velocities: np.ndarray) -> GameState:
    pos = bullets + velocities * t
    keep = ((pos[:, 0] >= -40) & (pos[:, 0] <= PLAYFIELD_WIDTH + 40)
            & (pos[:, 1] >= -40) & (pos[:, 1] <= PLAYFIELD_HEIGHT + 40))
    pos = pos[keep]
    vel = velocities[keep]
    raw = {"bullet_velocity": vel.tolist(), "source": "value_planner"}
    return GameState(
        frame=int(state.frame + t), in_game=True, stage=state.stage or 1,
        lives=state.lives, power=state.power, score=state.score, graze=state.graze,
        player_x=float(x), player_y=float(y), bullet_count=len(pos),
        bullets=[(float(a), float(b)) for a, b in pos],
        raw=raw,
    )


class ValuePlanner:
    """短视界 beam search + 学习到的 V(s) 终端评分。"""

    def __init__(self, value: ValueNet, *, dt: int = 3, depth: int = 10,
                 beam: int = 24, margin: float = 5.0,
                 switch_penalty: float = 0.02, edge_weight: float = 0.5) -> None:
        self.value = value
        self.dt = max(1, int(dt))
        self.depth = max(1, int(depth))
        self.beam = max(1, int(beam))
        self.switch_penalty = float(switch_penalty)
        self.edge_weight = float(edge_weight)
        self.core = MacroPlanner(macro_frames=self.dt, horizon=self.dt * self.depth,
                                 beam=self.beam, margin=margin, speed=4.4,
                                 near_radius=500.0)
        self.last_action = 0

    def plan(self, state: GameState) -> int:
        px = float(state.player_x if state.player_x is not None else PLAYFIELD_WIDTH / 2)
        py = float(state.player_y if state.player_y is not None else PLAYFIELD_HEIGHT - 32)
        bullets, velocities = self.core._bullets(state, px, py)
        if not len(bullets):
            return 0
        # beam: (bad, worst, x, y, first, last)
        beam: list[tuple[int, float, float, float, int, int]] = [
            (0, float("inf"), px, py, -1, self.last_action)
        ]
        for level in range(self.depth):
            t0 = level * self.dt
            cand: list[tuple[int, float, float, float, int, int]] = []
            for bad, worst, x, y, first, last in beam:
                for index, (ux, uy) in enumerate(_DIRECTIONS):
                    seg_bad, seg_min, nx, ny = self.core._segment(
                        x, y, ux, uy, bullets, velocities, t0)
                    change = 0.0 if last < 0 or index == last else 0.1
                    cand.append((bad + seg_bad, min(worst, seg_min) - change,
                                 nx, ny, index if first < 0 else first, index))
            cand.sort(key=lambda item: (
                item[0],
                -(item[1] - self.edge_weight * _edge_xy(item[2], item[3])),
            ))
            beam = cand[: self.beam]
        # 叶子: V(s) + 安全间隙; bad 重的路径直接压下去
        t_leaf = self.depth * self.dt
        feats = []
        meta = []
        leaves = []
        for bad, worst, x, y, first, last in beam:
            leaf = _leaf_state(state, x, y, t_leaf, bullets, velocities)
            feats.append(encode_state(leaf, **FEATURES))
            leaves.append(leaf)
            meta.append((bad, worst, first, last))
        values = self.value.predict(np.asarray(feats, dtype=np.float64))
        best = None
        for value, (bad, worst, first, last), leaf in zip(values, meta, leaves):
            score = float(value) - 0.5 * bad + 0.01 * min(worst, 20.0) \
                - self.edge_weight * edge_penalty(leaf) \
                - self.switch_penalty * (0 if last == self.last_action else 1)
            if best is None or score > best[0]:
                best = (score, first)
        action = 0 if best is None else int(best[1])
        self.last_action = action
        return action


#: 与 trainer / MacroPlanner 同序
_DIRECTIONS: tuple[tuple[int, int], ...] = (
    (0, 0), (0, -1), (0, 1), (-1, 0), (1, 0),
    (-1, -1), (1, -1), (-1, 1), (1, 1),
)


# ---------------------------------------------------------------------------
def generate_demos(value: ValueNet, *, patterns: tuple[str, ...] = PATTERNS,
                   seeds: tuple[int, ...] = (11,), max_frames: int = 2400,
                   dt: int = 3, depth: int = 10, beam: int = 24,
                   edge_weight: float = 0.5,
                   out: Path = ROOT / "demos.npz") -> dict[str, Any]:
    X: list[np.ndarray] = []
    A: list[int] = []
    rows: list[dict[str, Any]] = []
    for pattern in patterns:
        for seed in seeds:
            env = make_env(pattern, seed, step_frames=dt, randomize=True)
            state = env.get_state()
            planner = ValuePlanner(value, dt=dt, depth=depth, beam=beam,
                                   edge_weight=edge_weight)
            frames = 0
            prev = -1
            age = 0
            counts: dict[str, int] = {}
            while frames < max_frames:
                action = planner.plan(state)
                X.append(policy_obs(state, prev, age))
                A.append(action)
                name = ACTIONS[action]
                counts[name] = counts.get(name, 0) + 1
                result = env.step(name, frames=dt, auto_reset=False)
                state = result.state
                frames += dt
                age = update_action_age(prev, age, action, 12)
                prev = action
                if result.done:
                    break
            rows.append({"pattern": pattern, "seed": seed, "frames": frames,
                         "samples": sum(counts.values()), "actions": counts})
            print(f"  {pattern:14s} seed={seed:3d} frames={frames:4d} "
                  f"actions={counts}", flush=True)
            env.close()
    Xa = np.asarray(X, dtype=np.float64)
    Aa = np.asarray(A, dtype=np.int64)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, X=Xa, A=Aa)
    meta = {"rows": rows, "samples": int(len(Aa)), "dim": int(Xa.shape[1])}
    (out.with_suffix(".json")).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def train_policy(data: Path, out: Path, *, epochs: int = 40, batch: int = 128,
                 lr: float = 1e-3, seed: int = 0) -> dict[str, Any]:
    data = np.load(data, allow_pickle=False)
    X, A = data["X"], data["A"]
    policy = MLPPolicy(X.shape[1], len(ACTIONS), hidden=(128, 128),
                       seed=seed, lr=lr, entropy_coef=0.0)
    rng = np.random.default_rng(seed)
    for epoch in range(epochs):
        perm = rng.permutation(len(A))
        losses = []
        for start in range(0, len(perm), batch):
            idx = perm[start:start + batch]
            stats = policy.imitate_batch(X[idx], A[idx], entropy_coef=0.0)
            losses.append(stats["imitation_loss"])
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"  bc epoch {epoch+1:3d}/{epochs} loss={np.mean(losses):.4f}",
                  flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    policy.save(out)
    return {"samples": int(len(A)), "obs_dim": int(X.shape[1]),
            "actions": len(ACTIONS), "out": str(out)}


def eval_policy(policy_path: Path, *, patterns: tuple[str, ...] = PATTERNS,
                seeds: tuple[int, ...] = (11, 17, 23), max_frames: int = 2400,
                dt: int = 3) -> dict[str, Any]:
    policy = MLPPolicy.load(policy_path)
    results: dict[str, list[int]] = {}
    for pattern in patterns:
        vals = []
        for seed in seeds:
            env = make_env(pattern, seed, step_frames=dt, randomize=True)
            state = env.get_state()
            frames = 0
            prev = -1
            age = 0
            corner = total = 0
            while frames < max_frames:
                x = float(state.player_x or 0.0)
                y = float(state.player_y or 0.0)
                if x < 24 or x > PLAYFIELD_WIDTH - 24 or y > PLAYFIELD_HEIGHT - 32:
                    corner += 1
                total += 1
                obs = policy_obs(state, prev, age)
                action = ACTIONS[int(policy.greedy(obs))]
                result = env.step(action, frames=dt, auto_reset=False)
                state = result.state
                frames += dt
                index = ACTIONS.index(action)
                age = update_action_age(prev, age, index, 12)
                prev = index
                if result.done:
                    break
            vals.append(frames)
            env.close()
        results[pattern] = vals
        print(f"  {pattern:14s} {vals} mean={np.mean(vals):.0f} "
              f"corner={corner/max(1,total):.0%}", flush=True)
    return results


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="V(s) + 短视界规划 + 蒸馏")
    ap.add_argument("--stage", choices=["value", "distill", "eval", "all"],
                    default="all")
    ap.add_argument("--max-frames", type=int, default=2400)
    ap.add_argument("--seeds", default="11,17")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--value-epochs", type=int, default=30)
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--beam", type=int, default=24)
    ap.add_argument("--dt", type=int, default=3)
    ap.add_argument("--edge-weight", type=float, default=0.5,
                    help="边界惩罚权重 (避免学成贴角/贴边)")
    args = ap.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    seeds = tuple(int(x) for x in args.seeds.split(",") if x.strip())
    value_path = ROOT / "value.npz"
    demo_path = ROOT / "demos.npz"
    policy_path = ROOT / "policy.npz"

    if args.stage in ("value", "all"):
        print("== collect value data ==", flush=True)
        meta = collect_value_data(max_frames=args.max_frames, seeds=seeds,
                                  edge_weight=args.edge_weight)
        print(f"  samples={meta['samples']} dim={meta['dim']}", flush=True)
        data = np.load(ROOT / "value_data.npz", allow_pickle=False)
        print("== train V(s) ==", flush=True)
        t0 = time.perf_counter()
        value = ValueNet(data["X"].shape[1], hidden=128)
        stats = value.fit(data["X"], data["y"], epochs=args.value_epochs)
        value.save(value_path)
        print(f"  V: {stats}  ({time.perf_counter()-t0:.1f}s) -> {value_path}", flush=True)

    if args.stage in ("distill", "all"):
        if not value_path.exists():
            raise SystemExit(f"缺少 {value_path}; 先跑 --stage value")
        value = ValueNet.load(value_path)
        print("== generate planner demos ==", flush=True)
        demo = generate_demos(value, seeds=(seeds[0],), max_frames=args.max_frames,
                              dt=args.dt, depth=args.depth, beam=args.beam,
                              edge_weight=args.edge_weight)
        print(f"  demos={demo['samples']} -> {demo_path}", flush=True)
        print("== BC distill ==", flush=True)
        stats = train_policy(demo_path, policy_path, epochs=args.epochs)
        print(f"  policy: {stats}", flush=True)

    if args.stage in ("eval", "all"):
        if not policy_path.exists():
            raise SystemExit(f"缺少 {policy_path}; 先跑 --stage distill")
        print("== eval distilled policy ==", flush=True)
        results = eval_policy(policy_path, max_frames=args.max_frames, dt=args.dt)
        (ROOT / "eval.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
