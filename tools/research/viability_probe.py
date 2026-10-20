"""时间展开可行集规划器: 用"未来 H 帧是否还有活路"选动作。

和 `planner.py` 的 beam search 不同, 这里不做"离弹越远越好"的代价优化, 而是:

1. 把场地离散成 2D 网格;
2. 对每个未来时刻 t, 标记哪些格子会被弹幕覆盖 (`safe[t]`);
3. 从 H 反向递推 `viable[t][x,y]` = 此刻安全, 且存在一个动作能走到
   `viable[t+1]`;
4. 当前动作选"下一步仍在可行集里"的那个。

这样"先上移、再横移穿过环的缺口"会作为一条可行路径被保留; 贴底只有在它
仍然是唯一活路时才会被选。弹道用已知的 (位置 + 速度) 线性外推。

用法::

    python tools/research/viability_probe.py `
        --script .runtime/danmaku/th11_replay_full.json `
        --fracs 0.71,0.83,0.89 --dt 3 --horizon 45
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gap_trainer import apply_script_offset, script_length  # noqa: E402

from touhou_rl_mcp.config import Config  # noqa: E402
from touhou_rl_mcp.env import TouhouEnv  # noqa: E402
from touhou_rl_mcp.training.gap_label import MACRO_ACTIONS  # noqa: E402
from touhou_rl_mcp.types import PLAYFIELD_HEIGHT, PLAYFIELD_WIDTH  # noqa: E402

#: 与 trainer.DIRECTIONS 同序: (dx, dy)
MOVES: tuple[tuple[int, int], ...] = (
    (0, 0), (0, -1), (0, 1), (-1, 0), (1, 0),
    (-1, -1), (1, -1), (-1, 1), (1, 1),
)
PLAYER_SPEED = 4.4


def _shift(src: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """``out[y, x] = src[y + dy, x + dx]``, 边界补 False。"""
    rows, cols = src.shape
    out = np.zeros_like(src)
    if dy >= 0:
        sy0, sy1, dy0, dy1 = dy, rows, 0, rows - dy
    else:
        sy0, sy1, dy0, dy1 = 0, rows + dy, -dy, rows
    if dx >= 0:
        sx0, sx1, dx0, dx1 = dx, cols, 0, cols - dx
    else:
        sx0, sx1, dx0, dx1 = 0, cols + dx, -dx, cols
    if dy0 < dy1 and dx0 < dx1:
        out[dy0:dy1, dx0:dx1] = src[sy0:sy1, sx0:sx1]
    return out


class ViabilityPlanner:
    """时间展开 2D 可行集规划器。"""

    def __init__(self, *, dt: int = 3, horizon: int = 45, grid: float = 4.0,
                 hit_radius: float = 5.0, bottom_prior: float = 0.0) -> None:
        self.dt = max(1, int(dt))
        self.horizon = max(1, int(horizon))
        self.grid = float(grid)
        self.hit_radius = float(hit_radius)
        self.bottom_prior = float(bottom_prior)
        self.cols = int(math.ceil(PLAYFIELD_WIDTH / self.grid)) + 1
        self.rows = int(math.ceil(PLAYFIELD_HEIGHT / self.grid)) + 1
        self.radius_cells = max(1, int(math.ceil(hit_radius / self.grid)))
        self.move_cells = max(1, int(round(PLAYER_SPEED * self.dt / self.grid)))
        self.moves = [
            (max(-2, min(2, dx)) * self.move_cells,
             max(-2, min(2, dy)) * self.move_cells)
            for dx, dy in MOVES
        ]
        #: 玩家可活动范围 (格)
        self.y_min = int(math.ceil(32.0 / self.grid))
        self.y_max = int(math.floor((PLAYFIELD_HEIGHT - 16.0) / self.grid))
        self.x_min = int(math.ceil(8.0 / self.grid))
        self.x_max = int(math.floor((PLAYFIELD_WIDTH - 8.0) / self.grid))
        self.last_action = 0

    # ------------------------------------------------------------------
    def _occupancy(self, state) -> np.ndarray:
        """``(H+1, rows, cols)``: 每个未来层被弹幕覆盖的格子。"""
        occ = np.zeros((self.horizon + 1, self.rows, self.cols), dtype=bool)
        bullets = list(getattr(state, "bullets", []) or [])
        if not bullets:
            return occ
        pos = np.asarray(bullets, dtype=np.float64).reshape(-1, 2)
        raw = (getattr(state, "raw", None) or {}).get("bullet_velocity") or []
        try:
            vel = np.asarray(raw, dtype=np.float64).reshape(-1, 2)
            if vel.shape[0] != pos.shape[0]:
                raise ValueError
        except Exception:
            vel = np.zeros_like(pos)

        offsets = []
        r = self.radius_cells
        #: 只标记"格子中心到弹道中心 <= 命中半径"的格子; 不要用整个半径方块,
        #: 那会把 5px 的判定扩成 8~9px, 在密集弹里制造不存在的死路。
        limit = (self.hit_radius / self.grid) ** 2
        for oy in range(-r, r + 1):
            for ox in range(-r, r + 1):
                if ox * ox + oy * oy <= limit:
                    offsets.append((oy, ox))
        for t in range(self.horizon + 1):
            xs = pos[:, 0] + vel[:, 0] * (t * self.dt)
            ys = pos[:, 1] + vel[:, 1] * (t * self.dt)
            ix = np.rint(xs / self.grid).astype(np.intp)
            iy = np.rint(ys / self.grid).astype(np.intp)
            ok = (ix >= 0) & (ix < self.cols) & (iy >= 0) & (iy < self.rows)
            if not np.any(ok):
                continue
            ix, iy = ix[ok], iy[ok]
            for oy, ox in offsets:
                gy = iy + oy
                gx = ix + ox
                good = (gy >= 0) & (gy < self.rows) & (gx >= 0) & (gx < self.cols)
                occ[t, gy[good], gx[good]] = True
        return occ

    def _dp(self, safe: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """返回 (viable, life), 形状与 safe 相同。"""
        layers = safe.shape[0]
        viable = np.zeros_like(safe)
        life = np.zeros_like(safe, dtype=np.int16)
        viable[-1] = safe[-1]
        for t in range(layers - 2, -1, -1):
            reach = np.zeros((self.rows, self.cols), dtype=bool)
            best_life = np.zeros((self.rows, self.cols), dtype=np.int16)
            nxt_v = viable[t + 1]
            nxt_l = life[t + 1]
            for dy, dx in self.moves:
                reach |= _shift(nxt_v, dy, dx)
                best_life = np.maximum(best_life, _shift(nxt_l, dy, dx))
            viable[t] = safe[t] & reach
            life[t] = np.where(safe[t], best_life + 1, 0)
        return viable, life

    # ------------------------------------------------------------------
    def plan(self, state) -> int:
        """返回动作下标 (0~8, 与 MACRO_ACTIONS 同序)。"""
        t0 = time.perf_counter()
        occ = self._occupancy(state)
        safe = ~occ
        # 玩家活动范围之外不可站
        safe[:, : self.y_min, :] = False
        safe[:, self.y_max + 1:, :] = False
        safe[:, :, : self.x_min] = False
        safe[:, :, self.x_max + 1:] = False
        # 当前子弹所在格: 即使弹幕刚离开, 也不允许贴脸 (保守)
        safe[0] &= ~occ[0]

        viable, life = self._dp(safe)
        px = float(state.player_x if state.player_x is not None else PLAYFIELD_WIDTH / 2)
        py = float(state.player_y if state.player_y is not None else PLAYFIELD_HEIGHT - 32)
        ix = int(round(px / self.grid))
        iy = int(round(py / self.grid))
        ix = max(0, min(self.cols - 1, ix))
        iy = max(0, min(self.rows - 1, iy))

        best = None
        for index, (dy, dx) in enumerate(self.moves):
            ny, nx = iy + dy, ix + dx
            if not (0 <= ny < self.rows and 0 <= nx < self.cols):
                continue
            is_viable = bool(viable[1, ny, nx]) if self.horizon >= 1 else True
            score = (
                1 if is_viable else 0,
                int(life[1, ny, nx]),
                -int(round(self.bottom_prior * ny)),
                -abs(index - self.last_action),  # 轻微偏好不换动作
            )
            if best is None or score > best[0]:
                best = (score, index)
        action = 0 if best is None else int(best[1])
        self.last_action = action
        self.last_plan_ms = (time.perf_counter() - t0) * 1000.0
        return action


def make_env(script: str, dt: int) -> TouhouEnv:
    cfg = Config.load()
    cfg.patch({
        "backend": "mock", "game": "th11", "danmaku_pattern": "",
        "danmaku_script": script, "step_frames": int(dt),
        "max_episode_frames": 0, "auto_reset": False, "reward": {},
    })
    env = TouhouEnv(cfg)
    env.start()
    return env


def run(env: TouhouEnv, planner: ViabilityPlanner | None, frac: float,
        *, max_frames: int, dt: int) -> dict:
    apply_script_offset(env, frac)
    state = env.reset(seed=101).state
    survival = 0
    actions = []
    plan_ms = []
    while survival < max_frames:
        if planner is None:
            action = 0  # 站桩射击
        else:
            action = planner.plan(state)
            plan_ms.append(planner.last_plan_ms)
        actions.append(MACRO_ACTIONS[action])
        result = env.step(MACRO_ACTIONS[action], frames=dt, auto_reset=False)
        state = result.state
        survival += dt
        if result.done:
            break
    return {
        "frac": frac, "frames": survival, "steps": len(actions),
        "avg_plan_ms": round(float(np.mean(plan_ms)), 2) if plan_ms else 0.0,
        "max_plan_ms": round(float(np.max(plan_ms)), 2) if plan_ms else 0.0,
        "actions": {a: actions.count(a) for a in sorted(set(actions))},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="时间展开可行集规划器 · 真机回放")
    ap.add_argument("--script", default=".runtime/danmaku/th11_replay_full.json")
    ap.add_argument("--fracs", default="0.71,0.83,0.89")
    ap.add_argument("--max-frames", type=int, default=2400)
    ap.add_argument("--dt", type=int, default=3)
    ap.add_argument("--horizon", type=int, default=45)
    ap.add_argument("--grid", type=float, default=4.0)
    ap.add_argument("--hit-radius", type=float, default=5.0)
    ap.add_argument("--bottom-prior", type=float, default=0.0,
                    help=">0 时在同等可行/寿命下偏好低 y (回放里避免无谓上移)")
    ap.add_argument("--noop", action="store_true", help="同时跑站桩对照")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    fracs = [float(x) for x in args.fracs.split(",") if x.strip()]
    env = make_env(args.script, args.dt)
    print(f"script={args.script} length={script_length(env)}  dt={args.dt}  "
          f"horizon={args.horizon}  grid={args.grid}")
    results = []
    for frac in fracs:
        planner = ViabilityPlanner(dt=args.dt, horizon=args.horizon,
                                   grid=args.grid, hit_radius=args.hit_radius,
                                   bottom_prior=args.bottom_prior)
        result = run(env, planner, frac, max_frames=args.max_frames, dt=args.dt)
        results.append({"kind": "planner", **result})
        print(f"[planner] {frac:.2f}: {result['frames']:4d} 帧  "
              f"plan {result['avg_plan_ms']:.1f}ms  {result['actions']}")
    if args.noop:
        for frac in fracs:
            result = run(env, None, frac, max_frames=args.max_frames, dt=args.dt)
            results.append({"kind": "noop", **result})
            print(f"[noop]    {frac:.2f}: {result['frames']:4d} 帧")
    if args.json:
        Path(args.json).write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
