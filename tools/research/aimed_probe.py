"""自机狙探针: 用**单颗斜向自机狙**检查策略是否会主动躲避。

为什么需要它
------------
回放脚本里的弹幕太杂, "单颗朝自机飞来的弹"只占几帧, 很难从评估数字里看出策略
有没有躲。这个探针把 mock 的 `aimed_spread` 改成 `count=1`、固定发射点, 每 30 帧
朝自机当前位置打一颗斜向弹 —— 如果策略把"自机所在车道"错判成安全, 它会站着不动
然后被撞, 存活会明显缩短。

它也是 TTA 车道判定的回归测试: 斜着穿过 ±26px 高度带时, x 会横向移动十几像素,
必须用"最接近自机高度"的 x, 而不是"刚进带"的 x。

用法::

    python tools/research/aimed_probe.py --run .runtime/runs/gif-aimfix --seeds 11,17,23
    python tools/research/aimed_probe.py --run .runtime/runs/gif-aimfix --seeds 17 --gif
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gap_gif import draw_trace  # noqa: E402

from touhou_rl_mcp.config import Config  # noqa: E402
from touhou_rl_mcp.env import TouhouEnv  # noqa: E402
from touhou_rl_mcp.rl.reward import reward_profile  # noqa: E402
from touhou_rl_mcp.training.features import ObservationEncoder, gap_spec_from  # noqa: E402
from touhou_rl_mcp.training.gap_label import (  # noqa: E402
    COMMIT_FRAMES, MACRO_ACTIONS, update_action_age,
)
from touhou_rl_mcp.training.occupancy import (  # noqa: E402
    _bullets, _column_centers, _tta_profile,
)
from touhou_rl_mcp.training.policy import MLPPolicy  # noqa: E402


def make_env(seed: int, *, count: int, interval: int, speed: float,
             x: float, y: float) -> TouhouEnv:
    cfg = Config.load()
    cfg.patch({
        "backend": "mock", "game": "th11", "danmaku_pattern": "aimed_spread",
        "step_frames": COMMIT_FRAMES, "max_episode_frames": 0, "auto_reset": False,
        "reward": reward_profile("gap_survive"),
    })
    env = TouhouEnv(cfg)
    env.start()
    env.reset(seed=seed)
    pattern = env.backend.game.pattern
    pattern.count = int(count)
    pattern.spread = 0.0
    pattern.interval = int(interval)
    pattern.speed = float(speed)
    pattern.x = float(x)
    pattern.y = float(y)
    return env


def load_policy(run_dir: Path) -> tuple[MLPPolicy, ObservationEncoder, dict[str, Any]]:
    meta = json.loads((run_dir / "policy.json").read_text(encoding="utf-8"))
    cfg = meta["config"]
    name = "policy" if (run_dir / "policy.npz").exists() else "policy_best"
    return MLPPolicy.load(run_dir / f"{name}.npz"), ObservationEncoder.from_meta(cfg), cfg


def run_seed(env: TouhouEnv, policy: MLPPolicy, enc: ObservationEncoder,
             seed: int, *, max_frames: int, capture_every: int,
             args: argparse.Namespace) -> tuple[int, list, list]:
    env.reset(seed=seed)
    pattern = env.backend.game.pattern
    pattern.count = int(args.count)
    pattern.spread = 0.0
    pattern.interval = int(args.interval)
    pattern.speed = float(args.speed)
    pattern.x = float(args.x)
    pattern.y = float(args.y)
    state = env.get_state()
    spec = gap_spec_from(enc)
    frames = []
    rows = []
    prev = -1
    age = 0
    total = 0
    for i in range(max_frames // COMMIT_FRAMES):
        pos, vel = _bullets(state)
        tta = _tta_profile(float(state.player_y), pos, vel, spec) * spec.tta_horizon
        centers = _column_centers(spec)
        safe = np.flatnonzero(tta > 60.0)
        gap = (float(state.player_x) if not safe.size else
               float(centers[safe[np.argmin(np.abs(centers[safe] - state.player_x))]]))
        rows.append({"frame": int(state.frame), "px": float(state.player_x), "gap": gap})
        if capture_every and i % capture_every == 0:
            frames.append(env.capture_frame())
        action = int(policy.greedy(enc(state, prev_action=prev, action_age=age)))
        result = env.step(MACRO_ACTIONS[action], frames=COMMIT_FRAMES, auto_reset=False)
        state = result.state
        total += COMMIT_FRAMES
        age = update_action_age(prev, age, action, int(enc.gap_action_age_cap or 12))
        prev = action
        if result.done:
            break
    return total, frames, rows


def save_gif(frames: list, rows: list, path: Path, title: str) -> None:
    width = 360
    scaled = [f.resize((width, max(1, round(f.height * width / f.width)))) for f in frames]
    if scaled:
        scaled[0].save(path, save_all=True, append_images=scaled[1:],
                       duration=50, loop=0, optimize=True)
    draw_trace(rows, interval=0.0, title=title).save(path.with_name(path.stem + "_trace.png"))


def main() -> int:
    ap = argparse.ArgumentParser(description="单颗斜向自机狙探针")
    ap.add_argument("--run", required=True)
    ap.add_argument("--seeds", default="11,17,23")
    ap.add_argument("--count", type=int, default=1)
    ap.add_argument("--interval", type=int, default=30)
    ap.add_argument("--speed", type=float, default=2.4)
    ap.add_argument("--x", type=float, default=100.0, help="发射点 x (100=斜向, 192=正下)")
    ap.add_argument("--y", type=float, default=96.0)
    ap.add_argument("--max-frames", type=int, default=2400)
    ap.add_argument("--capture-every", type=int, default=0,
                    help=">0 时抓帧 (每 N 个决策); 配 --gif")
    ap.add_argument("--gif", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    run_dir = Path(args.run)
    policy, enc, _cfg = load_policy(run_dir)
    env = make_env(int(args.seeds.split(",")[0]), count=args.count,
                   interval=args.interval, speed=args.speed, x=args.x, y=args.y)
    out_dir = Path(args.out) if args.out else run_dir / "aimed_probe"
    if args.gif:
        out_dir.mkdir(parents=True, exist_ok=True)

    results: list[int] = []
    for seed_text in args.seeds.split(","):
        seed = int(seed_text)
        total, frames, rows = run_seed(
            env, policy, enc, seed, max_frames=args.max_frames,
            capture_every=(args.capture_every if args.gif else 0), args=args)
        results.append(total)
        print(f"seed {seed}: {total} 帧")
        if args.gif and frames:
            tag = f"{run_dir.name}_aimed_single_seed{seed}"
            gif_path = out_dir / f"{tag}.gif"
            save_gif(frames, rows, gif_path,
                     f"{run_dir.name}: single diagonal aimed bullet, survival {total}f")
            print(f"  -> {gif_path}")
    print(f"mean {sum(results) / len(results):.0f}  min {min(results)}  "
          f"max {max(results)}")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
