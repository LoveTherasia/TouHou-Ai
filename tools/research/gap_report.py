"""把若干 run 目录放到同一个基准上对比 —— 不只看存活帧数。

为什么需要它
------------
v2 的存活上限就是单局时限 (规则缺口跟随能无限跑), 所以**存活帧数会饱和**:
两个策略都能活满 2400 帧时, "换奖励有没有用"在存活曲线上完全看不出来。它实际改变
的是**飞行方式** —— 是主动贴着弹幕从缝里穿过去, 还是缩在缺口里苟着, 是持续横移还是
站着不动。这里用几个纯测量的量把这个差别量化出来。

指标 (全部在**未见过的**场景上贪心跑)
------------------------------------
``frames``     存活帧数 (上限 = ``--max-frames``)
``graze/1k``   每千帧擦弹数 —— "主动找缝"的密度
``min_d``      平均"到最近弹幕的距离" (px) —— 苟着不动通常更大, 贴缝穿过更小
``lateral``    横向移动 (left/right/斜向) 的决策占比 —— 站着不动 = 苟
``|x-gap|``    与最近缺口的横向距离 (px) —— 追踪误差
``agree``      与缺口标签的一致率

用法::

    python tools/research/gap_report.py --runs .runtime/runs/trainer-v2-* --bench v2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gap_rl import BENCHMARKS, make_env, reset_env  # noqa: E402

from touhou_rl_mcp.training.features import ObservationEncoder  # noqa: E402
from touhou_rl_mcp.training.gap_label import (  # noqa: E402
    COMMIT_FRAMES, MACRO_ACTIONS, gap_label, reach_target,
)
from touhou_rl_mcp.training.policy import MLPPolicy  # noqa: E402

#: 判定为"横向移动"的动作下标 (left / right / 四个斜向)
LATERAL = {3, 4, 5, 6, 7, 8}


def nearest_distance(state) -> float:
    if not state.bullets or state.player_x is None:
        return 999.0
    arr = np.asarray(state.bullets, dtype=np.float64).reshape(-1, 2)
    return float(np.hypot(arr[:, 0] - state.player_x, arr[:, 1] - state.player_y).min())


def evaluate_run(run_dir: Path, heldout, *, max_frames: int,
                 seeds: tuple[int, ...]) -> dict[str, float] | None:
    npz = run_dir / "policy.npz"
    meta = run_dir / "policy.json"
    if not npz.is_file() or not meta.is_file():
        return None
    policy = MLPPolicy.load(npz)
    config = json.loads(meta.read_text(encoding="utf-8")).get("config", {})
    enc = ObservationEncoder.from_meta(config)
    if enc.dim != policy.obs_dim:
        print(f"  ! {run_dir.name}: 观测维度不匹配 ({enc.dim} vs {policy.obs_dim})")
        return None
    label_mode = str(config.get("gap_label", "reach"))
    spec = enc.gap_spec

    scenarios = [dict(s) for s in heldout]
    env = make_env(**scenarios[0])
    frames: list[int] = []
    grazes: list[float] = []
    distances: list[float] = []
    lateral = total = 0
    errors: list[float] = []
    hits = 0
    for scenario in scenarios:
        for seed in seeds:
            state = reset_env(env, seed, scenario)
            survival = 0
            prev = -1
            while survival < max_frames:
                obs = enc(state, prev_action=prev)
                action = int(policy.greedy(obs))
                hits += int(action == gap_label(state, spec, label_mode))
                target = reach_target(state, spec)
                if target is not None and state.player_x is not None:
                    errors.append(abs(target - state.player_x))
                lateral += int(action in LATERAL)
                total += 1
                distances.append(nearest_distance(state))
                result = env.step(MACRO_ACTIONS[action], frames=COMMIT_FRAMES,
                                  auto_reset=False)
                state = result.state
                survival += COMMIT_FRAMES
                prev = action
                if result.done:
                    break
            frames.append(survival)
            grazes.append(float(state.graze or 0) / max(1.0, survival) * 1000.0)
    env.close()
    return {
        "frames": float(np.mean(frames)),
        "graze": float(np.mean(grazes)),
        "min_d": float(np.mean(distances)) if distances else 0.0,
        "lateral": lateral / total if total else 0.0,
        "error": float(np.mean(errors)) if errors else 0.0,
        "agree": hits / total if total else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="跨 run 的战术风格对比 (v2 已饱和时用)")
    ap.add_argument("--runs", nargs="+", required=True, help="run 目录 (支持通配符展开)")
    ap.add_argument("--bench", choices=tuple(BENCHMARKS), default="v2")
    ap.add_argument("--max-frames", type=int, default=2400)
    ap.add_argument("--seeds", default="101,202", help="逗号分隔的种子")
    args = ap.parse_args()

    heldout = BENCHMARKS[args.bench][1]
    seeds = tuple(int(s) for s in args.seeds.split(",") if s.strip())
    header = ("run", "frames", "graze/1k", "min_d", "lateral", "|x-gap|", "agree")
    print(f"{header[0]:34s} " + " ".join(f"{h:>9s}" for h in header[1:]))
    for raw in args.runs:
        run_dir = Path(raw)
        if not run_dir.is_dir():
            continue
        stats = evaluate_run(run_dir, heldout, max_frames=args.max_frames, seeds=seeds)
        if stats is None:
            continue
        print(f"{run_dir.name:34s} "
              f"{stats['frames']:9.0f} {stats['graze']:9.1f} {stats['min_d']:9.1f} "
              f"{stats['lateral']:8.0%} {stats['error']:9.1f} {stats['agree']:8.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
