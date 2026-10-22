"""量老师的"标签稳定性"：换向率 / 自一致率 / 动作-代价一致性 / focus 使用率.

为什么单独做一个工具
------------------
MPC 老师的问题从来不是"不会躲"，而是**它自己每一帧给的答案不一样**：
代价平坦处 CEM 的 argmax 随机跳，学生把这种噪声当真理学，天花板就卡在 20%。

所以任何"确定化"改动（承诺期、更多迭代、固定噪声）都必须先用同一批状态量出
下面四个数，再去训练 —— 否则训练结果好坏分不清是老师变好了还是样本变多了。

    python tools/teacher_check.py --patterns dense_ring,wall_gate --frames 150
    python tools/teacher_check.py --mpc-commit 3 --mpc-switch 5 --mpc-iters 5

指标
----
* ``换向率``     : 相邻两帧动作不同的比例（越低越稳）
* ``自一致率``   : 同一批状态，换成另一个 CEM 随机种子后动作仍相同的比例
* ``动作≈argmin``: 老师实际动作是否等于 ``action_costs`` 的最小项 —— 蒸馏教的是
  "代价"，学生 argmax 出来的就是 argmin；两者不一致说明教的和老师做的是两件事
* ``标签换向``   : 相邻两帧 **argmin(cost)** 变了没有 —— 这才是发给学生的标签噪声
* ``标签自一致`` : 两个 CEM 种子对同一状态的 argmin 是否一致
* ``focus``      : 老师用慢速动作（按住 Shift）的比例

状态序列由 ``RuleTeacher`` 驱动产生（不用 MPC），这样"稳定性"和"能不能活"解耦。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from touhou_rl_mcp.config import Config  # noqa: E402
from touhou_rl_mcp.env import TouhouEnv  # noqa: E402
from touhou_rl_mcp.games.registry import GameRegistry  # noqa: E402
from touhou_rl_mcp.training.mpc_teacher import MPCTeacher, to_trainer_name  # noqa: E402
from touhou_rl_mcp.training.rule_teacher import RuleTeacher  # noqa: E402
from touhou_rl_mcp.training.trainer import DIRECTIONS  # noqa: E402  (字符串方向表)

DEFAULT_PATTERNS = ("dense_ring", "closing_ring", "wall_gate")


def collect_states(pattern: str, frames: int, step_frames: int, seed: int) -> list:
    cfg = Config.load()
    cfg.patch({"backend": "mock", "game": "th11", "step_frames": step_frames,
               "danmaku_pattern": pattern, "start_stage": 1})
    env = TouhouEnv(cfg, registry=GameRegistry())
    env.start()
    state = env.reset(seed=seed).state
    drive = RuleTeacher()
    states = []
    try:
        for _ in range(frames):
            states.append(state)
            action = drive.choose(state, state.frame, None)[0]
            result = env.step(action, frames=step_frames)
            state = result.state
            if result.done or result.truncated:
                break
    finally:
        env.close()
    return states


def make_teacher(args: argparse.Namespace, seed: int, use_focus: bool) -> MPCTeacher:
    return MPCTeacher(
        horizon=args.mpc_horizon, dt=args.mpc_dt, n_samples=args.mpc_samples,
        iters=args.mpc_iters, gap_weight=args.mpc_gap, column_weight=args.mpc_column,
        adaptive=args.mpc_adaptive, long_horizon=args.long_horizon,
        long_dt=args.long_dt, use_focus=use_focus, plan_every=1,
        step_frames=args.step_frames, seed=seed,
        commit_steps=args.mpc_commit, switch_margin=args.mpc_switch,
        commit_frames=getattr(args, "mpc_commit_frames", 0),
        focus_margin=args.focus_margin, deterministic=not args.no_deterministic,
        smooth_clear=args.mpc_smooth, smooth_sigma=args.mpc_smooth_sigma,
    )


def direction_index(action: str) -> int:
    head = to_trainer_name(action).split("+shoot")[0].strip("+ ")
    try:
        return DIRECTIONS.index(head) if head else 0
    except ValueError:
        return 0


def run(teacher: MPCTeacher, states: list, use_focus: bool) -> dict:
    acts, dirs, focus_n, match, gaps, labels = [], [], 0, 0, [], []
    for state in states:
        action, _info = teacher.choose(state, state.frame, None)
        acts.append(action)
        dirs.append(direction_index(action))
        if "focus" in action:
            focus_n += 1
        costs = teacher.cost_vector(state)
        labels.append(-1 if costs is None else int(np.argmin(costs)))
        if costs is not None:
            n = len(DIRECTIONS)
            best = int(np.argmin(costs))
            n_act = len(costs)
            # 老师的动作在动作表里的位置 = 方向 (+N 若带 focus)
            idx = dirs[-1] + (n if ("focus" in action and n_act > n) else 0)
            match += int(idx == best)
            srt = np.sort(np.asarray(costs, dtype=np.float64))
            gaps.append(float(srt[min(1, len(srt) - 1)] - srt[0]))
    m = max(1, len(states))
    churn = float(np.mean([a != b for a, b in zip(acts[:-1], acts[1:])])) if len(acts) > 1 else 0.0
    label_churn = (float(np.mean([a != b for a, b in zip(labels[:-1], labels[1:])]))
                   if len(labels) > 1 else 0.0)
    return {
        "actions": acts,
        "labels": labels,
        "churn": churn,
        "label_churn": label_churn,
        "focus": focus_n / m,
        "match": match / m,
        "gap": float(np.mean(gaps)) if gaps else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patterns", default=",".join(DEFAULT_PATTERNS))
    ap.add_argument("--frames", type=int, default=150, help="每个弹型采集多少帧状态")
    ap.add_argument("--step-frames", type=int, default=2)
    ap.add_argument("--mpc-horizon", type=int, default=24)
    ap.add_argument("--mpc-dt", type=int, default=3)
    ap.add_argument("--mpc-samples", type=int, default=48)
    ap.add_argument("--mpc-iters", type=int, default=3)
    ap.add_argument("--mpc-gap", type=float, default=0.0)
    ap.add_argument("--mpc-column", type=float, default=0.0)
    ap.add_argument("--mpc-adaptive", action="store_true")
    ap.add_argument("--long-horizon", type=int, default=72)
    ap.add_argument("--long-dt", type=int, default=6)
    ap.add_argument("--mpc-commit", type=int, default=0,
                    help="承诺期 (决策步)")
    ap.add_argument("--mpc-commit-frames", type=int, default=0,
                    help="承诺期 (游戏帧, 优先级高于 --mpc-commit; 8~15 是实测区间)")
    ap.add_argument("--mpc-switch", type=float, default=0.0)
    ap.add_argument("--focus-margin", type=float, default=0.0)
    ap.add_argument("--mpc-smooth", type=float, default=0.0,
                    help="平滑安全余量权重 (>0 时代价随离弹距离连续变化, 打破并列)")
    ap.add_argument("--mpc-smooth-sigma", type=float, default=20.0)
    ap.add_argument("--no-deterministic", action="store_true")
    ap.add_argument("--no-focus", dest="focus", action="store_false")
    ap.add_argument("--seed", type=int, default=20261018)
    ap.add_argument("--json", default="", help="把结果写到 json")
    ap.set_defaults(focus=True)
    args = ap.parse_args()

    patterns = [p.strip() for p in args.patterns.split(",") if p.strip()]
    rows = []
    print(f"[check] iters={args.mpc_iters} commit={args.mpc_commit} "
          f"switch={args.mpc_switch} deterministic={not args.no_deterministic} "
          f"focus={args.focus} step_frames={args.step_frames}", flush=True)
    print(f"{'弹型':<14}{'换向率':>8}{'自一致':>8}{'标签换向':>10}{'标签自一致':>12}"
          f"{'动作≈argmin':>12}{'代价间隙':>9}{'focus':>7}", flush=True)
    for pat in patterns:
        states = collect_states(pat, args.frames, args.step_frames, args.seed)
        if not states:
            continue
        a = run(make_teacher(args, 1, args.focus), states, args.focus)
        b = run(make_teacher(args, 7, args.focus), states, args.focus)
        same = float(np.mean([x == y for x, y in zip(a["actions"], b["actions"])]))
        lab_same = float(np.mean([x == y for x, y in zip(a["labels"], b["labels"])]))
        rows.append({
            "pattern": pat, "states": len(states),
            "churn": round(a["churn"], 4), "self_consistency": round(same, 4),
            "label_churn": round(a["label_churn"], 4),
            "label_self_consistency": round(lab_same, 4),
            "action_argmin": round(a["match"], 4), "cost_gap": round(a["gap"], 3),
            "focus_rate": round(a["focus"], 4),
        })
        print(f"{pat:<14}{a['churn']*100:7.1f}%{same*100:7.1f}%{a['label_churn']*100:9.1f}%"
              f"{lab_same*100:11.1f}%{a['match']*100:11.1f}%{a['gap']:9.2f}"
              f"{a['focus']*100:6.1f}%", flush=True)

    if rows:
        n = len(rows)
        print(f"{'平均':<14}"
              f"{np.mean([r['churn'] for r in rows])*100:7.1f}%"
              f"{np.mean([r['self_consistency'] for r in rows])*100:7.1f}%"
              f"{np.mean([r['label_churn'] for r in rows])*100:9.1f}%"
              f"{np.mean([r['label_self_consistency'] for r in rows])*100:11.1f}%"
              f"{np.mean([r['action_argmin'] for r in rows])*100:11.1f}%"
              f"{np.mean([r['cost_gap'] for r in rows]):9.2f}"
              f"{np.mean([r['focus_rate'] for r in rows])*100:6.1f}%  ({n} 弹型)")
    if args.json:
        import json

        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[check] -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
