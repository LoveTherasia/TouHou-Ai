"""评估已保存的策略 (贪心, 不带探索) —— 用于确认"能通关第 1 面".

    python tools/eval_policy.py --policy .runtime/mock_clear3/policy_best --episodes 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from touhou_rl_mcp.config import Config  # noqa: E402
from touhou_rl_mcp.env import TouhouEnv  # noqa: E402
from touhou_rl_mcp.games.registry import GameRegistry  # noqa: E402
from touhou_rl_mcp.training import TrainConfig, Trainer  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--step-frames", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--hidden", default="128,128")
    ap.add_argument("--difficulty", type=int, default=1)
    ap.add_argument("--sample", action="store_true", help="用采样(而非贪心)评估")
    args = ap.parse_args()

    cfg = Config.load()
    cfg.patch({"backend": "mock", "game": "th11", "step_frames": args.step_frames})
    env = TouhouEnv(cfg, registry=GameRegistry())
    env.start()
    tc = TrainConfig(
        episodes=0,
        max_steps=args.max_steps,
        step_frames=args.step_frames,
        hidden=tuple(int(x) for x in args.hidden.split(",")),
        difficulty=args.difficulty,
        seed=1,
    )
    trainer = Trainer(env, tc, env_lock=env.lock)
    trainer.load(args.policy)

    clears = 0
    stages: list[int] = []
    steps: list[int] = []
    rets: list[float] = []
    for i in range(args.episodes):
        record, _ = trainer._run_episode(i, greedy=not args.sample)
        stages.append(record.max_stage or 0)
        steps.append(record.steps)
        rets.append(record.return_)
        if (record.max_stage or 0) >= 2:
            clears += 1

    mode = "采样" if args.sample else "贪心"
    print(f"[eval] 策略={args.policy}")
    print(f"[eval] {mode} {args.episodes} 局: 通关第1面 {clears}/{args.episodes} = {clears / args.episodes:.0%}")
    print(f"[eval] 平均步数={sum(steps) / len(steps):.1f} 平均回报={sum(rets) / len(rets):.1f} 最高到达面数={max(stages)}")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
