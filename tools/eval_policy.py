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
from touhou_rl_mcp.rl.reward import REWARD_PROFILES, reward_profile  # noqa: E402
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
    ap.add_argument("--k-bullets", type=int, default=16)
    ap.add_argument("--k-enemies", type=int, default=0)
    ap.add_argument("--bullet-velocity", action="store_true", help="观测含弹速 (真机/新版策略)")
    ap.add_argument("--danger-grad", action="store_true", help="观测含危险梯度")
    ap.add_argument("--wall-feat", action="store_true",
                    help="观测末尾含 12 维墙特征 (必须与训练时一致)")
    ap.add_argument("--no-focus", dest="focus", action="store_false",
                    help="动作表不含低速(focus); 必须与训练时一致")
    ap.set_defaults(focus=True)
    ap.add_argument("--reward-profile", default="", choices=[""] + sorted(REWARD_PROFILES),
                    help="按同一份奖励配置评估 (默认用 config.json)")
    ap.add_argument("--hier-switch", action="store_true",
                    help="开分层切换: 整幅横墙/多向收网时改用可行域规划器")
    args = ap.parse_args()

    cfg = Config.load()
    cfg.patch({"backend": "mock", "game": "th11", "step_frames": args.step_frames})
    if args.reward_profile:
        cfg.patch({"reward": reward_profile(args.reward_profile)})
    env = TouhouEnv(cfg, registry=GameRegistry())
    env.start()
    tc = TrainConfig(
        episodes=0,
        max_steps=args.max_steps,
        step_frames=args.step_frames,
        hidden=tuple(int(x) for x in args.hidden.split(",")),
        difficulty=args.difficulty,
        seed=1,
        k_bullets=args.k_bullets,
        k_enemies=args.k_enemies,
        bullet_velocity=args.bullet_velocity,
        use_danger_grad=args.danger_grad,
        use_focus=args.focus,
        use_wall_feat=args.wall_feat,
        hier_switch=args.hier_switch,
    )
    trainer = Trainer(env, tc, env_lock=env.lock)
    trainer.load(args.policy)

    clears = 0
    stages: list[int] = []
    steps: list[int] = []
    rets: list[float] = []
    damages: list[int] = []
    aligns: list[float] = []
    scores: list[int] = []
    planner_steps: list[int] = []
    for i in range(args.episodes):
        record, _ = trainer._run_episode(i, greedy=not args.sample)
        stages.append(record.max_stage or 0)
        steps.append(record.steps)
        rets.append(record.return_)
        damages.append(record.boss_damage)
        aligns.append(record.align_ratio or 0.0)
        scores.append(record.score_gain or 0)
        planner_steps.append(record.planner_steps)
        if (record.max_stage or 0) >= 2:
            clears += 1

    n = max(1, len(steps))
    mode = "采样" if args.sample else "贪心"
    print(f"[eval] 策略={args.policy}")
    print(f"[eval] {mode} {args.episodes} 局: 通关第1面 {clears}/{args.episodes} = {clears / args.episodes:.0%}")
    print(f"[eval] 平均步数={sum(steps) / n:.1f} 平均回报={sum(rets) / n:.1f} 最高到达面数={max(stages)}")
    print(f"[eval] 攻击意图: 均伤害={sum(damages) / n:.0f} 均净分={sum(scores) / n:.0f} "
          f"对准率={sum(aligns) / n:.0%}")
    if args.hier_switch:
        print(f"[eval] 分层切换: 均交给规划器 {sum(planner_steps) / n:.0f} 步 "
              f"({100.0 * sum(planner_steps) / max(1, sum(steps)):.1f}%)")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
