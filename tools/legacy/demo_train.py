"""训练 + 奖励调参演示 (mock 后端, 快且可见).

流程 (模拟 "Agent 当教练"):
  阶段 A: 用默认奖励训练 N 局;
  阶段 B: 在线打开 danger_scale / alive_bonus 等密集信号 (env.configure), 再训 N 局;
  最后保存学习曲线与仪表盘, 打印两阶段对比。

    python tools/demo_train.py --phase-episodes 120 --out .runtime/demo
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from touhou_rl_mcp.config import Config  # noqa: E402
from touhou_rl_mcp.env import TouhouEnv  # noqa: E402
from touhou_rl_mcp.games.registry import GameRegistry  # noqa: E402
from touhou_rl_mcp.training import TrainConfig, Trainer  # noqa: E402
from touhou_rl_mcp.training.visualize import save_png  # noqa: E402


def summarize(trainer: Trainer, since: int) -> dict:
    eps = trainer.episodes[since:]
    if not eps:
        return {}
    n = len(eps)
    return {
        "episodes": n,
        "mean_return": round(sum(e.return_ for e in eps) / n, 2),
        "mean_steps": round(sum(e.steps for e in eps) / n, 1),
        "best_stage": max((e.max_stage or 0) for e in eps),
        "mean_stage": round(sum((e.max_stage or 0) for e in eps) / n, 2),
        "clears": sum(1 for e in eps if e.cleared),
        "misses": sum(e.misses for e in eps),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase-episodes", type=int, default=120)
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--step-frames", type=int, default=4)
    ap.add_argument("--difficulty", type=int, default=1)
    ap.add_argument("--out", default=".runtime/demo")
    args = ap.parse_args()

    out_dir = Path(args.out)
    cfg = Config.load()
    cfg.patch({"backend": "mock", "game": "th11", "step_frames": args.step_frames})
    env = TouhouEnv(cfg, registry=GameRegistry())
    env.start()

    train_cfg = TrainConfig(
        episodes=10**9,
        max_steps=args.max_steps,
        step_frames=args.step_frames,
        batch_episodes=8,
        difficulty=args.difficulty,
        hidden=(64, 64),
        seed=20261017,
        capture_every=10,
        eval_every=40,
        eval_episodes=3,
        save_every=0,
        out_dir=str(out_dir),
    )
    trainer = Trainer(env, train_cfg, env_lock=env.lock, out_dir=out_dir)

    # 阶段 A: 故意用"稀疏"奖励 (关掉危险度/存活密集信号) —— 复现"学不动"
    env.configure({"reward": {"danger_scale": 0.0, "alive_bonus": 0.0}})
    print(f"[demo] reward(阶段A 稀疏): {env.reward_model.weights}")
    trainer.start()

    target = args.phase_episodes
    while len(trainer.episodes) < target and trainer.running:
        time.sleep(0.5)
    phase_a = summarize(trainer, 0)
    print(f"[demo] 阶段A (默认奖励) 完成: {phase_a}")
    curve_a = trainer.curve_image()
    save_png(curve_a, out_dir, name="curve_A_default.png")

    # === Agent 在线调参: 打开"危险度/存活"密集信号 ===
    patch = {"reward": {"danger_scale": 2.0, "alive_bonus": 1.0, "graze_scale": 0.1, "shaping_scale": 1.0}}
    env.configure(patch)
    print(f"[demo] 调参 -> {patch['reward']}")

    target_b = 2 * args.phase_episodes
    while len(trainer.episodes) < target_b and trainer.running:
        time.sleep(0.5)
    phase_b = summarize(trainer, args.phase_episodes)
    print(f"[demo] 阶段B (密集奖励) 完成: {phase_b}")

    status = trainer.stop(save=True)
    curve = trainer.curve_image()
    save_png(curve, out_dir, name="curve_final.png")
    dash = trainer.dashboard_image()
    save_png(dash, out_dir, name="dashboard.png")

    print(f"[demo] 总步数={status['total_steps']} 步/秒={status['steps_per_sec']} 最好回报={status['best_return']}")
    print(f"[demo] 曲线: {out_dir / 'curve_A_default.png'}")
    print(f"[demo] 曲线: {out_dir / 'curve_final.png'}")
    print(f"[demo] 仪表盘: {out_dir / 'dashboard.png'}")
    print(f"[demo] 策略: {status['policy_path']}.npz")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
