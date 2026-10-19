"""在 mock 后端把策略练到"能通关第 1 面" (打到第 2 面).

针对 REINFORCE 的两个坑做了处理:
  * 后期塌缩 —— **熵退火** (早期多探索, 后期收敛) + 全程保留最小熵;
  * 快照选错 —— 每 eval_every 局**暂停训练**, 用贪心跑几局评估, 只在贪心过面率
    提升时保存 policy_best (这样存下来的是"确定性能打"的策略)。

关键奖励: boss_damage_scale>0 (教"对准 Boss 输出") + danger/alive (教"别死")
+ 提高 stage_clear_bonus (明确过面目标)。

    python tools/mock_clear.py --episodes 2500
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from touhou_rl_mcp.config import Config  # noqa: E402
from touhou_rl_mcp.env import TouhouEnv  # noqa: E402
from touhou_rl_mcp.games.registry import GameRegistry  # noqa: E402
from touhou_rl_mcp.training import TrainConfig, Trainer  # noqa: E402
from touhou_rl_mcp.training.visualize import save_png  # noqa: E402

REWARD = {
    "score_scale": 1e-05,
    "graze_scale": 0.02,
    "power_scale": 0.1,
    "life_bonus": 5.0,
    "miss_penalty": -30.0,
    "death_penalty": -100.0,
    "stage_clear_bonus": 300.0,
    "game_clear_bonus": 500.0,
    "boss_damage_scale": 1.0,
    "danger_scale": 1.5,
    "danger_radius": 45.0,
    "alive_bonus": 0.5,
    "time_penalty": -0.001,
    "clip": 100.0,
    "shaping_scale": 1.0,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=2500)
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--step-frames", type=int, default=4)
    ap.add_argument("--hidden", default="128,128")
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--entropy", type=float, default=0.06, help="初始熵系数 (会退火到 0.01)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--difficulty", type=int, default=1)
    ap.add_argument("--out", default=".runtime/mock_clear")
    ap.add_argument("--eval-every", type=int, default=100, help="每多少局做一次贪心评估")
    ap.add_argument("--eval-episodes", type=int, default=10)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--early-stop", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "log.txt"

    def log(msg: str) -> None:
        print(msg, flush=True)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(msg + "\n")

    cfg = Config.load()
    cfg.patch({"backend": "mock", "game": "th11", "step_frames": args.step_frames})
    cfg.patch({"reward": REWARD})
    env = TouhouEnv(cfg, registry=GameRegistry())
    env.start()

    tc = TrainConfig(
        episodes=10**9,
        max_steps=args.max_steps,
        step_frames=args.step_frames,
        batch_episodes=args.batch,
        lr=args.lr,
        gamma=args.gamma,
        entropy_coef=args.entropy,
        hidden=tuple(int(x) for x in args.hidden.split(",")),
        difficulty=args.difficulty,
        seed=20261017,
        capture_every=0,
        eval_every=0,
        save_every=0,
        out_dir=str(out),
    )
    trainer = Trainer(env, tc, env_lock=env.lock, out_dir=out)
    log(f"[mock_clear] reward={json.dumps(REWARD, ensure_ascii=False)}")
    log(f"[mock_clear] hidden={tc.hidden} lr={tc.lr} batch={tc.batch_episodes} step_frames={tc.step_frames} 初始熵={tc.entropy_coef}")

    # 独立的评估环境: 不干扰训练线程 (训练线程一直在动同一个 env, 不能共用)
    eval_env = TouhouEnv(cfg, registry=GameRegistry())
    eval_env.start()
    eval_tc = TrainConfig(
        episodes=0,
        max_steps=args.max_steps,
        step_frames=args.step_frames,
        hidden=tc.hidden,
        difficulty=args.difficulty,
        seed=12345,
    )
    eval_trainer = Trainer(eval_env, eval_tc, env_lock=eval_env.lock)

    def greedy_clear_rate(count: int) -> float:
        """把当前策略存一份, 在独立环境里用贪心跑 count 局, 返回过面率."""
        trainer.save(out / "policy_snapshot")
        eval_trainer.load(out / "policy_snapshot")
        clears = 0
        for k in range(count):
            record, _ = eval_trainer._run_episode(k, greedy=True)
            if (record.max_stage or 0) >= 2:
                clears += 1
        return clears / max(1, count)

    trainer.start()
    first_clear: int | None = None
    last_report = 0
    last_eval = 0
    best_greedy = -1.0
    best_ep = 0
    while len(trainer.episodes) < args.episodes and trainer.running:
        time.sleep(1.0)
        n = len(trainer.episodes)
        if first_clear is None:
            for e in trainer.episodes:
                if e.max_stage and e.max_stage >= 2:
                    first_clear = e.index
                    log(f"[mock_clear] *** 首次通关第1面: episode {e.index} ***")
                    break
        # 熵退火
        if n > 0 and n % 50 == 0:
            frac = min(1.0, n / args.episodes)
            ent = max(0.01, args.entropy * (1.0 - frac) + 0.01 * frac)
            try:
                trainer.reconfigure({"entropy_coef": ent})
            except Exception:
                pass
        # 贪心评估 + 快照
        if n > 0 and n - last_eval >= args.eval_every:
            last_eval = n
            rate = greedy_clear_rate(args.eval_episodes)
            if rate > best_greedy:
                best_greedy = rate
                best_ep = n
                try:
                    trainer.save(out / "policy_best")
                except Exception:
                    pass
                log(f"[mock_clear] 新最佳(贪心): {rate:.0%} @ep{n} (已存 policy_best)")
        if first_clear is not None and args.early_stop:
            break
        if n - last_report >= args.log_every and n > 0:
            last_report = n
            recent = trainer.episodes[-50:]
            mean_ret = sum(e.return_ for e in recent) / len(recent)
            mean_steps = sum(e.steps for e in recent) / len(recent)
            best_stage = max((e.max_stage or 0) for e in recent)
            clears = sum(1 for e in recent if (e.max_stage or 0) >= 2)
            upd = trainer.status(tail=0).get("last_update") or {}
            log(
                f"[mock_clear] ep={n:5d} 回报={mean_ret:8.2f} 步数={mean_steps:6.1f} "
                f"最好面数={best_stage} 过1面={clears}/50 entropy={upd.get('entropy')}"
            )

    status = trainer.stop(save=True)
    curve = trainer.curve_image()
    save_png(curve, out, name="curve.png")
    dash = trainer.dashboard_image()
    save_png(dash, out, name="dashboard.png")
    total_clears = sum(1 for e in trainer.episodes if (e.max_stage or 0) >= 2)
    log(f"[mock_clear] 结束: 共 {len(trainer.episodes)} 局, 训练中过第1面 {total_clears} 局, 首次={first_clear}")
    log(f"[mock_clear] 最佳(贪心)过面率={best_greedy:.0%} @ep{best_ep} -> {out / 'policy_best.npz'}")
    log(f"[mock_clear] 曲线: {out / 'curve.png'}")
    eval_env.close()
    env.close()
    return 0 if best_greedy > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
