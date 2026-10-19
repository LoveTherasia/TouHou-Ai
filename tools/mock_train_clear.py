"""在 mock 后端把策略练到**稳定通过第一面**(max_stage>=2)。

比 tools/mock_clear.py 增强:
  * 观测加入 弹幕速度 + 敌机位置 (与真机一致 -> 便于迁移到真机);
  * 更大的网络 (默认 256x256);
  * 熵退火 + 独立评估环境 + 只在**贪心过面率**提升时保存 best;
  * 达到目标过面率自动停止。

    python tools/mock_train_clear.py --episodes 2000 --out .runtime/mock_v2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from touhou_rl_mcp.config import Config  # noqa: E402
from touhou_rl_mcp.env import TouhouEnv  # noqa: E402
from touhou_rl_mcp.games.registry import GameRegistry  # noqa: E402
from touhou_rl_mcp.training import TrainConfig, Trainer  # noqa: E402
from touhou_rl_mcp.training.visualize import save_png  # noqa: E402

#: 面向"存活 + 打 Boss + 过面"的奖励
REWARD = {
    "score_scale": 1e-05,
    "graze_scale": 0.02,
    "power_scale": 0.1,
    "life_bonus": 5.0,
    "miss_penalty": -30.0,
    "death_penalty": -150.0,
    "stage_clear_bonus": 300.0,
    "game_clear_bonus": 500.0,
    "boss_damage_scale": 1.0,
    "danger_scale": 1.5,
    "danger_radius": 45.0,
    "alive_bonus": 0.5,
    "time_penalty": -0.001,
    "clip": 300.0,
    "shaping_scale": 1.0,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--step-frames", type=int, default=4)
    ap.add_argument("--hidden", default="256,256")
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--entropy", type=float, default=0.06)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--difficulty", type=int, default=1)
    ap.add_argument("--k-enemies", type=int, default=8)
    ap.add_argument("--no-velocity", action="store_true")
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--eval-episodes", type=int, default=10)
    ap.add_argument("--target-clear", type=float, default=0.8)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--out", default=".runtime/mock_v2")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "log.txt"

    def log(msg: str) -> None:
        print(msg, flush=True)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(msg + "\n")

    velocity = not args.no_velocity
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
        bullet_velocity=velocity,
        k_enemies=args.k_enemies,
        difficulty=args.difficulty,
        seed=20261017,
        capture_every=0,
        eval_every=0,
        save_every=0,
        out_dir=str(out),
    )
    trainer = Trainer(env, tc, env_lock=env.lock, out_dir=out)
    log(f"[mock] reward={json.dumps(REWARD, ensure_ascii=False)}")
    log(f"[mock] hidden={tc.hidden} lr={tc.lr} batch={tc.batch_episodes} "
        f"velocity={velocity} k_enemies={tc.k_enemies} obs_dim={trainer.obs_dim}")

    eval_env = TouhouEnv(cfg, registry=GameRegistry())
    eval_env.start()
    eval_tc = TrainConfig(
        episodes=0, max_steps=args.max_steps, step_frames=args.step_frames,
        hidden=tc.hidden, bullet_velocity=velocity, k_enemies=args.k_enemies,
        difficulty=args.difficulty, seed=12345,
    )
    eval_trainer = Trainer(eval_env, eval_tc, env_lock=eval_env.lock)

    def greedy_clear_rate(count: int) -> float:
        trainer.save(out / "policy_snapshot")
        eval_trainer.load(out / "policy_snapshot")
        clears = 0
        for k in range(count):
            record, _ = eval_trainer._run_episode(k, greedy=True)
            if (record.max_stage or 0) >= 2:
                clears += 1
        return clears / max(1, count)

    trainer.start()
    first_clear = None
    last_report = 0
    last_eval = 0
    best = -1.0
    best_ep = 0
    t0 = time.time()
    while len(trainer.episodes) < args.episodes and trainer.running:
        time.sleep(1.0)
        n = len(trainer.episodes)
        if trainer.error:
            log(f"[mock] 训练出错: {trainer.error}")
            break
        if first_clear is None:
            for e in trainer.episodes:
                if e.max_stage and e.max_stage >= 2:
                    first_clear = e.index
                    log(f"[mock] *** 首次通过第1面: ep{e.index} ***")
                    break
        # 熵退火
        if n > 0 and n % 50 == 0:
            frac = min(1.0, n / max(1, args.episodes))
            ent = max(0.01, args.entropy * (1.0 - frac) + 0.01 * frac)
            try:
                trainer.reconfigure({"entropy_coef": ent})
            except Exception:
                pass
        # 贪心评估 + 选优
        if n > 0 and n - last_eval >= args.eval_every:
            last_eval = n
            rate = greedy_clear_rate(args.eval_episodes)
            if rate > best:
                best, best_ep = rate, n
                try:
                    trainer.save(out / "policy_best")
                except Exception:
                    pass
                log(f"[mock] 新最佳(贪心过面率)={rate:.0%} @ep{n} (已存 policy_best)")
            if rate >= args.target_clear:
                log(f"[mock] 达到目标过面率 {rate:.0%} >= {args.target_clear:.0%}, 停止")
                break
        if n - last_report >= args.log_every and n > 0:
            last_report = n
            recent = trainer.episodes[-50:]
            mean_ret = sum(e.return_ for e in recent) / len(recent)
            mean_steps = sum(e.steps for e in recent) / len(recent)
            best_stage = max((e.max_stage or 0) for e in recent)
            clears = sum(1 for e in recent if (e.max_stage or 0) >= 2)
            upd = trainer.status(tail=0).get("last_update") or {}
            log(f"[mock] ep={n:5d} 回报={mean_ret:8.1f} 步数={mean_steps:6.1f} "
                f"最好面数={best_stage} 过1面={clears}/50 ent={upd.get('entropy')} "
                f"{n / max(1, time.time() - t0):.0f}局/秒")

    status = trainer.stop(save=True)
    try:
        save_png(trainer.curve_image(), out, name="curve.png")
        save_png(trainer.progress_image(), out, name="progress.png")
    except Exception:
        pass
    total = sum(1 for e in trainer.episodes if (e.max_stage or 0) >= 2)
    log(f"[mock] 结束: {len(trainer.episodes)} 局, 训练中过1面 {total} 局, 首次={first_clear}")
    log(f"[mock] 最佳(贪心)过面率={best:.0%} @ep{best_ep} -> {out / 'policy_best.npz'}")
    log(f"[mock] obs_dim={trainer.obs_dim} 产物: {out}")
    eval_env.close()
    env.close()
    return 0 if best >= args.target_clear else 1


if __name__ == "__main__":
    raise SystemExit(main())
