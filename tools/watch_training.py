"""一边训练一边看 —— 实时监视窗口.

示例::

    # mock 后端, 开一个窗口看着它学 (每秒上千步)
    python tools/watch_training.py --backend mock --episodes 3000

    # 真实游戏 (游戏窗口必须在前台, 别点监视窗口)
    python tools/watch_training.py --backend memory --game th11 --episodes 50

    # 只看不练: 打开窗口显示已有的学习曲线与最近画面
    python tools/watch_training.py --replay-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from touhou_rl_mcp.config import Config  # noqa: E402
from touhou_rl_mcp.env import TouhouEnv  # noqa: E402
from touhou_rl_mcp.training import TrainConfig, Trainer  # noqa: E402
from touhou_rl_mcp.training.watch import run_watch_window  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Touhou RL 实时训练监视窗口")
    parser.add_argument("--backend", default="mock", choices=["mock", "memory"])
    parser.add_argument("--game", default="th11")
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--step-frames", type=int, default=4)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--hidden", default="64,64")
    parser.add_argument("--focus", action="store_true")
    parser.add_argument("--difficulty", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--out", default="")
    parser.add_argument("--load", default=None)
    parser.add_argument("--capture-every", type=int, default=5)
    parser.add_argument("--refresh-ms", type=int, default=700)
    parser.add_argument("--replay-only", action="store_true", help="不训练, 只看现有数据")
    args = parser.parse_args()

    config = Config.load()
    config.patch({"backend": args.backend, "game": args.game, "step_frames": args.step_frames})
    env = TouhouEnv(config)
    env.start()
    print(f"[watch] 后端={args.backend} 作品={env.spec.key} ({env.spec.name_zh})")

    train_config = TrainConfig(
        episodes=args.episodes,
        max_steps=args.max_steps,
        step_frames=args.step_frames,
        batch_episodes=args.batch,
        lr=args.lr,
        hidden=tuple(int(x) for x in args.hidden.split(",")),
        use_focus=args.focus,
        difficulty=args.difficulty,
        seed=args.seed,
        out_dir=args.out,
        capture_every=args.capture_every,
        eval_every=25,
        eval_episodes=3,
        save_every=50,
    )
    trainer = Trainer(env, train_config, env_lock=env.lock)
    if args.load:
        print(f"[watch] 加载策略: {trainer.load(args.load)}")

    if not args.replay_only:
        trainer.start()
        print("[watch] 训练已在后台线程启动; 窗口里 空格=暂停 S=保存 Esc=退出")

    def on_close() -> None:
        if trainer.running:
            print("[watch] 正在停止训练并保存策略...")
            trainer.stop(save=True)
        env.close()
        print("[watch] 已退出")

    try:
        run_watch_window(
            trainer,
            refresh_ms=args.refresh_ms,
            title=f"Touhou RL 训练监视 · {args.backend}/{env.spec.key}",
            on_close=on_close,
        )
    except KeyboardInterrupt:
        on_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
