"""训练入口 (命令行) —— 直接跑 RL, 不经过 MCP.

MCP 的 ``th_train_start`` 用的是同一个 :class:`~touhou_rl_mcp.training.Trainer`,
只是搬到后台线程里并暴露给 Agent 观测/调参。这个脚本适合:
  * 快速验证训练是否真的能学到东西
  * 长时间挂机训练 (配 --out 保存策略)

示例::

    # mock 后端快速迭代 (每秒上千步)
    python examples/train.py --backend mock --episodes 400

    # 真实游戏 (60fps 实时, 慢很多, 适合挂机)
    python examples/train.py --backend memory --game th11 --episodes 50

    # 加载已有策略继续训练
    python examples/train.py --load .runtime/training/policy.npz --episodes 200
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from touhou_rl_mcp.config import Config  # noqa: E402
from touhou_rl_mcp.env import TouhouEnv  # noqa: E402
from touhou_rl_mcp.training import TrainConfig, Trainer  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="mock", choices=["mock", "memory"])
    parser.add_argument("--game", default="th11")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--step-frames", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--entropy", type=float, default=0.01)
    parser.add_argument("--batch", type=int, default=8, help="累积多少局再更新一次参数")
    parser.add_argument("--hidden", default="64,64")
    parser.add_argument("--no-focus", dest="focus", action="store_false",
                        help="不加低速(focus)动作 (默认加: 9 快 + 9 慢 = 18 个动作)")
    parser.add_argument("--bomb", action="store_true", help="动作集加上放炸弹(方向+shoot+bomb), 再 +9 个动作")
    parser.add_argument("--algorithm", default="ppo", choices=["ppo", "reinforce_baseline"],
                        help="RL 算法 (默认 ppo)")
    parser.add_argument("--start-stage", type=int, default=1,
                        help="课程学习: mock 从第几面开局 (1=第一面)")
    parser.add_argument("--danmaku-script", default="",
                        help="用真机弹幕脚本代替 mock 内置弹型 (如 .runtime/danmaku/th11_stage1.json)")
    parser.add_argument("--danger-grad", action="store_true",
                        help="观测加入危险梯度(最近弹幕斥力合成)")
    parser.add_argument("--record-danmaku-dir", default="",
                        help="训练时同时把真机弹幕录到这个目录 (按难度/自机/机体命名)")
    parser.add_argument("--no-invincible", dest="invincible", action="store_false",
                        help="观测不含无敌帧特征 (默认含)")
    parser.add_argument("--ppo-clip", type=float, default=0.2)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--ppo-minibatch", type=int, default=256)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--log-steps", action="store_true", help="逐步写 steps.jsonl 日志(真机约6步/秒, 开销可接受)")
    parser.add_argument("--teacher", default="", choices=["", "guard", "dagger", "rule", "rule_dagger"],
                        help="模仿学习: guard/dagger=真机规则老师; rule/rule_dagger=mock 规则老师(与 dodge 宏同源)")
    parser.add_argument("--dagger-beta", type=float, default=0.2,
                        help="DAgger 中老师接管动作的比例 (0~1)")
    parser.add_argument("--bullet-velocity", action="store_true",
                        help="弹幕特征加入速度(vx,vy); 观测维 74->106, 旧的 74 维模型不再兼容")
    parser.add_argument("--k-enemies", type=int, default=0,
                        help="敌机特征个数 (0=只用数量; 8=追加最近 8 架敌机相对位置)")
    parser.add_argument("--difficulty", type=int, default=1)
    parser.add_argument("--character", type=int, choices=[0, 1], default=0)
    parser.add_argument("--shot", type=int, choices=[0, 1, 2], default=0)
    parser.add_argument("--rotate-loadouts", action="store_true", help="每局轮换 TH11 六机体（旧特征不含机体身份，建议分别训练）")
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--out", default="")
    parser.add_argument("--load", default=None)
    parser.add_argument("--restart-process", action="store_true", help="每局重启游戏进程，避免从 Demo 或上一局菜单开始")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--plot-every", type=int, default=10,
                        help="每多少局刷新一次 progress.png (存活步数/面数/回报/掉残; 0=关)")
    parser.set_defaults(focus=True, invincible=True)
    args = parser.parse_args()

    config = Config.load()
    config.patch({"backend": args.backend, "game": args.game, "step_frames": args.step_frames,
                  "danmaku_script": args.danmaku_script})
    if args.restart_process:
        config.patch({"restart_process": True})

    env = TouhouEnv(config)
    env.start()
    print(f"[train] 后端={args.backend} 作品={env.spec.key}({env.spec.name_zh})")

    train_config = TrainConfig(
        episodes=args.episodes,
        max_steps=args.max_steps,
        step_frames=args.step_frames,
        batch_episodes=args.batch,
        lr=args.lr,
        gamma=args.gamma,
        entropy_coef=args.entropy,
        hidden=tuple(int(x) for x in args.hidden.split(",")),
        use_focus=args.focus,
        use_bomb=args.bomb,
        use_invincible=args.invincible,
        use_danger_grad=args.danger_grad,
        algorithm=args.algorithm,
        start_stage=args.start_stage,
        ppo_clip=args.ppo_clip,
        ppo_epochs=args.ppo_epochs,
        ppo_minibatch=args.ppo_minibatch,
        gae_lambda=args.gae_lambda,
        log_steps=args.log_steps,
        teacher=args.teacher,
        dagger_beta=args.dagger_beta,
        bullet_velocity=args.bullet_velocity,
        k_enemies=args.k_enemies,
        difficulty=args.difficulty,
        character=args.character,
        shot=args.shot,
        rotate_loadouts=args.rotate_loadouts,
        seed=args.seed,
        out_dir=args.out,
        eval_every=max(1, args.log_every * 2),
        eval_episodes=3,
        save_every=50,
        plot_every=args.plot_every,
    )
    trainer = Trainer(env, train_config, env_lock=env.lock,
                      danmaku_dir=(args.record_danmaku_dir or None))
    if args.load:
        print(f"[train] 加载策略: {trainer.load(args.load)}")

    print(f"[train] 观测维度={trainer.obs_dim} 动作数={len(trainer.action_table)} 参数量={trainer.policy.describe()['params']}")
    print(f"[train] 动作表: {trainer.action_table}")

    trainer.start()
    started = time.perf_counter()
    last_report = 0
    try:
        while trainer.running:
            time.sleep(0.5)
            done = len(trainer.episodes)
            if done and done != last_report and done % args.log_every == 0:
                last_report = done
                status = trainer.status(tail=0)
                last = (status.get("recent_episodes") or [{}])[-1]
                if last.get("cleared"):
                    end = "clear"
                elif last.get("game_over"):
                    end = "death"
                elif last.get("truncated"):
                    end = "截断(max_steps)"
                else:
                    end = "?"
                print(
                    f"[train] ep={done:4d} 步={status['total_steps']:7d} "
                    f"回报(近50均)={status['mean_return_last50']:>9.2f} "
                    f"最好={status['best_return']:>9.2f} "
                    f"平均面数={status['mean_stage_last50']} "
                    f"最好面数={status['best_stage']} "
                    f"本局结束={end} 掉残={last.get('misses')} "
                    f"{status['steps_per_sec']}步/秒"
                )
                if status["last_evaluation"]:
                    ev = status["last_evaluation"]
                    print(
                        f"         └─ 贪心评估: 回报={ev['mean_return']} 面数={ev['best_stage']} 通关={ev['clears']}"
                    )
            if trainer.error:
                print(f"[train] 出错: {trainer.error}")
                break
    except KeyboardInterrupt:
        print("\n[train] 收到 Ctrl-C, 正在停止...")
    finally:
        status = trainer.stop(save=True)
        elapsed = time.perf_counter() - started
        print(
            f"\n[train] 结束: {status['episodes_done']} 局 / {status['total_steps']} 步 / "
            f"{elapsed:.1f}s ({status['total_steps']/max(elapsed,1e-9):.0f} 步/秒)"
        )
        print(f"[train] 策略已保存: {status['policy_path']}.npz")
        recent = trainer.episodes[-min(20, len(trainer.episodes)) :]
        if recent:
            early = trainer.episodes[: min(20, len(trainer.episodes))]
            print(
                f"[train] 前{len(early)}局平均回报={sum(r.return_ for r in early)/len(early):.2f} "
                f"-> 后{len(recent)}局平均回报={sum(r.return_ for r in recent)/len(recent):.2f}"
            )
        env.close()
    if trainer.error:
        print(f"[train] 失败: {trainer.error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
