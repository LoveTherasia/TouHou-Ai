"""在 mock 上训练到【完整通关】(打通全部 6 面)。

相对 examples/train.py 的增强 (针对"被弹刷输出"漏洞与多符卡 Boss):
  * 奖励**统一来自 config.json 的 reward 段** (不再自带一套参数);
  * 观测: 敌机位置 + 弹幕速度 + 无敌帧;
  * 动作: 9 快 + 9 慢(focus) 共 18 个;
  * 算法: PPO (GAE + clipped surrogate), 可选 reinforce_baseline;
  * 熵退火 + 贪心评估 + 最佳快照 (只在变好时存 policy_best);
  * 课程学习: --start-stage 从第 N 面开局, 逐面推进。

    python tools/mock_full_clear.py --episodes 20000
    python tools/mock_full_clear.py --start-stage 3 --episodes 8000 --out .runtime/mock_s3
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20000)
    ap.add_argument("--max-steps", type=int, default=12000)
    ap.add_argument("--step-frames", type=int, default=4)
    ap.add_argument("--hidden", default="256,256")
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--gamma", type=float, default=0.99,
                    help="折扣因子; step-frames=1(60Hz) 时建议 0.997~0.999")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--entropy", type=float, default=0.06)
    ap.add_argument("--min-entropy", type=float, default=0.01,
                    help="熵退火的下限 (防止过早塌缩到单一动作)")
    ap.add_argument("--k-enemies", type=int, default=8)
    ap.add_argument("--k-bullets", type=int, default=16,
                    help="观测里最近的弹幕数 (密集弹幕可调大到 32)")
    ap.add_argument("--danmaku-script", default="",
                    help="训练用的真机弹幕脚本 (空=内置弹型)")
    ap.add_argument("--eval-danmaku-script", default="",
                    help="贪心评估用的弹幕脚本 (空=内置弹型; 用来区分策略强弱)")
    ap.add_argument("--danmaku-offset", type=int, default=0,
                    help="弹幕脚本起始帧偏移 (用来直接练某一面)")
    ap.add_argument("--danmaku-pattern", default="",
                    help="弹幕类型库: dense_ring / expanding_homing_ring / aimed_trail / "
                         "static_swarm / spiral / fan / wall_gap / random_spray / flower / aimed_spread")
    ap.add_argument("--algorithm", default="ppo", choices=["ppo", "reinforce_baseline"])
    ap.add_argument("--teacher", default="", choices=["", "guard", "dagger", "rule", "rule_dagger"],
                    help="模仿学习: rule=规则老师(与 dodge 宏同源)接管并克隆; rule_dagger=策略驱动+老师打标签")
    ap.add_argument("--dagger-beta", type=float, default=0.2)
    ap.add_argument("--dagger-beta-min", type=float, default=0.15,
                    help="DAgger 老师接管比例的调度下限 (从 --dagger-beta 线性降到它)")
    ap.add_argument("--danger-weight", type=float, default=0.0,
                    help="危险状态加权: 离最近弹幕越近, 模仿损失权重越大 (0=不加权; 建议 3)")
    ap.add_argument("--teacher-epochs", type=int, default=8,
                    help="模仿学习每个 batch 重复几次梯度 (越大收敛越快)")
    ap.add_argument("--ppo-minibatch", type=int, default=512)
    ap.add_argument("--ppo-epochs", type=int, default=4)
    ap.add_argument("--value-coef", type=float, default=0.5)
    ap.add_argument("--target-kl", type=float, default=0.05)
    ap.add_argument("--start-stage", type=int, default=1)
    ap.add_argument("--no-focus", dest="focus", action="store_false",
                    help="不加低速(focus)动作 (默认加)")
    ap.add_argument("--no-invincible", dest="invincible", action="store_false",
                    help="观测不含无敌帧特征 (默认含)")
    ap.add_argument("--danger-grad", action="store_true",
                    help="观测加入危险梯度(最近弹幕斥力合成), 让反应式躲避更易被表示")
    ap.add_argument("--grid", action="store_true",
                    help="观测改用危险图(以自机为中心的弹幕位置栅格)而不是最近K颗弹的一维向量")
    ap.add_argument("--grid-size", type=int, default=16, help="危险图边长(格数, 默认 16)")
    ap.add_argument("--grid-span", type=float, default=110.0,
                    help="危险图半径(像素, 默认 110): 覆盖自机周围 ±span")
    ap.add_argument("--grid-scales", default="",
                    help="多尺度危险图 'size:span,...' 如 '12:90,12:330' (近场细+远场粗); 非空时忽略 --grid-size/--grid-span")
    ap.add_argument("--difficulty", type=int, default=0)
    ap.add_argument("--shaping-scale", type=float, default=1.0,
                    help="奖励整体缩放 (不改相对权重; PPO 数值稳定性用, 如 0.01)")
    ap.add_argument("--eval-every", type=int, default=400)
    ap.add_argument("--eval-episodes", type=int, default=20)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--out", default=".runtime/mock_full")
    ap.add_argument("--load", default="")
    ap.set_defaults(focus=True, invincible=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "train_log.txt"

    def log(msg: str) -> None:
        print(msg, flush=True)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(msg + "\n")

    cfg = Config.load()
    cfg.patch({"backend": "mock", "game": "th11", "step_frames": args.step_frames,
               "start_stage": args.start_stage,
               "danmaku_script": args.danmaku_script,
               "danmaku_offset": args.danmaku_offset,
               "danmaku_pattern": args.danmaku_pattern})
    if args.shaping_scale != 1.0:
        cfg.patch({"reward": {"shaping_scale": args.shaping_scale}})
    reward = dict(cfg.section("reward"))
    env = TouhouEnv(cfg, registry=GameRegistry())
    env.start()

    def make_config(**extra) -> TrainConfig:
        base = dict(
            max_steps=args.max_steps, step_frames=args.step_frames,
            hidden=tuple(int(x) for x in args.hidden.split(",")),
            gamma=args.gamma,
            k_bullets=args.k_bullets, bullet_velocity=True, k_enemies=args.k_enemies,
            use_focus=args.focus, use_invincible=args.invincible,
            use_danger_grad=args.danger_grad,
            use_grid=args.grid, grid_size=args.grid_size, grid_span=args.grid_span,
            grid_scales=args.grid_scales,
            algorithm=args.algorithm, start_stage=args.start_stage,
            ppo_minibatch=args.ppo_minibatch, ppo_epochs=args.ppo_epochs,
            value_coef=args.value_coef, target_kl=args.target_kl,
            teacher=args.teacher, dagger_beta=args.dagger_beta,
            danger_weight=args.danger_weight,
            teacher_epochs=args.teacher_epochs,
            difficulty=args.difficulty, seed=20261018,
        )
        base.update(extra)
        return TrainConfig(**base)

    tc = make_config(episodes=10 ** 9, batch_episodes=args.batch, lr=args.lr,
                     entropy_coef=args.entropy, capture_every=0, eval_every=0,
                     save_every=0, out_dir=str(out))
    trainer = Trainer(env, tc, env_lock=env.lock, out_dir=out)
    if args.load:
        log(f"[full] 载入 {trainer.load(args.load)}")

    eval_data = cfg.as_dict()
    eval_data["danmaku_script"] = args.eval_danmaku_script
    eval_cfg = Config(eval_data)
    eval_env = TouhouEnv(eval_cfg, registry=GameRegistry())
    eval_env.start()
    eval_tc = make_config(episodes=0, seed=999983)
    eval_trainer = Trainer(eval_env, eval_tc, env_lock=eval_env.lock)

    def greedy_eval(count: int):
        trainer.save(out / "policy_snapshot")
        eval_trainer.load(out / "policy_snapshot")
        clears = best_stage = 0
        rets = []
        steps = []
        for k in range(count):
            rec, _ = eval_trainer._run_episode(k, greedy=True)
            rets.append(rec.return_)
            steps.append(rec.steps)
            best_stage = max(best_stage, rec.max_stage or 0)
            if rec.cleared:
                clears += 1
        return (clears, best_stage, sum(rets) / max(1, len(rets)),
                sum(steps) / max(1, len(steps)))

    log(f"[full] reward={json.dumps(reward, ensure_ascii=False)}")
    log(f"[full] algo={tc.algorithm} start_stage={tc.start_stage} focus={tc.use_focus} "
        f"invincible={tc.use_invincible} hidden={tc.hidden} lr={tc.lr} batch={tc.batch_episodes} "
        f"k_enemies={tc.k_enemies} obs_dim={trainer.obs_dim} actions={len(trainer.action_table)}")
    trainer.start()
    best = (-1, -1, -1e9)
    last_eval = last_log = 0
    first_clear = None
    while len(trainer.episodes) < args.episodes and trainer.running:
        time.sleep(0.5)
        n = len(trainer.episodes)
        if n > 0 and n % 50 == 0:
            frac = min(1.0, n / max(1, args.episodes))
            try:
                trainer.reconfigure({
                    "entropy_coef": max(args.min_entropy, args.entropy * (1 - frac) + args.min_entropy * frac),
                    "dagger_beta": max(args.dagger_beta_min,
                                       args.dagger_beta * (1 - frac) + args.dagger_beta_min * frac),
                })
            except Exception:
                pass
        if n - last_eval >= args.eval_every and n > 0:
            last_eval = n
            clears, stage, mean_ret, mean_steps = greedy_eval(args.eval_episodes)
            if args.danmaku_script:
                # 脚本模式 = 弹幕生存: 面数不可靠, 用"存活步数"当指标
                key = (round(mean_steps, 1), round(mean_ret, 1))
                label = f"存活步数 {mean_steps:7.0f} 均回报 {mean_ret:8.1f}"
            else:
                key = (clears, stage, round(mean_ret, 1))
                label = (f"通关 {clears}/{args.eval_episodes} 最远 {stage} 面 "
                         f"均回报 {mean_ret:.1f}")
            if key > best:
                best = key
                try:
                    trainer.save(out / "policy_best")
                except Exception:
                    pass
                log(f"[full] *** 新最佳(贪心): {label} @ep{n} (已存 policy_best)")
            else:
                log(f"[full] 评估: {label} @ep{n}")
            if not args.danmaku_script and first_clear is None and clears > 0:
                first_clear = n
                log(f"[full] === 首次完整通关 (贪心) @ep{n} ===")
        if n - last_log >= args.log_every and n > 0:
            last_log = n
            recent = trainer.episodes[-50:]
            log(f"[full] ep={n:5d} 回报(近50均)={sum(e.return_ for e in recent)/len(recent):8.1f} "
                f"存活步数(近50均)={sum(e.steps for e in recent)/len(recent):7.0f} "
                f"掉残(均)={sum(e.misses for e in recent)/len(recent):.1f}")

    trainer.stop(save=True)
    save_png(trainer.progress_image(), out, name="progress.png")
    log(f"[full] 结束: {len(trainer.episodes)} 局 | 最佳(贪心)={best} | 首次通关={first_clear}")
    log(f"[full] 最佳策略: {out / 'policy_best.npz'}")
    eval_env.close()
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
