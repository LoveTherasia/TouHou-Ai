"""在 v2 缺口基准上用**真正的训练器**跑 (GapObservation + 新奖励 + 新标签)。

和 ``gap_rl.py`` 的分工
----------------------
``gap_rl.py`` 是一套平行实现 (自己的 Policy / rollout / 更新), 好处是能快速试
"标签口径"这种研究问题, 坏处是它和线上训练路径越走越远 —— 一个只在实验脚本里
成立的结论, 接不回 :class:`~touhou_rl_mcp.training.trainer.Trainer` 就没有意义。

这个脚本把三件东西一起换上, 走正式训练链路:

* **观测**: ``TrainConfig.use_gap`` —— 未来占用栅格 + 几何缺口剖面, 含上一动作 one-hot;
* **奖励**: ``reward_profile("gap_survive")`` —— 过面进度 + 擦弹 - 碰撞, **关掉斥力项**;
* **标签**: ``teacher="gap"`` / ``"gap_dagger"`` —— 几何缺口标签, 支持 DAgger。

分阶段跑 (每阶段内部按 chunk 增量训练, 每 chunk 结束在**留出场景**上评估一次):

    阶段 A  BC        teacher="gap"        (beta=1, 纯模仿, 数据来自随机轨迹)
    阶段 B  DAgger    teacher="gap_dagger" (beta=0.15, 策略驱动 + 老师打标签)
    阶段 C  RL        teacher=""           (PPO, 用新奖励)

用法::

    python tools/research/gap_trainer.py --bench v2 --label slack \
        --bc-episodes 30 --dagger-episodes 60 --rl-episodes 120
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gap_rl import BENCHMARKS  # noqa: E402

from touhou_rl_mcp.config import Config, runtime_dir          # noqa: E402
from touhou_rl_mcp.env import TouhouEnv                       # noqa: E402
from touhou_rl_mcp.rl.reward import reward_profile            # noqa: E402
from touhou_rl_mcp.training.features import gap_encoder, gap_spec_from  # noqa: E402
from touhou_rl_mcp.training.gap_label import (                # noqa: E402
    COMMIT_FRAMES, LABEL_MODES, MACRO_ACTIONS, gap_label,
)
from touhou_rl_mcp.training.trainer import (                  # noqa: E402
    TrainConfig, Trainer, apply_scenario,
)

#: 观测口径 -> TrainConfig 的三个开关
ARMS: dict[str, dict[str, bool]] = {
    "geo": dict(gap_grid=False, gap_geo=True, gap_clearance=True),
    "grid": dict(gap_grid=True, gap_geo=False, gap_clearance=False),
    "both": dict(gap_grid=True, gap_geo=True, gap_clearance=True),
}


# ----------------------------------------------------------------------
def build_env(scenario: dict[str, float], reward: dict[str, float]) -> TouhouEnv:
    cfg = Config.load()
    cfg.patch({
        "backend": "mock",
        "game": "th11",
        "danmaku_pattern": "wall_gate",
        "step_frames": COMMIT_FRAMES,
        "max_episode_frames": 0,        # 只按 game over 结束 (基准要靠存活帧数比较)
        "auto_reset": False,
        "reward": reward,
    })
    env = TouhouEnv(cfg)
    env.start()
    env.reset(seed=1)
    apply_scenario(env, scenario)
    return env


def make_trainer(env: TouhouEnv, args: argparse.Namespace,
                 scenario: dict[str, float], *, teacher: str,
                 out_dir: Path) -> Trainer:
    config = TrainConfig(
        algorithm="ppo",
        episodes=0,
        max_steps=max(1, args.max_frames // COMMIT_FRAMES),
        step_frames=COMMIT_FRAMES,
        batch_episodes=max(1, args.batch),
        lr=args.lr,
        gamma=args.gamma,
        entropy_coef=args.entropy,
        hidden=(args.hidden, args.hidden) if args.hidden > 0 else (),
        use_focus=False,               # 9 动作, 与缺口标签的动作空间同构
        use_bomb=False,
        use_gap=True,
        **ARMS[args.arm],
        gap_prev_action=True,
        gap_label=args.label,
        scenario=dict(scenario),
        # 缺口相位每局重采: 否则 2400 帧的基准退化成"同一个局面重复 N 次"
        scenario_random={"gap_phase": [0.0, 1.0]},
        teacher=teacher,
        dagger_beta=args.dagger_beta,
        teacher_epochs=args.teacher_epochs,
        ppo_epochs=args.ppo_epochs,
        ppo_minibatch=args.minibatch,
        eval_every=0,                  # 留出场景评估由本脚本控制
        plot_every=0,
        capture_every=0,
        save_every=0,
        log_steps=False,
        seed=args.seed,
        out_dir=str(out_dir),
    )
    return Trainer(env, config, out_dir=out_dir)


# ----------------------------------------------------------------------
def evaluate(env: TouhouEnv, trainer: Trainer,
             heldout: tuple[dict[str, float], ...], *, max_frames: int,
             label_mode: str, seeds: tuple[int, ...] = (101,)) -> dict[str, Any]:
    """在**未见过的**场景上贪心跑一局, 回来存活帧数 / 与标签的一致率。"""
    enc = gap_encoder(trainer.config)
    spec = gap_spec_from(trainer.config)
    frames: list[int] = []
    agree: list[float] = []
    for scenario in heldout:
        for seed in seeds:
            state = env.reset(seed=seed).state
            apply_scenario(env, scenario)
            survival = 0
            prev = -1
            hits = total = 0
            while survival < max_frames:
                obs = enc(state, prev_action=prev)
                with trainer.policy_lock:
                    action = int(trainer.policy.greedy(obs))
                hits += int(action == gap_label(state, spec, label_mode))
                total += 1
                result = env.step(MACRO_ACTIONS[action], frames=COMMIT_FRAMES,
                                  auto_reset=False)
                state = result.state
                survival += COMMIT_FRAMES
                prev = action
                if result.done:
                    break
            frames.append(survival)
            agree.append(hits / total if total else 0.0)
    return {
        "frames": frames,
        "mean": float(np.mean(frames)) if frames else 0.0,
        "agree": float(np.mean(agree)) if agree else 0.0,
    }


def run_chunk(trainer: Trainer, target_episodes: int) -> None:
    """把训练器跑到 ``target_episodes`` 局为止 (到点自己停线程)。"""
    trainer.config.episodes = int(target_episodes)
    trainer.start()
    while trainer.running:
        time.sleep(0.2)
    if trainer.error:
        raise RuntimeError(trainer.error)


def _write_row(path: Path, step: int, name: str, score: float,
               detail: list[int], **extra: Any) -> None:
    row = {"step": int(step), "name": name, "score": round(float(score), 1),
           "detail": [int(x) for x in detail]}
    row.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _fmt(result: dict[str, Any]) -> str:
    return (f"泛化 {result['mean']:6.0f} 帧  一致率 {result['agree']:4.0%}  "
            f"{[int(x) for x in result['frames']]}")


# ----------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="v2 缺口基准 · 正式训练链路")
    ap.add_argument("--arm", choices=tuple(ARMS), default="geo", help="观测口径")
    ap.add_argument("--bench", choices=tuple(BENCHMARKS), default="v2")
    ap.add_argument("--label", choices=LABEL_MODES, default="slack",
                    help="弱监督标签口径 (now/reach/future/slack)")
    ap.add_argument("--reward", default="gap_survive",
                    help="奖励 profile 名 (见 rl/reward.py 的 REWARD_PROFILES); "
                         "default = 不改奖励, 用 config.json 那套")
    ap.add_argument("--gap-margin", type=float, default=0.0,
                    help="缺口余量的势函数塑形系数 (0=关; 2~5 给'别站着不动'密集梯度)")
    ap.add_argument("--bc-episodes", type=int, default=30, help="纯模仿 (BC) 局数")
    ap.add_argument("--dagger-episodes", type=int, default=0,
                    help="DAgger 局数 (>0 时用策略自己的轨迹补标签)")
    ap.add_argument("--rl-episodes", type=int, default=0, help="BC/DAgger 之后的 PPO 局数")
    ap.add_argument("--batch", type=int, default=4, help="每多少局更新一次参数")
    ap.add_argument("--eval-every", type=int, default=20, help="每多少局评估一次留出场景")
    ap.add_argument("--max-frames", type=int, default=2400, help="单局帧数上限")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--rl-lr", type=float, default=3e-4,
                    help="PPO 微调的学习率。必须比监督阶段小一个量级 —— "
                         "实测用 2e-3 微调会把 DAgger 学到的感知一次冲掉")
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--entropy", type=float, default=0.01)
    ap.add_argument("--hidden", type=int, default=64, help="隐层宽度 (0=线性)")
    ap.add_argument("--teacher-epochs", type=int, default=6)
    ap.add_argument("--ppo-epochs", type=int, default=4)
    ap.add_argument("--ppo-minibatch", "--minibatch", type=int, default=256, dest="minibatch")
    ap.add_argument("--dagger-beta", type=float, default=0.15,
                    help="DAgger 时老师接管的概率 (其余时候策略自己走)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--run-dir", default="",
                    help="训练记录目录 (默认 .runtime/runs/<...>; 看板可直接读取)")
    args = ap.parse_args()

    train_scenario, heldout = BENCHMARKS[args.bench]
    # ``--reward default`` = 不动奖励 (用 config.json 里那套危险度惩罚), 作为对照组
    reward = {} if args.reward == "default" else reward_profile(args.reward)
    if args.gap_margin:
        reward["gap_margin_scale"] = float(args.gap_margin)

    out_dir = Path(args.run_dir) if args.run_dir else (
        runtime_dir() / "runs" /
        f"trainer-{args.bench}-{args.label}-{args.arm}-{time.strftime('%Y%m%d-%H%M%S')}")
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_path = out_dir / "eval.jsonl"

    env = build_env(train_scenario, reward)
    trainer = make_trainer(env, args, train_scenario, teacher="", out_dir=out_dir)

    print(f"bench={args.bench}  label={args.label}  arm={args.arm}  "
          f"reward={args.reward}  gap_margin={args.gap_margin}")
    print(f"obs_dim={trainer.obs_dim}  actions={len(trainer.action_table)}  "
          f"run={out_dir}")
    print(f"train scenario: {train_scenario}")

    meta: dict[str, Any] = {
        "script": "tools/research/gap_trainer.py",
        "bench": args.bench, "label": args.label, "arm": args.arm,
        "reward": args.reward, "gap_margin": args.gap_margin,
        "obs_dim": trainer.obs_dim,
        "train_scenario": train_scenario,
        "heldout": [dict(s) for s in heldout],
        "args": vars(args),
        "phases": [],
    }

    def flush_meta() -> None:
        (out_dir / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8")

    def step_index() -> int:
        return len(trainer.episodes)

    def eval_and_log(tag: str) -> dict[str, Any]:
        result = evaluate(env, trainer, heldout, max_frames=args.max_frames,
                          label_mode=args.label)
        _write_row(eval_path, step_index(), tag, result["mean"], result["frames"],
                   agree=round(result["agree"], 4))
        print(f"  [{tag}] {_fmt(result)}", flush=True)
        return result

    # ---------------- 阶段 A / B: 监督 ----------------
    phases = [
        ("bc", "gap", args.bc_episodes),
        ("dagger", "gap_dagger", args.dagger_episodes),
    ]
    for name, teacher, count in phases:
        if count <= 0:
            continue
        trainer.reconfigure({"teacher": teacher})
        print(f"\n== {name.upper()} ({teacher}, {count} 局) ==", flush=True)
        base, done = step_index(), 0
        chunk = max(1, args.eval_every)
        while done < count:
            done = min(count, done + chunk)
            run_chunk(trainer, base + done)      # 绝对局数 (episodes 不清零)
            result = eval_and_log(name)
            trainer.save()                       # 每段留一个 checkpoint, 崩了也不白跑
            meta["phases"].append({"phase": name, "episodes": step_index(),
                                   "mean_frames": round(result["mean"], 1),
                                   "agree": round(result["agree"], 4)})
        flush_meta()

    # ---------------- 阶段 C: RL ----------------
    if args.rl_episodes > 0:
        trainer.reconfigure({"teacher": "", "algorithm": "ppo", "lr": args.rl_lr})
        print(f"\n== RL (PPO, {args.rl_episodes} 局, lr={args.rl_lr:g}) ==", flush=True)
        base, done = step_index(), 0
        chunk = max(1, args.eval_every)
        while done < args.rl_episodes:
            done = min(args.rl_episodes, done + chunk)
            run_chunk(trainer, base + done)
            result = eval_and_log("rl")
            trainer.save()
            meta["phases"].append({"phase": "rl", "episodes": step_index(),
                                   "mean_frames": round(result["mean"], 1),
                                   "agree": round(result["agree"], 4)})
        flush_meta()

    if not trainer.episodes:
        eval_and_log("untrained")
    trainer.save()
    final = evaluate(env, trainer, heldout, max_frames=args.max_frames,
                     label_mode=args.label)
    meta["final"] = {"mean_frames": round(final["mean"], 1),
                     "agree": round(final["agree"], 4),
                     "detail": [int(x) for x in final["frames"]]}
    flush_meta()
    print(f"\n最终: {_fmt(final)}")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
