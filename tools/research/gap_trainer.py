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


def random_ranges(base: dict[str, float]) -> dict[str, list[float]]:
    """按基准场景派生**训练用**的参数区间 (域随机化)。

    为什么必须随机化: 只用 ``{"gap_phase": [0, 1]}`` 时, 间隔/弹速/漂移/缺口宽度
    都是写死的, 留出场景 (反向漂移、更快弹速、更短间隔) 全部落在训练分布之外 ——
    这时"泛化帧数"测的是运气, 不是能力。把它们的区间张开, 策略才没得背, 只能学
    "看缺口、提前占位"这件事本身。

    区间按相对比例取, 这样 v1/v2/v3 同一套代码都能用。
    """
    interval = float(base.get("interval", 75.0))
    speed = float(base.get("speed", 2.0))
    width = float(base.get("gap_width", 44.0))
    return {
        "gap_phase": [0.0, 1.0],
        # 漂移**含符号**: 左右横移都要会, 否则遇到反向的墙只会往错的方向跑
        "gap_drift": [-0.95, 0.95],
        "interval": [interval * 0.90, interval * 1.10],
        "speed": [speed * 0.93, speed * 1.10],
        "gap_width": [width * 0.90, width * 1.10],
    }


# ----------------------------------------------------------------------
def build_env(scenario: dict[str, float], reward: dict[str, float],
              script: str = "") -> TouhouEnv:
    cfg = Config.load()
    patch: dict[str, Any] = {
        "backend": "mock",
        "game": "th11",
        "danmaku_pattern": "" if script else "wall_gate",
        "step_frames": COMMIT_FRAMES,
        "max_episode_frames": 0,        # 只按 game over 结束 (基准要靠存活帧数比较)
        "auto_reset": False,
        "reward": reward,
    }
    if script:
        patch["danmaku_script"] = script
    cfg.patch(patch)
    env = TouhouEnv(cfg)
    env.start()
    env.reset(seed=1)
    if script:
        apply_script_offset(env, 0.0)
    else:
        apply_scenario(env, scenario)
    return env


def script_length(env: TouhouEnv) -> int:
    script = getattr(getattr(env, "backend", None), "_script", None) or {}
    return int(script.get("length") or 0)


def apply_script_offset(env: TouhouEnv, fraction: float) -> None:
    """把脚本起点设到 ``fraction`` 处 (0~1)。必须在 reset **之前**设, reset 会读它。"""
    length = script_length(env)
    if length <= 0:
        return
    #: 显式 int: 某些后端把配置值原样拿去取模, 浮点会静默失效
    env.backend.config.patch({"danmaku_offset": int(round(fraction * length))})


def make_trainer(env: TouhouEnv, args: argparse.Namespace,
                 scenario: dict[str, float], *, teacher: str,
                 out_dir: Path) -> Trainer:
    config = TrainConfig(        algorithm="ppo",
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
        gap_layers=args.gap_layers,
        gap_label=args.label,
        scenario=dict(scenario),
        # 缺口相位每局重采: 否则 2400 帧的基准退化成"同一个局面重复 N 次"
        # --rand 再把间隔/弹速/缺口宽度/漂移方向一起张开 (域随机化)
        scenario_random=(random_ranges(scenario) if args.rand else
                         ({"gap_phase": [0.0, 1.0]} if scenario else {})),
        # 脚本模式: 每局从录制的不同位置开始 (脚本会循环, 这是唯一的多样性来源)
        script_offset_range=list(args.script_train_range) if args.script else [],
        teacher=teacher,
        dagger_beta=args.dagger_beta,
        teacher_epochs=args.teacher_epochs,
        ppo_epochs=args.ppo_epochs,
        ppo_minibatch=args.minibatch,
        kl_anchor=args.kl_anchor,
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
             label_mode: str, phases: tuple[float, ...] = (0.0,)) -> dict[str, Any]:
    """在**未见过的**场景上贪心跑, 回来存活帧数 / 与标签的一致率。

    ``phases`` 是叠加在场景 ``gap_phase`` 上的相位偏移。**必须多于一个**: 弹幕墙
    每堵横移 ``drift*360`` px, 初始相位差 1/3 就是完全不同的死法, 3 局平均的
    方差足以把"学会"和"运气好"混在一起。v3 上实测同一策略的局间极差能到 900 帧。
    """
    enc = gap_encoder(trainer.config)
    spec = gap_spec_from(trainer.config)
    frames: list[int] = []
    agree: list[float] = []
    for scenario in heldout:
        for offset in phases:
            sc = dict(scenario)
            sc["gap_phase"] = float(sc.get("gap_phase", 0.0)) + float(offset)
            state = env.reset(seed=101).state
            apply_scenario(env, sc)
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
        "min": float(np.min(frames)) if frames else 0.0,
        "agree": float(np.mean(agree)) if agree else 0.0,
    }


def evaluate_script(env: TouhouEnv, trainer: Trainer, fractions: tuple[float, ...],
                    *, max_frames: int, label_mode: str) -> dict[str, Any]:
    """在**没训练过的脚本片段**上贪心跑 (真假弹幕模式的评估)。"""
    enc = gap_encoder(trainer.config)
    spec = gap_spec_from(trainer.config)
    frames: list[int] = []
    agree: list[float] = []
    for frac in fractions:
        apply_script_offset(env, float(frac))
        state = env.reset(seed=101).state
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
        "min": float(np.min(frames)) if frames else 0.0,
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
    return (f"泛化 {result['mean']:6.0f} 帧 (最差 {result['min']:4.0f})  "
            f"一致率 {result['agree']:4.0%}  "
            f"{[int(x) for x in result['frames']]}")


# ----------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="v2 缺口基准 · 正式训练链路")
    ap.add_argument("--arm", choices=tuple(ARMS), default="geo", help="观测口径")
    ap.add_argument("--bench", choices=tuple(BENCHMARKS), default="v2")
    ap.add_argument("--script", default="",
                    help="改用**真机录制的弹幕脚本** (如 .runtime/danmaku/th11_stage1_long.json)。"
                         "给了它就不再用 wall_gate 合成基准, 每局从脚本随机位置开始")
    ap.add_argument("--script-train-range", type=float, nargs=2, default=[0.0, 0.75],
                    metavar=("LO", "HI"),
                    help="训练可用的脚本区间 (占长度比例); 默认把最后 25%% 留作未见片段")
    ap.add_argument("--script-eval", default="0.80,0.87,0.94",
                    help="评估用的脚本位置 (逗号分隔比例), 应当在训练区间之外")
    ap.add_argument("--label", choices=LABEL_MODES, default="slack",
                    help="弱监督标签口径 (now/reach/future/slack)")
    ap.add_argument("--reward", default="gap_survive",
                    help="奖励 profile 名 (见 rl/reward.py 的 REWARD_PROFILES); "
                         "default = 不改奖励, 用 config.json 那套")
    ap.add_argument("--gap-margin", type=float, default=0.0,
                    help="缺口余量的势函数塑形系数 (0=关; 2~5 给'别站着不动'密集梯度)")
    ap.add_argument("--gap-layers", type=int, default=1, choices=(1, 2),
                    help="TTA 剖面层数: 2 = 再拼一层下一堵墙的到达时间 "
                         "(大漂移基准上, 第 1 层看不到下一堵墙的缺口)")
    ap.add_argument("--bc-episodes", type=int, default=30, help="纯模仿 (BC) 局数")
    ap.add_argument("--rand", action="store_true",
                    help="域随机化: 训练时随机化 间隔/弹速/缺口宽度/漂移方向 "
                         "(不开则只随机缺口相位, 留出场景基本在分布外)")
    ap.add_argument("--eval-phases", type=int, default=3,
                    help="每个留出场景叠加多少个相位偏移 (1=只有原相位, 噪声很大)")
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
    ap.add_argument("--kl-anchor", type=float, default=0.0,
                    help="RL 阶段到监督结束时策略的 KL 锚系数 (0=关)。"
                         "实测不开时 PPO 会把 DAgger 的缺口感知冲掉 "
                         "(v3: 1764 -> 1592)")
    ap.add_argument("--teacher-epochs", type=int, default=6)
    ap.add_argument("--ppo-epochs", type=int, default=4)
    ap.add_argument("--ppo-minibatch", "--minibatch", type=int, default=256, dest="minibatch")
    ap.add_argument("--dagger-beta", type=float, default=0.15,
                    help="DAgger 时老师接管的概率 (其余时候策略自己走)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--run-dir", default="",
                    help="训练记录目录 (默认 .runtime/runs/<...>; 看板可直接读取)")
    args = ap.parse_args()

    if args.script:
        train_scenario, heldout = {}, ()
        eval_fracs = tuple(float(x) for x in args.script_eval.split(",") if x.strip())
    else:
        train_scenario, heldout = BENCHMARKS[args.bench]
        eval_fracs = ()
    # ``--reward default`` = 不动奖励 (用 config.json 里那套危险度惩罚), 作为对照组
    reward = {} if args.reward == "default" else reward_profile(args.reward)
    if args.gap_margin:
        reward["gap_margin_scale"] = float(args.gap_margin)

    stem = Path(args.script).stem if args.script else args.bench
    out_dir = Path(args.run_dir) if args.run_dir else (
        runtime_dir() / "runs" /
        f"trainer-{stem}-{args.label}-{args.arm}-{time.strftime('%Y%m%d-%H%M%S')}")
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_path = out_dir / "eval.jsonl"

    env = build_env(train_scenario, reward, script=args.script)
    trainer = make_trainer(env, args, train_scenario, teacher="", out_dir=out_dir)

    where = f"script={args.script}" if args.script else f"bench={args.bench}"
    print(f"{where}  label={args.label}  arm={args.arm}  "
          f"reward={args.reward}  gap_margin={args.gap_margin}  rand={args.rand}")
    print(f"obs_dim={trainer.obs_dim}  actions={len(trainer.action_table)}  "
          f"run={out_dir}")
    if args.script:
        print(f"脚本长度 {script_length(env)} 帧, 训练区间 {args.script_train_range}, "
              f"留出片段 {eval_fracs}")
    else:
        print(f"train scenario: {train_scenario}")
        if args.rand:
            print(f"train ranges: {random_ranges(train_scenario)}")

    meta: dict[str, Any] = {
        "script": "tools/research/gap_trainer.py",
        "bench": ("script" if args.script else args.bench),
        "danmaku_script": str(args.script or ""),
        "eval_fracs": list(eval_fracs),
        "label": args.label, "arm": args.arm,
        "reward": args.reward, "gap_margin": args.gap_margin,
        "rand": bool(args.rand),
        "train_ranges": ([list(args.script_train_range)] if args.script else
                         (random_ranges(train_scenario) if args.rand
                          else {"gap_phase": [0.0, 1.0]})),
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

    #: 相位偏移: 0 加上均匀切分的其余相位, 每个场景多跑几局
    phase_offsets = tuple(
        i / max(1, args.eval_phases) for i in range(max(1, args.eval_phases)))

    best = {"mean": -1.0, "step": 0, "tag": ""}

    def eval_and_log(tag: str) -> dict[str, Any]:
        if args.script:
            result = evaluate_script(env, trainer, eval_fracs,
                                     max_frames=args.max_frames, label_mode=args.label)
        else:
            result = evaluate(env, trainer, heldout, max_frames=args.max_frames,
                              label_mode=args.label, phases=phase_offsets)
        _write_row(eval_path, step_index(), tag, result["mean"], result["frames"],
                   agree=round(result["agree"], 4),
                   min_frames=round(result["min"], 1))
        # 逐段择优: 训练只覆盖"最后一版"会丢掉中途更好的权重 (脚本模式实测
        # 2218 -> 1992 的回落), 而评估是唯一可信的选型依据。
        if result["mean"] > best["mean"]:
            best.update(mean=result["mean"], step=step_index(), tag=tag)
            trainer.save(out_dir / "policy_best")
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
    final = (evaluate_script(env, trainer, eval_fracs, max_frames=args.max_frames,
                             label_mode=args.label)
             if args.script else
             evaluate(env, trainer, heldout, max_frames=args.max_frames,
                      label_mode=args.label, phases=phase_offsets))
    meta["best"] = {"mean_frames": round(best["mean"], 1),
                    "step": best["step"], "phase": best["tag"]}
    meta["final"] = {"mean_frames": round(final["mean"], 1),
                     "min_frames": round(final["min"], 1),
                     "agree": round(final["agree"], 4),
                     "detail": [int(x) for x in final["frames"]]}
    flush_meta()
    print(f"\n最终: {_fmt(final)}")
    print(f"最好的一版: {best['mean']:.0f} 帧 (step {best['step']}, {best['tag']}) "
          f"-> {out_dir / 'policy_best.npz'}")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
