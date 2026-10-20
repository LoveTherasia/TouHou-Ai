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
    COMMIT_FRAMES, MACRO_ACTIONS, TEACHER_MODES, direction_sign, update_action_age,
)
from touhou_rl_mcp.training.gap_teacher import GapTeacher        # noqa: E402
from touhou_rl_mcp.training.occupancy import lane_safety         # noqa: E402
from touhou_rl_mcp.training.trainer import (                  # noqa: E402
    TrainConfig, Trainer, apply_scenario,
)

#: 观测口径 -> TrainConfig 的三个开关
ARMS: dict[str, dict[str, bool]] = {
    "geo": dict(gap_grid=False, gap_geo=True, gap_clearance=True),
    "grid": dict(gap_grid=True, gap_geo=False, gap_clearance=False),
    "both": dict(gap_grid=True, gap_geo=True, gap_clearance=True),
}

#: 动作名 -> 下标 (老师输出的是名字, 评估时和策略动作比较)
ACTION_INDEX: dict[str, int] = {name: i for i, name in enumerate(MACRO_ACTIONS)}


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
        gap_action_age=bool(getattr(args, "action_age", False)),
        gap_action_age_cap=int(getattr(args, "action_age_cap", 12) or 12),
        gap_layers=args.gap_layers,
        gap_label=args.label,
        gap_commit_frames=int(getattr(args, "commit_frames", 0) or 0),
        gap_hysteresis=float(getattr(args, "hysteresis", 0.0) or 0.0),
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
def _choose_eval_action(trainer: Trainer, obs: np.ndarray, state: Any,
                        spec: Any, *, sticky: int, hold: int,
                        last_action: int) -> tuple[int, int]:
    """评估时的动作选择: 可选"粘住上一步动作"的承诺包装。

    ``sticky`` 只影响推理, 不改训练。它用来回答一个独立问题: **如果只是让动作
    多保持几步, 抖动带来的损失能不能直接拿回来?** 当自机所在车道很危险
    (``lane_safety`` 接近 0) 时立刻解除保持, 避免把"粘住"变成"粘死"。
    """
    if sticky > 0 and hold > 0 and lane_safety(state, spec) > 0.05:
        return last_action, hold - 1
    with trainer.policy_lock:
        action = int(trainer.policy.greedy(obs))
    return action, max(0, sticky - 1)


def _rollout(env: TouhouEnv, trainer: Trainer, state: Any, *,
             max_frames: int, label_mode: str, commit_frames: int,
             hysteresis: float, sticky: int, action_age_cap: int
             ) -> tuple[int, float, float, float, float]:
    """从 ``state`` 开始贪心跑到死亡或帧数上限。

    返回 ``(存活帧数, 与老师一致率, 策略动作切换率, 老师动作切换率, 左右反向率)``。
    切换率是 P1 的核心指标: 抖动表现为"两个几乎等价的缺口之间每步换边"。
    左右反向率只看 left<->right 的翻转, 不会被"走到目标后变 stay"污染。
    """
    enc = gap_encoder(trainer.config)
    spec = gap_spec_from(trainer.config)
    # 老师必须每局新建: reach_hold / reach_hyst 的承诺状态不能跨局泄漏
    teacher = GapTeacher(spec, mode=label_mode,
                         commit_frames=commit_frames, hysteresis=hysteresis)
    survival = 0
    prev = -1
    prev_expected = -1
    age = 0
    hold = 0
    last_action = 0
    hits = total = 0
    policy_switches = label_switches = reversals = 0
    prev_sign = 0
    while survival < max_frames:
        obs = enc(state, prev_action=prev, action_age=age)
        action, hold = _choose_eval_action(
            trainer, obs, state, spec, sticky=sticky, hold=hold,
            last_action=last_action)
        expected_name, _info = teacher.choose(state, state.frame)
        expected = ACTION_INDEX[expected_name]
        hits += int(action == expected)
        if prev >= 0:
            policy_switches += int(action != prev)
        if prev_expected >= 0:
            label_switches += int(expected != prev_expected)
        sign = direction_sign(action)
        if prev_sign and sign and sign != prev_sign:
            reversals += 1
        if sign:
            prev_sign = sign
        total += 1
        result = env.step(MACRO_ACTIONS[action], frames=COMMIT_FRAMES,
                          auto_reset=False)
        state = result.state
        survival += COMMIT_FRAMES
        age = update_action_age(prev, age, action, action_age_cap)
        prev = action
        prev_expected = expected
        last_action = action
        if result.done:
            break
    denom = max(1, total - 1)
    return survival, (hits / total if total else 0.0), \
        (policy_switches / denom), (label_switches / denom), (reversals / denom)


def evaluate(env: TouhouEnv, trainer: Trainer,
             heldout: tuple[dict[str, float], ...], *, max_frames: int,
             label_mode: str, phases: tuple[float, ...] = (0.0),
             commit_frames: int = 0, hysteresis: float = 0.0,
             sticky: int = 0, action_age_cap: int = 12) -> dict[str, Any]:
    """在**未见过的**场景上贪心跑, 回来存活帧数 / 与标签的一致率。

    ``phases`` 是叠加在场景 ``gap_phase`` 上的相位偏移。**必须多于一个**: 弹幕墙
    每堵横移 ``drift*360`` px, 初始相位差 1/3 就是完全不同的死法, 3 局平均的
    方差足以把"学会"和"运气好"混在一起。v3 上实测同一策略的局间极差能到 900 帧。
    """
    frames: list[int] = []
    agree: list[float] = []
    policy_switch: list[float] = []
    label_switch: list[float] = []
    reversal: list[float] = []
    for scenario in heldout:
        for offset in phases:
            sc = dict(scenario)
            sc["gap_phase"] = (float(sc.get("gap_phase", 0.0)) + float(offset)) % 1.0
            state = env.reset(seed=101).state
            apply_scenario(env, sc)
            survival, rate, pswitch, lswitch, rev = _rollout(
                env, trainer, state, max_frames=max_frames, label_mode=label_mode,
                commit_frames=commit_frames, hysteresis=hysteresis,
                sticky=sticky, action_age_cap=action_age_cap)
            frames.append(survival)
            agree.append(rate)
            policy_switch.append(pswitch)
            label_switch.append(lswitch)
            reversal.append(rev)
    return {
        "frames": frames,
        "mean": float(np.mean(frames)) if frames else 0.0,
        "min": float(np.min(frames)) if frames else 0.0,
        "agree": float(np.mean(agree)) if agree else 0.0,
        "policy_switch": float(np.mean(policy_switch)) if policy_switch else 0.0,
        "label_switch": float(np.mean(label_switch)) if label_switch else 0.0,
        "reversal": float(np.mean(reversal)) if reversal else 0.0,
    }


def evaluate_script(env: TouhouEnv, trainer: Trainer, fractions: tuple[float, ...],
                    *, max_frames: int, label_mode: str,
                    commit_frames: int = 0, hysteresis: float = 0.0,
                    sticky: int = 0, action_age_cap: int = 12) -> dict[str, Any]:
    """在**没训练过的脚本片段**上贪心跑 (真假弹幕模式的评估)。"""
    frames: list[int] = []
    agree: list[float] = []
    policy_switch: list[float] = []
    label_switch: list[float] = []
    reversal: list[float] = []
    for frac in fractions:
        apply_script_offset(env, float(frac))
        state = env.reset(seed=101).state
        survival, rate, pswitch, lswitch, rev = _rollout(
            env, trainer, state, max_frames=max_frames, label_mode=label_mode,
            commit_frames=commit_frames, hysteresis=hysteresis,
            sticky=sticky, action_age_cap=action_age_cap)
        frames.append(survival)
        agree.append(rate)
        policy_switch.append(pswitch)
        label_switch.append(lswitch)
        reversal.append(rev)
    return {
        "frames": frames,
        "mean": float(np.mean(frames)) if frames else 0.0,
        "min": float(np.min(frames)) if frames else 0.0,
        "agree": float(np.mean(agree)) if agree else 0.0,
        "policy_switch": float(np.mean(policy_switch)) if policy_switch else 0.0,
        "label_switch": float(np.mean(label_switch)) if label_switch else 0.0,
        "reversal": float(np.mean(reversal)) if reversal else 0.0,
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
            f"切换 {result.get('policy_switch', 0.0):4.0%}  "
            f"反向 {result.get('reversal', 0.0):4.0%}  "
            f"{[int(x) for x in result['frames']]}")


def _split_fracs(text: str, default: str) -> tuple[float, ...]:
    raw = text.strip() or default
    return tuple(float(x) for x in raw.split(",") if x.strip())


def _phase_sets(count: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """bench 模式: dev 用 0,1/n,...; test 用错开半个步长的相位。

    这是 P0 的最低成本拆分 —— 场景本身仍是同一批 heldout, 只换了起始相位。
    真正的独立测试仍应换场景或换回放, 见 docs/GAP_LEARNING.md。
    """
    n = max(1, int(count))
    dev = tuple(i / n for i in range(n))
    test = tuple(((i + 0.5) / n) % 1.0 for i in range(n))
    return dev, test


def _eval_only(args: argparse.Namespace) -> int:
    """P0: 只做 dev/test 评估, 不改权重。用于给历史 run 一个诚实的数字。"""
    run_dir = Path(args.run_dir)
    meta_path = run_dir / "meta.json"
    policy_path = run_dir / "policy.json"
    if not meta_path.exists() or not policy_path.exists():
        raise SystemExit(f"--eval-only 需要 {run_dir} 下有 meta.json 和 policy.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    policy_meta = json.loads(policy_path.read_text(encoding="utf-8"))
    config = TrainConfig.from_dict(policy_meta["config"])
    script = args.script or str(meta.get("danmaku_script") or "")

    dev = _split_fracs(args.dev, args.script_eval)
    test = _split_fracs(args.test, args.script_eval)
    env = build_env({}, {}, script=script)
    trainer = Trainer(env, config, out_dir=run_dir)
    ckpt = run_dir / ("policy" if args.last else
                      ("policy_best" if (run_dir / "policy_best.npz").exists() else "policy"))
    trainer.load(ckpt)

    if script:
        dev_result = evaluate_script(
            env, trainer, dev, max_frames=args.max_frames, label_mode=config.gap_label,
            commit_frames=config.gap_commit_frames, hysteresis=config.gap_hysteresis,
            sticky=args.sticky, action_age_cap=config.gap_action_age_cap)
        test_result = evaluate_script(
            env, trainer, test, max_frames=args.max_frames, label_mode=config.gap_label,
            commit_frames=config.gap_commit_frames, hysteresis=config.gap_hysteresis,
            sticky=args.sticky, action_age_cap=config.gap_action_age_cap)
    else:
        heldout = tuple(meta.get("heldout") or ())
        dev_phases, test_phases = _phase_sets(args.eval_phases)
        dev_result = evaluate(
            env, trainer, heldout, max_frames=args.max_frames,
            label_mode=config.gap_label, phases=dev_phases,
            commit_frames=config.gap_commit_frames, hysteresis=config.gap_hysteresis,
            sticky=args.sticky, action_age_cap=config.gap_action_age_cap)
        test_result = evaluate(
            env, trainer, heldout, max_frames=args.max_frames,
            label_mode=config.gap_label, phases=test_phases,
            commit_frames=config.gap_commit_frames, hysteresis=config.gap_hysteresis,
            sticky=args.sticky, action_age_cap=config.gap_action_age_cap)

    report = {
        "run": str(run_dir), "checkpoint": str(ckpt),
        "dev": dev_result, "test": test_result,
        "sticky": int(args.sticky),
        "note": "dev 用于选择, test 只用于报告; 两者不应重叠",
    }
    out = run_dir / (f"eval-honest-sticky{int(args.sticky)}.json" if args.sticky
                     else "eval-honest.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str),
                   encoding="utf-8")
    print(f"checkpoint: {ckpt.name}")
    print(f"dev : {_fmt(dev_result)}")
    print(f"test: {_fmt(test_result)}")
    print(f"-> {out}")
    env.close()
    return 0


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
                    help="评估用的脚本位置 (逗号分隔比例), 应当在训练区间之外; "
                         "未给 --dev/--test 时同时用于两者 (旧行为)")
    ap.add_argument("--dev", default="",
                    help="选择 policy_best 用的 dev 片段 (逗号分隔比例)。"
                         "必须是训练区间之外的留出片段, 但**不能**和 --test 重叠")
    ap.add_argument("--test", default="",
                    help="只在最后报一次的 test 片段 (逗号分隔比例)。"
                         "默认沿用 --script-eval; 修 P0 时请显式给独立片段")
    ap.add_argument("--label", choices=TEACHER_MODES, default="slack",
                    help="弱监督标签口径 (now/reach/future/slack; "
                         "reach_hold/reach_hyst 需要 --commit-frames/--hysteresis)")
    ap.add_argument("--commit-frames", type=int, default=0,
                    help="reach_hold 的承诺时长 (帧); 0=关")
    ap.add_argument("--hysteresis", type=float, default=0.0,
                    help="reach_hyst 的滞回阈值 (帧): 新目标必须好这么多才切换")
    ap.add_argument("--action-age", action="store_true",
                    help="观测加入'连续同向决策步数' (1 维), 给策略可部署的承诺记忆")
    ap.add_argument("--action-age-cap", type=int, default=12,
                    help="动作年龄归一化上限 (决策步)")
    ap.add_argument("--sticky", type=int, default=0,
                    help="推理时动作保持步数 (0=关); 用于检验抖动本身的代价")
    ap.add_argument("--eval-only", action="store_true",
                    help="不训练: 读取 --run-dir 的 policy/policy_best, 在 dev/test 上评估")
    ap.add_argument("--last", action="store_true",
                    help="--eval-only 时读最后一版 policy 而不是 policy_best")
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

    if args.eval_only:
        return _eval_only(args)

    if args.script:
        train_scenario, heldout = {}, ()
        dev_fracs = _split_fracs(args.dev, args.script_eval)
        test_fracs = _split_fracs(args.test, args.script_eval)
        eval_fracs = dev_fracs
    else:
        train_scenario, heldout = BENCHMARKS[args.bench]
        dev_fracs = test_fracs = ()
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
              f"dev {dev_fracs}, test {test_fracs}")
    else:
        print(f"train scenario: {train_scenario}")
        if args.rand:
            print(f"train ranges: {random_ranges(train_scenario)}")

    meta: dict[str, Any] = {
        "script": "tools/research/gap_trainer.py",
        "bench": ("script" if args.script else args.bench),
        "danmaku_script": str(args.script or ""),
        "dev_fracs": list(dev_fracs),
        "test_fracs": list(test_fracs),
        "eval_fracs": list(eval_fracs),
        "label": args.label, "arm": args.arm,
        "commit_frames": int(args.commit_frames),
        "hysteresis": float(args.hysteresis),
        "action_age": bool(args.action_age),
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
        "protocol": "dev=选择 policy_best; test=只报告一次",
    }

    def flush_meta() -> None:
        (out_dir / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8")

    def step_index() -> int:
        return len(trainer.episodes)

    #: 相位偏移: dev 用 0,1/n,...; test 用错开半个步长的相位 (见 _phase_sets)
    phase_offsets, test_phase_offsets = _phase_sets(args.eval_phases)

    best = {"mean": -1.0, "step": 0, "tag": ""}

    def _evaluate_on(fractions: tuple[float, ...],
                     phases: tuple[float, ...]) -> dict[str, Any]:
        kwargs = dict(
            max_frames=args.max_frames, label_mode=args.label,
            commit_frames=args.commit_frames, hysteresis=args.hysteresis,
            sticky=args.sticky, action_age_cap=args.action_age_cap)
        if args.script:
            return evaluate_script(env, trainer, fractions, **kwargs)
        return evaluate(env, trainer, heldout, phases=phases, **kwargs)

    def eval_and_log(tag: str) -> dict[str, Any]:
        # P0: 只在一个评估集上选权重。默认 dev; 若用户没拆, dev=test, 会打印警告。
        result = _evaluate_on(dev_fracs, phase_offsets)
        _write_row(eval_path, step_index(), tag, result["mean"], result["frames"],
                   agree=round(result["agree"], 4),
                   switch=round(result.get("policy_switch", 0.0), 4),
                   min_frames=round(result["min"], 1))
        # 逐段择优: 训练只覆盖"最后一版"会丢掉中途更好的权重 (脚本模式实测
        # 2218 -> 1992 的回落), 而 dev 是唯一可信的选型依据。
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

    # P0: dev 上选出的 policy_best 与最后一版分开报告; test 只在这里看一次。
    dev_last = _evaluate_on(dev_fracs, phase_offsets)
    test_last = _evaluate_on(test_fracs, test_phase_offsets)
    _write_row(eval_path, step_index(), "dev-last", dev_last["mean"],
               dev_last["frames"], agree=round(dev_last["agree"], 4),
               switch=round(dev_last.get("policy_switch", 0.0), 4))
    _write_row(eval_path, step_index(), "test-last", test_last["mean"],
               test_last["frames"], agree=round(test_last["agree"], 4),
               switch=round(test_last.get("policy_switch", 0.0), 4))

    best_path = out_dir / "policy_best.npz"
    test_best: dict[str, Any] | None = None
    if best_path.exists():
        trainer.load(out_dir / "policy_best")
        test_best = _evaluate_on(test_fracs, test_phase_offsets)
        _write_row(eval_path, best["step"], "test-best", test_best["mean"],
                   test_best["frames"], agree=round(test_best["agree"], 4),
                   switch=round(test_best.get("policy_switch", 0.0), 4))

    meta["best"] = {
        "dev_mean_frames": round(best["mean"], 1),
        "step": best["step"], "phase": best["tag"],
        "test_mean_frames": None if test_best is None else round(test_best["mean"], 1),
        "test_detail": None if test_best is None else [int(x) for x in test_best["frames"]],
    }
    meta["dev"] = {"mean_frames": round(dev_last["mean"], 1),
                   "min_frames": round(dev_last["min"], 1),
                   "agree": round(dev_last["agree"], 4),
                   "switch": round(dev_last.get("policy_switch", 0.0), 4),
                   "reversal": round(dev_last.get("reversal", 0.0), 4),
                   "label_switch": round(dev_last.get("label_switch", 0.0), 4),
                   "detail": [int(x) for x in dev_last["frames"]]}
    meta["test"] = {"mean_frames": round(test_last["mean"], 1),
                    "min_frames": round(test_last["min"], 1),
                    "agree": round(test_last["agree"], 4),
                    "switch": round(test_last.get("policy_switch", 0.0), 4),
                    "reversal": round(test_last.get("reversal", 0.0), 4),
                    "label_switch": round(test_last.get("label_switch", 0.0), 4),
                    "detail": [int(x) for x in test_last["frames"]]}
    #: 兼容旧文档: final = **最后一版在 test 上**的结果, 不是 best
    meta["final"] = meta["test"]
    flush_meta()

    print(f"\ndev (last) : {_fmt(dev_last)}")
    print(f"test (last): {_fmt(test_last)}")
    if test_best is not None:
        print(f"test (best @ {best['step']}): {_fmt(test_best)}")
    print(f"选择依据: dev 最好 {best['mean']:.0f} 帧 (step {best['step']}, {best['tag']}) "
          f"-> {out_dir / 'policy_best.npz'}")
    if list(dev_fracs) == list(test_fracs):
        print("警告: dev 与 test 相同, 数字仍有选择偏差; 请显式给 --dev/--test")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
