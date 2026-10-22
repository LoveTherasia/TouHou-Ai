"""弹型基准: 用同一份策略在固定种子上逐个弹型跑, 看"到底会躲哪些、死在哪些".

单面的存活帧数只能给一个平均数, 掩盖了"某些弹型稳、某些弹型一碰就死"。这个工具
按弹型分别报 **均值 / 最差局 / 死亡数 / 伤害 / 对准率**, 是接下去做能力验收的入口。

    python tools/pattern_bench.py --policy .runtime/policies/attack_ppo_fixed.npz
    python tools/pattern_bench.py --policy ... --patterns dense_ring,ring_gate --seeds 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from touhou_rl_mcp.backends.patterns import PATTERN_DOCS, pattern_names  # noqa: E402
from touhou_rl_mcp.config import Config  # noqa: E402
from touhou_rl_mcp.env import TouhouEnv  # noqa: E402
from touhou_rl_mcp.games.registry import GameRegistry  # noqa: E402
from touhou_rl_mcp.rl.reward import REWARD_PROFILES, reward_profile  # noqa: E402
from touhou_rl_mcp.training import TrainConfig, Trainer  # noqa: E402
from touhou_rl_mcp.training.viability import ViabilityPlanner  # noqa: E402

#: 反应式策略的默认验收弹型 (由易到难; wall_gate 单独看, 它需要提前占位)
DEFAULT_PATTERNS = ("aimed_single", "aimed_spread", "fan", "spiral",
                    "dense_ring", "ring_gate", "closing_ring", "wall_gate")
#: 默认验收种子数 (与 STATUS 里的"固定 5 seeds"一致)
DEFAULT_SEEDS = 5


def run_chooser(env, chooser, seeds: int, max_steps: int, frames: int = 1) -> tuple:
    """用任何带 ``choose(state) -> action_name`` 的对象驱动若干局."""
    steps, dmg, align = [], [], []
    for i in range(seeds):
        state = env.reset(seed=1000 + i).state
        hp_prev = state.boss_hp
        d = 0.0
        n = 0
        aligned = boss = 0
        while n < max_steps:
            action = chooser.choose(state)
            result = env.step(action, frames=frames)
            state = result.state
            n += 1
            if state.boss_hp is not None and hp_prev is not None:
                delta = hp_prev - state.boss_hp
                if 0 < delta < 500:
                    d += delta
            hp_prev = state.boss_hp
            pos = (state.raw or {}).get("boss_position")
            if pos and state.player_x is not None:
                boss += 1
                if abs(state.player_x - float(pos[0])) <= 24:
                    aligned += 1
            if result.done or result.truncated:
                break
        steps.append(n)
        dmg.append(d)
        align.append(aligned / boss if boss else 0.0)
    return steps, dmg, align


def run_planner(env, planner: ViabilityPlanner, seeds: int, max_steps: int) -> tuple:
    """用规划器直接打若干局, 返回 (steps, damage, align) 三个列表."""
    steps, dmg, align = [], [], []
    for i in range(seeds):
        state = env.reset(seed=1000 + i).state
        hp_prev = state.boss_hp
        d = 0.0
        n = 0
        aligned = boss = 0
        while n < max_steps:
            action = planner.choose(state)[0]
            result = env.step(action, frames=1)
            state = result.state
            n += 1
            if state.boss_hp is not None and hp_prev is not None:
                delta = hp_prev - state.boss_hp
                if 0 < delta < 500:
                    d += delta
            hp_prev = state.boss_hp
            pos = (state.raw or {}).get("boss_position")
            if pos and state.player_x is not None:
                boss += 1
                if abs(state.player_x - float(pos[0])) <= 24:
                    aligned += 1
            if result.done or result.truncated:
                break
        steps.append(n)
        dmg.append(d)
        align.append(aligned / max(1, boss))
    return steps, dmg, align


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True,
                    help="策略 .npz / planner(可行域) / mpc(CEM) / potential(势能场)")
    ap.add_argument("--mpc-horizon", type=int, default=24, help="MPC 视界(帧)")
    ap.add_argument("--mpc-dt", type=int, default=3, help="MPC 每步推进帧数")
    ap.add_argument("--mpc-samples", type=int, default=48, help="CEM 采样数")
    ap.add_argument("--mpc-iters", type=int, default=3, help="CEM 迭代轮数")
    ap.add_argument("--mpc-clearance", type=float, default=10.0,
                    help="MPC 期望的最小安全余量(px)")
    ap.add_argument("--mpc-dynamics", default="",
                    help="训练好的动力学模型 .npz (tools/train_dynamics.py 的产物); "
                         "空=匀速外推")
    ap.add_argument("--mpc-gap", type=float, default=0.0,
                    help="MPC 的「缺口可达性」权重: 用弹幕穿过自机高度的时刻 "
                         "减去赶路所需帧数, 显式判断能不能赶到缺口列")
    ap.add_argument("--mpc-column", type=float, default=0.0,
                    help="MPC 的「正上方弹幕柱」惩罚权重 (墙/闭合环需要 >0)")
    ap.add_argument("--mpc-commit", type=int, default=0,
                    help="MPC 并列动作承诺 (选定方向后 N 帧不换向)")
    ap.add_argument("--mpc-switch", type=float, default=0.0,
                    help="承诺期内换向所需的代价优势")
    ap.add_argument("--mpc-focus-margin", type=float, default=0.0,
                    help="MPC: 慢速(focus)要便宜这么多才按 shift")
    ap.add_argument("--planner-horizon", type=int, default=200, help="规划器视野(帧)")
    ap.add_argument("--planner-dt", type=int, default=2, help="规划器每层帧数")
    ap.add_argument("--planner-aim-weight", type=float, default=0.0,
                    help="规划器: 安全位置里偏向 Boss 列的权重 (0=纯躲避)")
    ap.add_argument("--patterns", default="", help="逗号分隔; 空=默认验收集; all=全部 14 种")
    ap.add_argument("--seeds", type=int, default=DEFAULT_SEEDS, help="种子个数 (config.seed + 0..n-1)")
    ap.add_argument("--max-steps", type=int, default=3000, help="单局上限 (=100 秒 @1 帧/步)")
    ap.add_argument("--step-frames", type=int, default=1)
    ap.add_argument("--hidden", default="256,256")
    ap.add_argument("--difficulty", type=int, default=0)
    ap.add_argument("--k-bullets", type=int, default=16)
    ap.add_argument("--k-enemies", type=int, default=8)
    ap.add_argument("--bullet-velocity", action="store_true")
    ap.add_argument("--danger-grad", action="store_true")
    ap.add_argument("--gap", action="store_true", help="用缺口/占用栅格观测 (需与策略一致)")
    ap.add_argument("--gap-layers", type=int, default=1)
    ap.add_argument("--wall-feat", action="store_true",
                    help="观测末尾含 12 维墙特征 (必须与训练时一致)")
    ap.add_argument("--grid", action="store_true",
                    help="观测改用危险图(以自机为中心的弹幕栅格)")
    ap.add_argument("--grid-size", type=int, default=16, help="危险图边长(格数)")
    ap.add_argument("--grid-span", type=float, default=110.0, help="危险图半径(像素)")
    ap.add_argument("--grid-scales", default="", help="多尺度危险图 'size:span,...'")
    ap.add_argument("--no-focus", dest="focus", action="store_false")
    ap.set_defaults(focus=True)
    ap.add_argument("--hier-switch", action="store_true",
                    help="开分层切换: 检测到整幅横墙/多向收网时改问可行域规划器")
    ap.add_argument("--hier-replan-every", type=int, default=1,
                    help="分层切换时墙段内沿用上次规划方向的步数 (2~3 可提速)")
    ap.add_argument("--reward-profile", default="", choices=[""] + sorted(REWARD_PROFILES))
    ap.add_argument("--json", default="", help="把结果写到这个 json (便于对比两次实验)")
    args = ap.parse_args()

    if args.patterns.strip() == "all":
        names = pattern_names()
    elif args.patterns.strip():
        names = [p.strip() for p in args.patterns.split(",") if p.strip()]
    else:
        names = list(DEFAULT_PATTERNS)
    unknown = [n for n in names if n not in pattern_names()]
    if unknown:
        print(f"[bench] 未知弹型: {unknown}; 可选: {pattern_names()}")
        return 2

    rows = []
    for pat in names:
        cfg = Config.load()
        cfg.patch({"backend": "mock", "game": "th11", "step_frames": args.step_frames,
                   "danmaku_pattern": pat, "start_stage": 1})
        if args.reward_profile:
            cfg.patch({"reward": reward_profile(args.reward_profile)})
        env = TouhouEnv(cfg, registry=GameRegistry())
        env.start()
        tc = TrainConfig(
            episodes=0, max_steps=args.max_steps, step_frames=args.step_frames,
            hidden=tuple(int(x) for x in args.hidden.split(",")),
            difficulty=args.difficulty, k_bullets=args.k_bullets,
            k_enemies=args.k_enemies, bullet_velocity=args.bullet_velocity,
            use_danger_grad=args.danger_grad, use_focus=args.focus,
            use_gap=args.gap, gap_layers=args.gap_layers,
            use_wall_feat=args.wall_feat,
            use_grid=args.grid, grid_size=args.grid_size, grid_span=args.grid_span,
            grid_scales=args.grid_scales,
            hier_switch=args.hier_switch,
            planner_horizon=args.planner_horizon, planner_dt=args.planner_dt,
        )
        dyn = None
        if args.mpc_dynamics:
            from touhou_rl_mcp.training.dynamics import ResidualMLP

            dyn = ResidualMLP.load(args.mpc_dynamics)
            print(f"[bench] 动力学模型: {args.mpc_dynamics} ({dyn.describe()})", flush=True)
        if args.policy.strip().lower() in ("mpc-adaptive", "mpc_adaptive"):
            from touhou_rl_mcp.training.mpc import AdaptiveMPCPlanner

            planner = AdaptiveMPCPlanner(short_horizon=24, short_dt=3,
                                         long_horizon=args.mpc_horizon,
                                         long_dt=args.mpc_dt,
                                         long_column=args.mpc_column or 2.0,
                                         long_gap=args.mpc_gap,
                                         n_samples=args.mpc_samples,
                                         iters=args.mpc_iters, use_focus=args.focus,
                                         dynamics=dyn,
                                         commit_steps=args.mpc_commit,
                                         switch_margin=args.mpc_switch,
                                         focus_margin=args.mpc_focus_margin,
                                         deterministic=True)
            steps, dmg, align = run_chooser(env, planner, args.seeds, args.max_steps, 1)
        elif args.policy.strip().lower() == "mpc":
            from touhou_rl_mcp.training.mpc import MPCPlanner

            planner = MPCPlanner(horizon=args.mpc_horizon, dt=args.mpc_dt,
                                 n_samples=args.mpc_samples, iters=args.mpc_iters,
                                 clearance_margin=args.mpc_clearance,
                                 column_weight=args.mpc_column,
                                 gap_weight=args.mpc_gap,
                                 dynamics=dyn,
                                 use_focus=args.focus,
                                 commit_steps=args.mpc_commit,
                                 switch_margin=args.mpc_switch,
                                 focus_margin=args.mpc_focus_margin,
                                 deterministic=True)
            steps, dmg, align = run_chooser(env, planner, args.seeds, args.max_steps, 1)
        elif args.policy.strip().lower() == "potential":
            from touhou_rl_mcp.training.potential import PotentialFieldPlanner

            planner = PotentialFieldPlanner(use_focus=args.focus)
            steps, dmg, align = run_chooser(env, planner, args.seeds, args.max_steps, 1)
        elif args.policy.strip().lower() == "planner":
            planner = ViabilityPlanner(horizon=args.planner_horizon, dt=args.planner_dt,
                                       aim_weight=args.planner_aim_weight)
            steps, dmg, align = run_planner(env, planner, args.seeds, args.max_steps)
        else:
            trainer = Trainer(env, tc, env_lock=env.lock)
            trainer.load(args.policy)
            steps, dmg, align, tops, minys = [], [], [], [], []
            for i in range(args.seeds):
                record, _ = trainer._run_episode(i, greedy=True)
                steps.append(record.steps)
                dmg.append(record.boss_damage)
                align.append(record.align_ratio or 0.0)
                tops.append(record.top_ratio)
                minys.append(record.min_y if record.min_y is not None else 999.0)
        env.close()
        n = max(1, len(steps))
        rows.append({
            "pattern": pat,
            "mean_steps": round(sum(steps) / n, 1),
            "worst_steps": min(steps),
            "best_steps": max(steps),
            "deaths": sum(1 for s in steps if s < args.max_steps),
            "mean_damage": round(sum(dmg) / n, 1),
            "align": round(sum(align) / n, 3),
            "top_ratio": round(sum(tops) / n, 3) if tops else 0.0,
            "min_y": round(min(minys), 1) if minys else None,
            "steps": steps,
        })
        print("[bench] %-14s 均%6.0f 最差%6d 死亡%2d/%d 伤害%6.0f 对准%4.0f%% "
              "贴顶%4.0f%% min_y%5.0f  %s"
              % (pat, rows[-1]["mean_steps"], rows[-1]["worst_steps"], rows[-1]["deaths"],
                 args.seeds, rows[-1]["mean_damage"], 100 * rows[-1]["align"],
                 100 * rows[-1]["top_ratio"],
                 rows[-1]["min_y"] if rows[-1]["min_y"] is not None else -1, steps), flush=True)

    overall = sum(r["mean_steps"] for r in rows) / max(1, len(rows))
    worst = min(rows, key=lambda r: r["mean_steps"]) if rows else None
    print("\n[bench] 策略=%s  弹型=%d  种子=%d  上限=%d步"
          % (args.policy, len(rows), args.seeds, args.max_steps))
    avg_top = sum(r.get("top_ratio", 0.0) for r in rows) / max(1, len(rows))
    print("[bench] 平均贴顶占比 %.0f%% (越低越好; 上穿合法, 常驻天花板是退化)"
          % (100 * avg_top))
    print("[bench] 平均存活 %.0f 步 (%.0f%% 上限); 最弱弹型 = %s (%s)"
          % (overall, 100 * overall / max(1, args.max_steps),
             worst["pattern"] if worst else "-",
             PATTERN_DOCS.get(worst["pattern"], "") if worst else ""))

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"policy": args.policy, "seeds": args.seeds,
                                   "max_steps": args.max_steps, "rows": rows},
                                  ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[bench] 已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
