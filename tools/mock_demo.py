"""在 mock 上演示一局, 输出 GIF + 关键帧拼图 + 轨迹摘要。

    # 用训好的策略
    python tools/mock_demo.py --policy .runtime/mock_bc4/policy_best --step-frames 1 --seed 7

    # 用规则老师 (当前最强的 mock 玩家)
    python tools/mock_demo.py --teacher rule --step-frames 1 --seed 7 --out .runtime/demo_teacher
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image  # noqa: E402

from touhou_rl_mcp.config import Config  # noqa: E402
from touhou_rl_mcp.env import TouhouEnv  # noqa: E402
from touhou_rl_mcp.games.registry import GameRegistry  # noqa: E402
from touhou_rl_mcp.training.features import ObservationEncoder  # noqa: E402
from touhou_rl_mcp.training.policy import MLPPolicy  # noqa: E402
from touhou_rl_mcp.training.rule_teacher import RuleTeacher  # noqa: E402
from touhou_rl_mcp.training.trainer import build_action_table  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=".runtime/mock_bc4/policy_best")
    ap.add_argument("--hier-switch", action="store_true",
                    help="分层切换: 常规局面用策略, 整幅横墙/多向收网时改用可行域规划器")
    ap.add_argument("--hier-replan-every", type=int, default=1,
                    help="分层切换: 墙段内沿用上次规划方向的步数 (2~3 可提速)")
    ap.add_argument("--teacher", default="", choices=["", "rule", "score", "viability"],
                    help="用老师代替策略: rule=纯躲避; score=按奖励权重(会打 Boss)")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--every", type=int, default=6, help="每多少 step 抓一帧")
    ap.add_argument("--max-steps", type=int, default=20000)
    ap.add_argument("--step-frames", type=int, default=1, help="每个决策推进的帧数")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--gif-fps", type=float, default=12.0)
    ap.add_argument("--gif-width", type=int, default=384)
    ap.add_argument("--difficulty", type=int, default=0)
    ap.add_argument("--danmaku-script", default="",
                    help="用真机弹幕脚本复刻 (如 .runtime/danmaku/th11_stage1.json)")
    ap.add_argument("--danmaku-offset", type=int, default=0,
                    help="脚本起始帧偏移 (从回放的第几帧开始, 用来直接看某一面)")
    ap.add_argument("--danmaku-pattern", default="",
                    help="弹幕类型库: dense_ring / expanding_homing_ring / aimed_trail / "
                         "static_swarm / spiral / fan / wall_gap / random_spray / flower / aimed_spread")
    ap.add_argument("--out", default=".runtime/mock_demo")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    policy = None
    teacher = None
    controller = None
    actions: list[str] = []
    enc = None
    label = f"teacher:{args.teacher}"
    if not args.teacher:
        npz = Path(args.policy).with_suffix(".npz")
        policy = MLPPolicy.load(npz)
        meta_path = npz.with_suffix(".json")
        meta = json.loads(meta_path.read_text(encoding="utf-8")).get("config", {}) if meta_path.exists() else {}
        enc = ObservationEncoder.from_meta(meta)
        actions = build_action_table(use_focus=bool(meta.get("use_focus", False)),
                                     use_bomb=bool(meta.get("use_bomb", False)))
        if policy.obs_dim != enc.dim:
            raise SystemExit(f"维度不匹配: policy={policy.obs_dim} 期望={enc.dim}")
        label = str(npz)
        print(f"[demo] policy={npz} obs={policy.obs_dim} hidden={list(policy.hidden)} "
              f"actions={len(actions)} {enc}")
        if args.hier_switch:
            from touhou_rl_mcp.training.hierarchical import HierarchicalController

            controller = HierarchicalController(policy, enc, actions,
                                                replan_every=args.hier_replan_every)
            label += "+hier"

    cfg = Config.load()
    cfg.patch({"backend": "mock", "game": "th11", "step_frames": args.step_frames,
               "danmaku_script": args.danmaku_script,
               "danmaku_offset": args.danmaku_offset,
               "danmaku_pattern": args.danmaku_pattern,
               "visualize": {"scale": 1.0, "show_grid": True, "show_bullets": True, "show_hud": True}})
    env = TouhouEnv(cfg, registry=GameRegistry())
    env.start()
    if args.teacher == "rule":
        teacher = RuleTeacher()
        print("[demo] 使用规则老师 RuleTeacher (只躲, 不主动追 Boss)")
    elif args.teacher == "score":
        from touhou_rl_mcp.training.score_teacher import ScoreTeacher

        teacher = ScoreTeacher(weights=dict(cfg.section("reward")))
        print("[demo] 使用 ScoreTeacher (安全优先 + 按分数选位)")
    elif args.teacher == "viability":
        from touhou_rl_mcp.training.viability import ViabilityPlanner

        teacher = ViabilityPlanner(horizon=180, dt=2, aim_weight=0.15)
        print("[demo] 使用 ViabilityPlanner (时间展开可行域, 提前站位)")

    report = {"source": label, "step_frames": args.step_frames, "episodes": []}
    strips: list[Image.Image] = []
    for ep in range(1, args.episodes + 1):
        seed = args.seed + ep - 1
        res = env.reset(seed=seed, difficulty=args.difficulty)
        state = res.state
        frames: list[Image.Image] = []
        trace = []
        total = 0.0
        i = -1
        for i in range(args.max_steps):
            if teacher is not None:
                action, _ = teacher.choose(state, state.frame, "shoot")
            elif controller is not None:
                action = controller.act(state)
            else:
                obs = enc(state)
                action = actions[policy.greedy(obs)]
            res = env.step(action, frames=args.step_frames)
            state = res.state
            total += res.reward
            if i % args.every == 0:
                frames.append(env.capture_frame())
            if i % 100 == 0 or res.done or res.truncated:
                trace.append({"step": i, "stage": state.stage, "lives": state.lives,
                              "power": state.power, "score": state.score,
                              "bullets": state.bullet_count, "reason": res.info.get("done_reason")})
            if res.done or res.truncated:
                break
        ep_info = {"episode": ep, "seed": seed, "steps": i + 1, "return": round(total, 2),
                   "planner_steps": getattr(controller, "planner_steps", 0) if controller else 0,
                   "max_stage": state.stage, "lives": state.lives, "score": state.score,
                   "done_reason": res.info.get("done_reason"), "frames": len(frames), "trace": trace}
        report["episodes"].append(ep_info)
        switched = getattr(controller, "planner_steps", 0) if controller else 0
        extra = f" 交给规划器={switched} 步" if controller else ""
        print(f"[demo] 第 {ep} 局(seed={seed}): 步数={i + 1} 到达面数={state.stage} 残机={state.lives} "
              f"分数={state.score} 结束={res.info.get('done_reason')} 抓帧={len(frames)}{extra}")

        if frames:
            w = args.gif_width
            gif = [f.resize((w, max(1, round(f.height * w / f.width)))) for f in frames]
            gif_path = out / f"demo_ep{ep}.gif"
            gif[0].save(gif_path, save_all=True, append_images=gif[1:],
                        duration=int(1000 / args.gif_fps), loop=0)
            print(f"[demo] GIF: {gif_path}")
            n = 12
            picks = [frames[int(k * (len(frames) - 1) / max(1, n - 1))] for k in range(n)] if len(frames) > 1 else frames
            tw = 160
            thumbs = [f.resize((tw, max(1, round(f.height * tw / f.width)))) for f in picks]
            th, cols = thumbs[0].height, 4
            rows = (len(thumbs) + cols - 1) // cols
            strip = Image.new("RGB", (cols * tw, rows * th), (0, 0, 0))
            for k, t in enumerate(thumbs):
                strip.paste(t, ((k % cols) * tw, (k // cols) * th))
            sp = out / f"strip_ep{ep}.png"
            strip.save(sp)
            print(f"[demo] 关键帧拼图: {sp}")
            strips.append(strip)

    (out / "demo_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[demo] 报告: {out / 'demo_report.json'}")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
