"""把训练好的缺口策略放回 v3 场景里跑, 输出 GIF + 横向轨迹对比图。

GIF 直接回答"它到底在干什么": 是站着不动、是贴着缺口走、还是每次都等到最后一刻
才横移。轨迹图把这件事量化 —— 蓝线 = 自机 x, 橙线 = 当前这堵墙的缺口 x,
两者之间的垂直距离就是"还没挪到位"的量。

用法::

    python tools/research/gap_gif.py --run .runtime/runs/v3-anchor1 --scene 0 --phase 0
    python tools/research/gap_gif.py --run .runtime/runs/v3-anchor1 --all --out .runtime/gifs
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gap_diagnose import gap_x_at, load_run, make_env  # noqa: E402
from gap_rl import BENCHMARKS  # noqa: E402
from gap_trainer import apply_script_offset, build_env as build_script_env  # noqa: E402

from touhou_rl_mcp.capture.visualize import _font  # noqa: E402
from touhou_rl_mcp.training.features import ObservationEncoder  # noqa: E402
from touhou_rl_mcp.training.features import gap_spec_from  # noqa: E402
from touhou_rl_mcp.training.gap_label import COMMIT_FRAMES, update_action_age  # noqa: E402
from touhou_rl_mcp.training.occupancy import (  # noqa: E402
    PLAYER_SPEED, _bullets, _column_centers, _tta_profile,
)
from touhou_rl_mcp.training.trainer import build_action_table, apply_scenario  # noqa: E402
from touhou_rl_mcp.types import PLAYFIELD_WIDTH  # noqa: E402

TRACE_W, TRACE_H = 760, 240


def draw_trace(rows: list[dict[str, Any]], *, interval: float, title: str) -> Image.Image:
    """自机 x vs 当前墙缺口 x。竖直虚线 = 每堵墙生成。"""
    img = Image.new("RGB", (TRACE_W, TRACE_H), (22, 22, 26))
    d = ImageDraw.Draw(img)
    left, right, top, bottom = 46, TRACE_W - 12, 26, TRACE_H - 26
    fmax = max((r["frame"] for r in rows), default=1) or 1
    if interval > 0:                            # 合成基准: 每堵墙画一条竖线
        for f in range(0, int(fmax) + 1, int(interval)):
            x = left + (right - left) * f / fmax
            d.line([(x, top), (x, bottom)], fill=(48, 48, 56))
    d.rectangle([left, top, right, bottom], outline=(80, 80, 90))
    font = _font(12)
    d.text((left, 8), title, fill=(210, 210, 220), font=font)
    d.text((4, bottom - 12), "0", fill=(130, 130, 140), font=font)
    d.text((0, top + 2), f"{int(PLAYFIELD_WIDTH)}", fill=(130, 130, 140), font=font)

    def pts(key: str) -> list[tuple[float, float]]:
        out = []
        for r in rows:
            x = left + (right - left) * r["frame"] / fmax
            y = bottom - (bottom - top) * min(1.0, max(0.0, r[key] / PLAYFIELD_WIDTH))
            out.append((x, y))
        return out

    gap = pts("gap")
    if len(gap) > 1:
        d.line(gap, fill=(240, 160, 70), width=2)
    player = pts("px")
    if len(player) > 1:
        d.line(player, fill=(90, 170, 255), width=2)
    d.text((left, TRACE_H - 18),
           ("蓝=自机 x    橙=当前墙缺口 x    竖线=每堵墙生成" if interval > 0
            else "蓝=自机 x    橙=最近的安全车道"),
           fill=(160, 160, 170), font=font)
    return img


def main() -> int:
    ap = argparse.ArgumentParser(description="缺口策略的 GIF / 轨迹")
    ap.add_argument("--run", required=True)
    ap.add_argument("--scene", type=int, default=0)
    ap.add_argument("--phase", type=int, default=0)
    ap.add_argument("--phases", type=int, default=3, help="相位偏移总数 (与评估一致)")
    ap.add_argument("--all", action="store_true", help="跑全部 场景×相位 (或全部留出片段)")
    ap.add_argument("--max-frames", type=int, default=2400)
    ap.add_argument("--fracs", default="",
                    help="脚本模式的评估位置 (逗号分隔); 默认优先用 run 的 test_fracs")
    ap.add_argument("--stride", type=int, default=2, help="每 N 个宏动作抓一帧")
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--width", type=int, default=360)
    ap.add_argument("--out", default="")
    ap.add_argument("--last", action="store_true",
                    help="用最后一版权重 (默认优先用按评估择优的 policy_best)")
    args = ap.parse_args()

    run_dir = Path(args.run)
    weights = ("policy" if args.last or not (run_dir / "policy_best.npz").exists()
               else "policy_best")
    policy, meta = load_run(run_dir, weights)
    print(f"[gif] 权重: {weights}")
    cfg = meta["config"]
    enc = ObservationEncoder.from_meta(cfg)
    actions = build_action_table(use_focus=bool(cfg.get("use_focus", False)),
                                 use_bomb=bool(cfg.get("use_bomb", False)))
    run_meta = {}
    if (run_dir / "meta.json").exists():
        run_meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    bench = str(cfg.get("bench") or run_meta.get("bench") or "v3")
    spec = gap_spec_from(enc)
    #: 脚本路径在 run 的 meta.json 里 (TrainConfig 不存路径, 只存 offset 区间)
    script = str(cfg.get("danmaku_script") or run_meta.get("danmaku_script") or "")
    out_dir = Path(args.out) if args.out else run_dir / "gifs"
    out_dir.mkdir(parents=True, exist_ok=True)

    if script:                                  # 真机弹幕脚本模式
        env = build_script_env({}, {}, script=script)
        if args.fracs:
            fracs = [float(x) for x in args.fracs.split(",") if x.strip()]
        else:
            # P0 之后新 run 分开写 dev/test; 默认看 test (没参与选权重)
            fracs = [float(x) for x in (run_meta.get("test_fracs")
                                        or run_meta.get("eval_fracs")
                                        or [0.80, 0.87, 0.94])]
        jobs = ([(i, f) for i, f in enumerate(fracs)] if args.all
                else [(0, fracs[min(args.scene, len(fracs) - 1)])])
    else:                                       # wall_gate 合成基准
        env = make_env(meta)
        _, heldout = BENCHMARKS[bench]
        jobs = ([(s, k) for s in range(len(heldout)) for k in range(args.phases)]
                if args.all else [(args.scene, args.phase)])

    def target_x(state: Any) -> float:
        """橙色线: 合成基准上是"这堵墙的缺口", 脚本模式上是"最近的安全车道"。"""
        if not script:
            return gap_x_at(sc, int(state.frame // max(1.0, interval)))
        pos, vel = _bullets(state)
        tta = _tta_profile(float(state.player_y), pos, vel, spec) * spec.tta_horizon
        centers = _column_centers(spec)
        safe = np.flatnonzero(tta > 60.0)       # 一帧以上余量才算"安全"
        if not safe.size:
            return float(state.player_x)
        return float(centers[safe[np.argmin(np.abs(centers[safe] - state.player_x))]])

    for si, k in jobs:
        if script:
            apply_script_offset(env, float(k))
            st = env.reset(seed=101).state
            interval = 0.0
        else:
            sc = dict(heldout[si])
            sc["gap_phase"] = float(sc.get("gap_phase", 0.0)) + k / max(1, args.phases)
            st = env.reset(seed=101).state
            apply_scenario(env, sc)
            interval = float(sc.get("interval", 75.0))
        frames: list[Image.Image] = []
        rows: list[dict[str, Any]] = []
        prev = -1
        age = 0
        cap = int(cfg.get("gap_action_age_cap", 12) or 12)
        total = 0
        for i in range(args.max_frames // COMMIT_FRAMES):
            rows.append({"frame": int(st.frame), "px": float(st.player_x),
                         "gap": target_x(st)})
            if i % args.stride == 0:
                frames.append(env.capture_frame())
            obs = enc(st, prev_action=prev, action_age=age)
            a = int(policy.greedy(obs))
            res = env.step(actions[a], frames=COMMIT_FRAMES, auto_reset=False)
            st = res.state
            age = update_action_age(prev, age, a, cap)
            prev = a
            total += COMMIT_FRAMES
            if res.done:
                break
        tag = (f"{run_dir.name}_frac{k:.2f}" if script
               else f"{run_dir.name}_scene{si}_phase{k}")
        if frames:
            w = args.width
            scaled = [f.resize((w, max(1, round(f.height * w / f.width)))) for f in frames]
            gif_path = out_dir / f"{tag}.gif"
            scaled[0].save(gif_path, save_all=True, append_images=scaled[1:],
                           duration=int(1000 / args.fps), loop=0, optimize=True)
            print(f"[gif] {tag}: {total} 帧 / {len(frames)} 张 -> {gif_path}")
        trace = draw_trace(rows, interval=interval,
                           title=f"{tag}   存活 {total} 帧")
        png = out_dir / f"{tag}_trace.png"
        trace.save(png)
        print(f"[gif] 轨迹: {png}")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
