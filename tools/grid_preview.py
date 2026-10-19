"""把"危险图"观测画出来 —— 看清策略眼里的弹幕地图。

左: 游戏画面 (自机周围 ±span 的取样框)
右: 危险图 (16x16 栅格) 的密度热力图 + 平均速度箭头

    python tools/grid_preview.py --danmaku-script .runtime/danmaku/th11_replay_full.json \
        --offset 16200 --max-steps 900 --every 3 --out .runtime/grid_preview_s2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from touhou_rl_mcp.config import Config  # noqa: E402
from touhou_rl_mcp.env import TouhouEnv  # noqa: E402
from touhou_rl_mcp.games.registry import GameRegistry  # noqa: E402
from touhou_rl_mcp.training.features import ObservationEncoder  # noqa: E402
from touhou_rl_mcp.training.rule_teacher import RuleTeacher  # noqa: E402
from touhou_rl_mcp.types import PLAYFIELD_HEIGHT, PLAYFIELD_WIDTH  # noqa: E402

CELL = 12  # 每格在预览图里的像素大小


def render(state, enc: ObservationEncoder, frame: Image.Image) -> Image.Image:
    """把一帧画成 [游戏画面 | 危险图(各尺度)] 的对比图。"""
    # ---- 左: 游戏画面 + 各尺度取样框 ----
    left = frame.convert("RGB")
    scale_x = left.width / PLAYFIELD_WIDTH
    scale_y = left.height / PLAYFIELD_HEIGHT
    d = ImageDraw.Draw(left)
    px = float(state.player_x or PLAYFIELD_WIDTH / 2)
    py = float(state.player_y or PLAYFIELD_HEIGHT / 2)
    colors = [(0, 255, 128), (255, 190, 0), (120, 170, 255)]
    for k, (_, span) in enumerate(enc.scales):
        d.rectangle(
            [(px - span) * scale_x, (py - span) * scale_y,
             (px + span) * scale_x, (py + span) * scale_y],
            outline=colors[k % len(colors)], width=1,
        )

    # ---- 右: 各尺度危险图 ----
    vec = enc(state)
    ch = 3 if enc.bullet_velocity else 1
    panels = []
    off = 0
    for size, span in enc.scales:
        grid = vec[off : off + size * size * ch].reshape(size, size, ch)
        off += size * size * ch
        panel = Image.new("RGB", (size * CELL, size * CELL), (8, 8, 14))
        pd = ImageDraw.Draw(panel)
        for gy in range(size):
            for gx in range(size):
                dens = float(grid[gy, gx, 0])
                if dens <= 0.0:
                    continue
                v = int(255 * min(1.0, dens))
                pd.rectangle([gx * CELL, gy * CELL, gx * CELL + CELL - 1, gy * CELL + CELL - 1],
                             fill=(v, int(v * 0.35), 40))
        if ch >= 3:
            for gy in range(size):
                for gx in range(size):
                    if grid[gy, gx, 0] <= 0.0:
                        continue
                    vx = float(grid[gy, gx, 1])
                    vy = float(grid[gy, gx, 2])
                    cx = gx * CELL + CELL / 2
                    cy = gy * CELL + CELL / 2
                    pd.line([cx, cy, cx + vx * CELL * 3, cy + vy * CELL * 3],
                            fill=(120, 200, 255), width=1)
        c = size * CELL / 2
        pd.ellipse([c - 3, c - 3, c + 3, c + 3], fill=(255, 255, 255))
        pd.text((4, 4), f"{size}x{size}  +/-{span:.0f}px", fill=(200, 200, 200))
        panels.append(panel)

    right_w = sum(p.width for p in panels) + 6 * max(0, len(panels) - 1)
    right_h = max(p.height for p in panels)
    right = Image.new("RGB", (right_w, right_h), (0, 0, 0))
    x = 0
    for p in panels:
        right.paste(p, (x, 0))
        x += p.width + 6

    h = max(left.height, right_h)
    out = Image.new("RGB", (left.width + 6 + right_w, h), (0, 0, 0))
    out.paste(left, (0, 0))
    out.paste(right, (left.width + 6, 0))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--danmaku-script", default="")
    ap.add_argument("--danmaku-pattern", default="")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--step-frames", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--grid-size", type=int, default=16)
    ap.add_argument("--grid-span", type=float, default=110.0)
    ap.add_argument("--grid-scales", default="",
                    help="多尺度 'size:span,...' 如 '12:90,12:330'")
    ap.add_argument("--max-steps", type=int, default=900)
    ap.add_argument("--every", type=int, default=3)
    ap.add_argument("--teacher", default="rule", choices=["rule", "none"])
    ap.add_argument("--out", default=".runtime/grid_preview")
    args = ap.parse_args()

    cfg = Config.load()
    cfg.patch({"backend": "mock", "game": "th11", "step_frames": args.step_frames,
               "danmaku_script": args.danmaku_script,
               "danmaku_offset": args.offset,
               "danmaku_pattern": args.danmaku_pattern,
               "visualize": {"scale": 1.0, "show_grid": False,
                             "show_bullets": True, "show_hud": True}})
    env = TouhouEnv(cfg, registry=GameRegistry())
    env.start()
    env.reset(seed=args.seed)
    teacher = RuleTeacher() if args.teacher == "rule" else None
    enc = ObservationEncoder.from_meta({"use_grid": True, "grid_size": args.grid_size,
                                        "grid_span": args.grid_span,
                                        "grid_scales": args.grid_scales,
                                        "bullet_velocity": True, "use_invincible": True,
                                        "use_boss": True, "use_danger_grad": True})
    print(f"[grid] {enc}")

    frames: list[Image.Image] = []
    state = env.get_state()
    for i in range(args.max_steps):
        if teacher is not None:
            action, _ = teacher.choose(state, state.frame, "shoot")
        else:
            action = "noop"
        if i % args.every == 0:
            frames.append(render(state, enc, env.capture_frame()))
        res = env.step(action, frames=args.step_frames)
        state = res.state
        if res.done or res.truncated:
            break
    env.close()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    strip = frames[:8]
    sheet = Image.new("RGB", (strip[0].width, strip[0].height * len(strip)), (0, 0, 0))
    for k, img in enumerate(strip):
        sheet.paste(img, (0, k * img.height))
    sheet.save(out / "grid_strip.png")
    if frames:
        frames[0].save(out / "grid_demo.gif", save_all=True,
                       append_images=frames[1:], duration=60, loop=0)
    (out / "grid_report.json").write_text(json.dumps(
        {"frames": len(frames), "grid_size": args.grid_size, "grid_span": args.grid_span,
         "dim": enc.dim}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[grid] 拼图: {out / 'grid_strip.png'}")
    print(f"[grid] GIF : {out / 'grid_demo.gif'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
