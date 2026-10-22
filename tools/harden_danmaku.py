"""把一份弹幕脚本"加难": 加密、加速、缩短寿命, 用于高难度课程训练.

为什么不直接改 mock 的内置弹型: 内置弹型是"快而稀"的手写图案, 和真实回放的
"慢而密"分布差得远。这里的做法是**基于真实回放加难**, 保证弹道形状仍然真实
(都是直射/散射), 只把密度和速度往上推。

加难三件事 (都可单独关闭):

1. ``--density``: 每条弹额外复制 (density-1) 份, 速度方向按 ``--angle`` 旋转,
   位置按 ``--jitter`` 抖开, 生成帧按 ``--stagger`` 错开 —— 相当于把原来的
   单发变成 3 连发/5 连发;
2. ``--speed``: 速度整体乘一个系数 (1.0 不变);
3. ``--life``: 覆盖寿命上限 (帧), 保证加难后同屏弹数不会爆掉。

加密是**自适应**的: 先按 ``--max-onscreen`` 估算该时刻同屏存活弹数, 只在还有
余量时才补复制份。否则原始已经很密的段 (比如 F=15000 的符卡, 原脚本 480 颗)
直接翻倍会顶到 mock 的 ``MAX_BULLETS=900`` 上限并被截断 —— 那等于又造了一份
"失真环境"。

    python tools/harden_danmaku.py .runtime/danmaku/th11_replay_v2.json \\
        -o .runtime/danmaku/th11_replay_hard.json \\
        --density 2 --angle 9 --jitter 6 --stagger 6 --speed 1.2 --life 180
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--density", type=int, default=2,
                    help="每条弹复制成几份 (1=不加密); 2~3 属于\"可躲但明显更难\"")
    ap.add_argument("--angle", type=float, default=9.0,
                    help="复制份之间的速度方向夹角 (度, 总张角)")
    ap.add_argument("--jitter", type=float, default=6.0,
                    help="复制份的初始位置抖动 (像素, 单边)")
    ap.add_argument("--stagger", type=int, default=6,
                    help="复制份的生成帧最多错开多少帧 (形成连发)")
    ap.add_argument("--speed", type=float, default=1.2, help="速度倍数")
    ap.add_argument("--life", type=int, default=240,
                    help="寿命上限 (帧); 0 = 保留原脚本寿命")
    ap.add_argument("--max-onscreen", type=int, default=700,
                    help="自适应加密的上限: 估算同屏存活弹数超过它就不再补复制份 "
                         "(mock 的 MAX_BULLETS=900, 留 200 余量)")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    data = json.loads(Path(args.src).read_text(encoding="utf-8"))
    spawns = data["spawns"]
    rows_in = [r + [0] * (6 - len(r)) for r in spawns]     # 统一成 6 列

    angle_span = math.radians(args.angle)
    out: list[list[float]] = []

    def exit_frames(x: float, y: float, vx: float, vy: float, life: int) -> int:
        """这颗弹预计还能活多久 (帧): 有寿命就用寿命, 否则按飞出场地估算。"""
        if life > 0:
            return int(life)
        t = 600.0
        if vy > 0.05:
            t = min(t, max(1.0, (448.0 - y) / vy))
        if vx > 0.05:
            t = min(t, max(1.0, (384.0 - x) / vx))
        elif vx < -0.05:
            t = min(t, max(1.0, x / -vx))
        return int(min(600.0, t))

    # 贪心地保证"任一时刻同屏弹数 <= max_onscreen": 按帧序遍历, 维护一个
    # 死亡帧的最小堆; 只有剩余容量时才把这一条弹复制成多份。
    import heapq
    heap: list[int] = []
    rows_in.sort(key=lambda r: r[0])
    full_copies = 0
    for f, x, y, vx, vy, life in rows_in:
        while heap and heap[0] <= f:
            heapq.heappop(heap)
        want = max(1, int(args.density))
        copies = min(want, max(1, args.max_onscreen - len(heap)))
        if copies >= want:
            full_copies += 1
        for i in range(copies):
            # 复制份 i 相对原弹的角度偏移: copies==1 时不旋转
            frac = 0.0 if copies == 1 else (i / (copies - 1) - 0.5) * angle_span
            ca, sa = math.cos(frac), math.sin(frac)
            nvx = (vx * ca - vy * sa) * args.speed
            nvy = (vx * sa + vy * ca) * args.speed
            nx = x + (rng.uniform(-args.jitter, args.jitter) if copies > 1 else 0.0)
            ny = y + (rng.uniform(-args.jitter, args.jitter) if copies > 1 else 0.0)
            nf = int(f) + (rng.randint(0, args.stagger) if copies > 1 else 0)
            nl = int(life) or 0
            if args.life > 0:
                nl = args.life if nl <= 0 else min(nl, args.life)
            out.append([nf, round(nx, 2), round(ny, 2),
                        round(nvx, 4), round(nvy, 4), nl])
            heapq.heappush(heap, nf + exit_frames(nx, ny, nvx, nvy, nl))

    out.sort(key=lambda r: r[0])
    meta = {
        "source": str(Path(args.src)),
        "frames": data.get("frames", 0),
        "count": len(out),
        "dropped": data.get("dropped", 0),
        "hard": {
            "density": args.density, "angle": args.angle, "jitter": args.jitter,
            "stagger": args.stagger, "speed": args.speed, "life": args.life,
            "seed": args.seed,
        },
        "spawns": out,
    }
    Path(args.out).write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    # ---- 统计: 同屏弹数 (含寿命) 与速度分布 ----
    def onscreen(fr: int) -> int:
        n = 0
        for f, x, y, vx, vy, lf in out:
            if f > fr:
                continue
            dt = fr - f
            if lf > 0 and dt > lf:
                continue
            px, py = x + vx * dt, y + vy * dt
            if 0 <= px <= 384 and 0 <= py <= 448:
                n += 1
        return n

    probes = [5000, 10000, 15000, 20000, 40000, 60000, 80000]
    counts = {p: onscreen(p) for p in probes}
    speeds = sorted(math.hypot(r[3], r[4]) for r in out)
    print(f"[hard] {len(spawns)} -> {len(out)} 条弹 (density={args.density}, "
          f"speed={args.speed}, angle={args.angle}°, life<={args.life})")
    print(f"[hard] 自适应加密: 拿到满 density 的原弹 {full_copies}/{len(rows_in)} "
          f"(上限 {args.max_onscreen} 颗同屏)")
    print("[hard] 同屏弹数: " + ", ".join(f"F={k}:{v}" for k, v in counts.items()))
    print(f"[hard] 弹速 中位={speeds[len(speeds) // 2]:.2f} "
          f"p95={speeds[int(len(speeds) * .95)]:.2f} 最大={speeds[-1]:.2f} px/帧")
    print(f"[hard] 写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
