"""更强的弹幕数组探测: 要求 x,y **同时**每帧变化且在场地范围, 再按密集度聚类."""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from touhou_rl_mcp.backends.memory import MemoryBackend  # noqa: E402
from touhou_rl_mcp.config import Config  # noqa: E402
from touhou_rl_mcp.games.registry import GameRegistry  # noqa: E402
from touhou_rl_mcp.inputs.actions import parse  # noqa: E402

X0, X1 = -120.0, 640.0
Y0, Y1 = -120.0, 640.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="th11")
    ap.add_argument("--difficulty", type=int, default=1)
    ap.add_argument("--play-seconds", type=float, default=14.0)
    args = ap.parse_args()

    cfg = Config.load()
    cfg.patch({"backend": "memory", "game": args.game})
    be = MemoryBackend(GameRegistry().get(args.game), cfg)
    be.start()
    be.reset(options={"restart_process": True, "difficulty": args.difficulty})
    time.sleep(3)
    for _ in range(80):
        s = be.read_state()
        if s.in_game and (s.power or 0) > 0.3:
            break
        be.apply(parse("noop", default_frames=20))

    t0 = time.time()
    k = 0
    while time.time() - t0 < args.play_seconds:
        k += 1
        be.apply(parse("shoot", default_frames=6))
        be.apply(parse("left" if k % 2 else "right", default_frames=6))
        if not be.read_state().in_game:
            be.reset(options={"difficulty": args.difficulty})
            time.sleep(2)

    mem = be.mem
    regions = [
        (b, sz) for b, sz, p, ty in mem.memory_regions()
        if sz <= 64 * 1024 * 1024 and not (ty & 0x1000000)
    ]

    def cap():
        out = {}
        for b, sz in regions:
            try:
                out[b] = np.frombuffer(mem.read_bytes(b, sz), dtype="<f4").copy()
            except Exception:
                pass
        return out

    A = cap()
    be.apply(parse("noop", default_frames=2))
    B = cap()

    moved: dict[int, tuple[float, float]] = {}
    for b in set(A) & set(B):
        if A[b].shape != B[b].shape or len(A[b]) < 2:
            continue
        d = (A[b] != B[b])
        xv, yv = B[b][:-1], B[b][1:]
        pair = d[:-1] & d[1:] & (xv >= X0) & (xv <= X1) & (yv >= Y0) & (yv <= Y1)
        for i in np.flatnonzero(pair).tolist():
            moved[b + 4 * int(i)] = (float(xv[i]), float(yv[i]))
    print(f"[bul] x,y 同时变化且在范围的 (x,y) 对: {len(moved)}")

    # 按"地址相邻"聚类
    arr = sorted(moved)
    clusters: list[list[int]] = []
    cur = [arr[0]] if arr else []
    for a in arr[1:]:
        if a - cur[-1] <= 0x400:
            cur.append(a)
        else:
            clusters.append(cur)
            cur = [a]
    if cur:
        clusters.append(cur)
    clusters.sort(key=len, reverse=True)
    print(f"[bul] 聚类 {len(clusters)} 个, 最大的几个:")
    for cl in clusters[:6]:
        diffs = [cl[i + 1] - cl[i] for i in range(len(cl) - 1)]
        common = Counter(diffs).most_common(3)
        print(f"    span 0x{cl[0]:08X}..0x{cl[-1]:08X} 元素={len(cl)} 常见步长={[(hex(s), c) for s, c in common]}")
        for a in cl[:6]:
            print(f"        0x{a:08X} x={moved[a][0]:.2f} y={moved[a][1]:.2f}")

    # 对最大聚类找模块级指针链
    if clusters:
        base = clusters[0][0]
        print(f"\n[bul] 对 0x{base:08X} 找模块级指针链 ...")
        sc = be.scanner()
        arrays = dict(sc._capture_arrays(include_images=True))
        chains = sc.find_pointer_chain(base, arrays=arrays, limit=20)
        for c in chains[:10]:
            print(f"    ptr=0x{c['pointer']:X} offsets={[hex(o) for o in c['offsets']]}")
    be.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
