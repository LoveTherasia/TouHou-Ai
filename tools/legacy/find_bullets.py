"""探测 th11 弹幕数组: 步长(stride) / 元素字段(x,y) / 基址 / 数量.

原理: 弹幕每帧 x,y 都在变。间隔 1 帧抓两次内存, 找出"变了且新值落在场地范围"的 f32 槽,
      再按固定步长聚类 —— 弹幕是**结构体数组**, 会出现一个明显的公共步长。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np  # noqa: E402

from touhou_rl_mcp.backends.memory import MemoryBackend  # noqa: E402
from touhou_rl_mcp.config import Config  # noqa: E402
from touhou_rl_mcp.games.registry import GameRegistry  # noqa: E402
from touhou_rl_mcp.inputs.actions import parse  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="th11")
    ap.add_argument("--difficulty", type=int, default=0)
    ap.add_argument("--play-seconds", type=float, default=12.0)
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
    print(f"[bul] in_game={s.in_game} stage={s.stage} lives={s.lives} player=({s.player_x},{s.player_y})")

    # 玩一会, 攒弹幕
    t0 = time.time()
    k = 0
    while time.time() - t0 < args.play_seconds:
        k += 1
        be.apply(parse("shoot", default_frames=6))
        be.apply(parse(("left" if k % 2 else "right"), default_frames=6))
        if not be.read_state().in_game:
            break

    mem = be.mem
    regions = [
        (b, sz) for b, sz, p, ty in mem.memory_regions()
        if sz <= 64 * 1024 * 1024 and not (ty & 0x1000000)
    ]

    def cap_f32():
        out = {}
        for b, sz in regions:
            try:
                out[b] = np.frombuffer(mem.read_bytes(b, sz), dtype="<f4").copy()
            except Exception:
                pass
        return out

    def cap_u32():
        out = {}
        for b, sz in regions:
            try:
                out[b] = np.frombuffer(mem.read_bytes(b, sz), dtype="<u4").copy()
            except Exception:
                pass
        return out

    A = cap_f32()
    be.apply(parse("noop", default_frames=2))
    B = cap_f32()
    print(f"[bul] 抓取完成, 区域 {len(A)}")

    # 变化的槽 + 新值在场地范围
    changed: set[int] = set()
    for b in set(A) & set(B):
        if A[b].shape != B[b].shape:
            continue
        d = A[b] != B[b]
        v = B[b]
        m = d & (v >= -80.0) & (v <= 560.0)
        for i in np.flatnonzero(m).tolist():
            changed.add(b + 4 * int(i))
    print(f"[bul] 变化且落在范围的 f32 槽: {len(changed)}")

    # 候选步长
    strides = [0x1A0, 0x1B0, 0x1C0, 0x180, 0x150, 0x100, 0x80, 0x60, 0x50, 0x40, 0x38, 0x30, 0x28, 0x24, 0x20, 0x1C, 0x18, 0x14, 0x10]
    print("[bul] 步长 -> (a 和 a+stride 都在变化集里的数量):")
    ranked = []
    for st in strides:
        cnt = sum(1 for a in changed if (a + st) in changed)
        ranked.append((cnt, st))
    for cnt, st in sorted(ranked, reverse=True)[:8]:
        print(f"    0x{st:X} ({st}) : {cnt}")

    # 取最佳步长, 找连续的链
    best_st, best_cnt = max(((st, c) for c, st in ranked), key=lambda z: z[1])
    print(f"\n[bul] 最佳步长 0x{best_st:X} (匹配 {best_cnt})")
    arr = sorted(changed)
    aset = set(arr)
    used: set[int] = set()
    chains = []
    for a in arr:
        if a in used:
            continue
        chain = [a]
        x = a
        while (x + best_st) in aset:
            x += best_st
            chain.append(x)
        if len(chain) >= 4:
            for c in chain:
                used.add(c)
            chains.append(chain)
    chains.sort(key=len, reverse=True)
    print(f"[bul] 长度>=4 的链 {len(chains)} 条")
    for chain in chains[:6]:
        print(f"    base=0x{chain[0]:08X} 元素={len(chain)} 覆盖到 0x{chain[-1]:08X}")
        for a in chain[:5]:
            xv, yv = None, None
            for b in B:
                if b <= a < b + 4 * len(B[b]):
                    i = (a - b) // 4
                    xv = float(B[b][i])
                    yv = float(B[b][i + 1]) if i + 1 < len(B[b]) else None
                    break
            print(f"       0x{a:08X} x={xv} y+4={yv}")

    # 也看一下弹幕"数量"可能在哪: 在链 base 附近找小整数
    if chains:
        base0 = chains[0][0]
        print(f"\n[bul] 链 base 0x{base0:08X} 前 64 字节的 i32:")
        try:
            raw = mem.read_bytes(base0 - 64, 64)
            vals = np.frombuffer(raw, dtype="<i4")
            print("   ", [int(v) for v in vals])
        except Exception as exc:
            print("    读失败", exc)
    be.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
