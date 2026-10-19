"""找弹幕链表: 利用"节点的 next 指针指向另一个 (x,y) 每帧都在变的节点"这一信号.

步骤:
 1. 攒弹幕 -> 间隔 2 帧抓两次内存
 2. 收集 "x,y 同时变化且都在场地范围" 的地址集合 M (视为可能的节点, 假设 x 在 +0)
 3. 对每个偏移 k, 统计有多少 a∈M 满足 read_u32(a+k) ∈ M (即 next 指向另一个节点)
    -> 某个 k 会明显胜出 = next 字段偏移 = 找到链表
 4. 沿链走一遍, 校验节点 x,y; 找链表头; 再用 find_pointer_chain 固化
"""
from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np  # noqa: E402

from touhou_rl_mcp.backends.memory import MemoryBackend  # noqa: E402
from touhou_rl_mcp.config import Config  # noqa: E402
from touhou_rl_mcp.games.registry import GameRegistry  # noqa: E402
from touhou_rl_mcp.inputs.actions import parse  # noqa: E402
from touhou_rl_mcp.memory.cheats import LivesLock  # noqa: E402

X0, X1 = -60.0, 520.0
Y0, Y1 = -60.0, 560.0


def main() -> int:
    cfg = Config.load()
    cfg.patch({"backend": "memory", "game": "th11"})
    be = MemoryBackend(GameRegistry().get("th11"), cfg)
    be.start()
    be.reset(options={"restart_process": True, "difficulty": 1})   # Normal, 弹多
    time.sleep(3)
    for _ in range(80):
        s = be.read_state()
        if s.in_game and (s.power or 0) > 0.3:
            break
        be.apply(parse("noop", default_frames=20))
    lock = LivesLock(be.mem, be.reader.address_of("lives"), value=9, floor=1).start()

    t0 = time.time()
    k = 0
    while time.time() - t0 < 16:
        k += 1
        be.apply(parse("shoot", default_frames=8))
        be.apply(parse("left" if k % 2 else "right", default_frames=8))
        if not be.read_state().in_game:
            be.reset(options={"difficulty": 1})
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

    M: set[int] = set()
    for b in set(A) & set(B):
        if A[b].shape != B[b].shape or len(A[b]) < 2:
            continue
        d = A[b] != B[b]
        xv, yv = B[b][:-1], B[b][1:]
        pair = d[:-1] & d[1:] & (xv >= X0) & (xv <= X1) & (yv >= Y0) & (yv <= Y1)
        for i in np.flatnonzero(pair).tolist():
            M.add(b + 4 * int(i))
    print(f"[bl] 变化且成对在范围的节点候选 M = {len(M)}")
    Mlist = sorted(M)
    Mset = M

    def rd32(a: int) -> int | None:
        try:
            return int(mem.read_value(a, "u32"))
        except Exception:
            return None

    # 统计每个偏移 k 上, next 指向另一个 M 节点
    print("[bl] 偏移 k -> 命中数 (next 指向另一个候选节点):")
    ranked = []
    for koff in range(-0x40, 0x400, 4):
        cnt = 0
        for a in Mlist:
            v = rd32(a + koff)
            if v is not None and v in Mset and v != a:
                cnt += 1
        ranked.append((cnt, koff))
    for cnt, koff in sorted(ranked, reverse=True)[:10]:
        print(f"    k=0x{koff:X} : {cnt}")

    best_cnt, best_k = max(((c, kk) for c, kk in ranked), key=lambda z: z[0])
    print(f"[bl] 最佳 next 偏移 k=0x{best_k:X} (命中 {best_cnt})")
    if best_cnt < 5:
        print("[bl] 命中太少, 假设不成立 (可能 x 不在 +0, 或不是链表)")
        lock.stop()
        be.close()
        return 1

    # 沿链走
    targets = set()
    for a in Mlist:
        v = rd32(a + best_k)
        if v in Mset:
            targets.add(v)
    heads = [a for a in Mlist if a not in targets]
    print(f"[bl] 疑似链表头 {len(heads)} 个, 试走第一个 ...")
    if heads:
        node = heads[0]
        chain = []
        seen = set()
        while node and node in Mset and node not in seen and len(chain) < 200:
            seen.add(node)
            x = float(mem.read_value(node, "f32"))
            y = float(mem.read_value(node + 4, "f32"))
            chain.append((hex(node), round(x, 2), round(y, 2)))
            node = rd32(node + best_k)
        print(f"[bl] 链长 {len(chain)}: {chain[:12]}")
        # 找指向链表头的模块级指针链
        sc = be.scanner()
        arrays = dict(sc._capture_arrays(include_images=True))
        print(f"[bl] 对链表头 0x{heads[0]:08X} 找模块级指针链:")
        for c in sc.find_pointer_chain(heads[0], arrays=arrays, limit=10)[:8]:
            print(f"    ptr=0x{c['pointer']:X} offsets={[hex(o) for o in c['offsets']]}")

    lock.stop()
    be.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
