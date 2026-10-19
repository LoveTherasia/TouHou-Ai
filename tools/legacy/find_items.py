"""找掉落物(道具): 特征 = x 不变、y 缓慢匀速增加(下落), 且都在场地范围."""
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


def main() -> int:
    cfg = Config.load()
    cfg.patch({"backend": "memory", "game": "th11"})
    be = MemoryBackend(GameRegistry().get("th11"), cfg)
    be.start()
    be.reset(options={"restart_process": True, "difficulty": 0})
    time.sleep(3)
    for _ in range(80):
        s = be.read_state()
        if s.in_game and (s.power or 0) > 0.3:
            break
        be.apply(parse("noop", default_frames=20))
    lock = LivesLock(be.mem, be.reader.address_of("lives"), value=9, floor=1).start()

    # 打一会儿敌人攒道具
    for k in range(60):
        be.apply(parse("shoot", default_frames=8))
        be.apply(parse("left" if k % 2 else "right", default_frames=6))
        if not be.read_state().in_game:
            break

    mem = be.mem
    regions = [(b, sz) for b, sz, p, ty in mem.memory_regions()
               if sz <= 64 * 1024 * 1024 and not (ty & 0x1000000)]

    def cap():
        out = {}
        for b, sz in regions:
            try:
                out[b] = np.frombuffer(mem.read_bytes(b, sz), dtype="<f4").copy()
            except Exception:
                pass
        return out

    A = cap()
    time.sleep(0.35)
    B = cap()
    print(f"[item] 快照完成, 区域 {len(A)}")

    hits: dict[int, tuple[float, float, float]] = {}
    for b in set(A) & set(B):
        if A[b].shape != B[b].shape or len(A[b]) < 2:
            continue
        dx = np.abs(B[b][:-1] - A[b][:-1])          # 第 i 槽当作 x
        dy = B[b][1:] - A[b][1:]                    # 第 i+1 槽当作 y
        ok = (
            (dx < 0.05)                              # x 不变
            & (dy > 0.4) & (dy < 2.5)                # y 缓慢下落(道具速度)
            & (A[b][:-1] >= 16) & (A[b][:-1] <= 368)  # x 在场地内(排除 0.x 的速度表)
            & (B[b][1:] >= 0) & (B[b][1:] <= 460)    # y 在场地
        )
        for i in np.flatnonzero(ok).tolist():
            hits[b + 4 * int(i)] = (float(B[b][i]), float(B[b][i + 1]), float(dy[i]))
    print(f"[item] 'x不变+y下落' 槽: {len(hits)}")
    if not hits:
        lock.stop(); be.close(); return 1

    arr = sorted(hits)
    clusters, cur = [], [arr[0]]
    for a in arr[1:]:
        if a - cur[-1] <= 0x1000:
            cur.append(a)
        else:
            clusters.append(cur); cur = [a]
    clusters.append(cur)
    clusters.sort(key=len, reverse=True)
    print(f"[item] 聚类 {len(clusters)} 个:")
    for cl in clusters[:5]:
        diffs = Counter(cl[i + 1] - cl[i] for i in range(len(cl) - 1))
        print(f"    span 0x{cl[0]:08X}..0x{cl[-1]:08X} 元素={len(cl)} 常见步长={[(hex(k), v) for k, v in diffs.most_common(3)]}")
        for a in cl[:6]:
            x, y, d = hits[a]
            print(f"        0x{a:08X} x={x:.1f} y={y:.1f} dy={d:.2f}")

    base = clusters[0][0]
    print(f"\n[item] 对 0x{base:08X} 找模块级指针链:")
    sc = be.scanner()
    arrays = dict(sc._capture_arrays(include_images=True))
    for c in sc.find_pointer_chain(base, arrays=arrays, limit=12)[:8]:
        print(f"    ptr=0x{c['pointer']:X} offsets={[hex(o) for o in c['offsets']]}")
    lock.stop(); be.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
