"""在已知管理器里暴力找"另一条"带场地坐标的对象链 (可能是掉落物)."""
from __future__ import annotations

import struct
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

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
    for k in range(70):
        be.apply(parse("shoot", default_frames=8))
        be.apply(parse("left" if k % 2 else "right", default_frames=6))
        if not be.read_state().in_game:
            break
    print("state:", be.read_state().raw["entities"])

    mem = be.mem
    base = mem.main_module[1]

    def u32(a):
        try:
            return int(mem.read_value(a, "u32"))
        except Exception:
            return 0

    def f32(a):
        try:
            return float(mem.read_value(a, "f32"))
        except Exception:
            return None

    managers = {"bullets": u32(base + 0xA8D68), "enemies": u32(base + 0xA8D7C)}
    print("managers:", {k: hex(v) for k, v in managers.items()})

    results = []
    for mname, mgr in managers.items():
        if not mgr:
            continue
        for head_off in range(0, 0x100, 4):
            head = u32(mgr + head_off)
            if not (0x01000000 <= head <= 0x7FFFFFFF):
                continue
            # 走链: node -> [obj, next]; 试 obj 里几个可能的位置偏移
            for pos_off in (0x1070, 0x43C, 0x40, 0x44, 0x50, 0x64, 0x18):
                node = head
                seen = set()
                pts = []
                ok = True
                while node and len(seen) < 64:
                    if node in seen or not (0x01000000 <= node <= 0x7FFFFFFF):
                        ok = False
                        break
                    seen.add(node)
                    obj = u32(node)
                    nxt = u32(node + 4)
                    if not obj:
                        ok = False
                        break
                    x = f32(obj + pos_off)
                    y = f32(obj + pos_off + 4)
                    if x is None or y is None or not (-40 <= x <= 460 and -40 <= y <= 500):
                        ok = False
                        break
                    pts.append((round(x, 1), round(y, 1)))
                    node = nxt
                if ok and len(pts) >= 2:
                    results.append((mname, head_off, pos_off, len(pts), pts[:6]))
    print(f"候选链: {len(results)}")
    for mname, ho, po, n, sample in results[:20]:
        print(f"  [{mname}] head+0x{ho:X} pos+0x{po:X} 节点={n} 样例={sample}")
    lock.stop()
    be.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
