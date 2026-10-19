"""自动定位东方正作的**残机 (lives)** 偏移 —— 通用工作流的第一步.

思路:
  1. 残机是模块映像里的全局 i32, 平时不变, 只在**掉残时 -1**;
  2. 先在模块映像里筛"数值在 [1,9] 且等待若干秒仍不变"的地址 (候选很少);
  3. 让自机主动撞弹掉残, 记录每个候选的 "每次 -1" 事件; 掉残的那个会得到 2,1,0 序列;
  4. 把模块相对偏移写回 offsets json (跨重启稳定)。

    python tools/find_lives.py --game th11
    python tools/find_lives.py --game th11 --write          # 写回 offsets
    python tools/find_lives.py --game th11 --write --lock-seconds 30   # 并演示锁血
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from touhou_rl_mcp.config import Config  # noqa: E402
from touhou_rl_mcp.env import TouhouEnv  # noqa: E402
from touhou_rl_mcp.games.calibration import apply_calibration  # noqa: E402
from touhou_rl_mcp.games.registry import GameRegistry  # noqa: E402
from touhou_rl_mcp.inputs.actions import parse  # noqa: E402
from touhou_rl_mcp.memory.cheats import LivesLock  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="th11")
    ap.add_argument("--difficulty", type=int, default=0)
    ap.add_argument("--settle", type=float, default=4.0)
    ap.add_argument("--seconds", type=float, default=45.0, help="撞弹观察时长")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--lock-seconds", type=float, default=0.0, help="找到后锁血多少秒 (演示)")
    args = ap.parse_args()

    cfg = Config.load()
    cfg.patch({"backend": "memory", "game": args.game})
    env = TouhouEnv(cfg, registry=GameRegistry())
    env.start()
    backend = env.backend

    def press(buttons, frames):
        backend.apply(parse("+".join(buttons) if buttons else "noop", default_frames=int(frames)))

    st = env.reset(restart_process=True, difficulty=args.difficulty)
    print(f"[lives] 新一局: in_game={st.state.in_game} stage={st.state.stage} lives={st.state.lives}")
    scanner = backend.scanner()  # 必须在 reset(restart_process) 之后取 (否则句柄失效)

    cands = scanner.find_lives(settle=args.settle, log=lambda m: print("[lives]", m))
    if not cands:
        print("[lives] 没有候选, 退出")
        env.close()
        return 1
    print(f"[lives] 候选 {len(cands)} 个, 开始撞弹观察 {args.seconds:.0f}s ...")
    events = scanner.watch_decrements(
        cands, press=press, seconds=args.seconds, log=lambda m: print("[lives]", m)
    )
    if not events:
        print("[lives] 没观察到 -1 事件 (自机可能一直没掉残)")
        env.close()
        return 1

    print("[lives] 掉残事件:")
    ranked = sorted(events.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    for addr, seq in ranked[:15]:
        cand = next((c for c in cands if c.address == addr), None)
        off = f"{cand.module}+0x{cand.module_offset:X}" if cand and cand.module else "?"
        print(f"   0x{addr:08X} ({off})  -1 序列={seq}")

    # 选"严格 -1 递减"的那个 (掉残就是 2->1->0->-1)
    best = None
    for addr, seq in ranked:
        if len(seq) >= 2 and all(v <= 9 for v in seq) and all(seq[i + 1] == seq[i] - 1 for i in range(len(seq) - 1)):
            best = (addr, seq)
            break
    if best is None:  # 退而求其次: 只观察到一次
        for addr, seq in ranked:
            if all(v <= 9 for v in seq):
                best = (addr, seq)
                break
    if best is None:
        print("[lives] 无法确定 (事件值超范围)")
        env.close()
        return 1

    addr, seq = best
    cand = next(c for c in cands if c.address == addr)
    print(f"[lives] 判定 lives @ 0x{addr:08X} = {cand.module}+0x{cand.module_offset:X} (序列 {seq})")

    if args.write:
        result = apply_calibration(
            args.game,
            {"lives": {"offset": f"0x{cand.module_offset:X}", "type": "i32", "label": "残机(自动标定)"}},
        )
        print(f"[lives] 已写入 {result['path']}")

    if args.lock_seconds > 0:
        print(f"[lives] 锁血演示: 把残机锁到 5, 持续 {args.lock_seconds:.0f}s (自机应打不死)")
        lock = LivesLock(backend.mem, addr, value=5).start()
        t0 = time.time()
        while time.time() - t0 < args.lock_seconds:
            env.step("noop", frames=10)
        lock.stop()
        print(f"[lives] 锁血结束 (共写 {lock.writes} 次), 当前 lives 读数={backend.mem.read_value(addr,'i32')}")

    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
