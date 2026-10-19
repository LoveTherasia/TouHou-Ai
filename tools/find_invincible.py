"""自动定位东方正作的**无敌帧 (invincible)** 偏移.

无敌计数: 平时为 0, 被击中瞬间变成正值, 随后递减回 0。锁定它 -> 自机不再被击中 ->
**不再回位** -> 撞墙法/坐标标定变得可靠。

    python tools/find_invincible.py --game th11 --write
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from touhou_rl_mcp.config import Config  # noqa: E402
from touhou_rl_mcp.env import TouhouEnv  # noqa: E402
from touhou_rl_mcp.games.calibration import apply_calibration  # noqa: E402
from touhou_rl_mcp.games.registry import GameRegistry  # noqa: E402
from touhou_rl_mcp.inputs.actions import parse  # noqa: E402
from touhou_rl_mcp.memory.cheats import LivesLock, MemoryLock  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="th11")
    ap.add_argument("--difficulty", type=int, default=0)
    ap.add_argument("--hits", type=int, default=2)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--lock-value", type=int, default=5)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--lock-seconds", type=float, default=0.0, help="找到后锁定无敌多少秒演示")
    args = ap.parse_args()

    cfg = Config.load()
    cfg.patch({"backend": "memory", "game": args.game})
    env = TouhouEnv(cfg, registry=GameRegistry())
    env.start()
    backend = env.backend

    if not env.get_state().in_game:
        print("[inv] 开始一局")
        env.reset(restart_process=False, difficulty=args.difficulty)
        time.sleep(3.0)

    lives_addr = backend.reader.address_of("lives")
    if lives_addr is None:
        print("[inv] lives 未标定, 先跑 tools/find_lives.py --write")
        env.close()
        return 2

    lock = LivesLock(backend.mem, lives_addr, value=args.lock_value).start()
    print(f"[inv] 已锁血 @ 0x{lives_addr:08X}")
    scanner = backend.scanner()

    n = {"i": 0}

    def press(_b, _f):
        n["i"] += 1
        if n["i"] % 40 < 14:
            backend.apply(parse("up", default_frames=8))
        else:
            backend.apply(parse("noop", default_frames=3))

    # 等开局的无敌结束 (基线需为 0)
    print("[inv] 等开局无敌结束 (6s) ...")
    time.sleep(6.0)

    inv_lock: MemoryLock | None = None
    try:
        cands = scanner.find_invincible(
            lives_addr, hits=args.hits, press=press, timeout=args.timeout,
            log=lambda m: print("[inv]", m),
        )
        if not cands:
            print("[inv] 没找到 (可能没被击中)")
            return 1
        print(f"[inv] 候选 {len(cands)} 个 (被击中由 0 变正值且随后递减):")
        for c in sorted(cands, key=lambda c: -c.value)[:30]:
            off = f"{c.module}+0x{c.module_offset:X}" if c.module else "heap"
            print(f"   0x{c.address:08X} ({off}) 命中时值={c.value}")

        plausible = [c for c in cands if 1 <= c.value <= 900]
        best = (plausible or cands)[0]
        print(f"[inv] 判定 invincible @ 0x{best.address:08X} (值 {best.value})")
        if args.write:
            entry = (
                {"offset": f"0x{best.module_offset:X}", "type": "i32", "label": "无敌帧(自动标定)"}
                if best.module
                else {"address": f"0x{best.address:X}", "type": "i32", "label": "无敌帧(自动标定)"}
            )
            result = apply_calibration(args.game, {"invincible": entry})
            print(f"[inv] 已写入 {result['path']}")

        if args.lock_seconds > 0:
            lv0 = int(backend.mem.read_value(lives_addr, "i32"))
            print(f"[inv] 锁定无敌=60000, 持续 {args.lock_seconds:.0f}s (自机应打不中)")
            inv_lock = MemoryLock(backend.mem, best.address, 60000, interval=0.05).start()
            t0 = time.time()
            while time.time() - t0 < args.lock_seconds:
                env.step("noop", frames=10)
            inv_lock.stop()
            inv_lock = None
            lv1 = int(backend.mem.read_value(lives_addr, "i32"))
            print(f"[inv] 锁定期间 lives {lv0} -> {lv1}  ({'没被击中 ✓' if lv1 >= lv0 else '仍被击中 ✗'})")
        return 0
    finally:
        if inv_lock is not None:
            inv_lock.stop()
        lock.stop()
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
