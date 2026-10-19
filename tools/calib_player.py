"""标定自机坐标: 撞墙法 + 贴墙静止收尾, 可先锁血以争取时间.

流程:
  1. (可选) --lock-lives: 用已标定的 lives 偏移锁住残机 -> 自机打不死, 有充足时间;
  2. 撞墙法 (scanner.find_player_coordinates): "墙边不动期间不变 + 移动期会变" -> 候选;
  3. 收尾判据: 贴左墙后, 候选必须"很小且静止不变" (弹幕会动, 会被剔除)。

    python tools/calib_player.py --lock-lives
"""

from __future__ import annotations

import argparse
import struct
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
    ap.add_argument("--attempts", type=int, default=4)
    ap.add_argument("--lock-lives", action="store_true")
    ap.add_argument("--lives-value", type=int, default=5)
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()

    cfg = Config.load()
    cfg.patch({"backend": "memory", "game": args.game})
    env = TouhouEnv(cfg, registry=GameRegistry())
    env.start()
    backend = env.backend

    def press(buttons, frames):
        backend.apply(parse("+".join(buttons) if buttons else "noop", default_frames=int(frames)))

    lock: LivesLock | None = None
    try:
        for attempt in range(1, args.attempts + 1):
            if lock is not None:
                lock.stop()
                lock = None
            print(f"===== 尝试 {attempt} =====")
            env.reset(restart_process=True, difficulty=args.difficulty)
            scanner = backend.scanner()
            press(["confirm"], 6)
            press(["confirm"], 6)
            time.sleep(2.0)
            if not env.get_state().in_game:
                print("  未能进入局内, 重试")
                continue

            if args.lock_lives:
                addr = backend.reader.address_of("lives")
                if addr:
                    lock = LivesLock(backend.mem, addr, value=args.lives_value).start()
                    print(f"  锁血 @ 0x{addr:08X} = {args.lives_value}")
                else:
                    print("  (lives 未标定, 无法锁血; 先跑 tools/find_lives.py --write)")

            mem = backend.mem

            def rf(a: int) -> float | None:
                try:
                    return struct.unpack("<f", mem.read_bytes(a, 4))[0]
                except Exception:
                    return None

            print("  撞墙法找候选 ...")
            cands = scanner.find_player_coordinates(
                press, settle_frames=120, hold_frames=45, move_frames=80,
                low=1.0, high=448.0, include_images=False, log=lambda m: print("    " + m),
            )
            if not cands:
                print("  撞墙法无候选, 重试")
                continue

            # 收尾: 两次"左墙->右墙", 要求两次的左/右墙值都精确复现 (自机是精确 clamp)
            press(["left", "down"], 70)
            press(["left"], 90)
            l1 = {c.address: rf(c.address) for c in cands}
            press(["right"], 220)
            r1 = {c.address: rf(c.address) for c in cands}
            press(["left"], 260)
            l2 = {c.address: rf(c.address) for c in cands}
            press(["right"], 260)
            r2 = {c.address: rf(c.address) for c in cands}
            final = []
            for c in cands:
                a = c.address
                a_l1, a_r1, a_l2, a_r2 = l1.get(a), r1.get(a), l2.get(a), r2.get(a)
                if None in (a_l1, a_r1, a_l2, a_r2):
                    continue
                if a_l1 <= 80 and a_r1 >= 300 and abs(a_l1 - a_l2) < 0.5 and abs(a_r1 - a_r2) < 0.5:
                    final.append((a, a_l1, a_r1))
            print(f"  两次贴墙复现判据后: {len(final)} 个")
            for a, lv, rv in final[:20]:
                print(f"    0x{a:08X} 左={lv:.3f} 右={rv:.3f}")

            if final:
                address, lv, rv = final[0]
                print(f"  选定 0x{address:08X} (左{lv:.1f}/右{rv:.1f})")
                if not args.no_write:
                    result = apply_calibration(
                        args.game,
                        {
                            "player_x": {"address": f"0x{address:X}", "type": "f32"},
                            "player_y": {"address": f"0x{address + 4:X}", "type": "f32"},
                        },
                    )
                    print(f"  已写入 {result['path']}")
                return 0
            print("  无可靠候选, 重试")

        print("标定失败")
        return 1
    finally:
        if lock is not None:
            lock.stop()
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
