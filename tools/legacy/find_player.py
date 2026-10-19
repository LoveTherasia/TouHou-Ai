"""定位自机坐标: 锁血争取时间 + 撞墙法 ∩ "被击中跳回固定初始位置".

流程:
  1. 连到正在运行的游戏 (默认不重启进程);
  2. 读已标定的 lives 偏移, **锁血** (生命值一减少就写回) -> 不会 game over, 时间充足;
  3. 撞墙法得到"像自机 x"的候选;
  4. 观察被击中 (lives -1), 只保留"每次被击中后都跳回同一固定值"的地址;
  5. 写回 offsets。

    python tools/find_player.py --game th11
"""

from __future__ import annotations

import argparse
import struct
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
    ap.add_argument("--hits", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=150.0)
    ap.add_argument("--method", choices=["respawn", "combined"], default="respawn")
    ap.add_argument("--lock-value", type=int, default=5)
    ap.add_argument("--no-lock", action="store_true")
    ap.add_argument("--restart", action="store_true")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    cfg = Config.load()
    cfg.patch({"backend": "memory", "game": args.game})
    env = TouhouEnv(cfg, registry=GameRegistry())
    env.start()
    backend = env.backend

    st = env.get_state()
    print(f"[player] 当前: in_game={st.in_game} stage={st.stage} lives={st.lives}")
    if not st.in_game:
        print(f"[player] 不在局内, 开始一局 (restart_process={args.restart})")
        env.reset(restart_process=args.restart, difficulty=args.difficulty)
        time.sleep(3.0)

    lives_addr = backend.reader.address_of("lives")
    if lives_addr is None:
        print("[player] lives 未标定, 先跑 tools/find_lives.py --write")
        env.close()
        return 2
    print(f"[player] lives @ 0x{lives_addr:08X}")

    lock: LivesLock | None = None
    try:
        if not args.no_lock:
            lock = LivesLock(backend.mem, lives_addr, value=args.lock_value).start()
            print(f"[player] 已锁血: lives <- {args.lock_value} (不会再 game over)")

        scanner = backend.scanner()
        mem = backend.mem

        n = {"i": 0}

        def press(_buttons, _frames):
            n["i"] += 1
            k = n["i"] % 60
            # 周期性上移撞进弹幕, 其余时间不动 (被瞄准弹击中)
            if k < 20:
                backend.apply(parse("up", default_frames=8))
            elif k < 30:
                backend.apply(parse("down", default_frames=8))
            else:
                backend.apply(parse("noop", default_frames=3))

        if args.method == "combined":
            cands = scanner.find_player_by_respawn_and_wall(
                lives_addr, press=press, hits=args.hits, timeout=args.timeout,
                log=lambda m: print("[player]", m),
            )
        else:
            cands = scanner.find_respawn_coordinates(
                lives_addr, hits=args.hits, press=press, timeout=args.timeout,
                log=lambda m: print("[player]", m),
            )

        def rf(a: int) -> float | None:
            try:
                return struct.unpack("<f", mem.read_bytes(a, 4))[0]
            except Exception:
                return None

        print(f"[player] 候选 {len(cands)} 个 (撞墙法 ∩ 被击中后跳回同一固定值):")
        for c in cands[:40]:
            print(f"   0x{c.address:08X} 被击中后值={c.value:.3f}  +4={rf(c.address + 4)}  -4={rf(c.address - 4)}")

        likely = [c for c in cands if 120 <= c.value <= 260]
        if likely:
            best = likely[0]
            print(f"[player] 选定 x @ 0x{best.address:08X} = {best.value:.2f}")
            if args.write:
                result = apply_calibration(
                    args.game,
                    {
                        "player_x": {"address": f"0x{best.address:X}", "type": "f32"},
                        "player_y": {"address": f"0x{best.address + 4:X}", "type": "f32"},
                    },
                )
                print(f"[player] 已写入 {result['path']}")
        else:
            print("[player] 没有落在场地中心附近的候选, 请人工看上面的列表")
        return 0 if cands else 1
    finally:
        if lock is not None:
            lock.stop()
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
