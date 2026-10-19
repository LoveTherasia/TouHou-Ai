"""自动找自机坐标的**指针链** (CE 指针扫描的代码版, 全自动).

流程:
  1. 锁血保活;
  2. 回位法+左右验证 -> 自机 x 候选 (少量);
  3. 对每个候选找"模块级指针链" -> 能形成干净链的候选即真自机 x;
  4. 写回 offsets json (pointer + offsets, 跨重启稳定)。

    python tools/find_pointer_chain.py --game th11 --write
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


def resolve_chain(mem, mbase: int, pointer: int, offsets: list[int]) -> int | None:
    try:
        addr = int(mem.read_value(mbase + pointer, "u32"))
        for off in offsets[:-1]:
            addr = int(mem.read_value(addr + off, "u32"))
        return addr + offsets[-1] if offsets else addr
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="th11")
    ap.add_argument("--difficulty", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--lock-value", type=int, default=5)
    ap.add_argument("--max-offset", type=int, default=0x1000)
    ap.add_argument("--max-levels", type=int, default=4)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--rounds", type=int, default=5)
    args = ap.parse_args()

    cfg = Config.load()
    cfg.patch({"backend": "memory", "game": args.game})
    env = TouhouEnv(cfg, registry=GameRegistry())
    env.start()
    backend = env.backend
    if not env.get_state().in_game:
        env.reset(restart_process=False, difficulty=args.difficulty)
        time.sleep(3.0)

    lives_addr = backend.reader.address_of("lives")
    if lives_addr is None:
        print("[ptr] lives 未标定, 先跑 tools/find_lives.py --write")
        env.close()
        return 2

    lock = LivesLock(backend.mem, lives_addr, value=args.lock_value).start()
    print(f"[ptr] 已锁血 @ 0x{lives_addr:08X}")
    scanner = backend.scanner()
    mem = backend.mem
    mbase = backend.reader.base_address
    name, mb, ms = mem.main_module
    print(f"[ptr] 主模块 {name} base=0x{mb:X}")

    n = {"i": 0}

    def press(_b, frames):
        n["i"] += 1
        if _b and _b[0] == "up":
            backend.apply(parse("up", default_frames=int(frames)))
        else:
            backend.apply(parse("+".join(_b) if _b else "noop", default_frames=int(frames)))

    try:
        pool: set[int] = set()
        for r in range(1, args.rounds + 1):
            cs = scanner.find_player_by_respawn_lr(
                lives_addr, press=press, timeout=args.timeout, log=lambda m: None
            )
            for c in cs:
                pool.add(c.address)
            print(f"[ptr] 第 {r}/{args.rounds} 轮: 本轮 {len(cs)} 个, 候选池 {len(pool)} 个")
        if not pool:
            print("[ptr] 回位法多轮为空, 退回全内存回位法 ...")
            cs = scanner.find_respawn_coordinates(
                lives_addr, hits=2, press=press, timeout=args.timeout, log=lambda m: print("[ptr]", m)
            )
            for c in cs:
                pool.add(c.address)
        if not pool:
            print("[ptr] 没拿到任何候选")
            return 1
        print(f"[ptr] 候选池 {len(pool)} 个, 抓一次内存快照, 逐个找模块级指针链 ...")

        def rf(a: int) -> float | None:
            try:
                return struct.unpack("<f", mem.read_bytes(a, 4))[0]
            except Exception:
                return None

        # 静止稳定性 + 场地范围: 自机静止时 x 恒定且在 [8,376]; 弹幕会动/越界
        time.sleep(0.4)
        stable = []
        for a in sorted(pool):
            vs = []
            for _ in range(3):
                vs.append(rf(a))
                time.sleep(0.12)
            if all(v is not None and 8.0 <= v <= 376.0 for v in vs) and max(vs) - min(vs) < 0.5:
                stable.append(a)
        print(f"[ptr] 静止稳定且 x∈[8,376]: {len(stable)} 个 (原 {len(pool)})")
        if stable:
            pool = set(stable)

        # 写测试: 自机 x 被写入后游戏会"采纳"(自机移动, 值保持); 弹幕/派生量会被立刻覆盖
        def wf(a: int, v: float) -> None:
            try:
                mem.write_bytes(a, struct.pack("<f", v))
            except Exception:
                pass

        write_ok = []
        for a in sorted(pool):
            wf(a, 137.0)
            time.sleep(0.25)
            v1 = rf(a)
            wf(a, 211.0)
            time.sleep(0.25)
            v2 = rf(a)
            if v1 is not None and v2 is not None and abs(v1 - 137.0) < 1.0 and abs(v2 - 211.0) < 1.0:
                write_ok.append(a)
        print(f"[ptr] 写测试通过(值跟随写入=被游戏采纳): {len(write_ok)} 个 -> {[hex(x) for x in write_ok]}")
        if write_ok:
            pool = set(write_ok)

        arrays = dict(scanner._capture_arrays(include_images=True))
        best = None
        for a in sorted(pool):
            chains = scanner.find_pointer_chain(
                a, max_offset=args.max_offset, max_levels=args.max_levels,
                limit=20, arrays=arrays, log=lambda m: None,
            )
            ok = [ch for ch in chains if resolve_chain(mem, mbase, ch["pointer"], ch["offsets"]) == a]
            if ok:
                ok.sort(key=lambda ch: (ch["levels"], sum(ch["offsets"])))
                print(
                    f"   0x{a:08X}: 链 {len(chains)}, 可解析 {len(ok)}, "
                    f"最佳 {name}+0x{ok[0]['pointer']:X} offsets={[hex(o) for o in ok[0]['offsets']]}"
                )
                if best is None:
                    best = (a, ok[0])

        if best is None:
            print("[ptr] 没有候选形成可解析的模块级指针链")
            return 1

        a, ch = best
        print(
            f"[ptr] 选定 0x{a:08X} 链 = {ch['module']}+0x{ch['pointer']:X} "
            f"offsets={[hex(o) for o in ch['offsets']]} levels={ch['levels']}"
        )
        if args.write:
            px = {"pointer": f"0x{ch['pointer']:X}", "offsets": [f"0x{o:X}" for o in ch["offsets"]], "type": "f32"}
            offs_y = list(ch["offsets"])
            offs_y[-1] += 4
            py = {"pointer": f"0x{ch['pointer']:X}", "offsets": [f"0x{o:X}" for o in offs_y], "type": "f32"}
            result = apply_calibration(args.game, {"player_x": px, "player_y": py})
            print(f"[ptr] 已写入 {result['path']}")
            print(f"      player_x = {px}")
            print(f"      player_y = {py}")
        return 0
    finally:
        lock.stop()
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
