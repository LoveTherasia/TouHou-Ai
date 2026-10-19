"""多方法轮询 + 候选持久化, 逐轮求交逼近自机坐标.

思路 (消噪):
  * 始终**锁血** (残机降到 floor 才写回) -> 不会 game over, 可无限轮;
  * 每轮随机选一种方法产出候选: 撞墙法 / 回位法 / 两者结合;
  * 把候选写进一个**文本文件**, 下一轮与文件里的取**交集** -> 真自机 x 会一直活下来,
    噪声 (弹幕等) 会逐轮被剔除;
  * 文件变小到阈值就停并给出结论。

    python tools/find_player_iter.py --game th11 --rounds 8
    python tools/find_player_iter.py --game th11 --reset     # 清空候选文件重新开始
"""

from __future__ import annotations

import argparse
import random
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


def load_store(path: Path) -> set[int]:
    if not path.exists():
        return set()
    out = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.add(int(line.split()[0], 16))
            except ValueError:
                pass
    return out


def save_store(path: Path, addrs: set[int], mem) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for a in sorted(addrs):
        try:
            v = struct.unpack("<f", mem.read_bytes(a, 4))[0]
        except Exception:
            v = float("nan")
        lines.append(f"0x{a:08X}  {v:.3f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="th11")
    ap.add_argument("--difficulty", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--hits", type=int, default=2)
    ap.add_argument("--timeout", type=float, default=90.0)
    ap.add_argument("--lock-value", type=int, default=5)
    ap.add_argument("--stop-at", type=int, default=8, help="候选少于此数就停")
    ap.add_argument("--methods", default="respawn_lr,respawn_lr,wall,respawn_lr", help="轮换使用的方法序列")
    ap.add_argument("--store", default=".runtime/player_candidates/th11.txt")
    ap.add_argument("--reset", action="store_true", help="清空候选文件重新开始")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--write", action="store_true", help="收敛后写回 offsets")
    args = ap.parse_args()

    store_path = Path(args.store)
    if args.reset and store_path.exists():
        store_path.unlink()
    store = load_store(store_path)
    print(f"[iter] 起始候选文件: {store_path} ({len(store)} 个)")

    rng = random.Random(args.seed or None)

    cfg = Config.load()
    cfg.patch({"backend": "memory", "game": args.game})
    env = TouhouEnv(cfg, registry=GameRegistry())
    env.start()
    backend = env.backend

    if not env.get_state().in_game:
        print("[iter] 开始一局")
        env.reset(restart_process=False, difficulty=args.difficulty)
        time.sleep(3.0)

    lives_addr = backend.reader.address_of("lives")
    if lives_addr is None:
        print("[iter] lives 未标定, 先跑 tools/find_lives.py --write")
        env.close()
        return 2

    lock = LivesLock(backend.mem, lives_addr, value=args.lock_value).start()
    print(f"[iter] 已锁血 @ 0x{lives_addr:08X} (残机降到 1 才写回 {args.lock_value})")

    mem = backend.mem

    n = {"i": 0}

    def press(_buttons, _frames):
        n["i"] += 1
        k = n["i"] % 60
        if k < 18:
            backend.apply(parse("up", default_frames=8))
        elif k < 28:
            backend.apply(parse("down", default_frames=8))
        elif k < 40:
            backend.apply(parse("left", default_frames=8))
        else:
            backend.apply(parse("noop", default_frames=3))

    def readf(a: int) -> float | None:
        try:
            return struct.unpack("<f", mem.read_bytes(a, 4))[0]
        except Exception:
            return None

    def method_respawn_lr(scanner) -> set[int]:
        """回位法 + 左右验证: 被击中复位后 x≈中心/y≈底部, 再要求左移变小、右移变大."""
        prev = int(mem.read_value(lives_addr, "i32"))
        deadline = time.time() + args.timeout
        hit = False
        while time.time() < deadline:
            backend.apply(parse("up", default_frames=6))
            cur = int(mem.read_value(lives_addr, "i32"))
            if cur < prev:
                hit = True
                break
            prev = cur
        if not hit:
            return set()
        time.sleep(1.2)
        arrays = dict(scanner._capture_arrays(include_images=True))
        hits = []
        for base, arr in arrays.items():
            f = arr.view("<f4")
            for i in range(len(f) - 1):
                if 150.0 <= float(f[i]) <= 240.0 and 350.0 <= float(f[i + 1]) <= 448.0:
                    hits.append(base + 4 * i)
        backend.apply(parse("left", default_frames=110))
        keep = [a for a in hits if (readf(a) is not None and 1 <= readf(a) <= 60)]
        backend.apply(parse("right", default_frames=150))
        final = [a for a in keep if (readf(a) is not None and 300 <= readf(a) <= 384)]
        return set(final)

    def run_method(name: str, scanner) -> set[int]:
        try:
            if name == "respawn_lr":
                return method_respawn_lr(scanner)
            if name == "wall":
                cs = scanner.find_player_coordinates(
                    press, settle_frames=110, hold_frames=40, move_frames=70,
                    low=1.0, high=448.0, include_images=False, log=lambda m: None,
                )
            elif name == "respawn":
                cs = scanner.find_respawn_coordinates(
                    lives_addr, hits=args.hits, press=press, timeout=args.timeout, log=lambda m: None,
                )
            else:  # combined
                cs = scanner.find_player_by_respawn_and_wall(
                    lives_addr, press=press, hits=args.hits, timeout=args.timeout, log=lambda m: None,
                )
            return {c.address for c in cs}
        except Exception as exc:
            print(f"[iter]   方法 {name} 出错: {type(exc).__name__}: {exc}")
            return set()

    methods = args.methods.split(",")
    try:
        for rnd in range(1, args.rounds + 1):
            if not env.get_state().in_game:
                print(f"[iter] 第 {rnd} 轮: 不在局内, 重开一局")
                env.reset(restart_process=False, difficulty=args.difficulty)
                time.sleep(3.0)
                scanner = backend.scanner()
            else:
                scanner = backend.scanner()
            name = methods[(rnd - 1) % len(methods)]
            print(f"[iter] 第 {rnd}/{args.rounds} 轮: 方法={name} ...")
            got = run_method(name, scanner)
            if not got:
                print(f"[iter]   本轮无候选, 跳过 (不收缩)")
                continue
            if not store:
                store = got
            else:
                store = store & got
            save_store(store_path, store, mem)
            print(f"[iter]   本轮候选 {len(got)} -> 交集后 {len(store)} (已存 {store_path})")
            if len(store) <= args.stop_at:
                print("[iter] 候选已足够少, 提前结束")
                break
    finally:
        lock.stop()
        save_store(store_path, store, mem)

    print(f"[iter] 最终候选 {len(store)} 个 (文件 {store_path}):")
    rows = []
    for a in sorted(store):
        try:
            v = struct.unpack("<f", mem.read_bytes(a, 4))[0]
        except Exception:
            v = float("nan")
        rows.append((a, v))
    for a, v in rows[:60]:
        print(f"   0x{a:08X}  x={v:.3f}")

    likely = [(a, v) for a, v in rows if 120 <= v <= 260]
    if likely and len(store) <= args.stop_at:
        best = likely[0]
        print(f"[iter] 判定自机 x = 0x{best[0]:08X} (值 {best[1]:.2f})")
        if args.write:
            result = apply_calibration(
                args.game,
                {
                    "player_x": {"address": f"0x{best[0]:X}", "type": "f32"},
                    "player_y": {"address": f"0x{best[0] + 4:X}", "type": "f32"},
                },
            )
            print(f"[iter] 已写入 {result['path']}")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
