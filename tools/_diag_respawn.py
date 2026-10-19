"""诊断: 自机被击中复位后, 直接找"x 接近场地中心、y 接近底部"的 f32 地址."""

from __future__ import annotations

import struct
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from touhou_rl_mcp.config import Config  # noqa: E402
from touhou_rl_mcp.env import TouhouEnv  # noqa: E402
from touhou_rl_mcp.games.registry import GameRegistry  # noqa: E402
from touhou_rl_mcp.inputs.actions import parse  # noqa: E402
from touhou_rl_mcp.memory.cheats import LivesLock  # noqa: E402


def main() -> int:
    cfg = Config.load()
    cfg.patch({"backend": "memory", "game": "th11"})
    env = TouhouEnv(cfg, registry=GameRegistry())
    env.start()
    backend = env.backend
    if not env.get_state().in_game:
        env.reset(restart_process=False, difficulty=0)
        time.sleep(3.0)
    lives = backend.reader.address_of("lives")
    lock = LivesLock(backend.mem, lives, value=5).start()
    scanner = backend.scanner()
    mem = backend.mem
    try:
        prev = int(mem.read_value(lives, "i32"))
        print(f"等待被击中 (lives={prev}) ...")
        deadline = time.time() + 90
        while time.time() < deadline:
            backend.apply(parse("up", default_frames=6))
            cur = int(mem.read_value(lives, "i32"))
            if cur < prev:
                print(f"被击中! lives {prev}->{cur}, 等 1.2s 复位 ...")
                time.sleep(1.2)
                break
            prev = cur
        else:
            print("没等到被击中")
            return 1

        arrays = dict(scanner._capture_arrays(include_images=True))
        hits = []
        for base, arr in arrays.items():
            f = arr.view("<f4")
            for i in range(len(f) - 1):
                x = float(f[i])
                if 150.0 <= x <= 240.0:
                    y = float(f[i + 1])
                    if 350.0 <= y <= 448.0:
                        hits.append(base + 4 * i)
        print(f"复位后 x∈[150,240] 且 y(+4)∈[350,448]: {len(hits)} 个")

        def rf(a: int) -> float | None:
            try:
                return struct.unpack("<f", mem.read_bytes(a, 4))[0]
            except Exception:
                return None

        respawn = {a: rf(a) for a in hits}  # 复位瞬间的值 (在按左右之前)

        # 左移 -> 变小(到左墙); 右移 -> 变大(到右墙)
        backend.apply(parse("left", default_frames=110))
        after_left = {a: rf(a) for a in hits}
        keep = [a for a in hits if after_left.get(a) is not None and 1 <= after_left[a] <= 60]
        print(f"左移后 x∈[1,60]: {len(keep)} 个")
        backend.apply(parse("right", default_frames=150))
        after_right = {a: rf(a) for a in keep}
        final = [a for a in keep if after_right.get(a) is not None and 300 <= after_right[a] <= 384]
        print(f"右移后 x∈[300,384]: {len(final)} 个")
        for a in final[:20]:
            print(
                f"   player_x? 0x{a:08X} 复位={respawn.get(a)} 左={after_left[a]:.3f} 右={after_right[a]:.3f}"
            )
        # 自机复位值应接近场地中心 192
        center = [a for a in final if respawn.get(a) is not None and 185 <= respawn[a] <= 199]
        print(f"复位值接近中心(185~199): {len(center)} 个 -> {[hex(a) for a in center]}")
        return 0
    finally:
        lock.stop()
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
