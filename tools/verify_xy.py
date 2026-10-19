"""验证一个候选地址是不是自机坐标: 反复移动, 看 x/y 是否跟随。

    python tools/verify_xy.py 0x06213B80
"""

from __future__ import annotations

import struct
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from touhou_rl_mcp.config import Config  # noqa: E402
from touhou_rl_mcp.env import TouhouEnv  # noqa: E402
from touhou_rl_mcp.games.registry import GameRegistry  # noqa: E402


def main() -> int:
    addr = int(sys.argv[1], 16) if len(sys.argv) > 1 else 0x06213B80
    cfg = Config.load()
    cfg.patch({"backend": "memory", "game": "th11"})
    env = TouhouEnv(cfg, registry=GameRegistry())
    env.start()
    mem = env.backend.mem
    kb = env.backend.keyboard

    def rf(a: int) -> float | None:
        try:
            return struct.unpack("<f", mem.read_bytes(a, 4))[0]
        except Exception:
            return None

    def show(tag: str) -> None:
        st = env.get_state()
        print(f"{tag:14s} x={rf(addr)} y={rf(addr + 4)}  in_game={st.in_game} stage={st.stage}")

    show("初始")
    for label, keys in (("按左", ["left", "shoot"]), ("按右", ["right", "shoot"]),
                        ("按上", ["up", "shoot"]), ("按下", ["down", "shoot"])):
        kb.focus_window()
        kb.hold(keys)
        time.sleep(1.5)
        kb.release_all()
        show(label)
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
