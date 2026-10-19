"""对话/静默期快速标定自机坐标 (无弹幕时内存安静, 候选极少).

    python tools/find_player_dialogue.py --game th11 --write
"""

from __future__ import annotations

import argparse
import struct
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from touhou_rl_mcp.config import Config  # noqa: E402
from touhou_rl_mcp.env import TouhouEnv  # noqa: E402
from touhou_rl_mcp.games.calibration import apply_calibration  # noqa: E402
from touhou_rl_mcp.games.registry import GameRegistry  # noqa: E402
from touhou_rl_mcp.inputs.actions import parse  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="th11")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    cfg = Config.load()
    cfg.patch({"backend": "memory", "game": args.game})
    env = TouhouEnv(cfg, registry=GameRegistry())
    env.start()
    backend = env.backend
    scanner = backend.scanner()
    mem = backend.mem
    mbase = backend.reader.base_address

    def press(keys, frames):
        backend.apply(parse("+".join(keys) if keys else "noop", default_frames=int(frames)))

    def snap():
        return dict(scanner._capture_arrays(include_images=False))

    def rf(a):
        try:
            return struct.unpack("<f", mem.read_bytes(a, 4))[0]
        except Exception:
            return None

    print("[dlg] 快照 S0 (当前静止) ...")
    s0 = snap()
    press(["left"], 55)
    print("[dlg] 左移后 S1 ...")
    s1 = snap()
    press(["right"], 80)
    print("[dlg] 右移后 S2 ...")
    s2 = snap()

    cands = []
    for base, a0 in s0.items():
        a1 = s1.get(base)
        a2 = s2.get(base)
        if a1 is None or a2 is None or a1.shape != a0.shape or a2.shape != a0.shape:
            continue
        f0, f1, f2 = a0.view("<f4"), a1.view("<f4"), a2.view("<f4")
        with np.errstate(invalid="ignore"):
            m = (f0 >= 1) & (f0 <= 448) & (f1 >= 1) & (f1 <= 80) & (f2 >= 300) & (f2 <= 384)
        for i in np.flatnonzero(m).tolist():
            cands.append((base + 4 * int(i), float(f0[i]), float(f1[i]), float(f2[i])))
    print(f"[dlg] 候选 {len(cands)} 个 (左移变小 且 右移变大):")
    for a, x0, x1, x2 in cands[:40]:
        print(f"   0x{a:08X} S0={x0:7.2f} 左={x1:7.2f} 右={x2:7.2f}")

    if not cands:
        env.close()
        return 1

    # 找指针链
    arrays = dict(scanner._capture_arrays(include_images=True))
    best = None
    for a, *_ in cands:
        chains = scanner.find_pointer_chain(a, max_offset=0x1000, max_levels=2, limit=20, arrays=arrays)
        ok = []
        for ch in chains:
            try:
                addr = int(mem.read_value(mbase + ch["pointer"], "u32"))
                for off in ch["offsets"][:-1]:
                    addr = int(mem.read_value(addr + off, "u32"))
                r = addr + ch["offsets"][-1]
            except Exception:
                r = None
            if r == a:
                ok.append(ch)
        if ok:
            ok.sort(key=lambda ch: (ch["levels"], sum(ch["offsets"])))
            print(
                f"   0x{a:08X}: 链 {len(ok)}, 最佳 {ch_hex(ok[0])}"
            )
            if best is None:
                best = (a, ok[0])

    if best is None:
        print("[dlg] 候选都没形成可解析指针链")
        env.close()
        return 1
    a, ch = best
    print(f"[dlg] 选定 0x{a:08X} 链 = th11c.exe+0x{ch['pointer']:X} offsets={[hex(o) for o in ch['offsets']]}")
    if args.write:
        px = {"pointer": f"0x{ch['pointer']:X}", "offsets": [f"0x{o:X}" for o in ch["offsets"]], "type": "f32"}
        offs_y = list(ch["offsets"])
        offs_y[-1] += 4
        py = {"pointer": f"0x{ch['pointer']:X}", "offsets": [f"0x{o:X}" for o in offs_y], "type": "f32"}
        r = apply_calibration(args.game, {"player_x": px, "player_y": py})
        print(f"[dlg] 已写入 {r['path']}")
        print(f"      player_x = {px}")
        print(f"      player_y = {py}")
    env.close()
    return 0


def ch_hex(ch):
    return f"th11c.exe+0x{ch['pointer']:X} offsets={[hex(o) for o in ch['offsets']]}"


if __name__ == "__main__":
    raise SystemExit(main())
