"""TH11 Boss 结构只读探测: 校验 combat 代码签名 / Boss 指针 / 血量与坐标.

不按键、不改内存。Boss 未入场时 ``boss_ptr=0`` 是正常结果 —— 要看到血量与坐标,
得先进关并等到道中 Boss 出现 (第一面约 frame 2100)。

    python tools/probe_boss.py                    # 探测已在运行的进程
    python tools/probe_boss.py --launch           # 没跑就由本工具启动
    python tools/probe_boss.py --seconds 120      # 连续采样两分钟, 变化才打印
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from touhou_rl_mcp.games.registry import GameRegistry
from touhou_rl_mcp.memory.process import ProcessMemory, find_first_pid


def probe_once(mem: ProcessMemory, spec) -> dict:
    base = mem.main_module[1]
    layout = spec.entities["combat"]
    out: dict = {"base": hex(base)}
    sigs = []
    for s in layout.get("signatures", []):
        off = int(s["offset"], 16)
        want = bytes.fromhex(s["bytes"])
        try:
            got = mem.read_bytes(base + off, len(want))
            sigs.append({"offset": s["offset"], "ok": got == want,
                         "want": want.hex(), "got": got.hex()})
        except Exception as exc:  # noqa: BLE001
            sigs.append({"offset": s["offset"], "ok": False, "error": str(exc)})
    out["signatures"] = sigs

    enemies = spec.entities["enemies"]
    manager = mem.read_value(base + int(enemies["pointer"], 16), "u32")
    out["enemy_manager"] = hex(manager)
    if manager:
        boss = mem.read_value(manager + int(layout["boss"], 16), "u32")
        out["boss_ptr"] = hex(boss)
        out["enemy_count"] = mem.read_value(manager + 0x70, "i32")
        if boss:
            hp, mx = mem.read_value(boss + int(layout["hp"], 16), "i32"), \
                     mem.read_value(boss + int(layout["hp"], 16) + 4, "i32")
            flags = mem.read_value(boss + int(layout["flags"], 16), "u32")
            x, y = mem.read_value(boss + int(layout["position"], 16), "f32"), \
                   mem.read_value(boss + int(layout["position"], 16) + 4, "f32")
            out.update(hp=hp, max_hp=mx, flags=hex(flags), x=x, y=y,
                       plausible=(abs(x) < 8192 and abs(y) < 8192
                                  and -100000 <= hp <= 100000000))
    return out


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--launch", action="store_true")
    ap.add_argument("--seconds", type=float, default=0.0, help="持续采样秒数")
    ap.add_argument("--every", type=float, default=2.0)
    args = ap.parse_args()

    spec = GameRegistry().get("th11")
    pid = find_first_pid(["th11c.exe", "th11.exe"])
    if pid is None:
        if not args.launch:
            raise SystemExit("TH11 没在运行 (加 --launch 自动启动)")
        from touhou_rl_mcp.backends.memory import MemoryBackend
        from touhou_rl_mcp.config import Config
        be = MemoryBackend(spec, Config({"backend": "memory", "game": "th11"}))
        be.start()
        be.close()
        pid = find_first_pid(["th11c.exe", "th11.exe"])
        if pid is None:
            raise SystemExit("启动失败")
    print("[probe] pid =", pid, flush=True)

    with ProcessMemory(pid) as mem:
        if args.seconds <= 0:
            print(json.dumps(probe_once(mem, spec), indent=2, ensure_ascii=False))
            return 0
        deadline = time.perf_counter() + args.seconds
        last = None
        while time.perf_counter() < deadline:
            info = probe_once(mem, spec)
            sig_ok = all(s.get("ok") for s in info["signatures"])
            line = (f"sig={sig_ok} manager={info.get('enemy_manager')} "
                    f"boss={info.get('boss_ptr')} hp={info.get('hp')} "
                    f"xy={info.get('x')},{info.get('y')} flags={info.get('flags')}")
            if line != last:
                print(line, flush=True)
                last = line
            time.sleep(args.every)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
