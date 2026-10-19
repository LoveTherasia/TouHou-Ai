"""Read-only diagnostic for the TH11 entity layouts recovered from its x86 code."""
from __future__ import annotations

import collections
import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from touhou_rl_mcp.memory.process import ProcessMemory, find_first_pid


def main():
    pid = find_first_pid(["th11c.exe", "th11.exe"])
    if pid is None:
        raise SystemExit("th11 is not running")
    with ProcessMemory(pid) as mem:
        base = mem.main_module[1]
        read = mem.read_value
        bm = read(base + 0xA8D68, "u32")
        em = read(base + 0xA8D7C, "u32")
        pm = read(base + 0xA8EB4, "u32")
        result = {"pid": pid, "managers": [hex(v) for v in (bm, em, pm)],
                  "frame": read(base + 0xA5734, "i32")}
        if pm:
            result["player"] = struct.unpack("<ff", mem.read_bytes(pm + 0x3FC, 8))
        if bm:
            data = mem.read_bytes(bm + 0x64, 0x910 * 2000)
            states = collections.Counter()
            bullets = []
            for i in range(2000):
                off = i * 0x910
                state = struct.unpack_from("<H", data, off + 0x4B2)[0]
                states[state] += 1
                if state:
                    bullets.append({"slot": i, "state": state,
                                    "xy": struct.unpack_from("<ff", data, off + 0x43C),
                                    "radius": struct.unpack_from("<f", data, off + 0x45C)[0]})
            result["bullet_states"] = dict(states)
            result["manager_count"] = read(bm + 0x5C, "i32")
            result["bullets"] = bullets[:20]
        if em:
            node = read(em + 0x68, "u32")
            seen = set()
            enemies = []
            while node and node not in seen and len(seen) < 256:
                seen.add(node)
                obj, node = struct.unpack("<II", mem.read_bytes(node, 8))
                if obj:
                    enemies.append({"address": hex(obj),
                                    "xy": struct.unpack("<ff", mem.read_bytes(obj + 0x1070, 8)),
                                    "flags": hex(read(obj + 0x25BC, "u32"))})
            result["enemy_count"] = read(em + 0x70, "i32")
            result["enemies"] = enemies
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
