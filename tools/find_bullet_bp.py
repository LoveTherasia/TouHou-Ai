"""用 Cheat Engine 调试器定位弹幕结构: 对候选地址下访问断点, 记录指令与寄存器.

Python 侧: 攒弹幕 -> 从"x,y 同帧都变"的地址里挑一个"最密集"的当靶子 -> 写 target.txt -> 起 CE(autorun Lua)。
Lua 侧: 下 bptAccess 断点, 每次命中记录 EIP/反汇编/各寄存器相对靶子的偏移, 收集若干次后结束。
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from touhou_rl_mcp.backends.memory import MemoryBackend  # noqa: E402
from touhou_rl_mcp.config import Config  # noqa: E402
from touhou_rl_mcp.games.registry import GameRegistry  # noqa: E402
from touhou_rl_mcp.inputs.actions import parse  # noqa: E402
from touhou_rl_mcp.memory.cheats import LivesLock  # noqa: E402

CE_EXE = r"E:\Cheat Engine\cheatengine-i386.exe"
FLAG = Path(r"E:\TouHou-Ai\.runtime\ce")
AUTORUN = Path(r"E:\Cheat Engine\autorun\zz_find_bullet.lua")

LUA = r'''-- 弹幕结构断点探测 (由 tools/find_bullet_bp.py 生成)
local DIR = "E:\\TouHou-Ai\\.runtime\\ce\\"
local function log(s) local f=io.open(DIR.."bp.log","a") if f then f:write(os.date("%H:%M:%S ")..tostring(s).."\n") f:close() end end
local function touch(n) local f=io.open(DIR..n,"w") if f then f:write("1") f:close() end end

log("autorun start")
local pid = nil
for i = 1, 40 do pid = getProcessIDFromProcessName("th11c.exe") if pid then break end sleep(500) end
if not pid then log("no th11c") touch("bp_error") return end
openProcess(pid)
sleep(300)
local ok = debugProcess()
log("debugProcess -> "..tostring(ok).." pid="..tostring(pid))

local f = io.open(DIR.."target.txt", "r")
if not f then log("no target.txt") touch("bp_error") return end
local target = tonumber(f:read("*l"))
f:close()
log(string.format("target = 0x%X", target))

local hits = 0
local T = target
function debugger_onBreakpoint()
  hits = hits + 1
  local ins = "?"
  pcall(function() ins = disassemble(EIP) end)
  local parts = {}
  local regs = {EAX=EAX, EBX=EBX, ECX=ECX, EDX=EDX, ESI=ESI, EDI=EDI, EBP=EBP, ESP=ESP}
  for name, v in pairs(regs) do
    if v and v <= T then
      local d = T - v
      if d >= 0 and d <= 0x2000 then parts[#parts+1] = string.format("%s=0x%X(+0x%X)", name, v, d) end
    end
  end
  log(string.format("hit %d eip=0x%X |%s | %s", hits, EIP, ins, table.concat(parts, " ")))
  if hits >= 150 then
    debug_removeBreakpoint(T)
    touch("bp_done")
  end
  return 1
end

debug_setBreakpoint(target, 4, bptAccess)
log("breakpoint set")
local t = os.time()
while os.time() - t < 75 do
  local g = io.open(DIR.."bp_done", "r")
  if g then g:close() break end
  sleep(200)
end
pcall(function() debug_removeBreakpoint(T) end)
log("finished hits="..tostring(hits))
touch("bp_finished")
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--play-seconds", type=float, default=15.0)
    ap.add_argument("--difficulty", type=int, default=1)
    args = ap.parse_args()

    FLAG.mkdir(parents=True, exist_ok=True)
    for p in FLAG.glob("*"):
        p.unlink()
    AUTORUN.parent.mkdir(parents=True, exist_ok=True)
    AUTORUN.write_text(LUA, encoding="utf-8")

    cfg = Config.load()
    cfg.patch({"backend": "memory", "game": "th11"})
    be = MemoryBackend(GameRegistry().get("th11"), cfg)
    be.start()
    be.reset(options={"restart_process": True, "difficulty": args.difficulty})
    time.sleep(3)
    for _ in range(80):
        s = be.read_state()
        if s.in_game and (s.power or 0) > 0.3:
            break
        be.apply(parse("noop", default_frames=20))
    lock = LivesLock(be.mem, be.reader.address_of("lives"), value=9, floor=1).start()

    t0 = time.time()
    k = 0
    while time.time() - t0 < args.play_seconds:
        k += 1
        be.apply(parse("shoot", default_frames=8))
        be.apply(parse("left" if k % 2 else "right", default_frames=8))
        if not be.read_state().in_game:
            be.reset(options={"difficulty": args.difficulty})
            time.sleep(2)

    mem = be.mem
    regions = [(b, sz) for b, sz, p, ty in mem.memory_regions()
               if sz <= 64 * 1024 * 1024 and not (ty & 0x1000000)]

    def cap():
        out = {}
        for b, sz in regions:
            try:
                out[b] = np.frombuffer(mem.read_bytes(b, sz), dtype="<f4").copy()
            except Exception:
                pass
        return out

    A = cap()
    be.apply(parse("noop", default_frames=2))
    B = cap()
    M = set()
    for b in set(A) & set(B):
        if A[b].shape != B[b].shape or len(A[b]) < 2:
            continue
        d = A[b] != B[b]
        xv, yv = B[b][:-1], B[b][1:]
        pair = d[:-1] & d[1:] & (xv >= 0.0) & (xv <= 470.0) & (yv >= 0.0) & (yv <= 520.0)
        for i in np.flatnonzero(pair).tolist():
            M.add(b + 4 * int(i))
    print(f"[bp] 候选 {len(M)}")
    if not M:
        be.close()
        return 1
    Ms = sorted(M)
    # 挑"邻居最密集"的靶子 (越可能在弹幕数组里)
    best = None
    for a in Ms:
        n = sum(1 for x in Ms if abs(x - a) <= 0x400)
        if best is None or n > best[1]:
            best = (a, n)
    target = best[0]
    print(f"[bp] 靶子 = 0x{target:08X} (邻居 {best[1]})  值 x={mem.read_value(target,'f32')}")
    lock.stop()

    (FLAG / "target.txt").write_text(hex(target), encoding="ascii")
    print("[bp] 启动 CE ...")
    ce = subprocess.Popen([CE_EXE])
    try:
        deadline = time.time() + 180
        while time.time() < deadline:
            if (FLAG / "bp_finished").exists() or (FLAG / "bp_error").exists():
                break
            time.sleep(0.3)
        log_path = FLAG / "bp.log"
        print("[bp] ---- bp.log ----")
        if log_path.exists():
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            for ln in lines[:80]:
                print("   " + ln)
            print(f"   ... ({len(lines)} 行)")
    finally:
        try:
            ce.terminate()
        except Exception:
            pass
        try:
            AUTORUN.unlink()
        except Exception:
            pass
        be.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
