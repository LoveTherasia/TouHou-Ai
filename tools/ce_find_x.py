"""用 Cheat Engine + 移动跟踪, 自动找东方地灵殿的自机 X 坐标.

流程:
  1. CE (autorun Lua) 做"未知值 -> 未变化(贴墙不动) -> 变化(右移)"三轮扫描, 产出候选;
  2. 本项目用 SendInput 反复左右移动自机, 只保留"x 随按键大幅增减"的地址 —— 弹幕会被筛掉。

    python tools/ce_find_x.py
"""

from __future__ import annotations

import argparse
import struct
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from touhou_rl_mcp.config import Config  # noqa: E402
from touhou_rl_mcp.env import TouhouEnv  # noqa: E402
from touhou_rl_mcp.games.registry import GameRegistry  # noqa: E402
from touhou_rl_mcp.memory.cheats import LivesLock  # noqa: E402

FLAG = Path("E:/TouHou-Ai/.runtime/ce")
CE_EXE = r"E:\Cheat Engine\cheatengine-i386.exe"
AUTORUN_LUA = Path(r"E:\Cheat Engine\autorun\zz_touhou_findx.lua")

LUA_SOURCE = r'''-- 自动找 th11c.exe 的自机 X 坐标 (由 tools/ce_find_x.py 生成)
local DIR = "E:\\TouHou-Ai\\.runtime\\ce\\"
local function log(s) local f=io.open(DIR.."ce.log","a") if f then f:write(os.date("%H:%M:%S ")..tostring(s).."\n") f:close() end end
local function touch(n) local f=io.open(DIR..n,"w") if f then f:write("1") f:close() end end
local function wait_for(n, seconds)
  local t = os.time()
  while os.time() - t < seconds do
    local f = io.open(DIR..n, "r")
    if f then f:close() return true end
    sleep(100)
  end
  return false
end
log("autorun start")
local pid = nil
for i = 1, 20 do pid = getProcessIDFromProcessName("th11c.exe") if pid then break end sleep(500) end
if not pid then log("no th11c") touch("ce_error") return end
openProcess(pid)
sleep(300)
log("opened pid="..tostring(pid).." base=0x"..string.format("%X", getAddress("th11c.exe") or 0))
touch("ce_ready")
if not wait_for("phase_A", 90) then log("timeout A") touch("ce_error") return end
local ms = createMemScan()
ms.firstScan(soUnknownValue, vtDword, 0, "", "", "0", "7fffffff", "*X*C*W", 1, "4", false, false, false, false)
ms.waitTillDone()
log("firstScan done") touch("scan1_done")
if not wait_for("phase_B", 90) then log("timeout B") touch("ce_error") return end
ms.nextScan(soUnchanged, 0, "", "", false, false, false, false, false)
ms.waitTillDone()
log("nextScan(unchanged) done") touch("scan2_done")
if not wait_for("phase_C", 90) then log("timeout C") touch("ce_error") return end
ms.nextScan(soChanged, 0, "", "", false, false, false, false, false)
ms.waitTillDone()
log("nextScan(changed) done")
local fl = createFoundList(ms)
fl.initialize()
local out = io.open(DIR.."candidates.txt", "w")
local count = 0
for i = 0, fl.Count - 1 do
  local addr = tonumber(fl.getAddress(i), 16)
  if addr then
    local x = readFloat(addr)
    local y = readFloat(addr + 4)
    if x and y and x >= 1 and x <= 400 and y >= 1 and y <= 460 then
      out:write(string.format("0x%08X x=%.3f y=%.3f\n", addr, x, y))
      count = count + 1
    end
  end
end
out:close()
log("candidates="..tostring(count)) touch("ce_done")
'''



def touch(name: str) -> None:
    (FLAG / name).write_text("1", encoding="ascii")


def wait_for(name: str, timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if (FLAG / name).exists():
            return True
        time.sleep(0.2)
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ce-exe", default=CE_EXE)
    ap.add_argument("--difficulty", type=int, default=0)
    args = ap.parse_args()

    FLAG.mkdir(parents=True, exist_ok=True)
    for f in FLAG.glob("*"):
        f.unlink()
    AUTORUN_LUA.parent.mkdir(parents=True, exist_ok=True)
    AUTORUN_LUA.write_text(LUA_SOURCE, encoding="utf-8")

    cfg = Config.load()
    cfg.patch({"backend": "memory", "game": "th11"})
    env = TouhouEnv(cfg, registry=GameRegistry())
    env.start()
    env.reset(restart_process=True, difficulty=args.difficulty)
    time.sleep(3.0)
    st = env.get_state()
    print(f"[ce] 局内={st.in_game} stage={st.stage} lives={st.lives}")
    kb = env.backend.keyboard
    mem = env.backend.mem

    lives_addr = env.backend.reader.address_of("lives")
    lock = LivesLock(mem, lives_addr, value=5).start() if lives_addr else None
    if lock is not None:
        print(f"[ce] 已锁血 @ 0x{lives_addr:08X}")

    ce = subprocess.Popen([args.ce_exe])
    try:
        if not wait_for("ce_ready", 90):
            print("[ce] CE 未就绪")
            return 1
        kb.focus_window()
        time.sleep(0.3)
        kb.hold(["left", "down"]); time.sleep(2.5)
        kb.hold(["left", "shoot"]); time.sleep(1.5)
        touch("phase_A")
        if not wait_for("scan1_done", 180):
            print("[ce] 首次扫描超时")
            return 1
        touch("phase_B")
        if not wait_for("scan2_done", 180):
            print("[ce] 未变化扫描超时")
            return 1
        kb.hold(["right", "shoot"]); time.sleep(2.0)
        touch("phase_C")
        if not wait_for("ce_done", 180):
            print("[ce] 变化扫描超时")
            return 1
        kb.release_all()

        lines = (FLAG / "candidates.txt").read_text(encoding="utf-8").splitlines()
        addrs = [int(line.split()[0], 16) for line in lines if line.strip()]
        print(f"[ce] CE 候选 {len(addrs)} 个, 开始移动跟踪")

        def readf(a: int) -> float | None:
            try:
                return struct.unpack("<f", mem.read_bytes(a, 4))[0]
            except Exception:
                return None

        def snap() -> dict[int, float]:
            out = {}
            for a in addrs:
                v = readf(a)
                if v is not None:
                    out[a] = v
            return out

        cur = snap()
        for direction, cmp in (("left", "dec"), ("right", "inc"), ("left", "dec"), ("right", "inc")):
            kb.focus_window()
            kb.hold([direction, "shoot"])
            time.sleep(1.6)
            kb.release_all()
            nxt = snap()
            kept = {}
            for a, v in cur.items():
                nv = nxt.get(a)
                if nv is None or not (1 <= nv <= 400):
                    continue
                if cmp == "dec" and nv < v - 40:
                    kept[a] = nv
                if cmp == "inc" and nv > v + 40:
                    kept[a] = nv
            cur = kept
            print(f"[ce] 跟踪 {direction}({cmp}) 后剩 {len(cur)}")

        print(f"[ce] 跟踪后剩 {len(cur)} 个, 做判据")
        # 1) 贴左墙且静止 (弹幕会动)
        kb.focus_window()
        kb.hold(["left", "shoot"])
        time.sleep(2.5)
        p1 = {a: readf(a) for a in cur}
        time.sleep(0.5)
        p2 = {a: readf(a) for a in cur}
        kb.release_all()
        stage1 = {}
        for a in cur:
            v1, v2 = p1.get(a), p2.get(a)
            if v1 is None or v2 is None:
                continue
            if 1 <= v1 <= 60 and abs(v1 - v2) < 0.5:
                stage1[a] = v1
        print(f"[ce] 1)贴左墙静止: {len(stage1)}")

        # 2) 上下移动不应改变 x (自机 x 只受左右影响)
        kb.focus_window()
        kb.hold(["up", "shoot"]); time.sleep(1.5)
        u1 = {a: readf(a) for a in stage1}
        kb.hold(["down", "shoot"]); time.sleep(1.5)
        u2 = {a: readf(a) for a in stage1}
        kb.release_all()
        stage2 = {
            a: v for a, v in stage1.items()
            if u1.get(a) is not None and u2.get(a) is not None and abs(u1[a] - u2[a]) < 0.5
        }
        print(f"[ce] 2)上下不改变 x: {len(stage2)}")

        # 3) 右移后应变大 (到右墙)
        kb.focus_window()
        kb.hold(["right", "shoot"]); time.sleep(2.5)
        r = {a: readf(a) for a in stage2}
        kb.release_all()
        final = {a: r[a] for a in stage2 if r.get(a) is not None}
        print(f"[ce] 3)右移后候选 {len(final)} 个 (应到右墙 ~360):")
        for a, v in sorted(final.items())[:20]:
            print("   player_x? 0x%08X x=%.3f y+4=%.3f" % (a, v, readf(a + 4)))
        big = {a: v for a, v in final.items() if v >= 300}
        if big:
            best = sorted(big)[0]
            print(f"[ce] 建议: player_x = 0x{best:08X}")
        return 0 if big else 1
    finally:
        try:
            kb.release_all()
        except Exception:
            pass
        try:
            ce.terminate()
        except Exception:
            pass
        try:
            AUTORUN_LUA.unlink()
        except Exception:
            pass
        if lock is not None:
            lock.stop()
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
