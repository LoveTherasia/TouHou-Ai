import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import numpy as np

from touhou_rl_mcp.config import Config
from touhou_rl_mcp.env import TouhouEnv
from touhou_rl_mcp.games.registry import GameRegistry
from touhou_rl_mcp.memory.danmaku import DanmakuRecorder
from touhou_rl_mcp.training.rule_teacher import RuleTeacher

cfg = Config({"backend": "memory", "game": "th11",
              "game_dir": r"F:\[th11] 东方地灵殿 (汉化版+日文版)",
              "restart_process": True, "dialogue_assist": True,
              "capture": {"prefer_printwindow": False}})
env = TouhouEnv(cfg, registry=GameRegistry())
rec = DanmakuRecorder()
teacher = RuleTeacher()
seq = []
try:
    env.start()
    state = env.reset(difficulty=0, character=0, shot=0).state
    for i in range(500):
        diag = state.raw.get("entities", {}).get("bullets", {})
        slots = diag.get("active_slots", []) or []
        pts = state.bullets or []
        seq.append((i + 1, {int(s): (float(p[0]), float(p[1])) for s, p in zip(slots, pts)}))
        rec.update(state)
        a, _ = teacher.choose(state, state.frame, "shoot")
        r = env.step(a, frames=1)
        state = r.state
        if r.done:
            break
finally:
    env.close()

print("recorder.frame =", rec.frame, " 首个 spawn =", rec.spawns[0] if rec.spawns else None)
print("前 6 个 spawn:")
for e in rec.spawns[:6]:
    print("   frame=%d pos=(%.2f,%.2f) v=(%.3f,%.3f)" % (e.frame, e.x, e.y, e.vx, e.vy))

# 对前 2 个 spawn, 用真实序列核对速度
for e in rec.spawns[:2]:
    # 找到该 spawn 的 slot: 在 seq 中 frame==e.frame 且位置最接近 (e.x,e.y)
    best = None
    for frame, cur in seq:
        for slot, p in cur.items():
            d = abs(p[0] - e.x) + abs(p[1] - e.y)
            if d < 1.0:
                best = slot; break
        if best is not None:
            break
    print("\n=== spawn frame=%d slot=%s 记录 v=(%.3f,%.3f)" % (e.frame, best, e.vx, e.vy))
    if best is None:
        print("   未找到对应 slot"); continue
    prev = None
    for frame, cur in seq:
        if frame < e.frame or frame > e.frame + 60:
            continue
        if best in cur:
            p = cur[best]
            d = "" if prev is None else " Δ=(%.3f,%.3f)" % (p[0] - prev[0], p[1] - prev[1])
            print("   frame %d pos=(%.2f,%.2f)%s" % (frame, p[0], p[1], d))
            prev = p
