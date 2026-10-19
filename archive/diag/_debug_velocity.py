import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import numpy as np

from touhou_rl_mcp.config import Config
from touhou_rl_mcp.env import TouhouEnv
from touhou_rl_mcp.games.registry import GameRegistry
from touhou_rl_mcp.training.rule_teacher import RuleTeacher

cfg = Config({"backend": "memory", "game": "th11",
              "game_dir": r"F:\[th11] 东方地灵殿 (汉化版+日文版)",
              "restart_process": True, "dialogue_assist": True,
              "capture": {"prefer_printwindow": False}})
env = TouhouEnv(cfg, registry=GameRegistry())
teacher = RuleTeacher()
seq = []   # (frame, {slot: (x,y)})
try:
    env.start()
    state = env.reset(difficulty=0, character=0, shot=0).state
    for i in range(700):
        diag = state.raw.get("entities", {}).get("bullets", {})
        slots = diag.get("active_slots", []) or []
        pts = state.bullets or []
        seq.append((int(state.frame or 0), {int(s): (float(p[0]), float(p[1])) for s, p in zip(slots, pts)}))
        a, _ = teacher.choose(state, state.frame, "shoot")
        r = env.step(a, frames=1)
        state = r.state
        if r.done:
            break
finally:
    env.close()

# 找出前几个"新出现"的 slot, 打印它之后每帧的真实位移
seen = set()
shown = 0
for idx, (frame, cur) in enumerate(seq):
    for slot in cur:
        if slot in seen:
            continue
        seen.add(slot)
        if shown >= 4:
            continue
        shown += 1
        print("=== slot %d 首次出现在 frame %d, pos=%s" % (slot, frame, cur[slot]))
        prev = cur[slot]
        for j in range(1, 11):
            if idx + j >= len(seq):
                break
            f2, cur2 = seq[idx + j]
            if slot not in cur2:
                print("   frame %d: 消失" % f2); break
            p = cur2[slot]
            d = (p[0] - prev[0], p[1] - prev[1])
            print("   frame %d dt=%d pos=(%.2f,%.2f) Δ=(%.3f,%.3f)" % (f2, f2 - frame, p[0], p[1], d[0], d[1]))
            prev = p
