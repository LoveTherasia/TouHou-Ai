import sys, time
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
captured = {}
try:
    env.start()
    state = env.reset(difficulty=0, character=0, shot=0).state
    N = 1200
    for i in range(N):
        rec.update(state)
        captured[rec.frame] = list(state.bullets)   # 用 recorder 的 tick 帧号做 key
        a, _ = teacher.choose(state, state.frame, "shoot")
        r = env.step(a, frames=1)
        state = r.state
        if r.done:
            break
finally:
    env.close()

# 用脚本(线性运动)重建每一帧的弹幕
spawns = rec.spawns
frames_sorted = sorted(captured)
fN = frames_sorted[-1]
sim = {F: [] for F in frames_sorted}
for e in spawns:
    for F in frames_sorted:
        if F < e.frame:
            continue
        x = e.x + e.vx * (F - e.frame)
        y = e.y + e.vy * (F - e.frame)
        if -40 <= x <= 424 and -40 <= y <= 488:
            sim[F].append((x, y))

dists = []
per_frame = []
for F in frames_sorted:
    real = np.asarray(captured[F], dtype=float).reshape(-1, 2)
    s = np.asarray(sim[F], dtype=float).reshape(-1, 2)
    if len(real) == 0:
        continue
    if len(s) == 0:
        dists.extend([9999.0] * len(real)); continue
    d = np.linalg.norm(real[:, None, :] - s[None, :, :], axis=2).min(axis=1)
    dists.extend(d.tolist())
    per_frame.append((F, len(real), len(s), float(np.median(d))))
dists = np.asarray(dists)
print("captured frames=%d  spawns=%d  dropped=%d" % (len(frames_sorted), len(spawns), rec.dropped))
print("matched real bullets=%d" % len(dists))
print("nearest-dist: median=%.2f  mean=%.2f  p90=%.2f  p99=%.2f  <=3px=%.1f%%  <=6px=%.1f%%"
      % (np.median(dists), dists.mean(), np.percentile(dists, 90),
         np.percentile(dists, 99), (dists <= 3).mean() * 100, (dists <= 6).mean() * 100))
print("\n样本帧 (frame, real数, sim数, 中位误差):")
for row in per_frame[::max(1, len(per_frame) // 12)]:
    print("  ", row)

print("\n=== 逐帧明细 (real vs sim) ===")
for F in frames_sorted[::300][:5]:
    real = [(round(x, 1), round(y, 1)) for x, y in captured[F]]
    simv = [(round(x, 1), round(y, 1)) for x, y in sim.get(F, [])]
    print("frame %d  real(%d)=%s" % (F, len(real), real[:6]))
    print("           sim(%d)=%s" % (len(simv), simv[:6]))
