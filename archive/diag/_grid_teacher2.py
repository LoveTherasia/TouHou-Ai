import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from touhou_rl_mcp.config import Config
from touhou_rl_mcp.env import TouhouEnv
from touhou_rl_mcp.games.registry import GameRegistry
from touhou_rl_mcp.training.rule_teacher import RuleTeacher

SCRIPT = ".runtime/danmaku/th11_replay_full.json"
SEEDS = [1, 2]
CAP = 30000
env = TouhouEnv(Config({"backend": "mock", "game": "th11", "step_frames": 1,
                        "danmaku_script": SCRIPT}), registry=GameRegistry())
env.start()
results = []
grid = []
for radius in (28, 32, 36):
    for hw in (0.28, 0.34):
        for hy in (365, 375):
            for pf in (0.0, 8.0):
                grid.append((radius, hw, hy, pf))
for radius, hw, hy, pf in grid:
    ticks = []
    for seed in SEEDS:
        env.reset(seed=seed)
        g = RuleTeacher(radius=radius, home_weight=hw, home_y=hy, predict_frames=pf)
        n = 0
        for i in range(CAP):
            st = env.get_state()
            a, _ = g.choose(st, st.frame, "shoot")
            r = env.step(a, frames=1)
            n += 1
            if r.done or r.truncated:
                break
        ticks.append(n)
    avg = sum(ticks) / len(ticks)
    results.append((avg, radius, hw, hy, pf, ticks))
    print("r=%2d hw=%.2f hy=%3d pf=%3.0f  avg=%6.0f  %s" % (radius, hw, hy, pf, avg, ticks), flush=True)
env.close()
results.sort(reverse=True)
print("\nTOP5:")
for r in results[:5]:
    print("  avg=%6.0f  radius=%d hw=%.2f hy=%d predict=%.0f  %s" % (r[0], r[1], r[2], r[3], r[4], r[5]))
