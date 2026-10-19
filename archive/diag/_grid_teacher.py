import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from touhou_rl_mcp.config import Config
from touhou_rl_mcp.env import TouhouEnv
from touhou_rl_mcp.games.registry import GameRegistry
from touhou_rl_mcp.training.rule_teacher import RuleTeacher

seeds = [7, 11, 42, 101, 202]
env = TouhouEnv(Config({"backend": "mock", "game": "th11", "step_frames": 1}),
                registry=GameRegistry())
env.start()

results = []
for radius in (30, 34, 38):
    for hw in (0.30, 0.35):
        for hy in (370, 380):
            ticks = []
            clears = 0
            for seed in seeds:
                env.reset(seed=seed)
                g = RuleTeacher(planner="force", radius=radius, home_weight=hw, home_y=hy)
                n = 0
                for i in range(30000):
                    st = env.get_state()
                    a, _ = g.choose(st, st.frame, "shoot")
                    r = env.step(a, frames=1)
                    n += 1
                    if r.done:
                        break
                ticks.append(n)
                if (env.get_state().stage or 0) >= 2:
                    clears += 1
            avg = sum(ticks) / len(ticks)
            results.append((avg, clears, radius, hw, hy, ticks))
            print("radius=%2d hw=%.2f hy=%3d avg=%6.0f clears=%d %s"
                  % (radius, hw, hy, avg, clears, ticks), flush=True)

results.sort(reverse=True)
print("\nTOP5:")
for r in results[:5]:
    print("  avg=%6.0f clears=%d radius=%d hw=%.2f hy=%d" % (r[0], r[1], r[2], r[3], r[4]))
env.close()
