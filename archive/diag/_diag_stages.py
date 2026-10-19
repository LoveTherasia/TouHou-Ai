import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from touhou_rl_mcp.config import Config
from touhou_rl_mcp.env import TouhouEnv
from touhou_rl_mcp.games.registry import GameRegistry
from touhou_rl_mcp.training.rule_teacher import RuleTeacher

seeds = [7, 11, 42]
env = TouhouEnv(Config({"backend": "mock", "game": "th11", "step_frames": 1}),
                registry=GameRegistry())
env.start()
for start in (1, 2, 3, 4, 5, 6):
    rows = []
    for seed in seeds:
        env.reset(seed=seed, start_stage=start)
        g = RuleTeacher()
        n = 0
        for i in range(60000):
            st = env.get_state()
            a, _ = g.choose(st, st.frame, "shoot")
            r = env.step(a, frames=1)
            n += 1
            if r.done or r.truncated:
                break
        st = env.get_state()
        rows.append((n, st.stage or 0, st.raw.get("spells_cleared") or 0))
    print("start_stage=%d  ticks=%s  max_stage=%s  spells=%s"
          % (start, [r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows]))
env.close()
