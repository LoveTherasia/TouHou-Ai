import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from touhou_rl_mcp.config import Config
from touhou_rl_mcp.env import TouhouEnv
from touhou_rl_mcp.games.registry import GameRegistry
from touhou_rl_mcp.training.rule_teacher import RuleTeacher

for sf in (1, 2, 4):
    env = TouhouEnv(Config({"backend": "mock", "game": "th11", "step_frames": sf}),
                    registry=GameRegistry())
    env.start()
    for seed in (7, 11, 42):
        env.reset(seed=seed)
        g = RuleTeacher()
        tot = 0.0
        steps = 0
        for i in range(6000):
            st = env.get_state()
            a, _ = g.choose(st, st.frame, "shoot")
            r = env.step(a, frames=sf)
            tot += r.reward
            steps += 1
            if r.done:
                break
        st = env.get_state()
        print("sf=%d seed=%s steps=%s ticks=%s return=%.0f stage=%s spells=%s"
              % (sf, seed, steps, steps * sf, tot, st.stage, st.raw.get("spells_cleared")))
    env.close()
