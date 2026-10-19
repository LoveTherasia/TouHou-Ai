import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from touhou_rl_mcp.config import Config
from touhou_rl_mcp.env import TouhouEnv
from touhou_rl_mcp.games.registry import GameRegistry
from touhou_rl_mcp.training.tactical import TacticalGuard

env = TouhouEnv(Config({"backend": "mock", "game": "th11", "step_frames": 4}),
                registry=GameRegistry())
env.start()
for seed in (7, 11, 42):
    env.reset(seed=seed)
    g = TacticalGuard()
    tot = 0.0
    steps = 0
    for i in range(4000):
        st = env.get_state()
        try:
            a, _ = g.choose(st, st.frame, "shoot+focus")
        except Exception as exc:
            print("guard error", exc)
            break
        r = env.step(a, frames=4)
        tot += r.reward
        steps += 1
        if r.done:
            break
    st = env.get_state()
    print("seed=%s guard steps=%s ticks=%s return=%.0f stage=%s spells=%s"
          % (seed, steps, steps * 4, tot, st.stage, st.raw.get("spells_cleared")))
env.close()
