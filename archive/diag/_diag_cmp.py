import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from touhou_rl_mcp.config import Config
from touhou_rl_mcp.env import TouhouEnv
from touhou_rl_mcp.games.registry import GameRegistry
from touhou_rl_mcp.training.rule_teacher import RuleTeacher


def run(env, seed, mode, sf=1):
    env.reset(seed=seed)
    g = RuleTeacher(focus_radius=0.0) if mode == "rule_nofocus" else RuleTeacher()
    tot = 0.0
    steps = 0
    for i in range(8000):
        if mode == "macro":
            r = env.step("dodge", frames=sf, radius=40)
        else:
            st = env.get_state()
            a, _ = g.choose(st, st.frame, "shoot")
            r = env.step(a, frames=sf)
        tot += r.reward
        steps += 1
        if r.done:
            break
    st = env.get_state()
    return steps, steps * sf, tot, st.stage


for sf in (1, 4):
    env = TouhouEnv(Config({"backend": "mock", "game": "th11", "step_frames": sf}),
                    registry=GameRegistry())
    env.start()
    for mode in ("macro", "rule", "rule_nofocus"):
        for seed in (7, 11):
            steps, ticks, tot, stage = run(env, seed, mode, sf)
            print("sf=%d %-12s seed=%s ticks=%5s return=%7.0f stage=%s"
                  % (sf, mode, seed, ticks, tot, stage))
    env.close()
