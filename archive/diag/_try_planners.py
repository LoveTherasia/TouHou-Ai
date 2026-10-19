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
cands = [
    ("force 默认", dict(planner="force")),
    ("horizon", dict(planner="horizon")),
    ("horizon+slow", dict(planner="horizon", speed=2.0)),
    ("horizon r=34", dict(planner="horizon", radius=34)),
    ("horizon hw=0.5", dict(planner="horizon", home_weight=0.5)),
]
for name, kw in cands:
    ticks = []
    for seed in SEEDS:
        env.reset(seed=seed)
        g = RuleTeacher(**kw)
        n = 0
        for i in range(CAP):
            st = env.get_state()
            a, _ = g.choose(st, st.frame, "shoot")
            r = env.step(a, frames=1)
            n += 1
            if r.done or r.truncated:
                break
        ticks.append(n)
    print("%-16s avg=%6.0f  %s" % (name, sum(ticks) / len(ticks), ticks), flush=True)
env.close()
