import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from touhou_rl_mcp.config import Config
from touhou_rl_mcp.env import TouhouEnv
from touhou_rl_mcp.games.registry import GameRegistry
from touhou_rl_mcp.training.rule_teacher import RuleTeacher

variants = {
    "base":   dict(),
    "pred8":  dict(predict_frames=8),
    "pred12": dict(predict_frames=12),
    "pred20": dict(predict_frames=20),
}
seeds = [7, 11, 42, 101, 202]

env = TouhouEnv(Config({"backend": "mock", "game": "th11", "step_frames": 1}),
                registry=GameRegistry())
env.start()
for name, kw in variants.items():
    rows = []
    for seed in seeds:
        env.reset(seed=seed)
        g = RuleTeacher(**kw)
        steps = 0
        for i in range(30000):
            st = env.get_state()
            a, _ = g.choose(st, st.frame, "shoot")
            r = env.step(a, frames=1)
            steps += 1
            if r.done:
                break
        st = env.get_state()
        rows.append((seed, steps, st.stage or 0, st.raw.get("spells_cleared") or 0))
    avg_ticks = sum(r[1] for r in rows) / len(rows)
    best_stage = max(r[2] for r in rows)
    cleared = sum(1 for r in rows if r[2] >= 2)
    print("%-8s avg_ticks=%6.0f best_stage=%s stage1_clear=%d/%d  %s"
          % (name, avg_ticks, best_stage, cleared, len(rows), rows))
env.close()
