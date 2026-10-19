import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from touhou_rl_mcp.config import Config
from touhou_rl_mcp.env import TouhouEnv
from touhou_rl_mcp.games.registry import GameRegistry
from touhou_rl_mcp.training.features import encode_state
from touhou_rl_mcp.training.policy import MLPPolicy
from touhou_rl_mcp.training.trainer import build_action_table
from touhou_rl_mcp.training.rule_teacher import RuleTeacher

SCRIPT = ".runtime/danmaku/th11_replay_full.json"
F = dict(k_bullets=16, use_velocity=True, k_enemies=8,
         use_invincible=True, use_boss=True, use_danger_grad=True)
SEEDS = [1, 2, 3, 4, 5]
CAP = 30000
env = TouhouEnv(Config({"backend": "mock", "game": "th11", "step_frames": 1,
                        "danmaku_script": SCRIPT}), registry=GameRegistry())
env.start()
actions = build_action_table(use_focus=True)

cands = [
    ("老师 RuleTeacher", None),
    ("纯DAgger", ".runtime/mock_full_replay/policy_best.npz"),
    ("真人BC", ".runtime/bc_policy.npz"),
    ("真人BC+DAgger", ".runtime/bc_then_dagger/policy_best.npz"),
]
for name, path in cands:
    pol = MLPPolicy.load(path) if path and os.path.exists(path) else None
    if path and pol is None:
        print(name, "(missing)"); continue
    ticks = []
    for seed in SEEDS:
        env.reset(seed=seed)
        g = RuleTeacher()
        n = 0
        for i in range(CAP):
            st = env.get_state()
            if pol is None:
                a, _ = g.choose(st, st.frame, "shoot")
            else:
                a = actions[pol.greedy(encode_state(st, **F))]
            r = env.step(a, frames=1)
            n += 1
            if r.done or r.truncated:
                break
        ticks.append(n)
    print("%-18s 存活均值=%6.0f  %s" % (name, sum(ticks) / len(ticks), ticks), flush=True)
env.close()
