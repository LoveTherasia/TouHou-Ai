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
SEEDS = [1, 2, 3]
MAX = 20000
POLICY = sys.argv[1] if len(sys.argv) > 1 else ".runtime/mock_full_replay/policy_best.npz"

env = TouhouEnv(Config({"backend": "mock", "game": "th11", "step_frames": 1,
                        "danmaku_script": SCRIPT}), registry=GameRegistry())
env.start()
actions = build_action_table(use_focus=True)
teacher = RuleTeacher()
pol = MLPPolicy.load(POLICY) if os.path.exists(POLICY) else None
print("策略:", POLICY, "载入" if pol else "无")

for name in ("teacher", "policy"):
    ticks = []
    for seed in SEEDS:
        env.reset(seed=seed)
        n = 0
        for i in range(MAX):
            st = env.get_state()
            if name == "teacher":
                a, _ = teacher.choose(st, st.frame, "shoot")
            else:
                obs = encode_state(st, **F)
                a = actions[pol.greedy(obs)]
            r = env.step(a, frames=1)
            n += 1
            if r.done or r.truncated:
                break
        ticks.append(n)
    print("%-8s 存活步数=%s 均值=%.0f (max=%d)" % (name, ticks, sum(ticks)/len(ticks), MAX))
env.close()
