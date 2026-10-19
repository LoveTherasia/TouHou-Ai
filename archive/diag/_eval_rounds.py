import sys, glob, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from touhou_rl_mcp.config import Config
from touhou_rl_mcp.env import TouhouEnv
from touhou_rl_mcp.games.registry import GameRegistry
from touhou_rl_mcp.training.features import encode_state
from touhou_rl_mcp.training.policy import MLPPolicy
from touhou_rl_mcp.training.trainer import build_action_table

F = dict(k_bullets=16, use_velocity=True, k_enemies=8,
         use_invincible=True, use_boss=True, use_danger_grad=True)
SEEDS = [7, 11, 42, 101]
env = TouhouEnv(Config({"backend": "mock", "game": "th11", "step_frames": 1}),
                registry=GameRegistry())
env.start()
actions = build_action_table(use_focus=True)

cands = sorted(glob.glob('.runtime/auto_long/round*/policy_round.npz')) + \
        ['.runtime/auto_long/policy_current.npz']
for path in cands:
    if not os.path.exists(path):
        continue
    try:
        p = MLPPolicy.load(path)
    except Exception as exc:
        print(path, 'load fail', exc); continue
    ticks, stages = [], []
    for seed in SEEDS:
        env.reset(seed=seed)
        n = 0
        for i in range(20000):
            st = env.get_state()
            obs = encode_state(st, **F)
            r = env.step(actions[p.greedy(obs)], frames=1)
            n += 1
            if r.done or r.truncated:
                break
        ticks.append(n)
        stages.append(env.get_state().stage or 0)
    print("%-46s avg_ticks=%6.0f best_stage=%d stage>=2=%d/%d ticks=%s"
          % (path.replace('.runtime/auto_long/', ''), sum(ticks) / len(ticks),
             max(stages), sum(1 for s in stages if s >= 2), len(stages), ticks))
env.close()
