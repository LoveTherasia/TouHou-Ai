import sys, os
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
SEEDS = [7, 11, 42, 101, 202]
# 用"内置弹幕"(更难) 评估, 更能区分策略强弱; 可用第 2 个参数传脚本
script = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] != "-" else ""
paths = sys.argv[2:] if len(sys.argv) > 1 and sys.argv[1] != "-" else sys.argv[1:]
if not paths:
    paths = [".runtime/auto_long/policy_current.npz", ".runtime/mock_pd1/policy_best.npz"]

env = TouhouEnv(Config({"backend": "mock", "game": "th11", "step_frames": 1,
                        "danmaku_script": script}), registry=GameRegistry())
env.start()
actions = build_action_table(use_focus=True)
print("eval danmaku:", script or "(内置弹型)")
for path in paths:
    if not os.path.exists(path):
        print("%-44s (missing)" % path); continue
    p = MLPPolicy.load(path)
    ticks, stages = [], []
    for seed in SEEDS:
        env.reset(seed=seed)
        n = 0
        for i in range(30000):
            st = env.get_state()
            obs = encode_state(st, **F)
            r = env.step(actions[p.greedy(obs)], frames=1)
            n += 1
            if r.done or r.truncated:
                break
        ticks.append(n)
        stages.append(env.get_state().stage or 0)
    print("%-44s avg_ticks=%6.0f best_stage=%d stage>=2=%d/%d ticks=%s"
          % (os.path.basename(os.path.dirname(path)) + "/" + os.path.basename(path),
             sum(ticks) / len(ticks), max(stages),
             sum(1 for s in stages if s >= 2), len(stages), ticks))
env.close()
