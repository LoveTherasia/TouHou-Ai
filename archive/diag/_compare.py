import sys
import json
from pathlib import Path
from collections import Counter
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from touhou_rl_mcp.config import Config
from touhou_rl_mcp.env import TouhouEnv
from touhou_rl_mcp.games.registry import GameRegistry
from touhou_rl_mcp.training.features import encode_state, feature_dim
from touhou_rl_mcp.training.policy import MLPPolicy
from touhou_rl_mcp.training.trainer import build_action_table
from touhou_rl_mcp.training.rule_teacher import RuleTeacher

SEEDS = [7, 11, 42, 101, 202, 303]


def evaluate(policy, actions, meta, env):
    k = meta.get("k_bullets", 16); uv = meta.get("bullet_velocity", False)
    ke = meta.get("k_enemies", 0); ui = meta.get("use_invincible", False)
    ub = meta.get("use_boss", False); ud = meta.get("use_danger_grad", False)
    ticks = []
    stages = []
    for seed in SEEDS:
        env.reset(seed=seed)
        n = 0
        for i in range(20000):
            st = env.get_state()
            obs = encode_state(st, k_bullets=k, use_velocity=uv, k_enemies=ke,
                               use_invincible=ui, use_boss=ub, use_danger_grad=ud)
            r = env.step(actions[policy.greedy(obs)], frames=1)
            n += 1
            if r.done or r.truncated:
                break
        ticks.append(n)
        stages.append(env.get_state().stage or 0)
    return ticks, stages


def teacher_ticks(env):
    ticks = []
    stages = []
    for seed in SEEDS:
        env.reset(seed=seed)
        g = RuleTeacher()
        n = 0
        for i in range(20000):
            st = env.get_state()
            a, _ = g.choose(st, st.frame, "shoot")
            r = env.step(a, frames=1)
            n += 1
            if r.done or r.truncated:
                break
        ticks.append(n)
        stages.append(env.get_state().stage or 0)
    return ticks, stages


env = TouhouEnv(Config({"backend": "mock", "game": "th11", "step_frames": 1}),
                registry=GameRegistry())
env.start()

print("%-22s %8s %8s %s" % ("candidate", "avg_ticks", "max_stage", "stage>=2"))
tt, ts = teacher_ticks(env)
print("%-22s %8.0f %8d %d/%d  ticks=%s"
      % ("RuleTeacher(规则老师)", sum(tt) / len(tt), max(ts), sum(1 for s in ts if s >= 2), len(ts), tt))

for p in sys.argv[1:]:
    npz = Path(p) if p.endswith(".npz") else Path(p + ".npz")
    if not npz.exists():
        print("%-22s (missing)" % npz.name)
        continue
    pol = MLPPolicy.load(npz)
    meta = json.loads(npz.with_suffix(".json").read_text(encoding="utf-8"))["config"]
    actions = build_action_table(use_focus=meta.get("use_focus", False),
                                 use_bomb=meta.get("use_bomb", False))
    ticks, stages = evaluate(pol, actions, meta, env)
    print("%-22s %8.0f %8d %d/%d  ticks=%s"
          % (npz.parent.name, sum(ticks) / len(ticks), max(stages),
             sum(1 for s in stages if s >= 2), len(stages), ticks))
env.close()
