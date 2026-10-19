import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import json

from touhou_rl_mcp.config import Config
from touhou_rl_mcp.env import TouhouEnv
from touhou_rl_mcp.games.registry import GameRegistry
from touhou_rl_mcp.training.features import encode_state, feature_dim
from touhou_rl_mcp.training.policy import MLPPolicy
from touhou_rl_mcp.training.trainer import DIRECTIONS, build_action_table
from touhou_rl_mcp.training.rule_teacher import RuleTeacher

npz = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".runtime/mock_bc4/policy_best.npz")
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 7
pol = MLPPolicy.load(npz)
meta = json.loads(npz.with_suffix(".json").read_text(encoding="utf-8"))["config"]
actions = build_action_table(use_focus=meta.get("use_focus", False))
k = meta.get("k_bullets", 16); uv = meta.get("bullet_velocity", False)
ke = meta.get("k_enemies", 0); ui = meta.get("use_invincible", False); ub = meta.get("use_boss", False)
ud = meta.get("use_danger_grad", False)

env = TouhouEnv(Config({"backend": "mock", "game": "th11", "step_frames": 1}), registry=GameRegistry())
env.start()
env.reset(seed=seed)
teacher = RuleTeacher()

def tidx(a):
    head = a.split("+shoot")[0].strip("+ ")
    d = DIRECTIONS.index(head) if head in DIRECTIONS else 0
    if meta.get("use_focus", False) and "focus" in a:
        d += len(DIRECTIONS)
    return d

agree = 0
total = 0
misses = []
last = None
for i in range(4000):
    st = env.get_state()
    obs = encode_state(st, k_bullets=k, use_velocity=uv, k_enemies=ke, use_invincible=ui, use_boss=ub, use_danger_grad=ud)
    pa = pol.greedy(obs)
    ta, _ = teacher.choose(st, st.frame, "shoot")
    ti = tidx(ta)
    total += 1
    if pa == ti:
        agree += 1
    else:
        pts = np.asarray(st.bullets).reshape(-1, 2) if st.bullets else np.zeros((0, 2))
        near = float(np.linalg.norm(pts - [st.player_x, st.player_y], axis=1).min()) if len(pts) else 9999.0
        if len(misses) < 12:
            misses.append({"step": i, "policy": actions[pa], "teacher": ta,
                           "nearest": round(near, 1), "bullets": st.bullet_count,
                           "n_bullets_seen": min(k, st.bullet_count or 0)})
    r = env.step(actions[pa], frames=1)
    if r.done or r.truncated:
        break
    last = st

print("policy=%s seed=%d  动作一致率=%.3f (%d/%d)  存活=%d ticks  结束=%s"
      % (npz.parent.name, seed, agree / max(1, total), agree, total, total,
         env.get_state().raw.get("game_over")))
print("分歧示例:")
for m in misses:
    print("  ", m)
env.close()
