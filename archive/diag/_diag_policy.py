import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from touhou_rl_mcp.config import Config
from touhou_rl_mcp.env import TouhouEnv
from touhou_rl_mcp.games.registry import GameRegistry
from touhou_rl_mcp.training.features import encode_state, feature_dim
from touhou_rl_mcp.training.policy import MLPPolicy
from touhou_rl_mcp.training.trainer import build_action_table
import json

path = sys.argv[1]
policy = MLPPolicy.load(path)
meta = json.loads(Path(path).with_suffix(".json").read_text(encoding="utf-8"))["config"]
actions = build_action_table(use_focus=meta.get("use_focus", False), use_bomb=meta.get("use_bomb", False))
k = meta.get("k_bullets", 16); uv = meta.get("bullet_velocity", False)
ke = meta.get("k_enemies", 0); ui = meta.get("use_invincible", False); ub = meta.get("use_boss", False)
assert policy.obs_dim == feature_dim(k, uv, ke, ui, ub), (policy.obs_dim, feature_dim(k, uv, ke, ui, ub))
sf = int(sys.argv[2]) if len(sys.argv) > 2 else 1
env = TouhouEnv(Config({"backend": "mock", "game": "th11", "step_frames": sf}), registry=GameRegistry())
env.start()
from collections import Counter
for seed in (7, 11, 42, 101):
    env.reset(seed=seed)
    tot = 0.0; steps = 0; cnt = Counter()
    for i in range(20000):
        st = env.get_state()
        obs = encode_state(st, k_bullets=k, use_velocity=uv, k_enemies=ke, use_invincible=ui, use_boss=ub)
        a = actions[policy.greedy(obs)]
        cnt[a] += 1
        r = env.step(a, frames=sf)
        tot += r.reward; steps += 1
        if r.done:
            break
    st = env.get_state()
    print("seed=%s ticks=%5s return=%7.0f stage=%s spells=%s acts=%s"
          % (seed, steps * sf, tot, st.stage, st.raw.get("spells_cleared"), dict(cnt.most_common(3))))
env.close()
