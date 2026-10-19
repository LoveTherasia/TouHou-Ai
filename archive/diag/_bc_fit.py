import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np

from touhou_rl_mcp.config import Config
from touhou_rl_mcp.env import TouhouEnv
from touhou_rl_mcp.games.registry import GameRegistry
from touhou_rl_mcp.training.features import encode_state
from touhou_rl_mcp.training.policy import MLPPolicy
from touhou_rl_mcp.training.trainer import DIRECTIONS, build_action_table
from touhou_rl_mcp.training.rule_teacher import RuleTeacher

env = TouhouEnv(Config({"backend": "mock", "game": "th11", "step_frames": 1}),
                registry=GameRegistry())
env.start()
teacher = RuleTeacher()
X = []
A = []
for ep in range(25):
    env.reset(seed=1000 + ep)
    for i in range(4000):
        st = env.get_state()
        obs = encode_state(st, k_bullets=16, use_velocity=True, k_enemies=8,
                           use_invincible=True, use_boss=True)
        a_str, _ = teacher.choose(st, st.frame, "shoot")
        head = a_str.split("+shoot")[0].strip("+ ")
        idx = DIRECTIONS.index(head) if head in DIRECTIONS else 0
        X.append(obs)
        A.append(idx)
        r = env.step(a_str, frames=1)
        if r.done:
            break
env.close()
X = np.asarray(X)
A = np.asarray(A)
print("samples", X.shape, "action dist", np.bincount(A, minlength=9).tolist())

policy = MLPPolicy(X.shape[1], 9, hidden=(256, 256), seed=0, lr=2e-3, entropy_coef=0.0)
for epoch in range(241):
    idx = np.random.default_rng(epoch).permutation(len(X))
    for s in range(0, len(X), 512):
        b = idx[s:s + 512]
        policy.imitate_batch(X[b], A[b])
    if epoch % 20 == 0:
        logp, _, _ = policy.evaluate(X, A)
        acc = float((policy.logits_batch(X).argmax(1) == A).mean())
        print("epoch %3d loss %.4f acc %.4f" % (epoch, -float(logp.mean()), acc))
