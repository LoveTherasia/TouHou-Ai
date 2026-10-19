import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np

from touhou_rl_mcp.config import Config
from touhou_rl_mcp.env import TouhouEnv
from touhou_rl_mcp.games.registry import GameRegistry
from touhou_rl_mcp.training.rule_teacher import RuleTeacher
from touhou_rl_mcp.types import GameState


def truncate(state, k):
    """只保留离自机最近的 k 颗弹 (模拟策略的有限视野)。"""
    if not state.bullets or k <= 0:
        state.bullets = []
        return state
    arr = np.asarray(state.bullets, dtype=float).reshape(-1, 2)
    px = state.player_x if state.player_x is not None else 192.0
    py = state.player_y if state.player_y is not None else 416.0
    d = np.hypot(arr[:, 0] - px, arr[:, 1] - py)
    idx = np.argsort(d)[:k]
    state.bullets = [tuple(arr[i]) for i in idx]
    return state


seeds = [7, 11, 42, 101, 202]
env = TouhouEnv(Config({"backend": "mock", "game": "th11", "step_frames": 1}), registry=GameRegistry())
env.start()
for k in (8, 16, 24, 32, 64, 10 ** 9):
    ticks = []
    stages = []
    for seed in seeds:
        env.reset(seed=seed)
        g = RuleTeacher()
        n = 0
        for i in range(20000):
            st = env.get_state()
            st = truncate(st, k)
            a, _ = g.choose(st, st.frame, "shoot")
            r = env.step(a, frames=1)
            n += 1
            if r.done or r.truncated:
                break
        ticks.append(n)
        stages.append(env.get_state().stage or 0)
    print("teacher 视野 k=%-4s avg_ticks=%7.0f max_stage=%d 过第一面=%d/%d  ticks=%s"
          % (k if k < 10 ** 9 else "all", sum(ticks) / len(ticks), max(stages),
             sum(1 for s in stages if s >= 2), len(stages), ticks))
env.close()
