import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import numpy as np

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
SEEDS = [1, 2, 3, 4]
CAP = 40000
env = TouhouEnv(Config({"backend": "mock", "game": "th11", "step_frames": 1,
                        "danmaku_script": SCRIPT}), registry=GameRegistry())
env.start()
actions = build_action_table(use_focus=True)

OLD_TEACHER = dict(boss_home=True, aim_safe=70.0, home_weight=0.30, strafe_amp=70.0)
NEW_TEACHER = dict(boss_home=True, aim_safe=0.0, home_weight=0.80, strafe_amp=100.0)
cands = [
    ("旧老师(中线)", "teacher", OLD_TEACHER),
    ("新老师(追Boss)", "teacher", NEW_TEACHER),
    ("旧策略(bc_then_dagger)", ".runtime/bc_then_dagger/policy_best.npz", None),
    ("新策略(pd_new_teacher)", ".runtime/pd_new_teacher/policy_best.npz", None),
]
for name, kind, kw in cands:
    pol = None
    if kind != "teacher":
        if not os.path.exists(kind):
            print(name, "(missing)"); continue
        pol = MLPPolicy.load(kind)
    res, aligns, dmgs = [], [], []
    for seed in SEEDS:
        env.reset(seed=seed)
        g = RuleTeacher(**kw) if kw else None
        n = 0; al = 0; tot = 0; dmg = 0.0; last = None
        for i in range(CAP):
            st = env.get_state()
            bp = st.raw.get("boss_position")
            if bp:
                tot += 1
                if abs((st.player_x or 0) - bp[0]) < 22:
                    al += 1
            if last is not None and st.boss_hp is not None:
                dmg += max(0, last - st.boss_hp)
            if st.boss_hp is not None:
                last = st.boss_hp
            if pol is None:
                a, _ = g.choose(st, st.frame, "shoot")
            else:
                a = actions[pol.greedy(encode_state(st, **F))]
            r = env.step(a, frames=1)
            n += 1
            if r.done or r.truncated:
                break
        res.append(n); aligns.append(100 * al / max(1, tot)); dmgs.append(dmg)
    print("%-26s 存活=%-6.0f 对准=%.0f%% 伤害=%.0f  %s"
          % (name, np.mean(res), np.mean(aligns), np.mean(dmgs), res), flush=True)
env.close()
