import sys, os
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

SCRIPT = ".runtime/danmaku/th11_replay_full.json"
F = dict(k_bullets=16, use_velocity=True, k_enemies=8,
         use_invincible=True, use_boss=True, use_danger_grad=True)
POLICY = ".runtime/pd_new_teacher/policy_best.npz"
env = TouhouEnv(Config({"backend": "mock", "game": "th11", "step_frames": 1,
                        "danmaku_script": SCRIPT}), registry=GameRegistry())
env.start()
actions = build_action_table(use_focus=True)
pol = MLPPolicy.load(POLICY)


def tidx(a):
    head = a.split("+shoot")[0].strip("+ ")
    d = DIRECTIONS.index(head) if head in DIRECTIONS else 0
    if "focus" in a:
        d += len(DIRECTIONS)
    return d


def rollout(controller):
    """controller: 'teacher' 或 'policy'。返回 (agreement, n, 按最近弹距离分桶的一致率, 存活)"""
    env.reset(seed=1)
    g = RuleTeacher()
    agree = 0
    n = 0
    bins = {}   # dist_bin -> [agree, total]
    for i in range(40000):
        st = env.get_state()
        t_action = tidx(g.choose(st, st.frame, "shoot")[0])
        p_action = pol.greedy(encode_state(st, **F))
        near = 9999.0
        if st.bullets:
            arr = np.asarray(st.bullets, float).reshape(-1, 2)
            near = float(np.hypot(arr[:, 0] - st.player_x, arr[:, 1] - st.player_y).min())
        b = int(min(6, near // 10)) * 10
        e = bins.setdefault(b, [0, 0])
        e[1] += 1
        if t_action == p_action:
            e[0] += 1
        n += 1
        if controller == "teacher":
            agree += int(t_action == p_action)
            a = t_action
        else:
            agree += int(t_action == p_action)
            a = p_action
        r = env.step(actions[a], frames=1)
        if r.done or r.truncated:
            break
    return agree / max(1, n), n, bins


for ctrl in ("teacher", "policy"):
    rate, n, bins = rollout(ctrl)
    print("=== %s 驱动的轨迹 (存活 %d 步): 与老师逐帧一致率 = %.1f%%" % (ctrl, n, 100 * rate))
    for b in sorted(bins):
        a, t = bins[b]
        label = "%d-%dpx" % (b, b + 10) if b < 60 else ">=60px"
        print("     最近弹 %-8s 一致率 %.0f%%  (n=%d)" % (label, 100 * a / t, t))
env.close()
