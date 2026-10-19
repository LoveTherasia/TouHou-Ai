import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import numpy as np

from touhou_rl_mcp.config import Config
from touhou_rl_mcp.env import TouhouEnv
from touhou_rl_mcp.games.registry import GameRegistry
from touhou_rl_mcp.training.rule_teacher import RuleTeacher

SCRIPT = ".runtime/danmaku/th11_replay_full.json"
SEEDS = [1, 2]
CAP = 30000
env = TouhouEnv(Config({"backend": "mock", "game": "th11", "step_frames": 1,
                        "danmaku_script": SCRIPT}), registry=GameRegistry())
env.start()
rows = []
grid = []
for hw in (0.45, 0.60, 0.80):
    for aim in (0.0, 30.0):
        for amp in (70.0, 100.0):
            grid.append((hw, aim, amp))
for hw, aim, amp in grid:
    res, aligns, dmgs = [], [], []
    for seed in SEEDS:
        env.reset(seed=seed)
        g = RuleTeacher(boss_home=True, aim_safe=aim, home_weight=hw, strafe_amp=amp,
                        strafe_period=150)
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
            a, _ = g.choose(st, st.frame, "shoot")
            r = env.step(a, frames=1)
            n += 1
            if r.done or r.truncated:
                break
        res.append(n); aligns.append(100 * al / max(1, tot)); dmgs.append(dmg)
    rows.append((np.mean(res), np.mean(dmgs), np.mean(aligns), hw, aim, amp))
    print("hw=%.2f aim=%3.0f amp=%3.0f -> 存活=%6.0f 伤害=%6.0f 对准=%.0f%%"
          % (hw, aim, amp, np.mean(res), np.mean(dmgs), np.mean(aligns)), flush=True)
env.close()
rows.sort(key=lambda r: (r[0], r[1]), reverse=True)
print("\n按(存活, 伤害)排序 TOP:")
for r in rows[:5]:
    print("  存活=%6.0f 伤害=%6.0f 对准=%.0f%%  hw=%.2f aim=%.0f amp=%.0f" % r)
