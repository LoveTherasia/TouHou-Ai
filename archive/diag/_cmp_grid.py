"""对比: 危险图策略 vs 最近K弹策略 vs 新老师 (同种子 / 同弹幕 / 同口径)。

自动从 policy_*.json 读出观测配置, 所以"危险图策略"和"最近K弹策略"都能正确编码。

    python .runtime/_cmp_grid.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np  # noqa: E402

from touhou_rl_mcp.config import Config  # noqa: E402
from touhou_rl_mcp.env import TouhouEnv  # noqa: E402
from touhou_rl_mcp.games.registry import GameRegistry  # noqa: E402
from touhou_rl_mcp.training.features import ObservationEncoder  # noqa: E402
from touhou_rl_mcp.training.policy import MLPPolicy  # noqa: E402
from touhou_rl_mcp.training.rule_teacher import RuleTeacher  # noqa: E402
from touhou_rl_mcp.training.trainer import build_action_table  # noqa: E402

SCRIPT = ".runtime/danmaku/th11_replay_full.json"
SEEDS = [1, 2, 3, 4]
CAP = 40000


def load_policy(path: str):
    npz = Path(path).with_suffix(".npz")
    pol = MLPPolicy.load(npz)
    meta_path = npz.with_suffix(".json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")).get("config", {}) if meta_path.exists() else {}
    enc = ObservationEncoder.from_meta(meta)
    actions = build_action_table(use_focus=bool(meta.get("use_focus", True)),
                                 use_bomb=bool(meta.get("use_bomb", False)))
    assert pol.obs_dim == enc.dim, (path, pol.obs_dim, enc.dim)
    return pol, enc, actions


def main() -> int:
    env = TouhouEnv(Config({"backend": "mock", "game": "th11", "step_frames": 1,
                            "danmaku_script": SCRIPT}), registry=GameRegistry())
    env.start()
    cands = [("新老师(追Boss)", None, None, None),
             ("最近K弹策略 pd_new_teacher", *load_policy(".runtime/pd_new_teacher/policy_best")),
             ("危险图策略 pd_grid", *load_policy(".runtime/pd_grid/policy_best"))]
    rows = []
    for name, pol, enc, actions in cands:
        if enc is not None:
            print(f"[cmp] {name}: obs={pol.obs_dim} {enc}", flush=True)
        res, aligns, dmgs = [], [], []
        for seed in SEEDS:
            env.reset(seed=seed)
            g = RuleTeacher() if pol is None else None
            n = 0
            al = 0
            tot = 0
            dmg = 0.0
            last = None
            for _ in range(CAP):
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
                a = (g.choose(st, st.frame, "shoot")[0] if pol is None
                     else actions[pol.greedy(enc(st))])
                r = env.step(a, frames=1)
                n += 1
                if r.done or r.truncated:
                    break
            res.append(n)
            aligns.append(100 * al / max(1, tot))
            dmgs.append(dmg)
        rows.append((name, np.mean(res), np.mean(aligns), np.mean(dmgs), res))
        print("%-28s 存活=%-7.0f 对准=%3.0f%% 伤害=%-7.0f %s"
              % (name, np.mean(res), np.mean(aligns), np.mean(dmgs), res), flush=True)
    env.close()
    print("\n===== 汇总 (真机回放弹幕, %d 种子) =====" % len(SEEDS))
    for name, s, a, d, res in rows:
        print("  %-28s 存活=%7.0f 对准=%3.0f%% 伤害=%7.0f" % (name, s, a, d))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
