"""多种子对比: 最近K弹 vs 单尺度危险图 (消除"种子运气"带来的误判)。

包含训练器内部评估用的那批种子 (config.seed + index*1000), 便于与日志里的
"最佳评估"数字对齐。

    python .runtime/_cmp_big.py
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
from touhou_rl_mcp.training.trainer import build_action_table  # noqa: E402

SCRIPT = ".runtime/danmaku/th11_replay_full.json"
CAP = 40000
#: 普通种子 + 训练器内部评估种子 (20261018 + index*1000, index=25/50/75/100)
SEEDS = [1, 2, 3, 4, 5, 6, 7, 8,
         20261018 + 25000, 20261018 + 50000,
         20261018 + 75000, 20261018 + 100000]


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
    cands = [("最近K弹 pd_new_teacher", *load_policy(".runtime/pd_new_teacher/policy_best")),
             ("单尺度危险图 pd_grid", *load_policy(".runtime/pd_grid/policy_best")),
             ("多尺度危险图 pd_grid_ms", *load_policy(".runtime/pd_grid_ms/policy_best"))]
    out = {}
    for name, pol, enc, actions in cands:
        res = []
        for seed in SEEDS:
            env.reset(seed=seed)
            n = 0
            for _ in range(CAP):
                st = env.get_state()
                a = actions[pol.greedy(enc(st))]
                r = env.step(a, frames=1)
                n += 1
                if r.done or r.truncated:
                    break
            res.append(n)
        arr = np.array(res, float)
        out[name] = {"per_seed": res, "mean": float(arr.mean()),
                     "median": float(np.median(arr)), "max": float(arr.max()),
                     "min": float(arr.min()), "over_20k": int((arr >= 20000).sum())}
        print("%-24s 均=%7.0f 中位=%7.0f 最大=%7.0f 最小=%7.0f  >=20k: %d/%d"
              % (name, arr.mean(), np.median(arr), arr.max(), arr.min(),
                 (arr >= 20000).sum(), len(arr)), flush=True)
        print("   逐种子:", [int(x) for x in res], flush=True)
    env.close()
    Path(".runtime/_cmp_big.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                              encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
