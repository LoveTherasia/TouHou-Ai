"""Continue a preserved mock checkpoint separately on each real TH11 loadout.

Short exploratory training, not an evaluation of clear rates. Restarts TH11 six
times. No tactical overrides: REINFORCE must train on actions actually sampled.
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from touhou_rl_mcp.config import Config
from touhou_rl_mcp.env import TouhouEnv
from touhou_rl_mcp.training.trainer import Trainer, TrainConfig


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy", default=".runtime/mock_clear5/policy_best.npz")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--start-character", type=int, choices=[0, 1], default=0)
    args = ap.parse_args()
    root = Path(".runtime/loadout_training") / datetime.now().strftime("%Y%m%d_%H%M%S")
    root.mkdir(parents=True)
    results = []
    for character in range(args.start_character, 2):
        for shot in range(3):
            name = f"{'reimu' if character == 0 else 'marisa'}_{'ABC'[shot]}"
            seen = {"dialogue_pulses": 0, "max_bullets": 0, "requested": [character, shot]}

            class CheckedEnv(TouhouEnv):
                def reset(self, **options):
                    result = super().reset(**options)
                    seen["actual"] = [result.state.char_main, result.state.char_sub]
                    if seen["actual"] != seen["requested"]:
                        raise RuntimeError(f"Loadout mismatch {seen}")
                    return result

                def step(self, *a, **kw):
                    result = super().step(*a, **kw)
                    seen["dialogue_pulses"] += int(result.info.get("dialogue_assist", False))
                    seen["max_bullets"] = max(seen["max_bullets"], result.state.bullet_count or 0)
                    return result

            cfg = Config.load()
            cfg.patch({"backend": "memory", "game": "th11", "restart_process": True,
                       "slowdown": {"enabled": False},
                       "reward": {"boss_damage_scale": 0.01, "spell_capture_bonus": 20}})
            env = CheckedEnv(cfg)
            tc = TrainConfig(episodes=args.episodes, max_steps=args.steps, character=character,
                shot=shot, difficulty=0, lr=0.0001, entropy_coef=0.03,
                batch_episodes=1, eval_every=0, save_every=0, capture_every=0,
                out_dir=str(root/name))
            trainer = Trainer(env, tc, env_lock=env.lock)
            trainer.load(args.policy)
            trainer.policy.lr = tc.lr
            trainer.policy.entropy_coef = tc.entropy_coef
            before = [w.copy() for w in trainer.policy.W]
            try:
                trainer._loop()
                seen.update(name=name, error=trainer.error, steps=trainer.total_steps,
                    episodes=[e.to_dict() for e in trainer.episodes], update=trainer._last_update,
                    weights_changed=any(not np.array_equal(a,b) for a,b in zip(before,trainer.policy.W)))
                if not trainer.error:
                    seen["policy"] = str(trainer.save())
            finally:
                env.close()
            results.append(seen)
            (root/"report.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
            print(json.dumps(seen, ensure_ascii=True), flush=True)
            if trainer.error:
                print(f"Stopped at {name}: inspect menu selection before continuing", flush=True)
                return 1
    print(f"REPORT {root/'report.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
