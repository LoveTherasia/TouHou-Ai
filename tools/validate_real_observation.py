"""Read-only TH11 observation probe, or an explicitly requested short live training test.

python tools/validate_real_observation.py
python tools/validate_real_observation.py --train --steps 300
--train restarts the game, navigates menus, checks movement and runs one episode.
Artifacts go to a unique directory; existing policies are not overwritten.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from touhou_rl_mcp.config import Config
from touhou_rl_mcp.env import TouhouEnv
from touhou_rl_mcp.games.registry import GameRegistry
from touhou_rl_mcp.memory.process import ProcessMemory, find_first_pid
from touhou_rl_mcp.memory.reader import StateReader
from touhou_rl_mcp.training.features import encode_state
from touhou_rl_mcp.training.trainer import Trainer, TrainConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--steps", type=int, default=300)
    args = parser.parse_args()
    if not args.train:
        pid = find_first_pid(["th11c.exe", "th11.exe"])
        if pid is None:
            raise SystemExit("TH11 is not running")
        with ProcessMemory(pid) as mem:
            reader = StateReader(mem, GameRegistry().get("th11"))
            start = time.perf_counter()
            states = [reader.read_all() for _ in range(20)]
            print(json.dumps({"read_ms": (time.perf_counter()-start)*50,
                              "state": states[-1].to_dict(max_bullets=5)}, indent=2))
        return 0

    out = Path(".runtime/entity_validation") / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True)
    report = {"max_bullets": 0, "max_enemies": 0, "nonzero_bullet_observations": 0,
              "samples": 0, "movement": None}

    class ValidationEnv(TouhouEnv):
        def reset(self, **options):
            result = super().reset(**options)
            encode_state(result.state)
            before = result.state.player_x
            right = super().step("right", frames=12)
            left = super().step("left", frames=12)
            report["movement"] = [before, right.state.player_x, left.state.player_x]
            if right.state.player_x is None or right.state.player_x <= before + 5:
                raise RuntimeError(f"Movement check failed: {report['movement']}")
            return left

        def step(self, *values, **kwargs):
            result = super().step(*values, **kwargs)
            st = result.state
            report["samples"] += 1
            report["max_bullets"] = max(report["max_bullets"], st.bullet_count or 0)
            report["max_enemies"] = max(report["max_enemies"], st.enemies_alive or 0)
            if not result.done and not result.truncated:
                vec = encode_state(st)
                if not np.isfinite(vec).all():
                    raise RuntimeError("Non-finite features")
                report["nonzero_bullet_observations"] += int(bool(np.any(vec[13::4])))
            if st.bullets and "overlay" not in report:
                img = self.capture_frame()
                img.save(out / "game.png")
                draw = ImageDraw.Draw(img)
                sx, sy = img.width / 640, img.height / 480
                for points, color, radius in ((st.bullets, "cyan", 4),
                                              (st.enemy_positions, "red", 9),
                                              ([(st.player_x, st.player_y)], "lime", 5)):
                    for x, y in points:
                        x, y = (x+32)*sx, (y+16)*sy
                        draw.ellipse((x-radius, y-radius, x+radius, y+radius), outline=color, width=2)
                img.save(out / "overlay.png")
                report["overlay"] = str(out / "overlay.png")
                report["overlay_state"] = st.to_dict()
            return result

    cfg = Config.load()
    cfg.patch({"backend": "memory", "game": "th11", "restart_process": True,
               "slowdown": {"enabled": False}, "capture": {"prefer_printwindow": False}})
    env = ValidationEnv(cfg)
    try:
        env.start()
        trainer = Trainer(env, TrainConfig(episodes=1, max_steps=args.steps, difficulty=0,
            batch_episodes=1, hidden=(16,), capture_every=0, eval_every=0, save_every=0,
            out_dir=str(out)), env_lock=env.lock)
        trainer._loop()
        report["error"] = trainer.error
        report["steps"] = trainer.total_steps
        report["update"] = trainer._last_update
        report["policy"] = str(trainer.save()) if trainer.error is None else None
        report["success"] = (trainer.error is None and report["max_bullets"] > 0
            and report["max_enemies"] > 0 and bool(trainer._last_update)
            and all(np.isfinite(v) for v in trainer._last_update.values()))
    finally:
        env.close()
        (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "overlay_state"}, indent=2))
    print(f"Report: {out / 'report.json'}")
    return 0 if report.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
