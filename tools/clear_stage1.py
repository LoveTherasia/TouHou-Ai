"""TH11 stage-one evaluation: preserved mock policy plus optional tactical guard.

Runs real input, restarts the game, never continues after game over. Each attempt
has an isolated JSONL trace/report. Stage >= 2 is the success criterion.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from touhou_rl_mcp.config import Config
from touhou_rl_mcp.env import TouhouEnv
from touhou_rl_mcp.training.features import encode_state, feature_dim
from touhou_rl_mcp.training.policy import MLPPolicy
from touhou_rl_mcp.training.trainer import build_action_table
from touhou_rl_mcp.training.tactical import TacticalGuard


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy", default=".runtime/mock_clear5/policy_best.npz")
    ap.add_argument("--mode", choices=["policy", "guard"], default="policy")
    ap.add_argument("--max-steps", type=int, default=12000)
    ap.add_argument("--resume", action="store_true", help="继续当前测试局面，不重启；请勿用于标题 Demo")
    ap.add_argument("--step-frames", type=int, default=4)
    ap.add_argument("--bomb-radius", type=float, default=22)
    ap.add_argument("--difficulty", type=int, default=0)
    ap.add_argument("--character", type=int, choices=[0, 1], default=0)
    ap.add_argument("--shot", type=int, choices=[0, 1, 2], default=0)
    args = ap.parse_args()
    policy_path = Path(args.policy).with_suffix(".npz")
    policy = MLPPolicy.load(policy_path)
    meta = json.loads(policy_path.with_suffix(".json").read_text(encoding="utf-8"))["config"]
    actions = build_action_table(use_focus=meta.get("use_focus", False),
                             use_bomb=meta.get("use_bomb", False))
    k = meta.get("k_bullets", 16)
    use_vel = bool(meta.get("bullet_velocity", False))
    ke = int(meta.get("k_enemies", 0))
    use_inv = bool(meta.get("use_invincible", False))
    use_boss = bool(meta.get("use_boss", False))
    use_dg = bool(meta.get("use_danger_grad", False))
    if policy.obs_dim != feature_dim(k, use_vel, ke, use_inv, use_boss, use_dg) or policy.n_actions != len(actions):
        raise ValueError("Incompatible model metadata")
    out = Path(".runtime/stage1_attempts") / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + args.mode)
    out.mkdir(parents=True)
    cfg = Config.load()
    cfg.patch({"backend": "memory", "game": "th11", "restart_process": True,
               "reward": {"boss_damage_scale": 0.01, "spell_capture_bonus": 20},
               "slowdown": {"enabled": False}, "capture": {"prefer_printwindow": False}})
    env = TouhouEnv(cfg)
    report = {"args": vars(args), "policy": str(policy_path.resolve()), "cleared_stage1": False,
              "deaths": [], "bombs_requested": [], "max_bullets": 0, "max_stage": 0}
    last_bomb = -10000
    started = time.perf_counter()
    counts = Counter()
    guard = TacticalGuard()
    try:
        env.start()
        state = env.reset(difficulty=args.difficulty, character=args.character, shot=args.shot,
                          restart_process=not args.resume, menu="none" if args.resume else "auto").state
        print(f"START {out} player=({state.player_x},{state.player_y}) lives={state.lives}", flush=True)
        if [state.char_main, state.char_sub] != [args.character, args.shot]:
            raise RuntimeError("Actual loadout differs from requested loadout")
        with (out / "trace.jsonl").open("w", encoding="utf-8") as trace:
            for index in range(args.max_steps):
                obs = encode_state(state, k_bullets=k, use_velocity=use_vel, k_enemies=ke,
                                   use_invincible=use_inv, use_boss=use_boss, use_danger_grad=use_dg)
                action = actions[policy.greedy(obs)]
                points = np.asarray(state.bullets).reshape(-1, 2)
                near = float(np.linalg.norm(points - [state.player_x, state.player_y], axis=1).min()) if len(points) else 9999.0
                if args.mode == "guard":
                    frame = env.backend.mem.read_value(env.backend.reader.base_address+0xA5734, "i32")
                    action, tactical = guard.choose(state, frame, action)
                else:
                    tactical = {}
                threat = min(near, tactical.get("predicted_clearance", near))
                if threat < args.bomb_radius and (state.power or 0) >= 1 and index-last_bomb > 60:
                    action += "+bomb"
                    last_bomb = index
                    report["bombs_requested"].append({"step": index, "power": state.power, "nearest": near})
                counts[action] += 1
                result = env.step(action, frames=args.step_frames)
                nxt = result.state
                row = {"step": index, "seconds": round(time.perf_counter()-started, 3),
                       "stage": nxt.stage, "spell": nxt.spell_id, "lives": nxt.lives,
                       "power": nxt.power, "score": nxt.score, "graze": nxt.graze,
                       "x": nxt.player_x, "y": nxt.player_y, "bullets": nxt.bullet_count,
                       "enemies": nxt.enemies_alive, "near_before": round(near, 2), "action": action}
                row.update(boss_hp=nxt.boss_hp, boss_max_hp=nxt.boss_max_hp,
                           combat=nxt.raw.get("entities", {}).get("combat"),
                           reward_parts=result.info.get("reward_parts"),
                           dialogue_assist=result.info.get("dialogue_assist"),
                           character=nxt.char_main, shot=nxt.char_sub)
                row.update(tactical)
                trace.write(json.dumps(row)+"\n")
                report["max_bullets"] = max(report["max_bullets"], nxt.bullet_count or 0)
                report["max_stage"] = max(report["max_stage"], nxt.stage or 0)
                died = nxt.lives is not None and state.lives is not None and nxt.lives < state.lives
                if died:
                    report["deaths"].append(row)
                if index % 150 == 0 or died or result.done or (nxt.stage or 0) >= 2:
                    print(json.dumps(row), flush=True)
                    trace.flush()
                if died or (nxt.stage or 0) >= 2:
                    try:
                        env.capture_frame().save(out / f"step_{index}.png")
                    except Exception as exc:
                        report["capture_error"] = str(exc)
                state = nxt
                if (state.stage or 0) >= 2:
                    report["cleared_stage1"] = True
                    break
                if result.done or result.truncated:
                    report["done_reason"] = result.info.get("done_reason")
                    break
        report["last_state"] = state.to_dict(max_bullets=12)
        report["steps"] = index+1
        if report["cleared_stage1"] or (index+1 >= args.max_steps and state.in_game):
            report["stopped_at_limit"] = not report["cleared_stage1"]
            # Stage-transition screens may ignore Escape. Never claim a pause
            # merely because SendInput accepted it; verify the game frame stops.
            report["pause_verified"] = False
            address = env.backend.reader.base_address + 0xA5734
            for _ in range(10):
                env.backend.controller.tap(["pause"], hold=6, after=12)
                before = env.backend.mem.read_value(address, "i32")
                time.sleep(0.25)
                after = env.backend.mem.read_value(address, "i32")
                if before == after:
                    report["pause_verified"] = True
                    break
            env.capture_frame().save(out / "end_screen.png")
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        env.close()
        report["duration_s"] = time.perf_counter()-started
        report["actions"] = dict(counts)
        (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"REPORT {out / 'report.json'} cleared={report['cleared_stage1']}", flush=True)
    return 0 if report["cleared_stage1"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
