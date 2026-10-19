"""真机演示: 用 mock 上训练好的策略在真实游戏里贪心跑 (评估"通关一面")。

TH11 1.00a 已支持普通弹幕、敌人坐标和擦弹。激光/Boss 血量尚未支持。
读取失败会明确报错；旧 mock 策略不保证可以迁移到真实游戏。

    python tools/play_real.py --policy .runtime/mock_clear5/policy_best --difficulty 0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from touhou_rl_mcp.env import TouhouEnv  # noqa: E402
from touhou_rl_mcp.config import Config  # noqa: E402
from touhou_rl_mcp.games.registry import GameRegistry  # noqa: E402
from touhou_rl_mcp.training.features import encode_state, feature_dim  # noqa: E402
from touhou_rl_mcp.training.policy import MLPPolicy  # noqa: E402
from touhou_rl_mcp.training.trainer import build_action_table  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--game", default="th11")
    ap.add_argument("--difficulty", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=3000)
    ap.add_argument("--step-frames", type=int, default=4)
    ap.add_argument("--shot-every", type=int, default=120, help="每多少 step 存一张截图")
    args = ap.parse_args()

    cfg = Config.load()
    cfg.patch({"backend": "memory", "game": args.game})

    policy_path = Path(args.policy)
    npz = policy_path if policy_path.suffix == ".npz" else policy_path.with_suffix(".npz")
    policy = MLPPolicy.load(npz)
    meta_path = npz.with_suffix(".json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")).get("config", {}) if meta_path.exists() else {}
    k_bullets = int(meta.get("k_bullets", 16))
    use_vel = bool(meta.get("bullet_velocity", False))
    k_enemies = int(meta.get("k_enemies", 0))
    use_inv = bool(meta.get("use_invincible", False))
    use_boss = bool(meta.get("use_boss", False))
    use_dg = bool(meta.get("use_danger_grad", False))
    actions = build_action_table(use_focus=bool(meta.get("use_focus", False)),
                             use_bomb=bool(meta.get("use_bomb", False)))
    if policy.obs_dim != feature_dim(k_bullets, use_vel, k_enemies, use_inv, use_boss, use_dg) or policy.n_actions != len(actions):
        raise ValueError("策略观测维度不匹配，请提供对应的 JSON 元数据。")
    env = TouhouEnv(cfg, registry=GameRegistry())
    try:
        env.start()
        return play(env, args, policy, actions, k_bullets, use_vel, k_enemies, use_inv, use_boss, use_dg, npz)
    finally:
        env.close()


def play(env, args, policy, actions, k_bullets, use_vel, k_enemies, use_inv, use_boss, use_dg, npz) -> int:
    print(f"[play] 策略={npz} obs_dim={policy.obs_dim} 动作数={len(actions)}")
    print(f"[play] 动作表={actions}")

    out_dir = Path(".runtime/play_real")
    out_dir.mkdir(parents=True, exist_ok=True)

    best = 0
    for ep in range(1, args.episodes + 1):
        st = env.reset(difficulty=args.difficulty, restart_process=True).state
        print(f"\n[play] === 第 {ep} 局开始 ===")
        started = time.time()
        for step_index in range(args.max_steps):
            state = env.get_state()
            obs = encode_state(state, k_bullets=k_bullets, use_velocity=use_vel,
                               k_enemies=k_enemies, use_invincible=use_inv,
                               use_boss=use_boss, use_danger_grad=use_dg)
            action = policy.greedy(obs)
            result = env.step(actions[action], frames=args.step_frames)
            st = result.state
            if step_index % 30 == 0 or result.done or result.truncated:
                print(
                    f"  step={step_index:4d} 动作={actions[action]:<9} "
                    f"stage={st.stage} lives={st.lives} power={st.power} "
                    f"bullets={st.bullet_count} enemies={st.enemies_alive} "
                    f"player=({st.player_x},{st.player_y}) reward={result.reward:+.2f} "
                    f"reason={result.info.get('done_reason')}"
                )
            if args.shot_every and step_index % args.shot_every == 0:
                try:
                    img = env.capture_frame()
                    img.save(out_dir / f"ep{ep}_step{step_index:04d}.png")
                except Exception:
                    pass
            if result.done or result.truncated:
                print(f"  -> 结束: reason={result.info.get('done_reason')} "
                      f"stage={st.stage} lives={st.lives} score={st.score}")
                break
        best = max(best, st.stage or 0)
        print(f"[play] 第 {ep} 局用时 {time.time()-started:.1f}s 到达面数={st.stage}")

    print(f"\n[play] 最好到达面数={best} (>1 表示通关了第 1 面)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
