"""测"会不会上穿": 记录自机到达的最高位置 (min_y) 与停留在上半场的帧数.

这是验收口径里缺的那一项 —— 只看存活/伤害会漏掉"贴底横移也能活满 2000 步"这种
退化行为。``min_y`` 越靠近 0 说明越敢往场地上方走; 上半场帧数 = 0 说明整局都在
下半区游走, 也就是"不会上穿"。

实测口径 (2026-10-21): k_bullets 策略 5 个弹型 min_y 都在 325~368（场地 448）,
上半场帧数 0~61; 规划器 ``ring_gate`` min_y 44.6、连续 355 帧在上半场 ——
**上穿能力只有规划器有**。

    python tools/climb_probe.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from touhou_rl_mcp.config import Config
from touhou_rl_mcp.env import TouhouEnv
from touhou_rl_mcp.games.registry import GameRegistry
from touhou_rl_mcp.training import TrainConfig, Trainer
from touhou_rl_mcp.training.features import encode_observation
from touhou_rl_mcp.training.gap_label import update_action_age
from touhou_rl_mcp.training.viability import ViabilityPlanner

FIELD_H = 448.0
UPPER = 340.0          # y < 340 视为"已进入上半场"（起点约 416）


def policy_action(trainer: Trainer):
    """按训练器的贪心推理口径包装一个 act(state) -> action_name."""
    cfg = trainer.config
    prev_dir, age = -1, 0

    def act(state):
        nonlocal prev_dir, age
        obs = encode_observation(state, cfg, prev_action=prev_dir, action_age=age)
        action = trainer.policy.greedy(obs)
        next_dir = action % 9
        age = update_action_age(prev_dir, age, next_dir, getattr(cfg, "gap_action_age_cap", 12))
        prev_dir = next_dir
        return trainer.action_table[action]

    return act


def run(env, act, steps: int, frames: int) -> dict:
    state = env.reset(seed=1000).state
    min_y = state.player_y or FIELD_H
    upper = streak = best = 0
    n = 0
    while n < steps:
        r = env.step(act(state), frames=frames)
        state = r.state
        n += 1
        if state.player_y is not None:
            min_y = min(min_y, state.player_y)
            if state.player_y < UPPER:
                upper += 1
                streak += 1
                best = max(best, streak)
            else:
                streak = 0
        if r.done or r.truncated:
            break
    return {"steps": n, "min_y": round(min_y, 1), "upper": upper, "upper_run": best}


def make_env(pattern: str, step_frames: int) -> TouhouEnv:
    cfg = Config.load()
    cfg.patch({"backend": "mock", "game": "th11", "step_frames": step_frames,
               "danmaku_pattern": pattern, "start_stage": 1})
    env = TouhouEnv(cfg, registry=GameRegistry())
    env.start()
    return env


def main() -> int:
    rows = []
    patterns = ["aimed_single", "dense_ring", "ring_gate", "closing_ring", "wall_gate"]

    # ---- A. k_bullets 反应式策略 (1 帧/步) ----
    for pat in patterns:
        env = make_env(pat, 1)
        tc = TrainConfig(episodes=0, max_steps=2000, step_frames=1, k_bullets=16,
                         k_enemies=8, bullet_velocity=True, use_danger_grad=True,
                         use_focus=False)
        tr = Trainer(env, tc, env_lock=env.lock)
        tr.load(".runtime/policies/attack_ppo_fixed.npz")
        m = run(env, policy_action(tr), 2000, 1)
        rows.append(("k_bullets", pat, m))
        print(f"[climb] k_bullets  {pat:13s} {m}", flush=True)
        env.close()

    # ---- B. gap 策略 (4 帧/步) ----
    for pat in patterns:
        env = make_env(pat, 4)
        tc = TrainConfig(episodes=0, max_steps=500, step_frames=4, use_gap=True,
                         gap_layers=2, use_focus=True)
        tr = Trainer(env, tc, env_lock=env.lock)
        tr.load(".runtime/replay_hard_r1/policy_best.npz")
        m = run(env, policy_action(tr), 500, 4)
        rows.append(("gap822", pat, m))
        print(f"[climb] gap822      {pat:13s} {m}", flush=True)
        env.close()

    # ---- C. 规划器 (1 帧/步, 只跑 3 个关键型) ----
    pl = ViabilityPlanner(horizon=200, dt=2, aim_weight=0.0)
    for pat in ["aimed_single", "dense_ring", "ring_gate"]:
        env = make_env(pat, 1)
        m = run(env, lambda s: pl.choose(s)[0], 1200, 1)
        rows.append(("planner", pat, m))
        print(f"[climb] planner     {pat:13s} {m}", flush=True)
        env.close()

    print("\n| 方法 | 弹型 | 步数 | 最高点 min_y | 上半场帧数 | 最长连续 |")
    print("|---|---|---|---|---|---|")
    for who, pat, m in rows:
        print(f"| {who} | {pat} | {m['steps']} | {m['min_y']} | {m['upper']} | {m['upper_run']} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
