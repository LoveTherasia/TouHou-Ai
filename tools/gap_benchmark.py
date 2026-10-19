"""避弹能力基准: 对比"站着不动 / 斥力老师 / 时空规划老师"在缺口场景下的行为.

用途: 检验"未来推演 + 宏动作"能否产生"找缝"行为, 以及不同代价函数的副作用。
它不是训练脚本, 只做可复现的短跑评测。

用法::

    python tools/gap_benchmark.py --scenario static_ring --agent planner --escape-weight 0.3
    python tools/gap_benchmark.py --scenario closing_ring --agent force --frames 1800
    python tools/gap_benchmark.py --scenario dense_ring --agent planner

指标:
    steps        存活帧数 (死亡或达到上限)
    deaths       掉残次数
    success      仅 static_ring: 是否在时限内穿出环 (距离中心 > 半径 + 12)
    min_clear    全程与最近弹幕的最小距离 (越小说明越贴弹)
    mean_y       平均纵向位置 (448 为底边; 长期贴底说明"龟缩")
    corner       处于底角 (x<50 或 x>334, 且 y>380) 的时间占比
    up_frames    最长连续上移帧数 (衡量"承诺上穿"的能力)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from touhou_rl_mcp.config import Config
from touhou_rl_mcp.env import TouhouEnv
from touhou_rl_mcp.training.planner import MacroPlanner
from touhou_rl_mcp.training.rule_teacher import RuleTeacher

MACRO_FRAMES = 6
STATIC_RADIUS = 90.0


def build_env(scenario: str, gap_deg: float, seed: int) -> tuple[TouhouEnv, dict]:
    cfg = Config.load()
    pattern = "" if scenario == "static_ring" else scenario
    cfg.patch({
        "backend": "mock",
        "game": "th11",
        "danmaku_pattern": pattern,
        "step_frames": MACRO_FRAMES,
        "max_episode_frames": 0,
    })
    env = TouhouEnv(cfg)
    env.start()
    state = env.reset(seed=seed).state
    meta = {"center": (state.player_x, state.player_y)}
    if scenario == "static_ring":
        game = env.backend.game
        gap = math.radians(gap_deg)
        count = 16
        for i in range(count):
            angle = 2.0 * math.pi * i / count
            delta = abs(((angle - gap + math.pi) % (2.0 * math.pi)) - math.pi)
            if delta < math.radians(30.0):
                continue
            game.spawn(state.player_x + math.cos(angle) * STATIC_RADIUS,
                       state.player_y + math.sin(angle) * STATIC_RADIUS, 0.0, 0.0)
        state = env.get_state()
    return env, meta


def make_agent(name: str, escape_weight: float):
    if name == "noop":
        return lambda st: "shoot"
    if name == "force":
        teacher = RuleTeacher(planner="force")
        return lambda st: teacher.choose(st, int(st.frame), "")[0]
    planner = MacroPlanner(horizon=150, beam=20, probe_frames=60,
                           escape_weight=escape_weight)
    return lambda st: planner.plan(st).action


def run(scenario: str, agent_name: str, frames: int, seed: int,
        gap_deg: float, escape_weight: float) -> dict:
    env, meta = build_env(scenario, gap_deg, seed)
    act = make_agent(agent_name, escape_weight)
    state = env.get_state()
    cx, cy = meta["center"]
    xs, ys, deaths = [], [], 0
    lives = state.lives
    up = best_up = 0
    min_clear = float("inf")
    max_dist = 0.0
    try:
        while len(ys) * MACRO_FRAMES < frames:
            result = env.step(act(state), frames=MACRO_FRAMES)
            state = result.state
            xs.append(state.player_x)
            ys.append(state.player_y)
            if state.bullets:
                nearest = min(math.hypot(bx - state.player_x, by - state.player_y)
                              for bx, by in state.bullets)
                min_clear = min(min_clear, nearest)
            max_dist = max(max_dist, math.hypot(state.player_x - cx, state.player_y - cy))
            if len(ys) > 1 and ys[-1] < ys[-2] - 1.0:
                up += 1
                best_up = max(best_up, up)
            elif len(ys) > 1:
                up = 0
            if state.lives is not None and lives is not None and state.lives < lives:
                deaths += 1
            lives = state.lives
            if result.done:
                break
    finally:
        env.close()

    xs_arr, ys_arr = np.asarray(xs), np.asarray(ys)
    corner = 0.0
    if len(xs_arr):
        corner = float(np.mean(((xs_arr < 50) | (xs_arr > 334)) & (ys_arr > 380))) * 100.0
    success = None
    if scenario == "static_ring":
        success = bool(max_dist > STATIC_RADIUS + 12.0 and deaths == 0)
    return {
        "scenario": scenario,
        "agent": agent_name,
        "escape_weight": escape_weight if agent_name == "planner" else None,
        "steps": len(ys) * MACRO_FRAMES,
        "deaths": deaths,
        "success": success,
        "min_clear": round(min_clear, 1) if min_clear != float("inf") else None,
        "mean_y": round(float(ys_arr.mean()), 1) if len(ys_arr) else None,
        "max_dist": round(max_dist, 1),
        "corner_pct": round(corner, 1),
        "up_frames": best_up * MACRO_FRAMES,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenario", default="closing_ring",
                    choices=["static_ring", "closing_ring", "dense_ring", "spiral"])
    ap.add_argument("--agent", default="planner", choices=["noop", "force", "planner"])
    ap.add_argument("--frames", type=int, default=1800)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--gap-deg", type=float, default=-60.0,
                    help="static_ring 的缺口角度 (度; 0=右, -90=上)")
    ap.add_argument("--escape-weight", type=float, default=0.3,
                    help="规划器的'脱离包围'权重 (0=纯安全; 静态环需要 >0)")
    ap.add_argument("--json", action="store_true", help="只输出 JSON")
    args = ap.parse_args()
    result = run(args.scenario, args.agent, args.frames, args.seed,
                 args.gap_deg, args.escape_weight)
    if args.json:
        print(json.dumps(result, ensure_ascii=False))
    else:
        print(" | ".join(f"{k}={v}" for k, v in result.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
