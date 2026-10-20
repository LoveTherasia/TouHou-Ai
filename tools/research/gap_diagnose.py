"""把一局失败拆开看: 死在**哪一堵墙**、**来不来得及**、**是不是没动**。

为什么需要它: 评估只给"1830 帧 / 最差局 1368"。同样一个 1368, 可能是

* **来不及** —— 缺口横移 324px, 赶路要 74 帧, 而墙从出现到压下来只有 150 帧,
  再加上上一堵墙的躲避, 时间根本不够 (这是基准的上限, 不是策略的锅);
* **没及时** —— 时间够, 但策略在原地磨蹭, 或者先往反方向走了;
* **站错了** —— 走到了上一堵墙的缺口, 但那正是这一堵墙的弹幕密集处。

三种死法对应完全不同的下一步 (改基准 / 改标签 / 改奖励), 光看存活帧数分不出来。

用法::

    python tools/research/gap_diagnose.py --run .runtime/runs/v3-anchor1
    python tools/research/gap_diagnose.py --run .runtime/runs/v3-anchor1 --phases 3 --json out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gap_rl import BENCHMARKS  # noqa: E402

from touhou_rl_mcp.config import Config  # noqa: E402
from touhou_rl_mcp.env import TouhouEnv  # noqa: E402
from touhou_rl_mcp.rl.reward import reward_profile  # noqa: E402
from touhou_rl_mcp.training.features import ObservationEncoder  # noqa: E402
from touhou_rl_mcp.training.gap_label import (  # noqa: E402
    COMMIT_FRAMES, MACRO_ACTIONS, gap_label,
)
from touhou_rl_mcp.training.occupancy import PLAYER_SPEED  # noqa: E402
from touhou_rl_mcp.training.policy import MLPPolicy  # noqa: E402
from touhou_rl_mcp.training.trainer import apply_scenario, build_action_table  # noqa: E402
from touhou_rl_mcp.types import PLAYFIELD_HEIGHT, PLAYFIELD_WIDTH  # noqa: E402

#: wall_gate 从生成到压到自机位置需要的帧数 (y 108 -> 416)
WALL_TRAVEL_DEFAULT = (PLAYFIELD_HEIGHT - 32 - 108.0)


def load_run(run_dir: Path, name: str = "policy") -> tuple[MLPPolicy, dict[str, Any]]:
    """``name`` 可选 ``policy`` / ``policy_best`` (训练脚本按评估择优保存的那份)。"""
    meta = json.loads((run_dir / f"{name}.json").read_text(encoding="utf-8"))
    policy = MLPPolicy.load(run_dir / f"{name}.npz")
    return policy, meta


def make_env(meta: dict[str, Any]) -> TouhouEnv:
    args = meta.get("config", {})
    cfg = Config.load()
    reward = ({} if args.get("reward") == "default"
              else reward_profile(str(args.get("reward") or "gap_survive")))
    cfg.patch({
        "backend": "mock", "game": "th11", "danmaku_pattern": "wall_gate",
        "step_frames": COMMIT_FRAMES, "max_episode_frames": 0,
        "auto_reset": False, "reward": reward,
    })
    env = TouhouEnv(cfg)
    env.start()
    env.reset(seed=1)
    return env


def gap_x_at(scenario: dict[str, float], wave: int) -> float:
    """第 ``wave`` 堵墙的缺口横坐标 (与 ``WallGate.update`` 同一公式)。"""
    free = PLAYFIELD_WIDTH - 2 * 12.0
    phase = (float(scenario.get("gap_phase", 0.0)) + wave * float(scenario.get("gap_drift", 0.0)))
    phase %= 1.0
    return 12.0 + phase * free


def run_episode(env: TouhouEnv, policy: MLPPolicy | None, enc: ObservationEncoder,
                actions: list[str], scenario: dict[str, float], *,
                max_frames: int, label_mode: str) -> dict[str, Any]:
    state = env.reset(seed=101).state
    apply_scenario(env, scenario)
    interval = float(scenario.get("interval", 75.0))
    speed = float(scenario.get("speed", 2.05))
    wall_life = (PLAYFIELD_HEIGHT - 32 - 108.0) / speed

    trace: list[dict[str, Any]] = []
    frames = 0
    prev = -1
    disagreements = 0
    steps = 0
    last: dict[str, Any] | None = None
    while frames < max_frames:
        ideal = int(gap_label(state, enc_spec(enc), label_mode)) if label_mode else -1
        if policy is None:                     # 老师自己走: 用来看"标签的上限"
            action = ideal
        else:
            obs = enc(state, prev_action=prev)
            action = int(policy.greedy(obs))
        last = {"frame": int(state.frame), "px": float(state.player_x),
                "action": action, "ideal": ideal,
                "bullets": int(state.bullet_count)}
        trace.append(last)
        disagreements += int(ideal >= 0 and action != ideal)
        steps += 1
        result = env.step(actions[action], frames=COMMIT_FRAMES, auto_reset=False)
        state = result.state
        frames += COMMIT_FRAMES
        prev = action
        if result.done:
            break

    # 关键: 自机死亡后会**重置回场地中心**再继续, 所以 done 时刻的 player_x 恒为 192。
    # 死因必须用"致命那一步**之前**"的状态来算, 否则每一局看起来都死在正中间。
    assert last is not None
    death_frame = int(last["frame"])
    px = float(last["px"])
    wave = int(death_frame // max(1.0, interval))
    gx = gap_x_at(scenario, wave)
    # 这堵墙生成那一刻自机在哪 —— 只能从那时算"来不来得及"
    spawn_frame = wave * interval
    near = min(trace, key=lambda r: abs(r["frame"] - spawn_frame))
    px_at_spawn = near["px"]
    need = abs(gx - px_at_spawn) / PLAYER_SPEED
    prev_gap = gap_x_at(scenario, wave - 1) if wave > 0 else gx
    # 最后 30 帧有没有朝缺口方向挪 (反应延迟)
    back = [r for r in trace if r["frame"] >= death_frame - 30]
    moved = (float(back[-1]["px"]) - float(back[0]["px"])) if len(back) > 1 else 0.0
    toward = float(np.sign(gx - px)) * moved
    return {
        "frames": frames,
        "death_frame": death_frame,
        "wave": wave,
        "player_x": round(px, 1),
        "gap_x": round(gx, 1),
        "dx_at_death": round(abs(px - gx), 1),
        "px_at_spawn": round(px_at_spawn, 1),
        "need_frames": round(need, 1),
        "wall_life": round(wall_life, 1),
        "slack": round(wall_life - need, 1),
        "gap_shift": round(gx - prev_gap, 1),
        "moved_30f": round(moved, 1),
        "toward_gap_30f": round(toward, 1),
        "agree": round(1.0 - disagreements / max(1, steps), 3),
        "bullets_at_death": int(last["bullets"]),
    }


def enc_spec(enc: ObservationEncoder):
    from touhou_rl_mcp.training.features import gap_spec_from

    return gap_spec_from(enc)


def classify(row: dict[str, Any]) -> str:
    """死因分类 (阈值都来自几何, 不是拍脑袋)。"""
    if row["need_frames"] > row["wall_life"]:
        return "来不及"          # 从墙出现那一刻算, 跑到缺口都来不及
    if row["dx_at_death"] > 30.0:
        return "没及时"          # 时间够, 但死的时候离缺口还很远
    return "站错了"              # 已经贴着缺口却还是被弹 —— 说明进的是上一堵墙的缺口


def main() -> int:
    ap = argparse.ArgumentParser(description="缺口策略的失败诊断")
    ap.add_argument("--run", required=True, help="训练记录目录 (含 policy.npz / policy.json)")
    ap.add_argument("--teacher", action="store_true",
                    help="不加载策略, 直接用标签老师走 (给出'标签本身的上限')")
    ap.add_argument("--phases", type=int, default=3, help="每个留出场景叠加几个相位")
    ap.add_argument("--max-frames", type=int, default=2400)
    ap.add_argument("--label", default="", help="标签口径 (默认取 run 的配置)")
    ap.add_argument("--json", default="", help="把逐局结果写到这个 json")
    args = ap.parse_args()

    run_dir = Path(args.run)
    policy, meta = load_run(run_dir)
    if args.teacher:
        policy = None
    cfg_dict = meta["config"]
    enc = ObservationEncoder.from_meta(cfg_dict)
    actions = build_action_table(use_focus=bool(cfg_dict.get("use_focus", False)),
                                 use_bomb=bool(cfg_dict.get("use_bomb", False)))
    if policy is not None and policy.obs_dim != enc.dim:
        raise SystemExit(f"维度不匹配: policy={policy.obs_dim} enc={enc.dim}")
    bench = str(cfg_dict.get("bench") or json.loads(
        (run_dir / "meta.json").read_text(encoding="utf-8")).get("bench", "v3"))
    _, heldout = BENCHMARKS[bench]
    label_mode = args.label or str(cfg_dict.get("gap_label") or "reach")
    env = make_env(meta)

    print(f"run={run_dir.name}  bench={bench}  label={label_mode}  "
          f"obs={policy.obs_dim if policy is not None else 'teacher'}  "
          f"layers={enc.gap_layers}")
    rows: list[dict[str, Any]] = []
    for si, scenario in enumerate(heldout):
        for k in range(args.phases):
            sc = dict(scenario)
            sc["gap_phase"] = float(sc.get("gap_phase", 0.0)) + k / max(1, args.phases)
            row = run_episode(env, policy, enc, actions, sc,
                              max_frames=args.max_frames, label_mode=label_mode)
            row.update({"scene": si, "phase": k, "cause": classify(row)})
            rows.append(row)
            print(f"  scene{si} phase{k}  {row['frames']:5d} 帧 | 死于第 {row['wave']:2d} 堵墙 "
                  f"自机 x={row['player_x']:5.1f} 缺口 x={row['gap_x']:5.1f} "
                  f"| 赶路需 {row['need_frames']:5.1f} / 墙给 {row['wall_life']:5.1f} "
                  f"余 {row['slack']:6.1f} | 缺口横移 {row['gap_shift']:7.1f} | "
                  f"末30帧朝缺口 {row['toward_gap_30f']:6.1f} | {row['cause']}")

    causes: dict[str, int] = {}
    for r in rows:
        causes[r["cause"]] = causes.get(r["cause"], 0) + 1
    full = [r for r in rows if r["frames"] >= args.max_frames]
    print("\n汇总: " + "  ".join(f"{k}={v}" for k, v in sorted(causes.items()))
          + f"  打满={len(full)}/{len(rows)}")
    print(f"均值 {np.mean([r['frames'] for r in rows]):.0f} 帧  "
          f"最差 {min(r['frames'] for r in rows)} 帧")
    if args.json:
        Path(args.json).write_text(json.dumps(rows, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
        print(f"明细: {args.json}")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
