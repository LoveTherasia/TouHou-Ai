"""真机逐帧轨迹采集：给动力学模型补**曲线弹**数据.

为什么需要它
------------
`training/dynamics.py` 的残差模型学的是 ``p' − (p + v·dt)``，也就是"加速度带来的偏差"。
但 mock 与 replay 脚本里的弹幕全都是**匀速直线**（残差恒 0），所以模型在那些数据上
永远学不到东西（实测 MAE 0.00，见 `PLANNER_V2.md` §7.1）。要让它真的有增益，只有
一条路：在真机上把每一帧的弹幕坐标记下来，用差分算出真实的速度和加速度。

采集方式
--------
* 用**槽位号**当弹幕身份（`state.raw["entities"]["bullets"]["active_slots"]` 与
  `state.bullets` 同序），逐帧记 ``(frame, x, y)``；
* 可选 ``--lock-lives``：用 `memory.cheats.LivesLock` 把残机钉在 9，让采集不被死亡打断；
* 用 `--teacher rule` 让规则老师在场上跑动（自机狙会朝它来，能采到更多角度），
  `--teacher none` 则原地不动；
* 结束后按槽位拼轨迹，遇到"一帧跳 30px 以上"就当作槽位被回收，**断开成两条**；
  然后中心差分出 ``v``、二阶差分出 ``a``，写成

      {"tracks": [[[x, y, vx, vy, ax, ay], ...], ...], "meta": {...}}

  正好是 `tools/train_dynamics.py --tracks` 的输入格式。

用法
----
    # 采 90 秒，锁血，规则老师跑动
    python tools/collect_tracks.py --seconds 90 --lock-lives \
        --teacher rule --out .runtime/tracks/real_$(date +%H%M).json

    # 采完直接训练
    python tools/train_dynamics.py --tracks .runtime/tracks/*.json --epochs 300
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from touhou_rl_mcp.config import Config  # noqa: E402
from touhou_rl_mcp.env import TouhouEnv  # noqa: E402
from touhou_rl_mcp.games.registry import GameRegistry  # noqa: E402

#: 同一槽位相邻帧位移超过它 = 槽位被回收（换了一颗弹），断开轨迹
JUMP_BREAK = 30.0
#: 单帧速度上限（px/帧），超过视为脏数据
MAX_SPEED = 15.0


def split_runs(points: list[tuple[int, float, float]]) -> list[list[tuple[int, float, float]]]:
    """把一条槽位记录按"跳变"切成若干段（每段是一颗真实的弹）。"""
    runs: list[list[tuple[int, float, float]]] = []
    cur: list[tuple[int, float, float]] = []
    for item in points:
        if cur:
            pf, px, py = cur[-1]
            if (abs(item[0] - pf) > 2
                    or abs(item[1] - px) > JUMP_BREAK
                    or abs(item[2] - py) > JUMP_BREAK):
                runs.append(cur)
                cur = []
        cur.append(item)
    if cur:
        runs.append(cur)
    return runs


def track_to_rows(points: list[tuple[int, float, float]]) -> list[list[float]]:
    """``(frame, x, y)`` → ``[x, y, vx, vy, ax, ay]``（中心差分）。"""
    n = len(points)
    if n < 3:
        return []
    frames = np.array([p[0] for p in points], dtype=np.float64)
    xs = np.array([p[1] for p in points], dtype=np.float64)
    ys = np.array([p[2] for p in points], dtype=np.float64)

    def deriv(values: np.ndarray) -> np.ndarray:
        out = np.zeros_like(values)
        out[1:-1] = (values[2:] - values[:-2]) / np.maximum(1.0, frames[2:] - frames[:-2])
        out[0] = (values[1] - values[0]) / max(1.0, frames[1] - frames[0])
        out[-1] = (values[-1] - values[-2]) / max(1.0, frames[-1] - frames[-2])
        return out

    vx, vy = deriv(xs), deriv(ys)
    ax, ay = deriv(vx), deriv(vy)
    rows: list[list[float]] = []
    for i in range(n):
        if abs(vx[i]) > MAX_SPEED or abs(vy[i]) > MAX_SPEED:
            continue                      # 槽位复用造成的脏点，直接丢
        rows.append([round(float(xs[i]), 2), round(float(ys[i]), 2),
                     round(float(vx[i]), 4), round(float(vy[i]), 4),
                     round(float(ax[i]), 4), round(float(ay[i]), 4)])
    return rows


def collect(args: argparse.Namespace) -> dict:
    cfg = Config.load()
    cfg.patch({
        "backend": "memory", "game": args.game, "step_frames": 1,
        "restart_process": bool(args.launch),
        "dialogue_assist": True,
        "capture": {"prefer_printwindow": False},
    })
    env = TouhouEnv(cfg, registry=GameRegistry())
    env.start()
    state = env.backend.read_state()
    if not state.in_game:
        print("[tracks] 不在局内, 重开一局 ...", flush=True)
        state = env.reset(restart_process=bool(args.launch),
                          difficulty=0, character=0, shot=0).state

    lock = None
    if args.lock_lives:
        try:
            from touhou_rl_mcp.memory.cheats import LivesLock

            addr = env.backend.reader.address_of("lives")
            if addr:
                lock = LivesLock(env.backend.mem, addr, value=9, floor=1).start()
                print(f"[tracks] 锁血已开 (lives @ 0x{addr:X})", flush=True)
            else:
                print("[tracks] lives 未标定, 跳过锁血", flush=True)
        except Exception as exc:  # pragma: no cover - 真机环境
            print(f"[tracks] 锁血失败: {exc}", flush=True)

    teacher = None
    if args.teacher == "rule":
        from touhou_rl_mcp.training.rule_teacher import RuleTeacher

        teacher = RuleTeacher()

    slots: dict[int, list[tuple[int, float, float]]] = {}
    t0 = time.perf_counter()
    frames = steps = 0
    peak = 0
    try:
        while True:
            if time.perf_counter() - t0 > args.seconds:
                break
            if state.frame < frames and frames > 0:
                print("[tracks] 检测到重开, 停止采集", flush=True)
                break
            frames = max(frames, int(state.frame or 0))
            detail = (state.raw.get("entities", {}) or {}).get("bullets", {}) or {}
            active = detail.get("active_slots") or []
            bullets = state.bullets or []
            for i, (x, y) in enumerate(bullets):
                if i >= len(active):
                    break
                slots.setdefault(int(active[i]), []).append(
                    (int(state.frame or 0), float(x), float(y)))
            peak = max(peak, len(bullets))
            action = teacher.choose(state, state.frame, None)[0] if teacher else "shoot"
            result = env.step(action, frames=1)
            state = result.state
            steps += 1
            if steps % 300 == 0:
                print(f"[tracks] {steps} 步 / 帧 {state.frame} / 弹 {len(bullets)} / "
                      f"已记槽位 {len(slots)}", flush=True)
            if result.done or result.truncated:
                print("[tracks] 局终, 停止采集", flush=True)
                break
    finally:
        if lock is not None:
            lock.stop()
        env.close()

    tracks: list[list[list[float]]] = []
    for points in slots.values():
        for run in split_runs(points):
            rows = track_to_rows(run)
            if len(rows) >= 3:
                tracks.append(rows)
    return {
        "tracks": tracks,
        "meta": {"game": args.game, "steps": steps, "frames": frames,
                 "peak_bullets": peak, "slots": len(slots),
                 "seconds": round(time.perf_counter() - t0, 1),
                 "lock_lives": bool(lock), "teacher": args.teacher},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="th11")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--out", default=".runtime/tracks/real.json")
    ap.add_argument("--lock-lives", action="store_true",
                    help="锁残机 (需要 lives 已标定), 采集不被死亡打断")
    ap.add_argument("--teacher", default="rule", choices=["rule", "none"],
                    help="采集时的移动策略: rule=规则老师跑动, none=原地")
    ap.add_argument("--launch", action="store_true", help="游戏没开就先启动")
    args = ap.parse_args()

    data = collect(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    n = len(data["tracks"])
    pts = sum(len(t) for t in data["tracks"])
    print(f"[tracks] 轨迹 {n} 条 / 点 {pts} 个 -> {out}", flush=True)
    if n == 0:
        print("[tracks] 没采到轨迹 (不在局内? 弹幕未被识别?)", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
