"""训练弹幕动力学模型（残差 MLP），并在同一份数据上对比匀速外推。

    # mock 弹（匀速直线）：残差应当 ≈ 0，用来验证接口
    python tools/train_dynamics.py --script .runtime/danmaku/th11_replay_v2.json \
        --out .runtime/dynamics/mock.npz --epochs 40

    # 真机锁血采集的轨迹（曲线/加速弹）：这里才会看到真实收益
    python tools/train_dynamics.py --tracks .runtime/tracks/*.json --epochs 300

数据来源
--------
* ``--script``：弹幕脚本（``spawns`` = frame,x,y,vx,vy,life）。只能重建**匀速**轨迹。
* ``--tracks``：逐帧轨迹 JSON，形如 ``{"tracks": [[[x,y,vx,vy,ax,ay], ...], ...]}``
  （可选 4/6 列）。真机用锁血（invincible 开）逐帧记录槽位坐标即可产出。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from touhou_rl_mcp.training.dynamics import (  # noqa: E402
    ConstantVelocity,
    ResidualMLP,
    evaluate,
    pairs_from_tracks,
    tracks_from_script,
)


def load_tracks(paths: list[str]) -> list[list[tuple[float, ...]]]:
    tracks: list[list[tuple[float, ...]]] = []
    for item in paths:
        data = json.loads(Path(item).read_text(encoding="utf-8"))
        for seq in data.get("tracks", []):
            rows = []
            for i, p in enumerate(seq):
                vals = [float(v) for v in p]
                # 统一成 (frame, x, y, vx, vy, ax, ay)
                if len(vals) == 2:                      # 只有坐标 -> 差分出速度
                    rows.append((i, vals[0], vals[1], 0.0, 0.0, 0.0, 0.0))
                elif len(vals) == 4:                    # x,y,vx,vy
                    rows.append((i, vals[0], vals[1], vals[2], vals[3], 0.0, 0.0))
                else:                                   # 6 列: 含加速度
                    rows.append((i, *vals[:6]))
            if len(rows) > 2:
                tracks.append(rows)
    return tracks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", default="", help="弹幕脚本 (只能给出匀速轨迹)")
    ap.add_argument("--tracks", nargs="*", default=[], help="逐帧轨迹 JSON 列表")
    ap.add_argument("--out", default=".runtime/dynamics/model.npz")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--horizons", default="10,20,30")
    ap.add_argument("--limit", type=int, default=0, help="最多用多少条轨迹 (0=全部)")
    args = ap.parse_args()

    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    tracks: list[list[tuple[float, ...]]] = []
    if args.script:
        tracks += tracks_from_script(args.script)
    if args.tracks:
        tracks += load_tracks(args.tracks)
    if not tracks:
        print("[dyn] 没有数据: 需要 --script 或 --tracks")
        return 2
    if args.limit:
        tracks = tracks[:args.limit]

    X, Y, _ = pairs_from_tracks(tracks, horizons=horizons)
    if X.shape[0] == 0:
        print("[dyn] 样本为空 (轨迹太短?)")
        return 2
    print(f"[dyn] 轨迹 {len(tracks)} 条 -> 样本 {X.shape[0]} 个, "
          f"视界 {horizons}", flush=True)

    cv = evaluate(ConstantVelocity(), X, Y)
    print(f"[dyn] 匀速外推  MAE={cv['mae']:.3f}px "
          + " ".join(f"{k.split('@')[1]}帧={v:.2f}" for k, v in cv.items() if k.startswith('mae@')),
          flush=True)

    model = ResidualMLP(hidden=args.hidden, seed=0)
    model.fit(X, Y, epochs=args.epochs, batch=256, lr=5e-3)
    got = evaluate(model, X, Y)
    print(f"[dyn] 残差模型  MAE={got['mae']:.3f}px "
          + " ".join(f"{k.split('@')[1]}帧={v:.2f}" for k, v in got.items() if k.startswith('mae@')),
          flush=True)
    if cv["mae"] < 1e-6:
        ratio = None
        print("[dyn] 匀速外推误差已经为 0 —— 这份数据是匀速直线弹, "
              "动力学模型没有可学的残差 (需要真机锁血采集的曲线弹轨迹)",
              flush=True)
    else:
        ratio = got["mae"] / cv["mae"]
        print(f"[dyn] 相对匀速外推: {ratio * 100:.0f}% "
              f"({'有效' if ratio < 0.8 else '无增益: 数据本身是匀速直线'})", flush=True)

    out = Path(args.out)
    model.save(out)
    meta = {"source": {"script": args.script, "tracks": args.tracks},
            "samples": int(X.shape[0]), "horizons": horizons,
            "cv": cv, "model": got, "ratio": ratio,
            "describe": model.describe()}
    out.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
    print(f"[dyn] 已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
