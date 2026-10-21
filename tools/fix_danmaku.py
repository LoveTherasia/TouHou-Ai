"""修一份"弹幕脚本": 补回被记成静止的弹, 并去掉重复条目.

背景: 真机录制时速度来自**槽位跨帧差分** (memory/bullettrack.py)。槽位复用/漏读
会让一部分弹记成 ``vx=vy=0`` —— 在 mock 里它们永远不会飞出场地, 于是同屏弹数
单调涨到 2000+, 任何策略都会在 4 分钟处被"静态弹海"淹死。

修法 (保守, 不改轨迹形状):
  1. 同一帧里绝大多数弹属于同一套弹型, 速度几乎相同 -> 用同帧非零速度的**中位数**
     补给静止弹;
  2. 同帧内找不到样本时, 用全局中位数;
  3. 去掉同帧同坐标的重复条目 (槽位误判会同一颗弹记 20 次)。

    python tools/fix_danmaku.py .runtime/danmaku/th11_replay_full.json \
        -o .runtime/danmaku/th11_replay_full_fixed.json
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("-o", "--out", default="")
    ap.add_argument("--max-life", type=int, default=0,
                    help="给每颗弹写一个寿命上限 (帧)。录制端不记消失事件, 没有上限时"
                         "子弹会永远留在场上, 同屏数会从 1k 涨到 16k —— 完全失真。"
                         "建议 480~600 (8~10 秒)。")
    args = ap.parse_args()

    src = Path(args.src)
    data = json.loads(src.read_text(encoding="utf-8"))
    spawns = data["spawns"]

    # 1) 去重 (同帧 + 同坐标, 保留第一条)
    seen = set()
    uniq = []
    dropped_dup = 0
    for f, x, y, vx, vy in spawns:
        key = (f, round(x, 1), round(y, 1))
        if key in seen:
            dropped_dup += 1
            continue
        seen.add(key)
        uniq.append((f, x, y, vx, vy))

    # 2) 补速度
    by_frame: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for f, _x, _y, vx, vy in uniq:
        if abs(vx) > 1e-6 or abs(vy) > 1e-6:
            by_frame[f].append((vx, vy))
    all_vel = [v for vs in by_frame.values() for v in vs]
    global_vel = (statistics.median(v[0] for v in all_vel),
                  statistics.median(v[1] for v in all_vel)) if all_vel else (0.0, 1.0)

    fixed, frozen, fallback = [], 0, 0
    for f, x, y, vx, vy in uniq:
        if abs(vx) <= 1e-6 and abs(vy) <= 1e-6:
            frozen += 1
            pool = by_frame.get(f) or []
            if pool:
                vx = statistics.median(p[0] for p in pool)
                vy = statistics.median(p[1] for p in pool)
                fallback += 1
            else:
                vx, vy = global_vel
        row = [f, round(x, 2), round(y, 2), round(vx, 4), round(vy, 4)]
        if args.max_life > 0:
            row.append(int(args.max_life))
        fixed.append(row)

    data["spawns"] = fixed
    data["count"] = len(fixed)
    data["repair"] = {
        "dropped_duplicates": dropped_dup,
        "frozen_velocity": frozen,
        "filled_from_frame_median": fallback,
        "filled_from_global_median": frozen - fallback,
        "source": str(src),
        "max_life": args.max_life,
    }
    out = Path(args.out) if args.out else src.with_name(src.stem + "_fixed.json")
    out.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    zero_before = sum(1 for _f, _x, _y, vx, vy in spawns if abs(vx) <= 1e-6 and abs(vy) <= 1e-6)
    print(f"[fix] 原始 {len(spawns)} 条, 静止速度 {zero_before} 条 "
          f"({zero_before / max(1, len(spawns)):.0%})")
    print(f"[fix] 去重 -{dropped_dup}; 补速度 {frozen} 条 "
          f"(同帧中位 {fallback}, 全局 {frozen - fallback})")
    print(f"[fix] 写出 {out} ({out.stat().st_size / 1e6:.1f} MB)")
    # 复核: 修完之后同屏弹数应随时间收敛, 不再单调涨
    print(f"[fix] 每秒生成数(前 12 秒): "
          f"{[sum(1 for f, *_ in fixed if i * 60 <= f < (i + 1) * 60) for i in range(12)]}")
    dup = Counter((f, round(x, 1), round(y, 1)) for f, x, y, _vx, _vy in fixed)
    print(f"[fix] 残留重复: {sum(v - 1 for v in dup.values() if v > 1)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
