"""从训练快照里挑最优策略 —— 替代"靠 mean_ret 撞运气"。

为什么需要这个工具
==================
`mock_full_clear` 的 `policy_best` 是按**训练回报** `(clears, stage, mean_ret)` 选的，
而 `mean_ret` 里 `alive_bonus` 的分量远大于输出（见 `docs/PLANNER_V2.md` §15.4），
于是"苟活快照"经常被评为更优。实测 `ppo_cont` 的 902 步快照正好出现在 +600 局，
而后面每一步评估都在退化（伤害 558 → 277 → 114）——**这不代表 +600 局就是最优的，
只说明那一次撞对了**。

做法
====
1. 训练时 `--snapshot-every 100` 存下每 100 局的策略；
2. 训练后（或中断后）用本工具把每份快照跑一遍 6 弹型基准；
3. 按 `score = w_steps·(存活/上限) + w_dmg·(伤害/最大伤害)` 排序 —— **存活与输出
   联合打分**，任一单项极差都会被扣掉；
4. `--copy-best` 把最优那份复制成 `<dir>/policy_picked.npz`，后续流程直接用它。

用法
====
```bash
# 训练（每 100 局存一份）
python tools/mock_full_clear.py ... --snapshot-every 100 --out .runtime/run_x

# 训练后挑最优（跑评测）
python tools/pick_snapshot.py --dir .runtime/run_x --copy-best

# 已经有 bench json 时直接复用，不重复跑（省时间）
python tools/pick_snapshot.py --dir .runtime/run_x --reuse
```
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_PATTERNS = ("aimed_single,aimed_spread,dense_ring,ring_gate,"
                    "closing_ring,wall_gate")


def find_snapshots(run_dir: Path) -> list[Path]:
    """``run_dir`` 下的快照（``snapshots/policy_XXXX.npz`` 或直接平铺）。"""
    for pattern in ("snapshots/policy_*.npz", "policy_*.npz"):
        found = sorted(run_dir.glob(pattern))
        if found:
            return found
    return []


def bench_one(args: argparse.Namespace, snap: Path, out_json: Path) -> dict:
    """对一份快照跑 pattern_bench，返回汇总 dict。"""
    obs_flags = ["--k-bullets", str(args.k_bullets), "--bullet-velocity",
                 "--danger-grad"]
    if args.wall_feat:
        obs_flags.append("--wall-feat")
    if not args.focus:
        obs_flags.append("--no-focus")
    cmd = [
        sys.executable, str(ROOT / "tools" / "pattern_bench.py"),
        "--policy", str(snap), *obs_flags,
        "--patterns", args.patterns,
        "--seeds", str(args.seeds),
        "--max-steps", str(args.max_steps),
        "--step-frames", str(args.step_frames),
        "--json", str(out_json),
    ]
    res = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    if res.returncode != 0 or not out_json.exists():
        return {"pattern": "?", "mean_steps": 0.0, "mean_damage": 0.0,
                "align": 0.0, "top_ratio": 0.0, "error": (res.stderr or "")[-300:]}
    data = json.loads(out_json.read_text(encoding="utf-8"))
    rows = data.get("rows") if isinstance(data, dict) else data
    n = max(1, len(rows))
    return {
        "rows": rows,
        "mean_steps": sum(float(r["mean_steps"]) for r in rows) / n,
        "mean_damage": sum(float(r.get("mean_damage", 0.0)) for r in rows) / n,
        "align": sum(float(r.get("align", 0.0)) for r in rows) / n,
        "top_ratio": sum(float(r.get("top_ratio", 0.0) or 0.0) for r in rows) / n,
    }


def score_of(row: dict, max_steps: int, dmg_ref: float) -> float:
    """**联合打分**：存活与输出各占一部分，任一单项极差都会被扣分。"""
    steps = float(row["mean_steps"]) / max(1.0, float(max_steps))
    dmg = float(row["mean_damage"]) / max(1.0, float(dmg_ref))
    return 0.7 * steps + 0.3 * dmg


def main() -> int:
    ap = argparse.ArgumentParser(description="从训练快照里按 6 弹型基准挑最优策略")
    ap.add_argument("--dir", default="", help="训练输出目录（含 snapshots/）")
    ap.add_argument("--patterns", default=DEFAULT_PATTERNS,
                    help="基准用的弹型列表（逗号分隔）")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument("--step-frames", type=int, default=2)
    ap.add_argument("--k-bullets", type=int, default=32)
    ap.add_argument("--wall-feat", action="store_true",
                    help="快照观测含 12 维墙特征时打开")
    ap.add_argument("--no-focus", dest="focus", action="store_false",
                    help="快照是 9 动作（无 focus）时打开")
    ap.add_argument("--w-steps", type=float, default=0.7,
                    help="联合打分里存活占的权重（默认 0.7）")
    ap.add_argument("--reuse", action="store_true",
                    help="复用同目录下已有的 bench_<快照名>.json，不重新评测")
    ap.add_argument("--copy-best", action="store_true",
                    help="把最优快照复制成 <dir>/policy_picked.npz")
    ap.add_argument("--json", default="", help="把汇总结果写到这个文件")
    ap.set_defaults(focus=True)
    args = ap.parse_args()

    run_dir = Path(args.dir or ".runtime/run")
    if not run_dir.exists():
        print(f"[pick] 目录不存在: {run_dir}")
        return 1
    snaps = find_snapshots(run_dir)
    if not snaps:
        print(f"[pick] {run_dir} 下没有 policy_*.npz 快照；训练时请加 --snapshot-every")
        return 1

    print(f"[pick] {run_dir} 找到 {len(snaps)} 份快照，开始评测 "
          f"({len(args.patterns.split(','))} 弹型 × {args.seeds} seeds)")
    results = []
    for snap in snaps:
        out_json = snap.parent / f"bench_{snap.stem}.json"
        if args.reuse and out_json.exists():
            data = json.loads(out_json.read_text(encoding="utf-8"))
            rows = data.get("rows") if isinstance(data, dict) else data
            n = max(1, len(rows))
            info = {
                "rows": rows,
                "mean_steps": sum(float(r["mean_steps"]) for r in rows) / n,
                "mean_damage": sum(float(r.get("mean_damage", 0.0)) for r in rows) / n,
                "align": sum(float(r.get("align", 0.0)) for r in rows) / n,
                "top_ratio": sum(float(r.get("top_ratio", 0.0) or 0.0) for r in rows) / n,
            }
        else:
            info = bench_one(args, snap, out_json)
        info["snapshot"] = snap.name
        results.append(info)
        print(f"[pick] {snap.name:24s} 存活{info['mean_steps']:6.0f} "
              f"伤害{info['mean_damage']:6.0f} 对准{info['align']:5.0%} "
              f"贴顶{info['top_ratio']:4.0%}", flush=True)

    dmg_ref = max((r["mean_damage"] for r in results), default=1.0) or 1.0
    for r in results:
        steps = float(r["mean_steps"]) / max(1.0, float(args.max_steps))
        dmg = float(r["mean_damage"]) / dmg_ref
        r["score"] = round(args.w_steps * steps + (1 - args.w_steps) * dmg, 4)
    results.sort(key=lambda r: -r["score"])

    print("\n[pick] 按联合打分排序 (存活 %.1f + 伤害 %.1f):"
          % (args.w_steps, 1 - args.w_steps))
    print(f"{'快照':<26}{'存活':>7}{'伤害':>7}{'对准':>7}{'贴顶':>7}{'score':>8}")
    for r in results:
        print(f"{r['snapshot']:<26}{r['mean_steps']:7.0f}{r['mean_damage']:7.0f}"
              f"{r['align']*100:6.0f}%{r['top_ratio']*100:6.0f}%{r['score']:8.3f}")

    best = results[0]
    print(f"\n[pick] 最优 = {best['snapshot']} (score {best['score']:.3f})")
    if args.copy_best:
        src = run_dir / "snapshots" / best["snapshot"]
        if not src.exists():
            src = run_dir / best["snapshot"]
        dst = run_dir / "policy_picked.npz"
        shutil.copyfile(src, dst)
        print(f"[pick] 已复制 -> {dst}")
    if args.json:
        Path(args.json).write_text(
            json.dumps({"dir": str(run_dir), "weights": [args.w_steps, 1 - args.w_steps],
                        "results": results}, ensure_ascii=False, indent=2),
            encoding="utf-8")
        print(f"[pick] 汇总 -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
