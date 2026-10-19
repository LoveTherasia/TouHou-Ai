"""分析训练日志 (train.jsonl / steps.jsonl), 帮助找出智能体的不足并据此调参。

    python tools/analyze_training_log.py --dir .runtime/real_verify/run4
    python tools/analyze_training_log.py --dir .runtime/real_verify/run4 --tail 10
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    out = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def pct(a: float, b: float) -> str:
    return f"{a / b:.0%}" if b else "-"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--tail", type=int, default=8)
    args = ap.parse_args()
    d = Path(args.dir)

    eps = load_jsonl(d / "train.jsonl")
    steps = load_jsonl(d / "steps.jsonl")

    print(f"===== {d} =====")
    print(f"train.jsonl: {len(eps)} 局 | steps.jsonl: {len(steps)} 步")

    if eps:
        print("\n--- 最近几局 ---")
        print(f"{'ep':>4} {'char/shot':>9} {'steps':>6} {'return':>9} {'stage':>5} "
              f"{'lives':>5} {'graze':>6} {'miss':>4} {'end':>9} {'sps':>5}")
        for e in eps[-args.tail:]:
            end = "cleared" if e.get("cleared") else ("game_over" if e.get("game_over") else
                                                      ("truncated" if e.get("truncated") else "?"))
            print(f"{e.get('episode'):>4} {str(e.get('character'))+'/'+str(e.get('shot')):>9} "
                  f"{e.get('steps'):>6} {e.get('return'):>9.2f} {str(e.get('max_stage')):>5} "
                  f"{str(e.get('misses')):>5} {str(e.get('total_graze')):>6} {'':>4} {end:>9} "
                  f"{str(e.get('steps_per_sec')):>5}")

        n = len(eps)
        last = eps[-min(20, n):]
        print(f"\n--- 汇总 (全部 {n} 局 / 最近 {len(last)} 局) ---")
        print(f"回报: 全部均={sum(e['return'] for e in eps)/n:.2f}  "
              f"最近{len(last)}局均={sum(e['return'] for e in last)/len(last):.2f}  "
              f"最好={max(e['return'] for e in eps):.2f}")
        print(f"面数: 最好={max((e.get('max_stage') or 0) for e in eps)}  "
              f"平均(最近)={sum((e.get('max_stage') or 0) for e in last)/len(last):.2f}")
        print(f"结束: cleared={sum(1 for e in eps if e.get('cleared'))} "
              f"game_over={sum(1 for e in eps if e.get('game_over'))} "
              f"truncated={sum(1 for e in eps if e.get('truncated'))}")
        print(f"掉残(最近{len(last)}局合计)={sum(e.get('misses', 0) for e in last)}")
        rp = defaultdict(float)
        for e in eps:
            for k, v in (e.get("reward_parts") or {}).items():
                rp[k] += v
        print("奖励构成合计:", {k: round(v, 1) for k, v in sorted(rp.items(), key=lambda z: z[1])})
        ac = Counter()
        for e in eps:
            for k, v in (e.get("actions") or {}).items():
                ac[k] += v
        if ac:
            tot = sum(ac.values())
            print("动作分布(全局):", {k: f"{v}({pct(v, tot)})" for k, v in ac.most_common()})
        bomb = sum(v for k, v in ac.items() if "bomb" in k)
        print(f"炸弹动作总计: {bomb} 次 ({pct(bomb, sum(ac.values()))})")

    if steps:
        print("\n--- 逐步日志分析 ---")
        ac = Counter(s.get("action", "?") for s in steps)
        tot = sum(ac.values())
        print("动作分布:", {k: f"{v}({pct(v, tot)})" for k, v in ac.most_common()})
        bomb = sum(v for k, v in ac.items() if "bomb" in k)
        print(f"策略炸弹动作: {bomb} 次 ({pct(bomb, tot)})")
        guard = sum(1 for s in steps if s.get("bomb_guard"))
        dlg = sum(1 for s in steps if s.get("dialogue_assist"))
        print(f"紧急保护放弹(规则): {guard} 次 ({pct(guard, tot)}) | 对话辅助步: {dlg}")
        rp = defaultdict(float)
        for s in steps:
            for k, v in (s.get("reward_parts") or {}).items():
                rp[k] += float(v)
        print("每步奖励构成合计:", {k: round(v, 1) for k, v in sorted(rp.items(), key=lambda z: z[1])})
        deaths = [s for s in steps if s.get("done_reason")]
        print(f"结束步: {len(deaths)}")
        for s in deaths[:args.tail]:
            print(f"  ep{s.get('episode')} step{s.get('step')} reason={s.get('done_reason')} "
                  f"stage={s.get('stage')} lives={s.get('lives')} bullets={s.get('bullets')}")
        # 危险度/存活/掉残 与 弹幕数 的关系
        with_b = [s for s in steps if (s.get("bullets") or 0) > 0]
        print(f"有弹幕的步占比: {pct(len(with_b), len(steps))}  "
              f"平均弹幕数(有弹幕时)={sum(s['bullets'] for s in with_b)/len(with_b):.1f}" if with_b else "无弹幕步")
        alive = sum(1 for s in steps if (s.get("reward_parts") or {}).get("alive"))
        print(f"存活奖励步占比: {pct(alive, len(steps))}")
        print("面数分布:", dict(Counter(str(s.get('stage')) for s in steps).most_common()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
