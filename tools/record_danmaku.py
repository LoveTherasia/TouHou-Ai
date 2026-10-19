"""录制真机弹幕 -> 生成脚本 (供 mock 复刻)。支持**多局游玩**。

东方 .rpy 只存输入不存弹幕, 所以只能"进游戏逐帧读弹幕"来复刻。
本工具用规则老师边躲边录, 以 60Hz 逐帧采集, 每局一个文件, 并按
**难度 / 自机角色 / 机体** 区分命名 (弹幕与它们强相关)。

    python tools/record_danmaku.py --runs 3 --difficulty 0 --character 0 --shot 0 `
        --out-dir .runtime/danmaku

产物: ``.runtime/danmaku/th11_d0_c0_s0_run1.json`` ...

前提: TH11 已安装且能被本项目的 memory 后端启动 (窗口化)。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from touhou_rl_mcp.config import Config  # noqa: E402
from touhou_rl_mcp.env import TouhouEnv  # noqa: E402
from touhou_rl_mcp.games.registry import GameRegistry  # noqa: E402
from touhou_rl_mcp.memory.danmaku import DanmakuRecorder  # noqa: E402
from touhou_rl_mcp.training.rule_teacher import RuleTeacher  # noqa: E402


def record_one(env, args, run: int, out_dir: Path) -> dict:
    rec = DanmakuRecorder()
    teacher = RuleTeacher()
    state = env.reset(difficulty=args.difficulty, character=args.character,
                      shot=args.shot, restart_process=args.restart).state
    if [state.char_main, state.char_sub] != [args.character, args.shot]:
        print(f"[rec] 警告: 实际自机 {(state.char_main, state.char_sub)} != 请求 "
              f"{(args.character, args.shot)}", flush=True)
    print(f"[rec] run{run} 进入游戏: stage={state.stage} lives={state.lives} "
          f"char={state.char_main}{state.char_sub}", flush=True)
    max_frames = int(args.seconds * 60)
    started = time.perf_counter()
    last_stage = state.stage
    for i in range(max_frames):
        rec.update(state)
        action, _ = teacher.choose(state, state.frame, "shoot")
        result = env.step(action, frames=1)
        state = result.state
        if state.stage and state.stage != last_stage:
            last_stage = state.stage
            print(f"[rec] run{run} 进入第 {state.stage} 面 @frame {i}", flush=True)
        if i % 600 == 0:
            print(f"[rec] run{run} {i:5d}帧 stage={state.stage} lives={state.lives} "
                  f"bullets={state.bullet_count} 生成={len(rec.spawns)}", flush=True)
        if result.done or result.truncated:
            print(f"[rec] run{run} 结束 @frame {i}: {result.info.get('done_reason')}", flush=True)
            break
    path = out_dir / (f"{env.spec.key}_d{args.difficulty}_c{args.character}_s{args.shot}"
                      f"_run{run}.json")
    rec.save(path, game=env.spec.key, difficulty=args.difficulty,
             character=args.character, shot=args.shot)
    summary = {"run": run, "path": str(path), "frames": rec.frame,
               "spawns": len(rec.spawns), "max_stage": state.stage,
               "cleared": bool(result.done and result.info.get("done_reason") == "cleared"),
               "seconds": round(time.perf_counter() - started, 1)}
    print(f"[rec] run{run} 完成: 帧={rec.frame} 生成={len(rec.spawns)} "
          f"最远面={state.stage} -> {path}", flush=True)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3, help="游玩/录制几局")
    ap.add_argument("--seconds", type=float, default=200.0, help="每局最多录多少游戏内秒")
    ap.add_argument("--out-dir", default=".runtime/danmaku")
    ap.add_argument("--game-dir", default=r"F:\[th11] 东方地灵殿 (汉化版+日文版)")
    ap.add_argument("--difficulty", type=int, default=0)
    ap.add_argument("--character", type=int, default=0)
    ap.add_argument("--shot", type=int, default=0)
    ap.add_argument("--no-restart", dest="restart", action="store_false",
                    help="每局不重启进程 (默认重启, 保证全新一局)")
    ap.set_defaults(restart=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = Config({
        "backend": "memory", "game": "th11",
        "game_dir": args.game_dir,
        "restart_process": bool(args.restart),
        "dialogue_assist": True,
        "capture": {"prefer_printwindow": False},
    })
    env = TouhouEnv(cfg, registry=GameRegistry())
    summaries = []
    try:
        env.start()
        for run in range(1, args.runs + 1):
            summaries.append(record_one(env, args, run, out_dir))
    except Exception as exc:
        import traceback
        print("ERROR:", type(exc).__name__, exc)
        traceback.print_exc()
        return 1
    finally:
        env.close()
    best = max((s["max_stage"] or 0) for s in summaries) if summaries else 0
    print(f"\n[rec] 全部完成: {len(summaries)} 局, 最远面={best}")
    for s in summaries:
        print(f"   run{s['run']}: 面={s['max_stage']} 通关={s['cleared']} "
              f"生成={s['spawns']} 帧={s['frames']} ({s['seconds']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
