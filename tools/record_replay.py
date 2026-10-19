r"""在游戏里**播放 replay**, 逐帧录制完整弹幕 (全 6 面)。

东方 .rpy 只存输入不存弹幕, 但**游戏自带回放播放器**能完整重演;
我们只要让游戏播回放, 然后逐帧读弹幕即可 (不需要解析 .rpy)。

    python tools/record_replay.py --seconds 2400 --out .runtime/danmaku/th11_replay_full.json

菜单: 标题(=主菜单, Game Start 高亮) -> 下 N 次到 Replay -> confirm -> confirm 开始播放。
播放期间**不发任何按键**, 只逐帧读状态。
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from touhou_rl_mcp.config import Config  # noqa: E402
from touhou_rl_mcp.env import TouhouEnv  # noqa: E402
from touhou_rl_mcp.games.registry import GameRegistry  # noqa: E402
from touhou_rl_mcp.memory.danmaku import DanmakuRecorder  # noqa: E402
from touhou_rl_mcp.training.features import encode_state  # noqa: E402

#: 行为克隆用的观测配置 (须与训练/策略一致)
BC_FEATURES = dict(k_bullets=16, use_velocity=True, k_enemies=8,
                   use_invincible=True, use_boss=True, use_danger_grad=True)
_DIRMAP = {(0, 0): 0, (0, -1): 1, (0, 1): 2, (-1, 0): 3, (1, 0): 4,
           (-1, -1): 5, (1, -1): 6, (-1, 1): 7, (1, 1): 8}


def infer_action(p0, p1):
    """从真人自机的逐帧位移**反推**动作索引 (与 build_action_table 的 18 动作一致)。"""
    if p0 is None or p1 is None or p0[0] is None or p1[0] is None:
        return None
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    sp = math.hypot(dx, dy)
    if sp < 0.6:
        return 0  # 原地
    ux = 1 if dx > 0.35 * sp else (-1 if dx < -0.35 * sp else 0)
    uy = 1 if dy > 0.35 * sp else (-1 if dy < -0.35 * sp else 0)
    d = _DIRMAP.get((ux, uy), 0)
    focus = sp < 2.8  # 真机: 高速~4.5 / 低速~2.0
    return d + (9 if focus else 0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=2400.0, help="最多录多少游戏内秒")
    ap.add_argument("--out", default=".runtime/danmaku/th11_replay_full.json")
    ap.add_argument("--game-dir", default=r"F:\[th11] 东方地灵殿 (汉化版+日文版)")
    ap.add_argument("--menu-downs", type=int, default=3,
                    help="从 Game Start 往下按几次到 Replay")
    ap.add_argument("--replay-index", type=int, default=0, help="回放列表里选第几个")
    ap.add_argument("--bc-out", default="",
                    help="同时保存 (obs, action) 数据集供行为克隆 (学真人操作)")
    ap.add_argument("--restart", action="store_true", default=True)
    ap.add_argument("--wait", type=float, default=12.0, help="重启后等多久再导航 (秒)")
    ap.add_argument("--min-spawns", type=int, default=500,
                    help="生成事件少于它就认为失败, 不覆盖已有脚本")
    args = ap.parse_args()

    cfg = Config({"backend": "memory", "game": "th11",
                  "game_dir": args.game_dir, "restart_process": bool(args.restart),
                  "dialogue_assist": False,
                  "capture": {"prefer_printwindow": False}})
    env = TouhouEnv(cfg, registry=GameRegistry())
    rec = DanmakuRecorder()
    try:
        env.start()
        env.reset(difficulty=0, character=0, shot=0, restart_process=args.restart,
                  menu="none")  # 只重启, 不做菜单导航
        print("[rep] 菜单导航到 Replay (全部短按; 失败会重试)", flush=True)

        def navigate() -> bool:
            c = env.backend.controller
            c.release_all()
            for _ in range(max(0, args.menu_downs)):
                c.tap(["down"], hold=2, after=16)
            c.tap(["confirm"], hold=2, after=90)      # -> 回放页面
            time.sleep(0.6)
            for _ in range(max(0, args.replay_index)):
                c.tap(["down"], hold=2, after=16)
            c.tap(["confirm"], hold=2, after=90)      # -> 选中该回放
            time.sleep(0.6)
            c.tap(["confirm"], hold=2, after=90)      # -> 从第 1 面开始播放
            time.sleep(2.5)
            return env.backend.read_state().in_game

        ok = False
        for attempt in range(3):
            if attempt:
                print("[rep] 重试 %d: 重启游戏" % attempt, flush=True)
                env.reset(difficulty=0, character=0, shot=0,
                          restart_process=True, menu="none")
            time.sleep(args.wait)
            if navigate():
                ok = True
                break
            print("[rep] 未进入回放 (attempt %d)" % (attempt + 1), flush=True)
        if not ok:
            print("[rep] 3 次都没能进入回放, 放弃 (不覆盖已有脚本)", flush=True)
            return 1
        st = env.backend.read_state()
        print("[rep] 开始播放: in_game=%s stage=%s lives=%s bullets=%s"
              % (st.in_game, st.stage, st.lives, st.bullet_count), flush=True)
        max_frames = int(args.seconds * 60)
        out_of_game = 0
        frozen = 0
        last_game_frame = None
        started = time.perf_counter()
        last_save = started
        use_bc = bool(args.bc_out)
        bc_obs: list = []
        bc_act: list = []
        i = 0
        for i in range(max_frames):
            rec.update(st)
            obs_cur = None
            if use_bc:
                try:
                    obs_cur = encode_state(st, **BC_FEATURES)
                except Exception:
                    obs_cur = None
            env.backend.advance(1)
            nxt = env.backend.read_state()
            if use_bc and obs_cur is not None:
                act = infer_action((st.player_x, st.player_y), (nxt.player_x, nxt.player_y))
                if act is not None:
                    bc_obs.append(obs_cur)
                    bc_act.append(act)
            st = nxt
            if i % 1800 == 0:
                print("[rep] t=%5.0fs frame=%6d stage=%s lives=%s bullets=%s 生成=%d"
                      % (time.perf_counter() - started, i, st.stage, st.lives,
                         st.bullet_count, len(rec.spawns)), flush=True)
            if time.perf_counter() - last_save > 300:
                last_save = time.perf_counter()
                try:
                    rec.save(args.out, game="th11", fps=60, difficulty=0,
                             character=0, shot=0)
                    print("[rep] 中途保存 (%d 生成)" % len(rec.spawns), flush=True)
                except Exception as exc:
                    print("[rep] 中途保存失败:", exc, flush=True)
                if use_bc and bc_act:
                    try:
                        Path(args.bc_out).parent.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(args.bc_out,
                                            obs=np.stack(bc_obs).astype(np.float32),
                                            action=np.asarray(bc_act, dtype=np.int64))
                        print("[rep] BC 中途保存 (%d 样本)" % len(bc_act), flush=True)
                    except Exception as exc:
                        print("[rep] BC 中途保存失败:", exc, flush=True)
            if st.stage is None:          # 回到菜单 = 回放结束
                print("[rep] 回到菜单, 回放结束", flush=True)
                break
            if st.frame == last_game_frame:
                frozen += 1
                if frozen > 180:          # 帧计数冻结 -> 结算/结束画面
                    print("[rep] 游戏帧冻结, 回放结束", flush=True)
                    break
            else:
                frozen = 0
                last_game_frame = st.frame
            if not st.in_game:
                out_of_game += 1
                if out_of_game > 600 and i > 600:
                    print("[rep] 已离开局内 (回放结束?)", flush=True)
                    break
            else:
                out_of_game = 0
        if len(rec.spawns) < args.min_spawns:
            print("[rep] 录制太少 (%d 生成), 不覆盖 %s" % (len(rec.spawns), args.out), flush=True)
            return 1
        if use_bc and bc_act:
            Path(args.bc_out).parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.bc_out,
                                obs=np.stack(bc_obs).astype(np.float32),
                                action=np.asarray(bc_act, dtype=np.int64))
            print("[rep] BC 数据: %d 样本 -> %s" % (len(bc_act), args.bc_out), flush=True)
        path = rec.save(args.out, game="th11", fps=60, difficulty=0,
                        character=0, shot=0)
        print("[rep] 完成: 帧=%d 生成=%d 丢弃=%d -> %s"
              % (rec.frame, len(rec.spawns), rec.dropped, path), flush=True)
    except Exception as exc:
        import traceback
        print("ERROR:", type(exc).__name__, exc)
        traceback.print_exc()
        return 1
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
