r"""循环自动化流水线: 真机 + mock 双进程 -> 合并 -> 继续 -> 再合并 ... 往返循环。

每一轮 (round):
  1. 从当前策略出发, 启动**独立进程**的 mock worker (用最新录到的真机弹幕);
  2. 主进程跑**真机训练** (规则老师 DAgger, 同时把弹幕录到 --danmaku-dir);
  3. 每隔 --merge-every 秒: 真机策略 + mock 策略 **按权重平均** -> merged.npz,
     真机进程换上合并策略继续 (mock worker 也会自动载入);
  4. 本轮结束: 再合并一次, 作为**下一轮的起点** (policy_current.npz);
  5. 回到 1) 开始下一轮, 直到 --rounds 用尽 (0 = 无限, Ctrl-C 停止)。

    # 一键循环 (默认 Normal 难度, 每轮真机 10 分钟)
    .\.venv\Scripts\python.exe tools\auto_pipeline.py --rounds 0 --real-seconds 600

    # 只跑 mock (不占用真机, 用于验证流程)
    .\.venv\Scripts\python.exe tools\auto_pipeline.py --no-real --rounds 2 --real-seconds 60
"""
from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from touhou_rl_mcp.config import Config  # noqa: E402
from touhou_rl_mcp.env import TouhouEnv  # noqa: E402
from touhou_rl_mcp.games.registry import GameRegistry  # noqa: E402
from touhou_rl_mcp.training import TrainConfig, Trainer, blend_policies  # noqa: E402
from touhou_rl_mcp.training.features import feature_dim  # noqa: E402
from touhou_rl_mcp.training.policy import MLPPolicy  # noqa: E402

#: 必须与 tools/mock_worker.py 的 FEATURES 保持一致
FEATURES = dict(k_bullets=16, bullet_velocity=True, k_enemies=8,
                use_invincible=True, use_boss=True, use_danger_grad=True,
                use_focus=True)
HERE = Path(__file__).resolve().parent


def log(msg: str) -> None:
    print(f"[auto] {msg}", flush=True)


def newest_script(directory: str) -> str:
    files = glob.glob(str(Path(directory) / "*.json"))
    return max(files, key=lambda p: Path(p).stat().st_mtime) if files else ""


def write_meta(path: str | Path, policy: MLPPolicy) -> None:
    meta = {"config": {**FEATURES, "hidden": list(policy.hidden), "use_bomb": False}}
    Path(path).with_suffix(".json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def build_dims() -> tuple[int, int]:
    obs = feature_dim(FEATURES["k_bullets"], FEATURES["bullet_velocity"], FEATURES["k_enemies"],
                      FEATURES["use_invincible"], FEATURES["use_boss"], FEATURES["use_danger_grad"])
    return obs, 9 * (2 if FEATURES["use_focus"] else 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=0, help="跑几轮 (0=无限, Ctrl-C 停)")
    ap.add_argument("--real-seconds", type=float, default=600.0, help="每轮真机训练秒数")
    ap.add_argument("--mock-seconds", type=float, default=0.0, help="每轮 mock 秒数 (0=跟真机一致)")
    ap.add_argument("--merge-every", type=float, default=120.0)
    ap.add_argument("--merge-alpha", type=float, default=0.5, help="合并时真机策略权重")
    ap.add_argument("--difficulty", type=int, default=1)
    ap.add_argument("--character", type=int, default=0)
    ap.add_argument("--shot", type=int, default=0)
    ap.add_argument("--danmaku-dir", default=".runtime/danmaku_real")
    ap.add_argument("--game-dir", default=r"F:\[th11] 东方地灵殿 (汉化版+日文版)")
    ap.add_argument("--load", default="", help="初始策略 (可选)")
    ap.add_argument("--out", default=".runtime/auto_pipeline")
    ap.add_argument("--no-real", action="store_true", help="只跑 mock (不占用真机)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    obs_dim, n_actions = build_dims()
    mock_seconds = args.mock_seconds or args.real_seconds

    # ---- 当前策略 (循环的"接力棒") ----
    current_path = out / "policy_current.npz"
    if args.load and Path(args.load).with_suffix(".npz").exists():
        policy = MLPPolicy.load(Path(args.load).with_suffix(".npz"))
    elif current_path.exists():
        policy = MLPPolicy.load(current_path)
        log(f"从上一轮的当前策略继续: {current_path}")
    else:
        policy = MLPPolicy(obs_dim, n_actions, hidden=(256, 256), seed=20261019,
                           lr=1e-3, entropy_coef=0.0)
    if policy.obs_dim != obs_dim or policy.n_actions != n_actions:
        raise SystemExit(f"策略维度不匹配: {policy.obs_dim}/{policy.n_actions} != {obs_dim}/{n_actions}")
    policy.save(current_path)
    write_meta(current_path, policy)
    log(f"初始/当前策略 obs={obs_dim} actions={n_actions} -> {current_path}")

    # ---- 真机环境 (跨轮复用) ----
    real_env = None
    real_tr = None
    lock = threading.RLock()
    if not args.no_real:
        real_cfg = Config({"backend": "memory", "game": "th11",
                           "game_dir": args.game_dir, "step_frames": 1,
                           "restart_process": True, "dialogue_assist": True,
                           "capture": {"prefer_printwindow": False}})
        real_env = TouhouEnv(real_cfg, registry=GameRegistry())
        real_env.start()

    rounds_done = 0
    try:
        while args.rounds == 0 or rounds_done < args.rounds:
            rounds_done += 1
            rdir = out / f"round{rounds_done:03d}"
            rdir.mkdir(parents=True, exist_ok=True)
            log(f"========== ROUND {rounds_done} ==========")
            script = newest_script(args.danmaku_dir)
            log(f"mock 弹幕脚本: {script or '(内置弹型)'}")

            # 1) 独立进程 mock worker
            mock_out = rdir / "mock"
            worker = subprocess.Popen([
                sys.executable, str(HERE / "mock_worker.py"),
                "--init", str(current_path), "--danmaku-script", script,
                "--out", str(mock_out), "--seconds", str(mock_seconds),
                "--merge-every", str(args.merge_every),
            ], cwd=str(HERE.parent))
            log(f"启动 mock worker (pid={worker.pid}, {mock_seconds:.0f}s)")

            merged_path = rdir / "merged.npz"
            real_path = rdir / "real_policy.npz"
            mock_path = mock_out / "mock_policy.npz"

            # 2) 真机训练 (或 mock-only 时直接等 worker)
            if args.no_real:
                while worker.poll() is None:
                    time.sleep(5.0)
                log("(no-real) mock worker 结束")
            else:
                # 让真机每轮至少能打完 1 局 (否则本轮不产生训练更新):
                # 真机约 60 步/秒, 再扣掉重启/菜单的固定开销。
                real_max_steps = max(4000, int(max(60.0, args.real_seconds - 80.0) * 55.0))
                real_tc = TrainConfig(episodes=0, max_steps=real_max_steps, step_frames=1,
                                      batch_episodes=1, lr=1e-3, entropy_coef=0.0,
                                      hidden=(256, 256), teacher="rule", danger_weight=3.0,
                                      capture_every=0, eval_every=0, save_every=0,
                                      difficulty=args.difficulty, character=args.character,
                                      shot=args.shot, **FEATURES)
                real_tr = Trainer(real_env, real_tc, env_lock=real_env.lock,
                                  out_dir=rdir / "real", danmaku_dir=args.danmaku_dir,
                                  policy=policy, policy_lock=lock)
                real_tr.start()
                log(f"真机训练启动 (难度={args.difficulty})")
                started = time.perf_counter()
                last_merge = started
                while time.perf_counter() - started < args.real_seconds:
                    time.sleep(5.0)
                    now = time.perf_counter()
                    if now - last_merge >= args.merge_every and mock_path.exists():
                        last_merge = now
                        try:
                            real_tr.save(real_path)
                            rp = MLPPolicy.load(real_path)
                            mp = MLPPolicy.load(mock_path)
                            blended = blend_policies([rp, mp],
                                                     [args.merge_alpha, 1.0 - args.merge_alpha])
                            blended.save(merged_path)
                            write_meta(merged_path, blended)
                            with lock:
                                real_tr.policy = blended
                            log(f"t={now-started:5.0f}s 合并 real+mock -> {merged_path}")
                        except Exception as exc:
                            log(f"合并失败: {exc}")
                    last = real_tr.episodes[-1] if real_tr.episodes else None
                    info = f" last={last.steps}步 面={last.max_stage}" if last else ""
                    log(f"t={now-started:5.0f}s 真机eps={len(real_tr.episodes)} "
                        f"steps={real_tr.total_steps}{info}")
                real_tr.stop(save=True)
                real_tr.save(real_path)
                log("真机本轮结束, 等 mock worker...")
                try:
                    worker.wait(timeout=max(30.0, mock_seconds))
                except Exception:
                    worker.terminate()

            # 3) 本轮合并 -> 下一轮起点
            try:
                rp = MLPPolicy.load(real_path) if real_path.exists() else policy
                if mock_path.exists():
                    mp = MLPPolicy.load(mock_path)
                    nxt = blend_policies([rp, mp], [args.merge_alpha, 1.0 - args.merge_alpha])
                else:
                    nxt = rp
                policy = nxt
                policy.save(current_path)
                write_meta(current_path, policy)
                (rdir / "policy_round").parent.mkdir(parents=True, exist_ok=True)
                policy.save(rdir / "policy_round")
                write_meta(rdir / "policy_round", policy)
                log(f"ROUND {rounds_done} 完成 -> 下一轮起点 {current_path}")
            except Exception as exc:
                log(f"本轮合并失败: {exc}")
    except KeyboardInterrupt:
        log("Ctrl-C: 收尾保存...")
    finally:
        if real_tr is not None:
            try:
                real_tr.stop(save=True)
            except Exception:
                pass
        if real_env is not None:
            try:
                real_env.close()
            except Exception:
                pass
        try:
            policy.save(out / "policy_final")
            write_meta(out / "policy_final", policy)
            log(f"最终策略: {out / 'policy_final.npz'}")
        except Exception:
            pass
    log(f"结束: 共 {rounds_done} 轮")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
