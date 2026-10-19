"""真机 + mock **双进程**训练 + 周期合并 (model soup)。

流程:
  1. 建一个初始策略 -> <out>/init.npz, 真机进程与 mock 进程都从它出发;
  2. 启动 **独立进程** 的 mock worker (tools/mock_worker.py) —— 绕开 GIL, 不拖慢真机;
  3. 主进程跑**真机训练** (更高难度, 规则老师 DAgger, 同时录弹幕);
  4. 每隔 --merge-every 秒: 真机策略 + mock 策略 **按权重平均** -> <out>/merged.npz,
     主进程换上合并后的策略继续真机训练 (mock worker 也会自动载入它);
  5. 结束后再合并一次, 保存 <out>/policy_final.npz。

    python tools/dual_proc_train.py --real-seconds 900 --mock-seconds 900 `
        --difficulty 1 --danmaku-dir .runtime/danmaku_real --out .runtime/dualproc
"""
from __future__ import annotations

import argparse
import glob
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


def newest_script(directory: str) -> str:
    files = glob.glob(str(Path(directory) / "*.json"))
    return max(files, key=lambda p: Path(p).stat().st_mtime) if files else ""


def write_meta(path: str | Path, policy) -> None:
    """写策略元数据 json (供 clear_stage1 / play_real 等真机工具读取维度)。"""
    import json

    meta = {"config": {**FEATURES, "hidden": list(policy.hidden), "use_bomb": False}}
    Path(path).with_suffix(".json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-seconds", type=float, default=900.0)
    ap.add_argument("--mock-seconds", type=float, default=900.0)
    ap.add_argument("--merge-every", type=float, default=90.0)
    ap.add_argument("--merge-alpha", type=float, default=0.5,
                    help="合并时真机策略的权重 (其余给 mock)")
    ap.add_argument("--difficulty", type=int, default=1, help="真机难度 0=Easy..3=Lunatic")
    ap.add_argument("--character", type=int, default=0)
    ap.add_argument("--shot", type=int, default=0)
    ap.add_argument("--danmaku-dir", default=".runtime/danmaku_real")
    ap.add_argument("--game-dir", default=r"F:\[th11] 东方地灵殿 (汉化版+日文版)")
    ap.add_argument("--load", default="", help="初始策略 (可选)")
    ap.add_argument("--out", default=".runtime/dualproc")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    obs_dim = feature_dim(FEATURES["k_bullets"], FEATURES["bullet_velocity"],
                          FEATURES["k_enemies"], FEATURES["use_invincible"],
                          FEATURES["use_boss"], FEATURES["use_danger_grad"])
    n_actions = 9 * (2 if FEATURES["use_focus"] else 1)

    if args.load and Path(args.load).with_suffix(".npz").exists():
        policy = MLPPolicy.load(Path(args.load).with_suffix(".npz"))
    else:
        policy = MLPPolicy(obs_dim, n_actions, hidden=(256, 256), seed=20261019,
                           lr=1e-3, entropy_coef=0.0)
    if policy.obs_dim != obs_dim or policy.n_actions != n_actions:
        raise SystemExit(f"策略维度不匹配: {policy.obs_dim}/{policy.n_actions} != {obs_dim}/{n_actions}")
    init_path = out / "init.npz"
    policy.save(init_path)
    print(f"[coord] 初始策略 obs={obs_dim} actions={n_actions} -> {init_path}", flush=True)

    script = newest_script(args.danmaku_dir)
    print(f"[coord] mock 使用弹幕脚本: {script or '(内置弹型)'}", flush=True)
    worker_cmd = [
        sys.executable, str(Path(__file__).with_name("mock_worker.py")),
        "--init", str(init_path), "--danmaku-script", script, "--out", str(out / "mock"),
        "--seconds", str(args.mock_seconds), "--merge-every", str(args.merge_every),
    ]
    worker = subprocess.Popen(worker_cmd, cwd=str(Path(__file__).resolve().parent.parent))
    print(f"[coord] 已启动 mock worker (pid={worker.pid})", flush=True)

    # ---- 真机训练 (主进程) ----
    real_cfg = Config({"backend": "memory", "game": "th11",
                       "game_dir": args.game_dir, "step_frames": 1,
                       "restart_process": True, "dialogue_assist": True,
                       "capture": {"prefer_printwindow": False}})
    real_env = TouhouEnv(real_cfg, registry=GameRegistry())
    lock = threading.RLock()
    real_tc = TrainConfig(episodes=0, max_steps=30000, step_frames=1, batch_episodes=1,
                          lr=1e-3, entropy_coef=0.0, hidden=(256, 256),
                          teacher="rule", danger_weight=3.0,
                          capture_every=0, eval_every=0, save_every=0,
                          difficulty=args.difficulty, character=args.character,
                          shot=args.shot, **FEATURES)
    real_tr = Trainer(real_env, real_tc, env_lock=real_env.lock, out_dir=out / "real",
                      danmaku_dir=args.danmaku_dir, policy=policy, policy_lock=lock)
    merged_path = out / "merged.npz"
    real_path = out / "real_policy.npz"
    mock_path = out / "mock" / "mock_policy.npz"

    try:
        real_env.start()
        real_tr.start()
        print(f"[coord] 真机训练启动 (难度={args.difficulty})", flush=True)
        started = time.perf_counter()
        last_merge = started
        while time.perf_counter() - started < args.real_seconds:
            time.sleep(5.0)
            now = time.perf_counter()
            if now - last_merge >= args.merge_every:
                last_merge = now
                real_tr.save(real_path)
                if mock_path.exists():
                    try:
                        rp = MLPPolicy.load(real_path)
                        mp = MLPPolicy.load(mock_path)
                        blended = blend_policies([rp, mp],
                                                 [args.merge_alpha, 1.0 - args.merge_alpha])
                        blended.save(merged_path)
                        with lock:
                            real_tr.policy = blended
                        print(f"[coord] t={now-started:5.0f}s 合并完成 "
                              f"(real+mock) -> {merged_path} | 真机eps={len(real_tr.episodes)}",
                              flush=True)
                    except Exception as exc:
                        print(f"[coord] 合并失败: {exc}", flush=True)
                else:
                    print(f"[coord] t={now-started:5.0f}s 等待 mock 策略...", flush=True)
            last = real_tr.episodes[-1] if real_tr.episodes else None
            info = (f" last={last.steps}步/{last.duration_s:.0f}s 面={last.max_stage}"
                    if last else "")
            print(f"[coord] t={now-started:5.0f}s 真机eps={len(real_tr.episodes)} "
                  f"steps={real_tr.total_steps}{info}", flush=True)
    except KeyboardInterrupt:
        print("[coord] Ctrl-C", flush=True)
    finally:
        real_tr.stop(save=True)
        real_tr.save(real_path)
        print("[coord] 真机线程已停, 等 mock worker 结束...", flush=True)
        try:
            worker.wait(timeout=max(30.0, args.mock_seconds))
        except Exception:
            worker.terminate()
        # 最终合并
        try:
            rp = MLPPolicy.load(real_path)
            if mock_path.exists():
                mp = MLPPolicy.load(mock_path)
                final = blend_policies([rp, mp], [args.merge_alpha, 1.0 - args.merge_alpha])
            else:
                final = rp
            saved = final.save(out / "policy_final")
            write_meta(saved, final)
            print(f"[coord] 最终策略: {saved}", flush=True)
        except Exception as exc:
            print(f"[coord] 最终合并失败: {exc}", flush=True)
        real_env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
