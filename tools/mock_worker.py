"""独立进程的 mock 训练 worker —— 配合 tools/dual_proc_train.py 使用。

从 ``--init`` 载入策略, 用 ``--danmaku-script`` 跑 mock (规则老师 DAgger);
每隔 ``--merge-every`` 秒把策略写到 ``<out>/mock_policy.npz``, 并检查
``<out>/merged.npz`` 是否更新 (更新了就载入继续训练)。

放在独立进程里是为了**彻底绕开 GIL**: 真机进程需要 60fps 的实时性, 不能被
CPU 密集的 mock 线程饿死。
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from touhou_rl_mcp.config import Config  # noqa: E402
from touhou_rl_mcp.env import TouhouEnv  # noqa: E402
from touhou_rl_mcp.games.registry import GameRegistry  # noqa: E402
from touhou_rl_mcp.training import TrainConfig, Trainer  # noqa: E402
from touhou_rl_mcp.training.features import feature_dim  # noqa: E402
from touhou_rl_mcp.training.policy import MLPPolicy  # noqa: E402

#: 必须与 tools/dual_proc_train.py 的 FEATURES 保持一致
FEATURES = dict(k_bullets=16, bullet_velocity=True, k_enemies=8,
                use_invincible=True, use_boss=True, use_danger_grad=True,
                use_focus=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True, help="初始策略 npz")
    ap.add_argument("--danmaku-script", default="", help="真机弹幕脚本 (空=内置弹型)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seconds", type=float, default=900.0)
    ap.add_argument("--merge-every", type=float, default=90.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--step-sleep", type=float, default=0.0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    policy = MLPPolicy.load(Path(args.init).with_suffix(".npz"))
    obs_dim = feature_dim(FEATURES["k_bullets"], FEATURES["bullet_velocity"],
                          FEATURES["k_enemies"], FEATURES["use_invincible"],
                          FEATURES["use_boss"], FEATURES["use_danger_grad"])
    n_actions = 9 * (2 if FEATURES["use_focus"] else 1)
    if policy.obs_dim != obs_dim or policy.n_actions != n_actions:
        raise SystemExit(f"策略维度不匹配 {policy.obs_dim}/{policy.n_actions} != {obs_dim}/{n_actions}")

    cfg = Config({"backend": "mock", "game": "th11", "step_frames": 1,
                  "danmaku_script": args.danmaku_script})
    env = TouhouEnv(cfg, registry=GameRegistry())
    env.start()
    lock = threading.RLock()
    tc = TrainConfig(episodes=0, max_steps=24000, step_frames=1, batch_episodes=1,
                     lr=args.lr, entropy_coef=0.0, hidden=tuple(policy.hidden),
                     teacher="rule", danger_weight=3.0, step_sleep=args.step_sleep,
                     capture_every=0, eval_every=0, save_every=0,
                     difficulty=0, character=0, shot=0, **FEATURES)
    tr = Trainer(env, tc, env_lock=env.lock, out_dir=out, policy=policy, policy_lock=lock)
    print(f"[worker] 启动: script={args.danmaku_script or '(内置弹型)'} obs={obs_dim} "
          f"actions={n_actions}", flush=True)
    tr.start()

    merged_path = out / "merged.npz"
    mock_path = out / "mock_policy.npz"
    last_merged = merged_path.stat().st_mtime if merged_path.exists() else 0.0
    started = time.perf_counter()
    last_merge = started
    try:
        while time.perf_counter() - started < args.seconds:
            time.sleep(2.0)
            now = time.perf_counter()
            if now - last_merge >= args.merge_every:
                last_merge = now
                tr.save(mock_path)  # 供主进程合并
                if merged_path.exists() and merged_path.stat().st_mtime > last_merged:
                    try:
                        blended = MLPPolicy.load(merged_path)
                        with lock:
                            tr.policy = blended
                        last_merged = merged_path.stat().st_mtime
                        print(f"[worker] 载入合并策略 @t={now-started:.0f}s "
                              f"eps={len(tr.episodes)}", flush=True)
                    except Exception as exc:
                        print(f"[worker] 载入合并失败: {exc}", flush=True)
                print(f"[worker] t={now-started:5.0f}s eps={len(tr.episodes)} "
                      f"steps={tr.total_steps}", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        tr.stop(save=True)
        tr.save(mock_path)
        env.close()
        print(f"[worker] 结束: episodes={len(tr.episodes)} steps={tr.total_steps}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
