"""双线程训练: 真机线程 + mock 线程 **共享同一个策略**。

思路:
  * **真机线程** —— 在真机上用规则老师做 DAgger, 同时把真机弹幕录成脚本;
  * **mock 线程** —— 用录制下来的**真机弹幕脚本**跑 mock (可高频刷量), 同样 DAgger;
  * 两条线程共享同一个 ``MLPPolicy`` (用一把锁保护 forward/backward), 于是策略
    同时从"真机分布"和"真机弹幕的 mock 回放"里学习, 得到更稳的拟合。

    python tools/dual_train.py --seconds 900 --danmaku-dir .runtime/danmaku_real `
        --load .runtime/mock_dagger5/policy_best --out .runtime/dual

注意: 真机是 60fps 实时, 训练循环会占用键盘/窗口; mock 线程在 CPU 上高速跑。
"""
from __future__ import annotations

import argparse
import glob
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

#: 两条线程必须用**完全一致**的观测/动作配置, 才能共享策略
FEATURES = dict(k_bullets=16, bullet_velocity=True, k_enemies=8,
                use_invincible=True, use_boss=True, use_danger_grad=True,
                use_focus=True)


def newest_script(directory: str) -> str | None:
    files = [f for f in glob.glob(str(Path(directory) / "*.json"))]
    if not files:
        return None
    return max(files, key=lambda p: Path(p).stat().st_mtime)


def build_tc(**extra) -> TrainConfig:
    base = dict(episodes=0, max_steps=24000, step_frames=1, batch_episodes=1,
                lr=1e-3, entropy_coef=0.0, hidden=(256, 256),
                teacher="rule", danger_weight=3.0,
                capture_every=0, eval_every=0, save_every=0,
                difficulty=0, character=0, shot=0, **FEATURES)
    base.update(extra)
    return TrainConfig(**base)


def main() -> int:
    # 多线程共享 GIL: 缩短切换间隔, 让真机线程更及时拿到 GIL
    sys.setswitchinterval(0.001)
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=900.0, help="跑多久 (秒)")
    ap.add_argument("--danmaku-dir", default=".runtime/danmaku_real")
    ap.add_argument("--game-dir", default=r"F:\[th11] 东方地灵殿 (汉化版+日文版)")
    ap.add_argument("--load", default="", help="初始共享策略 (两线程都从这里开始)")
    ap.add_argument("--out", default=".runtime/dual")
    ap.add_argument("--difficulty", type=int, default=0)
    ap.add_argument("--character", type=int, default=0)
    ap.add_argument("--shot", type=int, default=0)
    ap.add_argument("--reload-every", type=float, default=120.0,
                    help="mock 线程每隔多少秒换用最新录到的弹幕脚本")
    ap.add_argument("--mock-step-sleep", type=float, default=0.003,
                    help="mock 线程每步 sleep 秒数 (给真机线程让出 GIL; 0=不让)")
    ap.add_argument("--mock-only", action="store_true", help="只跑 mock 线程 (不占用真机)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    script = newest_script(args.danmaku_dir)
    print(f"[dual] 初始真机弹幕脚本: {script}", flush=True)

    # ---- 共享策略 ----
    obs_dim = feature_dim(FEATURES["k_bullets"], FEATURES["bullet_velocity"],
                          FEATURES["k_enemies"], FEATURES["use_invincible"],
                          FEATURES["use_boss"], FEATURES["use_danger_grad"])
    n_actions = 9 * (2 if FEATURES["use_focus"] else 1)
    if args.load:
        policy = MLPPolicy.load(Path(args.load).with_suffix(".npz"))
        if policy.obs_dim != obs_dim or policy.n_actions != n_actions:
            raise SystemExit(f"载入策略维度不匹配: {policy.obs_dim}/{policy.n_actions} "
                             f"期望 {obs_dim}/{n_actions}")
        print(f"[dual] 载入共享策略 {args.load} ({policy.describe()})", flush=True)
    else:
        policy = MLPPolicy(obs_dim, n_actions, hidden=(256, 256), seed=20261018,
                           lr=1e-3, entropy_coef=0.0)
        print("[dual] 新建共享策略", flush=True)
    lock = threading.RLock()

    trainers: list[tuple[str, Trainer, TouhouEnv]] = []

    # ---- mock 线程 ----
    mock_cfg = Config({"backend": "mock", "game": "th11", "step_frames": 1,
                       "danmaku_script": script or ""})
    mock_env = TouhouEnv(mock_cfg, registry=GameRegistry())
    mock_env.start()
    mock_tr = Trainer(mock_env, build_tc(step_sleep=args.mock_step_sleep), env_lock=mock_env.lock,
                      out_dir=out / "mock", policy=policy, policy_lock=lock)
    trainers.append(("mock", mock_tr, mock_env))

    # ---- 真机线程 ----
    if not args.mock_only:
        real_cfg = Config({"backend": "memory", "game": "th11",
                           "game_dir": args.game_dir, "step_frames": 1,
                           "restart_process": True, "dialogue_assist": True,
                           "capture": {"prefer_printwindow": False}})
        real_env = TouhouEnv(real_cfg, registry=GameRegistry())
        real_tc = build_tc(difficulty=args.difficulty, character=args.character,
                           shot=args.shot)
        real_tr = Trainer(real_env, real_tc, env_lock=real_env.lock,
                          out_dir=out / "real", danmaku_dir=args.danmaku_dir,
                          policy=policy, policy_lock=lock)
        trainers.append(("real", real_tr, real_env))

    try:
        for name, tr, env in trainers:
            env.start()
            tr.start()
            print(f"[dual] 启动 {name} 线程", flush=True)

        started = time.perf_counter()
        last_reload = started
        while time.perf_counter() - started < args.seconds:
            time.sleep(5.0)
            # mock 线程换用最新录到的真机弹幕
            now = time.perf_counter()
            if not args.mock_only and now - last_reload >= args.reload_every:
                last_reload = now
                newest = newest_script(args.danmaku_dir)
                if newest and newest != script:
                    script = newest
                    mock_env.backend.set_danmaku_script(script)
                    print(f"[dual] mock 换用新弹幕脚本: {script}", flush=True)
            line = [f"[dual] t={now-started:6.0f}s"]
            for name, tr, _env in trainers:
                eps = len(tr.episodes)
                stages = [e.max_stage or 0 for e in tr.episodes[-20:]]
                last = tr.episodes[-1] if tr.episodes else None
                extra = ""
                if last is not None and last.duration_s > 0:
                    extra = (f" last={last.steps}步/{last.duration_s:.0f}s"
                             f"({last.steps / last.duration_s:.0f}步/秒)")
                line.append(f"{name}: eps={eps} steps={tr.total_steps} "
                            f"best_stage={max(stages) if stages else 0}{extra}")
            print(" | ".join(line), flush=True)

        for name, tr, _env in trainers:
            st = tr.stop(save=True)
            print(f"[dual] {name} 停止: episodes={st['episodes_done']} "
                  f"steps={st['total_steps']} error={st['error']}", flush=True)
    except KeyboardInterrupt:
        print("[dual] Ctrl-C, 停止...", flush=True)
        for name, tr, _env in trainers:
            try:
                tr.stop(save=True)
            except Exception:
                pass
    finally:
        for _name, _tr, env in trainers:
            try:
                env.close()
            except Exception:
                pass
        saved = policy.save(out / "policy_shared")
        print(f"[dual] 共享策略已保存: {saved}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
