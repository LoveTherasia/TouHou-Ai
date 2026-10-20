"""按弹幕类型**分别训练专属策略** (每类一个)。

每种弹型用"横向游走规则老师"做 DAgger, 得到该弹型的专属策略, 存到
``<out>/<pattern>/policy_best.npz``。训练完再各自做贪心评估 (存活步数)。

    python tools/train_patterns.py --episodes 120 --out .runtime/patterns
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from touhou_rl_mcp.backends.patterns import PATTERN_DOCS, pattern_names  # noqa: E402
from touhou_rl_mcp.config import Config  # noqa: E402
from touhou_rl_mcp.env import TouhouEnv  # noqa: E402
from touhou_rl_mcp.games.registry import GameRegistry  # noqa: E402
from touhou_rl_mcp.training import TrainConfig, Trainer  # noqa: E402
from touhou_rl_mcp.training.features import encode_state  # noqa: E402
from touhou_rl_mcp.training.trainer import build_action_table  # noqa: E402

F = dict(k_bullets=16, use_velocity=True, k_enemies=8,
         use_invincible=True, use_boss=True, use_danger_grad=True)
#: TrainConfig 用的是 bullet_velocity (名字不同)
TC_F = dict(k_bullets=16, bullet_velocity=True, k_enemies=8,
            use_invincible=True, use_boss=True, use_danger_grad=True)


def make_env(pattern: str) -> TouhouEnv:
    env = TouhouEnv(Config({"backend": "mock", "game": "th11", "step_frames": 1,
                            "danmaku_pattern": pattern}), registry=GameRegistry())
    env.start()
    return env


def greedy_survival(env: TouhouEnv, policy, actions, seeds=(101, 202, 303), cap=6000) -> float:
    ticks = []
    for seed in seeds:
        env.reset(seed=seed)
        n = 0
        for i in range(cap):
            st = env.get_state()
            a = actions[policy.greedy(encode_state(st, **F))]
            r = env.step(a, frames=1)
            n += 1
            if r.done or r.truncated:
                break
        ticks.append(n)
    return sum(ticks) / len(ticks)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=120)
    ap.add_argument("--max-steps", type=int, default=6000)
    ap.add_argument("--out", default=".runtime/patterns")
    ap.add_argument("--patterns", default="", help="只训这些 (逗号分隔; 空=全部)")
    ap.add_argument("--teacher", default="rule_dagger", choices=["rule_dagger", "none"],
                    help="rule_dagger=规则老师+DAgger (旧默认); none=纯 PPO")
    args = ap.parse_args()

    names = [p.strip() for p in args.patterns.split(",") if p.strip()] or pattern_names()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = []
    for pat in names:
        d = out / pat
        d.mkdir(parents=True, exist_ok=True)
        print("\n========== 弹型 %s : %s ==========" % (pat, PATTERN_DOCS.get(pat, "")), flush=True)
        env = make_env(pat)
        tc = TrainConfig(episodes=args.episodes, max_steps=args.max_steps, step_frames=1,
                         batch_episodes=8, lr=1e-3, entropy_coef=0.02, hidden=(256, 256),
                         teacher=("" if args.teacher == "none" else args.teacher),
                         dagger_beta=0.4, danger_weight=3.0,
                         teacher_epochs=20, ppo_minibatch=512,
                         capture_every=0, eval_every=0, save_every=40,
                         out_dir=str(d), **TC_F)
        tr = Trainer(env, tc, env_lock=env.lock, out_dir=d)
        tr.start()
        t0 = time.perf_counter()
        while tr.running and len(tr.episodes) < args.episodes and time.perf_counter() - t0 < 900:
            time.sleep(1.0)
        tr.stop(save=True)
        # 贪心评估 (策略单独)
        try:
            tr.load(d / "policy")
        except Exception:
            pass
        actions = build_action_table(use_focus=True)
        surv = greedy_survival(env, tr.policy, actions)
        # 老师对照
        from touhou_rl_mcp.training.rule_teacher import RuleTeacher
        g = RuleTeacher()
        ticks = []
        for seed in (101, 202, 303):
            env.reset(seed=seed)
            n = 0
            for i in range(6000):
                st = env.get_state()
                a, _ = g.choose(st, st.frame, "shoot")
                r = env.step(a, frames=1)
                n += 1
                if r.done or r.truncated:
                    break
            ticks.append(n)
        teach = sum(ticks) / len(ticks)
        env.close()
        print("[%s] 策略存活=%.0f  老师存活=%.0f  -> %s"
              % (pat, surv, teach, d / "policy_best.npz"), flush=True)
        summary.append((pat, surv, teach))
    print("\n===== 汇总 (策略 vs 老师, 存活步数) =====")
    for pat, s, t in summary:
        print("  %-24s 策略=%6.0f  老师=%6.0f  %s" % (pat, s, t, "OK" if s >= t * 0.9 else "偏弱"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
