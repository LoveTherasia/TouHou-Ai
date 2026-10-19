"""示例: 直接在 Python 里驱动环境 (不经过 MCP).

这是 RL 训练循环的样子 —— 之后接入 SB3 / 自研 PPO 时, 把这个 while 换成
"策略前向 -> env.step -> 收集 rollout -> 更新参数" 即可。

    python examples/rl_loop.py --backend mock --episodes 3
    python examples/rl_loop.py --backend memory --game th11 --episodes 1
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from touhou_rl_mcp.config import Config  # noqa: E402
from touhou_rl_mcp.env import TouhouEnv  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="mock", choices=["mock", "memory"])
    parser.add_argument("--game", default="th11")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--frames", type=int, default=8, help="每个动作持续的帧数")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = Config.load()
    config.patch({"backend": args.backend, "game": args.game})

    env = TouhouEnv(config)
    env.start()
    rng = random.Random(args.seed)

    # 极简的"手写策略": 大多数时候待机射击 + 定期用躲避辅助。
    # 换成神经网络就是标准 RL。
    policy = ["noop", "shoot", "shoot", "focus_shoot", "dodge", "move_to"]

    try:
        for episode in range(1, args.episodes + 1):
            result = env.reset(seed=rng.randrange(1 << 30))
            print(
                f"\n=== episode {episode} 开始: stage={result.state.stage} "
                f"lives={result.state.lives} power={result.state.power} ==="
            )
            total = 0.0
            steps = 0
            while True:
                action = rng.choice(policy)
                extra: dict = {}
                if action == "move_to":
                    extra = {"x": rng.uniform(60, 324), "y": rng.uniform(180, 420)}
                result = env.step(action, frames=args.frames, **extra)
                total += result.reward
                steps += 1
                if result.done or result.truncated:
                    print(
                        f"  结束: reason={result.info.get('done_reason')} "
                        f"steps={steps} frames={result.info.get('episode_frames')} "
                        f"回报={total:+.2f}"
                    )
                    print(f"  明细: {result.info.get('reward_parts')}")
                    break
            print(f"  本局统计: {env.stats.episodes[-1]}")
        print(f"\n训练统计: {env.stats.summary()}")
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
