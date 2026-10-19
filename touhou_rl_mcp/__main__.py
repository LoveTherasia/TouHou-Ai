"""入口: ``python -m touhou_rl_mcp``.

    python -m touhou_rl_mcp                 # 以 stdio 启动 MCP 服务器
    python -m touhou_rl_mcp selftest        # 用内置模拟器跑一局, 自检
    python -m touhou_rl_mcp selftest --dodge
    python -m touhou_rl_mcp --backend memory --game th11 --transport stdio
"""

from __future__ import annotations

import argparse
import json
import sys

from . import __version__
from .config import Config, default_config_path


def _build_config(args: argparse.Namespace) -> Config:
    config = Config.load(args.config)
    patch: dict = {}
    if args.backend:
        patch["backend"] = args.backend
    if args.game:
        patch["game"] = args.game
    if args.game_dir:
        patch["game_dir"] = args.game_dir
    if args.step_frames:
        patch["step_frames"] = int(args.step_frames)
    if patch:
        config.patch(patch)
        if args.save_config:
            config.save()
    return config


def _selftest(args: argparse.Namespace) -> int:
    from .env import TouhouEnv

    config = Config.load(args.config)
    config.patch({"backend": "mock", "game": args.game or "th11"})
    env = TouhouEnv(config)
    env.start()
    print(f"[selftest] backend=mock game={env.spec.key} ({env.spec.name_zh})")

    result = env.reset(seed=args.seed, difficulty=args.difficulty)
    print(f"[selftest] reset -> lives={result.state.lives} stage={result.state.stage}")

    total_reward = 0.0
    steps = 0
    done = False
    while steps < args.max_steps and not done:
        action = "dodge" if args.dodge else args.action
        step = env.step(action, frames=args.frames)
        total_reward += step.reward
        steps += 1
        done = step.done or step.truncated
        if steps % 50 == 0:
            state = step.state
            print(
                f"[selftest] step={steps:4d} frame={state.frame:5d} stage={state.stage} "
                f"lives={state.lives} power={state.power} score={state.score} "
                f"bullets={state.bullet_count} reward={step.reward:+.3f}"
            )

    state = env.get_state()
    print(
        f"[selftest] 结束: steps={steps} 总回报={total_reward:+.2f} done={done} "
        f"stage={state.stage} lives={state.lives} score={state.score} graze={state.graze}"
    )

    image = env.visualize()
    from .capture.visualize import save_png

    path = save_png(image, None, prefix="selftest")
    print(f"[selftest] 状态可视化已保存: {path}")
    print(json.dumps(env.stats.summary(), ensure_ascii=False, indent=2))
    env.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="touhou_rl_mcp", description="Touhou RL MCP server")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--config", default=None, help=f"配置文件路径 (默认 {default_config_path()})")
    parser.add_argument("--backend", default=None, choices=["mock", "memory"], help="后端")
    parser.add_argument("--game", default=None, help="作品 key, 例如 th11")
    parser.add_argument("--game-dir", default=None, help="游戏安装目录")
    parser.add_argument("--step-frames", default=None, help="默认每次 step 推进的帧数")
    parser.add_argument("--transport", default="stdio", choices=["stdio", "sse", "streamable-http"])
    parser.add_argument("--save-config", action="store_true", help="把命令行覆盖项写回 config.json")

    sub = parser.add_subparsers(dest="command")
    test = sub.add_parser("selftest", help="用 mock 后端跑一局自检")
    test.add_argument("--steps", dest="max_steps", type=int, default=600)
    test.add_argument("--frames", type=int, default=4)
    test.add_argument("--action", default="shoot")
    test.add_argument("--dodge", action="store_true", help="用内置躲避辅助跑")
    test.add_argument("--seed", type=int, default=1234)
    test.add_argument("--difficulty", type=int, default=1)

    args = parser.parse_args(argv)

    if args.command == "selftest":
        return _selftest(args)

    from . import server

    if args.backend or args.game or args.step_frames or args.game_dir:
        config = _build_config(args)
        server._ENV = None
        from .env import TouhouEnv as _Env

        server._ENV = _Env(config, registry=server._REGISTRY)
        server._ENV.start()

    server.run(args.transport)
    return 0


if __name__ == "__main__":
    sys.exit(main())
