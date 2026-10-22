"""真机 focus（Shift 慢速）自检：按住 Shift 时自机位移应明显变小。

为什么必须在真机训练之前跑一次
------------------------------
mock 里 focus 速度是写死的 ``2.0 px/帧``（全速 4.4），比例 0.45。真机完全不是
"写死"的：Shift 有没有被游戏窗口收到、游戏内是否开了"慢速键"、置前/扫描码是否
正常，任何一环断了，训练里那 9 个带 focus 的动作在真机上就等于全速。

后果很隐蔽：策略学的是"按住 Shift 走半步、刚好卡进缝里"，真机却走了两步 ——
看起来像"策略泛化差"，其实是**输入没生效**。所以真机训练前先跑这个。

用法::

    python tools/check_focus.py                  # 游戏需已进入关卡
    python tools/check_focus.py --frames 12 --rounds 3 --backend memory

判据
----
* 全速位移应接近 ``4.4 px/帧 × frames``（真机可能略有出入，正作约 4~4.5）
* 慢速/全速 比例落在 **0.30 ~ 0.65** 之间视为 PASS（正作与 mock 都在 0.45 附近）
* 比例 ≈ 1.0 = Shift 没生效 → 真机训练前必须先修输入链路
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from touhou_rl_mcp.config import Config  # noqa: E402
from touhou_rl_mcp.env import TouhouEnv  # noqa: E402
from touhou_rl_mcp.games.registry import GameRegistry  # noqa: E402

#: 正作/mock 的自机速度 (px/帧) 与 focus 比例，用于打印对比
REF_SPEED = 4.4
REF_RATIO = 2.0 / 4.4


def _x(env: TouhouEnv) -> float | None:
    state = env.step("noop", frames=1).state
    return state.player_x


def _shift(env: TouhouEnv, buttons: str, frames: int) -> float | None:
    """按 ``buttons`` 走 ``frames`` 帧，返回横向位移（右为正）。"""
    start = _x(env)
    if start is None:
        return None
    env.step(buttons, frames=frames)
    end = _x(env)
    if end is None:
        return None
    return end - start


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="", help="空=用 config 里的 backend (真机应为 memory)")
    ap.add_argument("--game", default="th11")
    ap.add_argument("--frames", type=int, default=10, help="每次按住多少帧")
    ap.add_argument("--rounds", type=int, default=3, help="左右各测几次，取均值")
    args = ap.parse_args()

    cfg = Config.load()
    patch = {"game": args.game}
    if args.backend:
        patch["backend"] = args.backend
    cfg.patch(patch)
    env = TouhouEnv(cfg, registry=GameRegistry())
    env.start()
    try:
        if _x(env) is None:
            print("[focus] 读不到自机坐标 —— 请确认：\n"
                  "  1) 游戏已经进入关卡（不是标题/菜单/暂停）；\n"
                  "  2) backend 是 memory（真机）而不是 mock；\n"
                  "  3) 游戏窗口没有最小化。")
            return 2

        fast, slow = [], []
        for _ in range(max(1, args.rounds)):
            for direction in ("right+shoot", "left+shoot"):
                d = _shift(env, direction, args.frames)
                if d is not None:
                    fast.append(abs(d))
            for direction in ("right+shoot+focus", "left+shoot+focus"):
                d = _shift(env, direction, args.frames)
                if d is not None:
                    slow.append(abs(d))
    finally:
        env.close()

    if not fast or not slow:
        print("[focus] 采样失败（位移读不到），请检查游戏状态。")
        return 2
    f = sum(fast) / len(fast)
    s = sum(slow) / len(slow)
    ratio = s / f if f > 1e-6 else 0.0
    f_ps, s_ps = f / args.frames, s / args.frames

    print(f"[focus] frames={args.frames} rounds={args.rounds}")
    print(f"[focus] 全速位移 {f:6.2f} px ({f_ps:.2f} px/帧, 参考 {REF_SPEED})")
    print(f"[focus] 慢速位移 {s:6.2f} px ({s_ps:.2f} px/帧)")
    print(f"[focus] 慢速/全速 = {ratio:.2f} (参考 {REF_RATIO:.2f})")

    if ratio < 0.15:
        print("[focus] FAIL: 慢速几乎不动 —— Shift 可能绑到了别的键，或一直是按下状态。")
        return 1
    if ratio > 0.85:
        print("[focus] FAIL: 慢速和全速一样快 —— Shift 没有被游戏收到。\n"
              "  先确认键盘映射 focus=Shift(0x10)、游戏在前台；再确认游戏内没有\n"
              "  把 Slow 键改掉。真机训练前必须修好，否则 9 个 focus 动作全是全速。")
        return 1
    if not 0.30 <= ratio <= 0.65:
        print("[focus] WARN: 比例偏离参考值，训练里 MPC 的 focus 速度按 0.45 建模，\n"
              "  偏差大时老师算出来的'半步位'会不准（可在 mpc.FOCUS_SPEED_RATIO 调）。")
    print("[focus] PASS: 按住 Shift 时确实在慢速移动，真机训练可以启用 focus 动作。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
