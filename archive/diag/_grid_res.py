"""危险图分辨率够不够? —— 看密集环时期"每格塞了几颗弹"。

网格边长 2*span/size 像素; 如果密集弹幕的间距 < 格宽, 危险图就会把多颗弹
糊成一格, 丢失"缝隙在哪"的信息。

    python .runtime/_grid_res.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np  # noqa: E402

from touhou_rl_mcp.config import Config  # noqa: E402
from touhou_rl_mcp.env import TouhouEnv  # noqa: E402
from touhou_rl_mcp.games.registry import GameRegistry  # noqa: E402
from touhou_rl_mcp.training.rule_teacher import RuleTeacher  # noqa: E402

SPAN, SIZE = 110.0, 16
CELL = 2 * SPAN / SIZE


def occupancy(state) -> tuple[int, int, int]:
    """返回 (窗口内弹数, 最大单格弹数, 有>1颗弹的格数)。"""
    if not state.bullets:
        return 0, 0, 0
    arr = np.asarray(state.bullets, float).reshape(-1, 2)
    dx = arr[:, 0] - state.player_x
    dy = arr[:, 1] - state.player_y
    m = (np.abs(dx) <= SPAN) & (np.abs(dy) <= SPAN)
    if not np.any(m):
        return 0, 0, 0
    ix = np.clip(((dx[m] + SPAN) / (2 * SPAN) * (SIZE - 1)).astype(int), 0, SIZE - 1)
    iy = np.clip(((dy[m] + SPAN) / (2 * SPAN) * (SIZE - 1)).astype(int), 0, SIZE - 1)
    counts = np.zeros((SIZE, SIZE), int)
    np.add.at(counts, (iy, ix), 1)
    return int(m.sum()), int(counts.max()), int((counts > 1).sum())


def main() -> int:
    env = TouhouEnv(Config({"backend": "mock", "game": "th11", "step_frames": 1,
                            "danmaku_script": ".runtime/danmaku/th11_replay_full.json"}),
                    registry=GameRegistry())
    env.start()
    env.reset(seed=1)
    g = RuleTeacher()
    st = env.get_state()
    rows = []
    for i in range(17600):
        a, _ = g.choose(st, st.frame, "shoot")
        res = env.step(a, frames=1)
        st = res.state
        if 17000 <= i <= 17360 and i % 20 == 0:
            rows.append((i, st.bullet_count, *occupancy(st)))
        if res.done or res.truncated:
            break
    env.close()
    print(f"格宽 = {CELL:.2f} px (span={SPAN:.0f}, size={SIZE})")
    print(f"{'帧':>6} {'全屏弹数':>8} {'窗口内弹数':>10} {'最大单格':>8} {'多弹格数':>8}")
    for i, total, inwin, mx, multi in rows:
        print(f"{i:>6} {total:>8} {inwin:>10} {mx:>8} {multi:>8}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
