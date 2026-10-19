"""临时脚本: 聚焦游戏窗口并发送按键 (用于从 Game Over 菜单重开一局)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from touhou_rl_mcp.inputs.keyboard import Keyboard  # noqa: E402
from touhou_rl_mcp.memory.process import find_first_pid, find_game_window  # noqa: E402


def main() -> int:
    keys = sys.argv[1:] or ["down", "down", "down", "confirm"]
    pid = find_first_pid(["th11c.exe", "th11.exe"])
    if pid is None:
        print("no th11 process")
        return 1
    win = find_game_window(pid, ["地霊殿", "地灵殿", "東方", "东方", "Touhou"])
    print(f"pid={pid} window={win.title if win else None}")
    kb = Keyboard(win.hwnd if win else None, focus=True)
    kb.focus_window()
    time.sleep(0.5)
    for key in keys:
        kb.tap(key, hold_seconds=0.09)
        time.sleep(0.35)
    kb.release_all()
    print("sent:", keys)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
