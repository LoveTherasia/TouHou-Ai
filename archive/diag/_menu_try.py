import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from touhou_rl_mcp.config import Config
from touhou_rl_mcp.env import TouhouEnv
from touhou_rl_mcp.games.registry import GameRegistry

cfg = Config({"backend": "memory", "game": "th11",
              "game_dir": r"F:\[th11] 东方地灵殿 (汉化版+日文版)",
              "restart_process": False, "dialogue_assist": False,
              "capture": {"prefer_printwindow": False}})
env = TouhouEnv(cfg, registry=GameRegistry())
try:
    env.start()
    time.sleep(3.0)
    ctrl = env.backend.controller

    def back_to_title():
        for _ in range(4):
            ctrl.tap(["cancel"], hold=6, after=30)

    def probe(name, seq):
        back_to_title()
        time.sleep(1.0)
        for btn, cnt in seq:
            for _ in range(cnt):
                ctrl.tap([btn], hold=6, after=18)
            time.sleep(0.3)
        time.sleep(4.0)
        st = env.backend.read_state()
        print("%-16s in_game=%-5s stage=%-2s lives=%-3s bullets=%s"
              % (name, st.in_game, st.stage, st.lives, st.bullet_count), flush=True)

    probe("down3+c+c", [("down", 3), ("confirm", 1), ("confirm", 1)])
    probe("down2+c+c", [("down", 2), ("confirm", 1), ("confirm", 1)])
    probe("down4+c+c", [("down", 4), ("confirm", 1), ("confirm", 1)])
    probe("c+down3+c+c", [("confirm", 1), ("down", 3), ("confirm", 1), ("confirm", 1)])
    probe("down3+c+down+c", [("down", 3), ("confirm", 1), ("down", 1), ("confirm", 1)])
    back_to_title()
finally:
    env.close()
