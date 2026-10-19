import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from touhou_rl_mcp.config import Config
from touhou_rl_mcp.env import TouhouEnv
from touhou_rl_mcp.games.registry import GameRegistry

out = Path(".runtime/_menu_shots"); out.mkdir(parents=True, exist_ok=True)
cfg = Config({"backend": "memory", "game": "th11",
              "game_dir": r"F:\[th11] 东方地灵殿 (汉化版+日文版)",
              "restart_process": True, "dialogue_assist": False,
              "capture": {"prefer_printwindow": False}})
env = TouhouEnv(cfg, registry=GameRegistry())
try:
    env.start()
    env.reset(difficulty=0, character=0, shot=0, restart_process=True, menu="none")
    ctrl = env.backend.controller
    ctrl.release_all()

    def shot(name):
        img, method = env.screenshot()
        p = out / f"{name}.png"
        img.save(p)
        print("shot", name, method, p, flush=True)

    time.sleep(1.0); shot("00_title")
    for i in range(1, 5):
        ctrl.tap(["down"], hold=6, after=14)
        time.sleep(0.3); shot("01_down%d" % i)
    ctrl.tap(["confirm"], hold=6, after=90)
    time.sleep(0.5); shot("02_confirm")
    ctrl.tap(["confirm"], hold=6, after=90)
    time.sleep(0.5); shot("03_confirm2")
    st = env.backend.read_state()
    print("final state: in_game=%s stage=%s lives=%s" % (st.in_game, st.stage, st.lives))
finally:
    env.close()
