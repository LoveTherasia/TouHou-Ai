import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from touhou_rl_mcp.config import Config
from touhou_rl_mcp.env import TouhouEnv
from touhou_rl_mcp.games.registry import GameRegistry

out = Path(".runtime/_menu_shots"); out.mkdir(parents=True, exist_ok=True)
cfg = Config({"backend": "memory", "game": "th11",
              "game_dir": r"F:\[th11] 东方地灵殿 (汉化版+日文版)",
              "restart_process": False, "dialogue_assist": False,
              "capture": {"prefer_printwindow": False}})
env = TouhouEnv(cfg, registry=GameRegistry())
try:
    st = env.start()
    print("pid", env.backend.pid, "window", env.backend.window.to_dict() if env.backend.window else None)
    time.sleep(8.0)
    img, method = env.screenshot()
    p = out / "00_title.png"
    img.save(p)
    print("shot", method, p, img.size)
    ctrl = env.backend.controller
    for i in range(1, 4):
        ctrl.tap(["down"], hold=6, after=16)
        time.sleep(0.3)
        img, m = env.screenshot(); img.save(out / f"01_down{i}.png")
    ctrl.tap(["confirm"], hold=6, after=90); time.sleep(0.6)
    img, m = env.screenshot(); img.save(out / "02_confirm.png")
    print("after confirm: in_game=%s stage=%s" % (env.backend.read_state().in_game, env.backend.read_state().stage))
finally:
    env.close()
