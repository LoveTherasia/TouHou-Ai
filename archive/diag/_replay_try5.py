import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from touhou_rl_mcp.config import Config
from touhou_rl_mcp.env import TouhouEnv
from touhou_rl_mcp.games.registry import GameRegistry

out = Path(".runtime/_replay_shots"); out.mkdir(parents=True, exist_ok=True)
cfg = Config({"backend": "memory", "game": "th11",
              "game_dir": r"F:\[th11] 东方地灵殿 (汉化版+日文版)",
              "restart_process": True, "dialogue_assist": False,
              "capture": {"prefer_printwindow": False}})
env = TouhouEnv(cfg, registry=GameRegistry())

DOWN = dict(hold=2, after=16)     # 短按!
CONF = dict(hold=2, after=90)     # 短按 Z


def shot(name):
    try:
        img, _ = env.screenshot(); img.save(out / f"{name}.png"); print("shot", name, flush=True)
    except Exception as e:
        print("shot fail", name, e, flush=True)


def st():
    s = env.backend.read_state()
    print("   in_game=%s stage=%s lives=%s bullets=%s" % (s.in_game, s.stage, s.lives, s.bullet_count), flush=True)
    return s


try:
    env.start()
    env.reset(difficulty=0, character=0, shot=0, restart_process=True, menu="none")
    ctrl = env.backend.controller
    ctrl.release_all()
    time.sleep(9.0)
    shot("50_title"); st()
    time.sleep(1.0)

    for _ in range(3):
        ctrl.tap(["down"], **DOWN)
    time.sleep(0.6); shot("51_down3_replay"); st()

    ctrl.tap(["confirm"], **CONF)
    time.sleep(1.0); shot("52_replay_page"); st()

    ctrl.tap(["confirm"], **CONF)
    time.sleep(1.0); shot("53_replay_selected"); st()

    ctrl.tap(["confirm"], **CONF)
    time.sleep(3.0); shot("54_stage_start"); st()
finally:
    env.close()
