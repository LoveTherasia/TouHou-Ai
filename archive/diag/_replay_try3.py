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


def shot(name):
    try:
        img, _ = env.screenshot(); img.save(out / f"{name}.png"); print("shot", name, flush=True)
    except Exception as e:
        print("shot fail", name, e, flush=True)


try:
    env.start()
    env.reset(difficulty=0, character=0, shot=0, restart_process=True, menu="none")
    time.sleep(3.0)
    ctrl = env.backend.controller
    ctrl.release_all()
    shot("30_title")
    for _ in range(3):
        ctrl.tap(["down"], hold=6, after=22)
    time.sleep(0.4); shot("31_down3_replay")
    ctrl.tap(["confirm"], hold=6, after=200); time.sleep(1.2); shot("32_replay_list")
    st = env.backend.read_state(); print("列表: in_game=%s stage=%s" % (st.in_game, st.stage), flush=True)
    ctrl.tap(["confirm"], hold=6, after=200); time.sleep(1.2); shot("33_select_replay")
    st = env.backend.read_state(); print("选回放: in_game=%s stage=%s" % (st.in_game, st.stage), flush=True)
    ctrl.tap(["confirm"], hold=6, after=200); time.sleep(3.0); shot("34_start1")
    st = env.backend.read_state(); print("确认#3: in_game=%s stage=%s lives=%s bullets=%s" % (st.in_game, st.stage, st.lives, st.bullet_count), flush=True)
    if not st.in_game:
        ctrl.tap(["confirm"], hold=6, after=200); time.sleep(3.0); shot("35_start2")
        st = env.backend.read_state(); print("确认#4: in_game=%s stage=%s lives=%s" % (st.in_game, st.stage, st.lives), flush=True)
finally:
    env.close()
