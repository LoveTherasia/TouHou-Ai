import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from touhou_rl_mcp.config import Config
from touhou_rl_mcp.env import TouhouEnv
from touhou_rl_mcp.games.registry import GameRegistry

out = Path(".runtime/_replay_shots"); out.mkdir(parents=True, exist_ok=True)
cfg = Config({"backend": "memory", "game": "th11",
              "game_dir": r"F:\[th11] 东方地灵殿 (汉化版+日文版)",
              "restart_process": False, "dialogue_assist": False,
              "capture": {"prefer_printwindow": False}})
env = TouhouEnv(cfg, registry=GameRegistry())


def shot(name):
    try:
        img, _ = env.screenshot(); img.save(out / f"{name}.png")
    except Exception as e:
        print("shot fail", e)


try:
    env.start(); time.sleep(3.0)
    ctrl = env.backend.controller
    for _ in range(4):
        ctrl.tap(["cancel"], hold=6, after=30)
    time.sleep(1.0)

    for _ in range(3):
        ctrl.tap(["down"], hold=6, after=20)
    time.sleep(0.5); shot("10_after_down3")
    ctrl.tap(["confirm"], hold=6, after=150)
    time.sleep(1.0); shot("11_list")
    st = env.backend.read_state()
    print("列表后: in_game=%s stage=%s" % (st.in_game, st.stage), flush=True)

    ctrl.tap(["confirm"], hold=6, after=150)
    time.sleep(5.0); shot("12_play")
    st = env.backend.read_state()
    print("确认后: in_game=%s stage=%s lives=%s bullets=%s" % (st.in_game, st.stage, st.lives, st.bullet_count), flush=True)

    # 再试: 列表里下移一格再确认
    ctrl.tap(["cancel"], hold=6, after=120); time.sleep(1.0)
    ctrl.tap(["down"], hold=6, after=20)
    ctrl.tap(["confirm"], hold=6, after=150)
    time.sleep(5.0); shot("13_play2")
    st = env.backend.read_state()
    print("下移+确认后: in_game=%s stage=%s lives=%s bullets=%s" % (st.in_game, st.stage, st.lives, st.bullet_count), flush=True)
finally:
    env.close()
