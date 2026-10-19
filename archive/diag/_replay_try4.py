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


def st():
    s = env.backend.read_state()
    print("   in_game=%s stage=%s lives=%s bullets=%s" % (s.in_game, s.stage, s.lives, s.bullet_count), flush=True)
    return s


try:
    env.start()
    env.reset(difficulty=0, character=0, shot=0, restart_process=True, menu="none")
    ctrl = env.backend.controller
    ctrl.release_all()
    print("等待标题稳定...", flush=True)
    time.sleep(9.0)
    shot("40_title"); st()

    # 先不动
    time.sleep(2.0)

    # 下 x3 -> Replay
    for _ in range(3):
        ctrl.tap(["down"], hold=6, after=28)
    time.sleep(1.0)
    shot("41_down3_replay"); st()

    # Z -> 进入 Replay 页面
    ctrl.tap(["confirm"], hold=6, after=200)
    time.sleep(2.0)
    shot("42_replay_page"); st()

    # 选第一个回放
    ctrl.tap(["confirm"], hold=6, after=200)
    time.sleep(2.0)
    shot("43_replay_selected"); st()

    # 选第 1 面开始
    ctrl.tap(["confirm"], hold=6, after=200)
    time.sleep(4.0)
    shot("44_stage1_start"); st()

    if not env.backend.read_state().in_game:
        ctrl.tap(["confirm"], hold=6, after=200)
        time.sleep(4.0)
        shot("45_more"); st()
finally:
    env.close()
