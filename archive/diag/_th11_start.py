import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from touhou_rl_mcp.config import Config
from touhou_rl_mcp.env import TouhouEnv
from touhou_rl_mcp.games.registry import GameRegistry

cfg = Config({
    "backend": "memory", "game": "th11",
    "game_dir": r"F:\[th11] 东方地灵殿 (汉化版+日文版)",
    "restart_process": False, "dialogue_assist": False,
})
env = TouhouEnv(cfg, registry=GameRegistry())
try:
    env.start()
    print("pid", env.backend.pid, "in_game(before)=", env.get_state().in_game)
    res = env.reset(difficulty=0, character=0, shot=0)
    st = res.state
    print("reset -> in_game=%s stage=%s lives=%s char=%s%s" %
          (st.in_game, st.stage, st.lives, st.char_main, st.char_sub))
    # 推进 5 秒, 看弹幕
    for i in range(60):
        r = env.step("shoot", frames=5)
        st = r.state
        if i % 12 == 0:
            print("t=%5.1fs in_game=%s stage=%s bullets=%s enemies=%s player=(%s,%s)" %
                  (i * 5 / 60, st.in_game, st.stage, st.bullet_count, st.enemies_alive,
                   st.player_x, st.player_y))
    print("done. raw entities:", env.get_state().raw.get("entities"))
except Exception as exc:
    import traceback
    print("ERROR:", type(exc).__name__, exc)
    traceback.print_exc()
finally:
    env.close()
