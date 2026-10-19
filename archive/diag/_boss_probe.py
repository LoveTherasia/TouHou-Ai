import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from touhou_rl_mcp.config import Config
from touhou_rl_mcp.env import TouhouEnv
from touhou_rl_mcp.games.registry import GameRegistry
from touhou_rl_mcp.training.rule_teacher import RuleTeacher

cfg = Config({
    "backend": "memory", "game": "th11",
    "game_dir": r"F:\[th11] 东方地灵殿 (汉化版+日文版)",
    "restart_process": True, "dialogue_assist": True,
    "capture": {"prefer_printwindow": False},
})
env = TouhouEnv(cfg, registry=GameRegistry())
teacher = RuleTeacher()  # 现在默认 boss_home=True
try:
    env.start()
    state = env.reset(difficulty=0, character=0, shot=0).state
    print("进入游戏 stage=%s lives=%s" % (state.stage, state.lives), flush=True)
    boss_seen = 0
    started = time.perf_counter()
    for i in range(14000):
        action, _ = teacher.choose(state, state.frame, "shoot")
        r = env.step(action, frames=1)
        state = r.state
        bp = state.raw.get("boss_position")
        if bp is not None:
            boss_seen += 1
            if boss_seen == 1 or i % 600 == 0:
                dx = abs((state.player_x or 0) - bp[0])
                print("i=%5d BOSS pos=(%.0f,%.0f) hp=%s | 自机=(%.0f,%.0f) |Δx|=%.0f"
                      % (i, bp[0], bp[1], state.boss_hp, state.player_x or -1,
                         state.player_y or -1, dx), flush=True)
        if i % 1200 == 0:
            print("i=%5d stage=%s bullets=%s enemies=%s boss_id=%s 位置读数=%d"
                  % (i, state.stage, state.bullet_count, state.enemies_alive,
                     state.raw.get("entities", {}).get("combat", {}).get("boss_id"), boss_seen),
                  flush=True)
        if r.done or r.truncated:
            print("结束 i=%d %s" % (i, r.info.get("done_reason")), flush=True)
            break
    print("完成: %.0fs 读到 Boss 位置 %d 次, 最远 stage=%s"
          % (time.perf_counter() - started, boss_seen, env.get_state().stage))
except Exception as exc:
    import traceback
    print("ERROR:", type(exc).__name__, exc)
    traceback.print_exc()
finally:
    env.close()
