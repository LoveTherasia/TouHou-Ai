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
    "restart_process": True,
    "dialogue_assist": True,
    "dialogue_tap_hold": 2, "dialogue_tap_after": 6, "dialogue_max_taps": 14,
    "capture": {"prefer_printwindow": False},
})
env = TouhouEnv(cfg, registry=GameRegistry())
teacher = RuleTeacher()
try:
    env.start()
    state = env.reset(difficulty=0, character=0, shot=0).state
    print("进入游戏: stage=%s lives=%s" % (state.stage, state.lives), flush=True)
    dialogue_steps = 0
    dialogue_taps = 0
    started = time.perf_counter()
    for i in range(5400):
        action, _ = teacher.choose(state, state.frame, "shoot")
        r = env.step(action, frames=1)
        info = r.info
        if info.get("dialogue_assist"):
            dialogue_steps += 1
            dialogue_taps += int((info.get("backend_frames") or {}).get("dialogue_taps", 0))
        state = r.state
        if i % 600 == 0:
            print("i=%5d stage=%s lives=%s bullets=%s enemies=%s 对话步=%d 点按=%d"
                  % (i, state.stage, state.lives, state.bullet_count,
                     state.enemies_alive, dialogue_steps, dialogue_taps), flush=True)
        if r.done or r.truncated:
            print("结束于 i=%d: %s" % (i, info.get("done_reason")), flush=True)
            break
    print("完成: 用时=%.0fs 对话步=%d 对话点按=%d 最终 stage=%s score=%s"
          % (time.perf_counter() - started, dialogue_steps, dialogue_taps,
             env.get_state().stage, env.get_state().score))
except Exception as exc:
    import traceback
    print("ERROR:", type(exc).__name__, exc)
    traceback.print_exc()
finally:
    env.close()
