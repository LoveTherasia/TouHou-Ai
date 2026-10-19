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
    "restart_process": True, "capture": {"prefer_printwindow": False},
})
env = TouhouEnv(cfg, registry=GameRegistry())
teacher = RuleTeacher()
try:
    env.start()
    st = env.reset(difficulty=0, character=0, shot=0).state
    print("start: stage=%s lives=%s" % (st.stage, st.lives), flush=True)
    dlg_frames = 0
    run_start = None
    windows = []
    for i in range(9000):
        a, _ = teacher.choose(st, st.frame, "shoot")
        r = env.step(a, frames=1)
        st = r.state
        if r.info.get("dialogue_assist"):
            dlg_frames += 1
            if run_start is None:
                run_start = i
        else:
            if run_start is not None:
                windows.append((run_start, i - 1, i - run_start))
                run_start = None
        if i % 600 == 0:
            print("frame=%5d stage=%s lives=%s bullets=%s enemies=%s spell=%s dlg=%s"
                  % (i, st.stage, st.lives, st.bullet_count, st.enemies_alive,
                     st.spell_id, r.info.get("dialogue_assist")), flush=True)
        if r.done or r.truncated:
            print("ended @", i, r.info.get("done_reason"), flush=True)
            break
    if run_start is not None:
        windows.append((run_start, i, i - run_start))
    print("总帧=%d 对话辅助帧=%d 对话窗口(起,止,长)=%s" % (i, dlg_frames, windows), flush=True)
except Exception as exc:
    import traceback
    print("ERROR:", type(exc).__name__, exc); traceback.print_exc()
finally:
    env.close()
