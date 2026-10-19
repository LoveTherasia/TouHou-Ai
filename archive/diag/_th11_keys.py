import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from touhou_rl_mcp.config import Config
from touhou_rl_mcp.env import TouhouEnv
from touhou_rl_mcp.games.registry import GameRegistry
from touhou_rl_mcp.inputs.actions import parse

cfg = Config({
    "backend": "memory", "game": "th11",
    "game_dir": r"F:\[th11] 东方地灵殿 (汉化版+日文版)",
    "restart_process": False, "dialogue_assist": False,
    "capture": {"prefer_printwindow": False},
})
env = TouhouEnv(cfg, registry=GameRegistry())


def snap(tag):
    s = env.get_state()
    print("%-10s frame=%6s stage=%s lives=%s bullets=%s enemies=%s spell=%s in_game=%s"
          % (tag, s.frame, s.stage, s.lives, s.bullet_count, s.enemies_alive, s.spell_id, s.in_game),
          flush=True)
    return s


try:
    env.start()
    st = snap("current")
    if not st.in_game:
        print("不在局内, 退出")
        raise SystemExit(0)
    # 依次尝试不同按键, 每次按住 30 帧, 看对话是否推进 (spell_id 变化 / 弹幕出现)
    for key in ["shoot", "confirm", "bomb", "up", "down", "left", "right", "focus"]:
        before = env.get_state()
        env.backend.apply(parse(key, default_frames=30))
        time.sleep(0.1)
        after = env.get_state()
        changed = (after.spell_id != before.spell_id) or (after.bullet_count != before.bullet_count) \
                  or (after.enemies_alive != before.enemies_alive) or (after.stage != before.stage)
        print("  key=%-8s frame %s->%s bullets %s->%s enemies %s->%s spell %s->%s  changed=%s"
              % (key, before.frame, after.frame, before.bullet_count, after.bullet_count,
                 before.enemies_alive, after.enemies_alive, before.spell_id, after.spell_id, changed),
              flush=True)
except SystemExit:
    pass
except Exception as exc:
    import traceback
    print("ERROR:", type(exc).__name__, exc); traceback.print_exc()
finally:
    env.close()
