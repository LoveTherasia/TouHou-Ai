import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from touhou_rl_mcp.config import Config
from touhou_rl_mcp.env import TouhouEnv
from touhou_rl_mcp.games.registry import GameRegistry

cfg = Config({
    "backend": "memory", "game": "th11",
    "game_dir": r"F:\[th11] 东方地灵殿 (汉化版+日文版)",
    "restart_process": False,
})
env = TouhouEnv(cfg, registry=GameRegistry())
try:
    status = env.start()
    print("started. pid=", env.backend.pid)
    print("window=", (env.backend.window.to_dict() if env.backend.window else None))
    st = env.get_state()
    print("state:", st.to_dict(max_bullets=3))
except Exception as exc:
    print("ERROR:", type(exc).__name__, exc)
finally:
    env.close()
