"""Fail closed for real-game control when entity observations are unavailable."""
from __future__ import annotations

import math

from .errors import BackendError
from .types import GameState


def require_entities(state: GameState) -> None:
    """Empty, verified collections are valid; missing/failed reads are not.

    Mock and externally constructed states without memory provenance retain their
    existing behavior. Call before choosing an action, not on terminal states.
    """
    if state.raw.get("backend") != "memory" and "entities" not in state.raw:
        return
    if not state.in_game:
        raise BackendError("尚未进入可观测的游戏局内，请检查菜单流程或使用 restart_process=True。")
    if any(v is None or not math.isfinite(v) for v in (state.player_x, state.player_y)):
        raise BackendError("自机坐标读取失败，已停止控制。")
    diagnostics = state.raw.get("entities", {})
    for name in ("bullets", "enemies"):
        detail = diagnostics.get(name, {"status": "unconfigured"})
        if detail.get("status") != "ok":
            raise BackendError(f"实机观测 {name} 不可用: {detail}；请检查游戏版本和实体偏移配置。")
