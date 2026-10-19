"""动作解析.

对外 (MCP) 的动作表示刻意做成"人类/LLM 友好"的字符串或对象, 内部再翻译成
虚拟按键集合或宏动作:

    "right"                     -> 右移
    "right+shoot"               -> 右移并射击
    "shoot,focus"               -> 同上 (逗号/加号/空格分隔都可以)
    "bomb"                      -> 放 Bomb (点按 X)
    "noop"                      -> 不按键, 只推进帧
    "move_to" / {"macro": "move_to", "x": 192, "y": 400}
    "dodge"                     -> 内置的最近弹幕躲避辅助 (需要 player_x/y + bullets)
    "wait_in_game"              -> 推进到进入游戏局内
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .keyboard import ALIASES, resolve_button

#: 方向键组合名 -> 按键集合
DIRECTION_SETS: dict[str, tuple[str, ...]] = {
    "up": ("up",),
    "down": ("down",),
    "left": ("left",),
    "right": ("right",),
    "up_left": ("up", "left"),
    "up_right": ("up", "right"),
    "down_left": ("down", "left"),
    "down_right": ("down", "right"),
    "up-left": ("up", "left"),
    "up-right": ("up", "right"),
    "down-left": ("down", "left"),
    "down-right": ("down", "right"),
    "none": (),
    "center": (),
}

#: 简单动词 -> 按键集合
VERBS: dict[str, tuple[str, ...]] = {
    "noop": (),
    "wait": (),
    "idle": (),
    "stand": (),
    "shoot": ("shoot",),
    "fire": ("shoot",),
    "shot": ("shoot",),
    "focus": ("focus",),
    "slow": ("focus",),
    "focus_shoot": ("focus", "shoot"),
    "shoot_focus": ("focus", "shoot"),
    "focused_shoot": ("focus", "shoot"),
    "confirm": ("confirm",),
    "menu": ("confirm",),
    "cancel": ("cancel",),
    "bomb": ("bomb",),
    "pause": ("pause",),
}

#: 需要"闭环控制"的宏
MACROS = {"move_to", "dodge", "wait_in_game", "run_sequence", "escape_bullets", "autopilot"}

_SIMPLE_BUTTONS = {"up", "down", "left", "right", "shoot", "focus", "bomb", "confirm", "cancel", "pause", "enter", "space"}

_SPLIT = re.compile(r"[+,;|\s]+")
_KV = re.compile(r"(\w+)\s*=\s*(-?[\d.]+)")


@dataclass
class Action:
    """解析后的动作."""

    kind: str  # "buttons" | "macro"
    name: str
    buttons: tuple[str, ...] = ()
    frames: int | None = None
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def is_macro(self) -> bool:
        return self.kind == "macro"

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "name": self.name,
            "buttons": list(self.buttons),
            "frames": self.frames,
            "params": self.params,
        }


def _parse_object(payload: dict[str, Any], default_frames: int | None) -> Action:
    frames = payload.get("frames", default_frames)
    frames = int(frames) if frames is not None else None

    macro = payload.get("macro") or payload.get("action")
    if isinstance(macro, str) and resolve_button(macro) in MACROS:
        params = {
            k: v
            for k, v in payload.items()
            if k not in {"macro", "action", "frames"} and not k.startswith("_")
        }
        return Action(kind="macro", name=macro.strip().lower(), frames=frames, params=params)

    raw = payload.get("buttons") or payload.get("keys") or macro or ""
    if isinstance(raw, (list, tuple, set)):
        buttons = tuple(resolve_button(str(b)) for b in raw)
        name = "+".join(buttons) or "noop"
    else:
        return parse(str(raw), default_frames=default_frames, extra_params={
            k: v for k, v in payload.items() if k not in {"buttons", "keys", "frames"}
        })
    return Action(kind="buttons", name=name, buttons=buttons, frames=frames)


def parse(spec: Any, *, default_frames: int | None = None, extra_params: dict[str, Any] | None = None) -> Action:
    """把任意形式的动作表示解析成 :class:`Action`."""
    if spec is None:
        return Action(kind="buttons", name="noop", buttons=(), frames=default_frames)
    if isinstance(spec, dict):
        action = _parse_object(spec, default_frames)
        if extra_params:
            action.params = {**extra_params, **action.params}
        return action
    if isinstance(spec, (list, tuple, set)):
        buttons = tuple(resolve_button(str(b)) for b in spec)
        return Action(
            kind="buttons",
            name="+".join(buttons) or "noop",
            buttons=buttons,
            frames=default_frames,
            params=dict(extra_params or {}),
        )

    text = str(spec).strip()
    params: dict[str, Any] = dict(extra_params or {})

    # 形如 "move_to x=192 y=400" 或 "dodge radius=40"
    match = _KV.findall(text)
    if match:
        for key, value in match:
            params[key] = float(value) if "." in value else int(value)
        text = _KV.sub(" ", text).strip()

    lowered = text.lower()
    canonical = lowered.replace("-", "_")

    if canonical in MACROS or resolve_button(canonical) in MACROS:
        return Action(
            kind="macro",
            name=resolve_button(canonical),
            frames=default_frames,
            params=params,
        )

    if not text or canonical in {"", "none"}:
        return Action(kind="buttons", name="noop", buttons=(), frames=default_frames, params=params)

    if canonical in VERBS:
        buttons = VERBS[canonical]
        return Action(kind="buttons", name=canonical, buttons=buttons, frames=default_frames, params=params)

    if canonical in DIRECTION_SETS:
        buttons = DIRECTION_SETS[canonical]
        return Action(kind="buttons", name=canonical, buttons=buttons, frames=default_frames, params=params)

    # 组合: "right+shoot", "up,left,bomb"
    tokens = [t for t in _SPLIT.split(lowered) if t]
    buttons: list[str] = []
    for token in tokens:
        if token in DIRECTION_SETS:
            buttons.extend(DIRECTION_SETS[token])
            continue
        if token in VERBS:
            buttons.extend(VERBS[token])
            continue
        name = resolve_button(token)
        if name in _SIMPLE_BUTTONS or ALIASES.get(token):
            buttons.append(name)
            continue
        raise ValueError(
            f"无法解析的动作片段: {token!r}。可用方向: {', '.join(sorted(DIRECTION_SETS))}; "
            f"可用动词: {', '.join(sorted(VERBS))}; 宏: {', '.join(sorted(MACROS))}"
        )

    seen: list[str] = []
    for button in buttons:
        if button not in seen:
            seen.append(button)
    return Action(kind="buttons", name="+".join(seen) or "noop", buttons=tuple(seen), frames=default_frames, params=params)


def parse_sequence(steps: Any) -> list[Action]:
    """解析 ``send_keys`` 的按键序列: ``[{"buttons": ["confirm"], "frames": 8, "after": 20}, ...]``"""
    out: list[Action] = []
    if isinstance(steps, (str, dict)):
        steps = [steps]
    for step in steps:
        if isinstance(step, dict):
            action = parse(step, default_frames=int(step.get("frames", 6)))
            after = step.get("after") or step.get("wait_frames") or step.get("after_frames")
            if after:
                action.params["after"] = int(after)
            out.append(action)
        else:
            out.append(parse(step))
    return out
