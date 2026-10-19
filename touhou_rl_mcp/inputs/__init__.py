"""键盘注入与帧同步控制器."""

from __future__ import annotations

from .actions import DIRECTION_SETS, MACROS, VERBS, Action, parse, parse_sequence  # noqa: F401
from .controller import Controller, ExecutionReport, InputSink, NullSink  # noqa: F401
from .keyboard import ALIASES, VK_CODES, Keyboard, resolve_button, vk_of  # noqa: F401

__all__ = [
    "ALIASES",
    "DIRECTION_SETS",
    "MACROS",
    "VERBS",
    "VK_CODES",
    "Action",
    "Controller",
    "ExecutionReport",
    "InputSink",
    "Keyboard",
    "NullSink",
    "parse",
    "parse_sequence",
    "resolve_button",
    "vk_of",
]
