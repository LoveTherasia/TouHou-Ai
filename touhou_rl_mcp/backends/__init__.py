"""后端: 内置模拟器 / 真实游戏."""

from __future__ import annotations

from .base import Backend  # noqa: F401
from .memory import MemoryBackend  # noqa: F401
from .mock import MockBackend, MockGame  # noqa: F401

BACKENDS: dict[str, type] = {
    "mock": MockBackend,
    "memory": MemoryBackend,
}

__all__ = ["BACKENDS", "Backend", "MemoryBackend", "MockBackend", "MockGame"]
