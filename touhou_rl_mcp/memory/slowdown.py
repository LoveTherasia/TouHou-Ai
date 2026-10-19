"""真机时间缩放钩子 (可选) —— 让真实游戏按 factor 倍速运行.

* ``factor > 1`` 加速 —— 真机训练时提高每秒帧数, 从而提高训练吞吐;
* ``factor < 1`` 减速 —— 便于观察, 或给慢速 Agent 留出决策时间。

实现: 把 ``native/speedhack/bin/speedhack32.dll`` 注入游戏进程 (用同目录的 32 位
``injector32.exe``); DLL 钩住 ``timeGetTime`` 并按 factor 缩放时间。速度因子写在
``%TEMP%/touhou_speedhack.cfg``, DLL 每 200ms 重读, 所以运行中可动态改。

需要在有游戏运行时构建/使用; 构建见 ``native/speedhack/README.md``。
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Any

from ..config import PROJECT_ROOT
from ..errors import TouhouError

CONFIG_NAME = "touhou_speedhack.cfg"
LOG_NAME = "touhou_speedhack.log"
MIN_FACTOR = 0.01
MAX_FACTOR = 100.0


def default_binary_dir() -> Path:
    return PROJECT_ROOT / "native" / "speedhack" / "bin"


def speed_config_path() -> Path:
    return Path(tempfile.gettempdir()) / CONFIG_NAME


def speed_log_path() -> Path:
    return Path(tempfile.gettempdir()) / LOG_NAME


class SpeedHook:
    """管理"注入时间缩放 DLL + 设置速度因子"的小封装."""

    def __init__(self, binary_dir: str | Path | None = None) -> None:
        self.binary_dir = Path(binary_dir) if binary_dir else default_binary_dir()
        self.last_result: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    @property
    def dll_path(self) -> Path:
        return self.binary_dir / "speedhack32.dll"

    @property
    def injector_path(self) -> Path:
        return self.binary_dir / "injector32.exe"

    def available(self) -> bool:
        return self.dll_path.exists() and self.injector_path.exists()

    def missing(self) -> list[str]:
        return [str(p) for p in (self.dll_path, self.injector_path) if not p.exists()]

    # ------------------------------------------------------------------
    def write_factor(self, factor: float) -> Path:
        """把速度因子写进配置文件 (DLL 会重读, 可在不重新注入的情况下改速度)."""
        factor = float(factor)
        if not (MIN_FACTOR < factor < MAX_FACTOR):
            raise TouhouError(f"速度因子应在 ({MIN_FACTOR}, {MAX_FACTOR}) 内, 收到 {factor}")
        path = speed_config_path()
        path.write_text(f"{factor:.3f}", encoding="ascii")
        return path

    def apply(self, pid: int, factor: float) -> dict[str, Any]:
        """注入 DLL 并设置速度因子 (幂等: 重复注入会加载第二份, 但通常无害)."""
        if pid is None or int(pid) <= 0:
            raise TouhouError("需要有效的游戏进程 pid 才能注入时间缩放钩子")
        if not self.available():
            raise TouhouError(
                "时间缩放钩子尚未构建: "
                + ", ".join(self.missing())
                + " (运行 native/speedhack/build.ps1)"
            )
        self.write_factor(factor)
        proc = subprocess.run(
            [str(self.injector_path), str(int(pid)), str(self.dll_path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        result: dict[str, Any] = {
            "pid": int(pid),
            "factor": float(factor),
            "returncode": proc.returncode,
            "stdout": (proc.stdout or "").strip(),
            "stderr": (proc.stderr or "").strip(),
            "config": str(speed_config_path()),
            "dll": str(self.dll_path),
        }
        self.last_result = result
        if proc.returncode != 0:
            raise TouhouError(
                f"注入失败 (exit={proc.returncode}): {result['stderr'] or result['stdout']}"
            )
        return result

    def status(self) -> dict[str, Any]:
        return {
            "available": self.available(),
            "missing": self.missing(),
            "binary_dir": str(self.binary_dir),
            "config": str(speed_config_path()),
            "log": str(speed_log_path()),
            "last_result": self.last_result,
        }
