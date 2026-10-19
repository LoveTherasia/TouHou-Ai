"""训练进程的启动与状态跟踪.

设计取舍
--------
* **子进程不随看板退出**。关掉看板不该让跑了半小时的训练消失, 所以进程独立启动,
  只把 pid 记进 ``launch.json``。反过来, 这也意味着看板重启后只能"观察"旧进程,
  不能再停它 —— 停止只对**本进程启起来的**任务开放 (pid 复用会误杀别的程序)。
* 输出全部写进 run 目录下的 ``console.log``, 前端按需 tail, 不占内存。
* 状态靠 ``Popen.poll()``; 看板重启后用 pid 探活兜底, 但拿不到退出码, 标为
  ``unknown`` 而不是猜。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .recipes import PROJECT_ROOT, Recipe, default_run_dir, get

#: 与 registry 共用的文件名约定
LAUNCH_FILE = "launch.json"
LOG_FILE = "console.log"

#: Windows 下不弹新控制台; 其它平台无此概念
_CREATE_NO_WINDOW = 0x08000000

#: 状态文案 (前端直接显示)
_RUNNING, _EXITED, _FAILED, _UNKNOWN = "running", "exited", "failed", "unknown"


@dataclass
class _Proc:
    proc: subprocess.Popen
    run_dir: Path
    started: float
    recipe: str


def _pid_alive(pid: int | None) -> bool:
    """跨平台探活。

    **不要用 ``os.kill(pid, 0)``**: Windows 上它走的是 TerminateProcess,
    sig=0 会真的把进程杀掉。
    """
    if not pid or pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes

        kernel32 = ctypes.windll.kernel32                 # type: ignore[attr-defined]
        handle = kernel32.OpenProcess(0x1000, False, int(pid))   # QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return code.value == 259                       # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(int(pid), 0)
    except OSError:
        return False
    return True


@dataclass
class LaunchManager:
    """持有本进程启动的子进程, 并提供启动 / 查询 / 停止 / 读日志。"""

    roots: list[Path]
    python: str = field(default_factory=lambda: sys.executable)
    project_root: Path = PROJECT_ROOT
    #: True 时只算 argv 不真的启动 (测试用)
    dry_run: bool = False
    _procs: dict[str, _Proc] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    # ------------------------------------------------------------------
    def _run_root(self) -> Path:
        root = Path(self.roots[0]) if self.roots else Path(".runtime")
        return root.expanduser().resolve()

    def resolve_run_dir(self, run_id: str) -> Path | None:
        """把前端传来的 run id 解析成目录, 且必须在扫描根目录之内。"""
        if not run_id:
            return None
        candidate = (self._run_root() / run_id).resolve()
        try:
            candidate.relative_to(self._run_root())
        except ValueError:
            return None
        return candidate if candidate.is_dir() else None

    # ------------------------------------------------------------------
    def start(self, recipe_id: str, params: dict[str, Any],
              *, stamp: str | None = None) -> dict[str, Any]:
        recipe: Recipe = get(recipe_id)
        script = (self.project_root / recipe.script).resolve()
        if not script.is_file():
            raise ValueError(f"脚本不存在: {recipe.script}")

        resolved = {p.key: p.coerce(params.get(p.key, p.default)) for p in recipe.params}
        run_dir = default_run_dir(
            self._run_root(), recipe, stamp=stamp or time.strftime("%Y%m%d-%H%M%S"),
            params=resolved)
        argv = recipe.argv(params, run_dir=run_dir)
        # -u: 强制无缓冲。子进程的 stdout 是文件, 默认块缓冲会把日志攒到 8KB 才落盘,
        # 看板上就变成"训练跑了半天控制台还是空的"。
        cmd = [self.python, "-u", str(script), *argv]
        run_dir.mkdir(parents=True, exist_ok=True)

        payload: dict[str, Any] = {
            "recipe": recipe.id,
            "recipe_name": recipe.name,
            "argv": argv,
            "cmd": cmd,
            "cwd": str(self.project_root),
            "started": time.time(),
            "started_text": time.strftime("%Y-%m-%d %H:%M:%S"),
            "params": resolved,
            "pid": None,
            "state": _RUNNING,
        }
        if self.dry_run:
            payload["state"] = "dry-run"
            return {"run_id": str(run_dir.relative_to(self._run_root())).replace("\\", "/"),
                    "run_dir": str(run_dir), "launch": payload, "dry_run": True}

        log_path = run_dir / LOG_FILE
        creationflags = _CREATE_NO_WINDOW if sys.platform == "win32" else 0
        with open(log_path, "wb") as log:
            proc = subprocess.Popen(          # noqa: S603 - argv 由白名单食谱生成
                cmd, cwd=str(self.project_root), stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT,
                creationflags=creationflags,
                start_new_session=(sys.platform != "win32"),
            )
        payload["pid"] = proc.pid
        _write_json(run_dir / LAUNCH_FILE, payload)
        with self._lock:
            self._procs[str(run_dir.resolve())] = _Proc(
                proc=proc, run_dir=run_dir, started=payload["started"], recipe=recipe.id)
        return {"run_id": str(run_dir.relative_to(self._run_root())).replace("\\", "/"),
                "run_dir": str(run_dir), "launch": payload}

    # ------------------------------------------------------------------
    def status(self, run_dir: str | Path) -> dict[str, Any] | None:
        path = Path(run_dir).resolve()
        info = _read_json(path / LAUNCH_FILE)
        if not info:
            return None
        info = dict(info)
        with self._lock:
            entry = self._procs.get(str(path))
        if entry is not None:
            code = entry.proc.poll()
            if code is None:
                info["state"] = _RUNNING
            else:
                info["state"] = _EXITED if code == 0 else _FAILED
                info["exit_code"] = code
                info["elapsed"] = round(time.time() - entry.started, 1)
                info["can_stop"] = False
                _patch_json(path / LAUNCH_FILE, {"state": info["state"],
                                                 "exit_code": code,
                                                 "elapsed": info["elapsed"]})
                return info
            info["elapsed"] = round(time.time() - entry.started, 1)
            info["can_stop"] = True
            return info
        # 看板重启过: 只能探活, 拿不到退出码
        pid = info.get("pid")
        if _pid_alive(pid):
            info["state"] = _RUNNING
            info["elapsed"] = round(max(0.0, time.time() - float(info.get("started") or 0)), 1)
        elif info.get("exit_code") is not None:
            info["state"] = _EXITED if info["exit_code"] == 0 else _FAILED
        else:
            info["state"] = _UNKNOWN
        info["can_stop"] = False
        return info

    # ------------------------------------------------------------------
    def stop(self, run_dir: str | Path) -> bool:
        path = Path(run_dir).resolve()
        with self._lock:
            entry = self._procs.get(str(path))
        if entry is None or entry.proc.poll() is not None:
            return False
        entry.proc.terminate()
        try:
            entry.proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            entry.proc.kill()
        return True

    # ------------------------------------------------------------------
    def log(self, run_dir: str | Path, *, lines: int = 200,
            max_bytes: int = 512 * 1024) -> str:
        path = Path(run_dir) / LOG_FILE
        if not path.is_file():
            return ""
        size = path.stat().st_size
        with open(path, "rb") as fh:
            if size > max_bytes:
                fh.seek(size - max_bytes)
            raw = fh.read()
        text = raw.decode("utf-8", errors="replace")
        rows = text.splitlines()
        return "\n".join(rows[-max(1, int(lines)):])


# ----------------------------------------------------------------------
def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _patch_json(path: Path, extra: dict[str, Any]) -> None:
    data = _read_json(path)
    if not data:
        return
    data.update(extra)
    _write_json(path, data)
