"""真实游戏后端 (读进程内存 + SendInput 注入按键 + 窗口截图).

工作流::

    start()   ->  找到/启动 thXXc.exe, 定位主窗口, 建立内存读取器
    reset()   ->  (可选)重启进程, 然后按 menu_sequence 导航到"开始游戏"
    advance() ->  按 60fps 推进时间; 若标定了 frame 偏移则严格等待帧数递增
    read_state() -> 按 offsets json 里的字段定义读内存
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Any

from PIL import Image

from ..config import Config
from ..errors import BackendError, GameNotFoundError, NotInGameError, OffsetUnavailableError
from ..games.registry import GameSpec
from ..inputs.actions import Action
from ..inputs.controller import Controller, sleep_until
from ..inputs.keyboard import Keyboard
from ..memory.process import (
    ProcessMemory,
    find_first_pid,
    find_game_window,
    terminate_pid,
    wait_for_process,
)
from ..memory.reader import StateReader
from ..memory.scanner import MemoryScanner
from ..types import GameState
from .base import Backend


class MemoryBackend(Backend):
    """真实游戏后端."""

    name = "memory"

    def __init__(self, spec: GameSpec, config: Config) -> None:
        super().__init__(spec, config)
        self.pid: int | None = None
        self.mem: ProcessMemory | None = None
        self.reader: StateReader | None = None
        self.keyboard: Keyboard | None = None
        self.game_dir: str = str(config.get("game_dir") or "")
        self.exe_path: Path | None = None
        self._scanner: MemoryScanner | None = None
        self._deadline = time.perf_counter()
        self._started = False
        self.hz = int(config.get("control.tick_hz", 60))
        self._speedhook: Any = None
        self._speedhook_result: dict[str, Any] | None = None
        self._speedhook_error: str | None = None
        self._title_hints = list(spec.folder_hints) + [
            "東方", "东方", "touhou", "地霊殿", "地灵殿",
        ]

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    @property
    def window(self):
        if self.pid is None:
            return None
        return find_game_window(self.pid, self._title_hints)

    def start(self) -> None:
        """找到(或启动)游戏进程并建立连接."""
        if self._started and self.mem is not None and self.mem.is_alive():
            return

        self._resolve_exe_path()
        self.pid = find_first_pid(self.spec.exes)
        if self.pid is None:
            if self.config.get("backend") == "memory" and self.exe_path is not None:
                self._launch()
            else:
                raise GameNotFoundError(
                    f"未找到 {self.spec.key} 进程 ({', '.join(self.spec.exes)})。"
                    f"先手动启动游戏, 或确认 config.game_dir 指向安装目录后由 MCP 启动。"
                )

        self.mem = ProcessMemory(self.pid)
        self.reader = StateReader(self.mem, self.spec)

        window = find_game_window(self.pid, self._title_hints)
        self.keyboard = Keyboard(window.hwnd if window else None, focus=bool(self.config.get("control.focus_window", True)))
        self.controller = Controller(
            self.keyboard,
            self.advance,
            hz=self.hz,
            state_fn=self.read_state,
            game=self.spec.key,
        )
        self._scanner = MemoryScanner(
            self.mem,
            max_region_bytes=int(self.config.get("scanner.max_region_mb", 64)) * 1024 * 1024,
        )
        self._deadline = time.perf_counter()
        self._started = True
        self._apply_speedhook()

    def _apply_speedhook(self) -> None:
        """按 config.slowdown 给游戏注入时间缩放钩子 (可选, 尽力而为)."""
        self._speedhook_result = None
        self._speedhook_error = None
        if not bool(self.config.get("slowdown.enabled", False)):
            return
        factor = float(self.config.get("slowdown.factor", 1.0) or 1.0)
        if factor == 1.0 or self.pid is None:
            return
        from ..memory.slowdown import SpeedHook

        hook = SpeedHook()
        if not hook.available():
            self._speedhook_error = "钩子未构建: " + ", ".join(hook.missing())
            return
        try:
            self._speedhook_result = hook.apply(self.pid, factor)
            self._speedhook = hook
        except Exception as exc:  # 注入失败不应阻断连接
            self._speedhook_error = f"{type(exc).__name__}: {exc}"

    def _resolve_exe_path(self) -> None:
        if self.exe_path is not None and self.exe_path.exists():
            return
        game_dir = self.game_dir
        if not game_dir:
            from ..games.registry import GameRegistry

            detected = GameRegistry().detect_dir(
                self.spec.key,
                list(self.config.get("search_roots") or []),
                int(self.config.get("search_depth", 2)),
            )
            game_dir = detected or ""
            self.game_dir = game_dir
        if game_dir:
            from ..games.registry import GameRegistry

            self.exe_path = GameRegistry().find_exe(self.spec.key, game_dir)

    def _launch(self) -> None:
        assert self.exe_path is not None
        if find_first_pid(self.spec.exes) is not None:
            return
        subprocess.Popen([str(self.exe_path)], cwd=str(self.exe_path.parent))
        self.pid = wait_for_process(self.spec.exes, timeout=30.0)
        time.sleep(float(self.config.get("launch_wait_seconds", 6.0)))

    # ------------------------------------------------------------------
    def reset(self, *, options: dict[str, Any] | None = None) -> GameState:
        options = dict(options or {})
        self.start()
        assert self.mem is not None and self.controller is not None

        restart = bool(options.get("restart_process", self.config.get("restart_process", False)))
        if restart:
            self._restart()

        if self.keyboard is not None and self.keyboard.hwnd is None:
            window = self.window
            if window is not None:
                self.keyboard.set_window(window.hwnd)
        if self.keyboard is not None and self.keyboard.focus_enabled:
            self.keyboard.focus_window()

        self._deadline = time.perf_counter()

        state = self.read_state()
        skip_if_in_game = bool(options.get("skip_if_in_game", False))
        mode = str(options.get("menu") or "auto")

        # 注意: 标题画面闲置约 45 秒后游戏会自己播 Demo, Demo 期间读到的也是
        # "局内"数值, 所以 skip_if_in_game 默认关闭 —— 想保证全新一局请用
        # restart_process=True (会杀进程重启, 在 Demo 开始之前完成菜单导航)。
        if mode == "none" or (skip_if_in_game and state.in_game and not restart):
            self.controller.release_all()
            return self.read_state()

        if mode == "keys":
            steps = options.get("steps") or []
            values = options.get("values") or {}
            self.controller.run_menu_sequence(steps, values)
        else:
            values = {
                "difficulty": int(options.get("difficulty", 1)),
                "character": int(options.get("character", 0)),
                "shot": int(options.get("shot", 0)),
                "menu_index": int(options.get("menu_index", 0)),
            }
            sequence = [dict(step) for step in self.spec.menu_sequence]
            if self.spec.key == "th11" and restart:
                for step in sequence:
                    if step.get("index_from") in {"character", "shot"}:
                        step["anchor_times"] = 0
                    if step.get("index_from") == "character":
                        step["direction"] = "right"
            self.controller.run_menu_sequence(sequence, values)

        self.controller.release_all()
        self._deadline = time.perf_counter()
        result = self.read_state()
        if self.spec.key == "th11" and restart and mode == "auto":
            expected = (int(options.get("character", 0)), int(options.get("shot", 0)))
            if (result.char_main, result.char_sub) != expected:
                raise BackendError(f"选机校验失败: 请求 {expected}, 实际 {(result.char_main, result.char_sub)}")
        return result

    def _restart(self) -> None:
        pid = find_first_pid(self.spec.exes)
        if pid is not None:
            terminate_pid(pid)
        if self.mem is not None:
            self.mem.close()
        self.mem = None
        self.reader = None
        self._scanner = None  # 必须一起丢掉: 它持有旧的(已关闭)进程句柄
        self.keyboard = None
        self.controller = None
        self._started = False
        self._resolve_exe_path()
        self._launch()
        self.start()

    # ------------------------------------------------------------------
    def read_state(self) -> GameState:
        if self.reader is None:
            raise BackendError("尚未 attach 到游戏进程")
        state = self.reader.read_all()
        state.raw["game"] = self.spec.key
        state.raw["pid"] = self.pid
        return state

    # ------------------------------------------------------------------
    def advance(self, frames: int) -> int:
        frames = max(0, int(frames))
        if frames == 0:
            return 0

        frame_address = self.reader.address_of("frame") if self.reader else None
        if frame_address is not None and self.mem is not None:
            return self._advance_by_frame_counter(frames, frame_address)
        return self._advance_wall_clock(frames)

    def _advance_wall_clock(self, frames: int) -> int:
        """按墙钟推进 (没有 frame 偏移时的回退)。"""
        now = time.perf_counter()
        if self._deadline < now:
            self._deadline = now
        self._deadline += frames / self.hz
        sleep_until(self._deadline)
        return frames

    def _advance_by_frame_counter(self, frames: int, address: int) -> int:
        assert self.mem is not None
        try:
            start = int(self.mem.read_value(address, "i32"))
        except Exception:
            # 读帧失败: 回退墙钟 (注意不能调 self.advance, 否则无限递归)
            self._deadline = time.perf_counter()
            return self._advance_wall_clock(frames)
        target = start + frames
        guard = time.perf_counter() + frames / self.hz * 4 + 0.5
        while time.perf_counter() < guard:
            try:
                current = int(self.mem.read_value(address, "i32"))
            except Exception:
                break
            if current >= target:
                return current - start
            time.sleep(0.0005)
        return frames

    # ------------------------------------------------------------------
    def screenshot(self) -> tuple[Image.Image, str]:
        window = self.window
        if window is None:
            raise BackendError("找不到游戏窗口, 无法截图")
        from ..capture.screenshot import grab_window

        return grab_window(window.hwnd, prefer_printwindow=bool(self.config.get("capture.prefer_printwindow", True)))

    # ------------------------------------------------------------------
    def scanner(self) -> MemoryScanner:
        if self._scanner is None:
            self.start()
        assert self._scanner is not None
        return self._scanner

    # ------------------------------------------------------------------
    def release_keys(self) -> None:
        if self.keyboard is not None:
            self.keyboard.release_all()

    def close(self) -> None:
        self.release_keys()
        if self._speedhook is not None:
            try:  # 断开时把游戏恢复原速
                self._speedhook.write_factor(1.0)
            except Exception:
                pass
            self._speedhook = None
        if self.mem is not None:
            self.mem.close()
        self.mem = None
        self.reader = None
        self._started = False
        super().close()

    # ------------------------------------------------------------------
    def status(self) -> dict[str, Any]:
        info = super().status()
        window = None
        try:
            window = self.window
        except Exception:
            window = None
        info.update(
            {
                "pid": self.pid,
                "game_dir": self.game_dir,
                "exe": str(self.exe_path) if self.exe_path else None,
                "window": window.to_dict() if window else None,
                "module_base": hex(self.reader.base_address) if self.reader else None,
                "entity_profile": self.spec.entities.get("profile"),
                "calibrated_fields": sorted(
                    name for name, field in self.spec.fields.items() if field.calibrated
                ),
                "uncalibrated_fields": self.spec.uncalibrated(),
                "slowdown": {
                    "enabled": bool(self.config.get("slowdown.enabled", False)),
                    "factor": float(self.config.get("slowdown.factor", 1.0) or 1.0),
                    "applied": self._speedhook_result,
                    "error": self._speedhook_error,
                },
            }
        )
        return info

    def apply(self, action: Action) -> dict[str, Any]:
        if self.keyboard is not None and self.keyboard.focus_enabled:
            self.keyboard.focus_window()
        return super().apply(action)
