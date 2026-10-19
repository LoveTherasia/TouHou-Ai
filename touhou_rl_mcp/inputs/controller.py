"""控制器: 把"动作"翻译成按键 + 推进游戏帧.

关键设计 (为 agentic RL 服务):
  * LLM Agent 通常只能以 1~10 Hz 决策, 而游戏跑在 60fps。因此控制器提供
    **宏动作** (move_to / dodge / wait_in_game), 由内部 60Hz 闭环执行,
    Agent 只需要给出高层意图。
  * ``advance()`` 是唯一的"时间推进"入口, 由后端实现:
      - 真实游戏: 按帧率 sleep (若有帧计数偏移则等帧数递增)
      - 模拟器: 直接跑仿真
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from ..errors import OffsetUnavailableError
from ..types import GameState
from ..observation import require_entities
from .actions import Action
from .keyboard import resolve_button


class InputSink(Protocol):
    """能接收"当前按住的按键集合"的对象."""

    def hold(self, names: list[str]) -> None: ...

    def release_all(self) -> None: ...

    @property
    def held_names(self) -> set[str]: ...


class NullSink:
    """不真的发按键, 只记录状态 (给模拟器后端用)."""

    def __init__(self) -> None:
        self._names: set[str] = set()

    def hold(self, names: list[str]) -> None:
        self._names = {resolve_button(n) for n in names}

    def release_all(self) -> None:
        self._names.clear()

    @property
    def held_names(self) -> set[str]:
        return set(self._names)


@dataclass
class ExecutionReport:
    action: str
    frames: int
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {"action": self.action, "frames": self.frames, "notes": self.notes}


class Controller:
    """按键 + 帧推进的统合器."""

    def __init__(
        self,
        sink: InputSink,
        advance_fn: Callable[[int], int],
        *,
        hz: int = 60,
        state_fn: Callable[[], GameState] | None = None,
        focus_window: bool = True,
        game: str = "?",
    ) -> None:
        self.sink = sink
        self.advance_fn = advance_fn
        self.hz = max(1, int(hz))
        self.state_fn = state_fn
        self.focus_window = focus_window
        self.game = game
        self.total_frames = 0
        #: 每次 advance 之后调用 (可视化采样轨迹用)
        self.on_advance: Callable[[], None] | None = None

    # ------------------------------------------------------------------
    @property
    def frame_seconds(self) -> float:
        return 1.0 / self.hz

    def state(self) -> GameState:
        if self.state_fn is None:
            raise RuntimeError("Controller 未绑定状态读取函数")
        return self.state_fn()

    def advance(self, frames: int) -> int:
        frames = max(0, int(frames))
        if frames == 0:
            return 0
        advanced = self.advance_fn(frames)
        self.total_frames += advanced
        if self.on_advance is not None:
            try:
                self.on_advance()
            except Exception:
                pass
        return advanced

    # ------------------------------------------------------------------
    def focus(self) -> bool:
        """把输入焦点交给游戏窗口 (DirectInput 只读前台窗口的按键)."""
        focus_fn = getattr(self.sink, "focus_window", None)
        if callable(focus_fn) and self.focus_window:
            try:
                return bool(focus_fn())
            except Exception:
                return False
        return False

    def hold(self, buttons: list[str] | tuple[str, ...]) -> None:
        self.sink.hold(list(buttons))

    def release_all(self) -> None:
        self.sink.release_all()

    def tap(self, buttons: list[str] | tuple[str, ...], *, hold: int = 4, after: int = 0) -> int:
        self.focus()
        spent = 0
        self.hold(list(buttons))
        spent += self.advance(hold)
        self.release_all()
        if after:
            spent += self.advance(after)
        return spent

    def tap_repeat(
        self,
        buttons: list[str] | tuple[str, ...],
        *,
        keep: list[str] | tuple[str, ...] = (),
        hold: int = 2,
        after: int = 6,
        max_taps: int = 12,
        stop: Callable[[], bool] | None = None,
    ) -> tuple[int, int]:
        """连续点按 —— 用于**快进对话**。

        东方是**边沿触发**: 长按 Z 不会连跳, 必须反复"按下-松开"。
        每次: 按住 ``buttons`` ``hold`` 帧 -> 松开(但保留 ``keep``) ``after`` 帧。
        ``stop()`` 返回 True 时立即结束 (例如"敌人又出现了")。
        返回 ``(总推进帧数, 实际点按次数)``。
        """
        spent = 0
        taps = 0
        for _ in range(max(0, int(max_taps))):
            self.focus()
            self.hold(list(buttons))
            spent += self.advance(hold)
            self.hold(list(keep))
            if after:
                spent += self.advance(after)
            taps += 1
            if stop is not None and stop():
                break
        return spent, taps

    # ------------------------------------------------------------------
    def run(self, action: Action) -> ExecutionReport:
        """执行一个动作."""
        self.focus()
        notes: list[str] = []
        frame_budget = action.frames or 1

        if not action.is_macro:
            self.hold(list(action.buttons))
            spent = self.advance(frame_budget)
            after = action.params.get("after")
            if after:
                spent += self.advance(int(after))
            return ExecutionReport(action.name, spent, notes)

        name = action.name
        if name == "move_to":
            return self._macro_move_to(action, notes)
        if name in {"dodge", "escape_bullets", "autopilot"}:
            return self._macro_dodge(action, notes)
        if name == "wait_in_game":
            return self._macro_wait_in_game(action, notes)
        if name == "run_sequence":
            steps = action.params.get("steps") or []
            values = action.params.get("values") or {}
            log = self.run_menu_sequence(steps, values)
            return ExecutionReport(name, sum(int(s.get("frames", 0)) for s in log), [f"executed {len(log)} steps"])
        raise ValueError(f"未知的宏动作: {name}")

    # ------------------------------------------------------------------
    # 宏动作
    # ------------------------------------------------------------------
    def _require_player(self) -> tuple[float, float]:
        state = self.state()
        if state.player_x is None or state.player_y is None:
            raise OffsetUnavailableError(
                "player_x/player_y",
                game=self.game,
                hint=(
                    "找不到自机坐标偏移。先调用 start_new_game 进入游戏, 让自机移动, "
                    "再用 memory_scan(值=自机 x 坐标的 f32) + memory_filter 定位地址, "
                    "然后写入 games/offsets/<game>.json 的 fields.player_x / player_y。"
                ),
            )
        return float(state.player_x), float(state.player_y)

    def _macro_move_to(self, action: Action, notes: list[str]) -> ExecutionReport:
        params = action.params
        target_x = float(params.get("x", 192))
        target_y = float(params.get("y", 384))
        tolerance = float(params.get("tolerance", 8.0))
        timeout = int(params.get("timeout_frames") or action.frames or 240)
        keep_shooting = bool(params.get("shoot", True))
        keep_focus = bool(params.get("focus", False))

        spent = 0
        for _ in range(timeout):
            px, py = self._require_player()
            dx, dy = target_x - px, target_y - py
            if abs(dx) <= tolerance and abs(dy) <= tolerance:
                notes.append(f"到达目标 (误差 {abs(dx):.1f},{abs(dy):.1f})")
                break
            buttons: list[str] = []
            if dx > tolerance * 0.5:
                buttons.append("right")
            elif dx < -tolerance * 0.5:
                buttons.append("left")
            if dy > tolerance * 0.5:
                buttons.append("down")
            elif dy < -tolerance * 0.5:
                buttons.append("up")
            if keep_focus:
                buttons.append("focus")
            if keep_shooting:
                buttons.append("shoot")
            self.hold(buttons)
            spent += self.advance(1)
        else:
            notes.append(f"超时 ({timeout} 帧) 未到达目标, 当前位置 {(px, py)}")
        self.hold(["shoot"] if keep_shooting else [])
        return ExecutionReport(action.name, spent, notes)

    def _macro_dodge(self, action: Action, notes: list[str]) -> ExecutionReport:
        """启发式弹幕躲避 —— 对最近的若干弹做斥力求和, 再叠加"回到安全区"的吸引力."""
        params = action.params
        frames = int(params.get("frames") or action.frames or 60)
        radius = float(params.get("radius", 34.0))
        keep_shooting = bool(params.get("shoot", True))
        home_x = float(params.get("home_x", 192.0))
        home_y = float(params.get("home_y", 400.0))
        home_weight = float(params.get("home_weight", 0.35))
        assist = bool(params.get("assist", True))

        spent = 0
        for _ in range(frames):
            state = self.state()
            if assist:
                try:
                    require_entities(state)
                except Exception:
                    self.release_all()
                    raise
            px, py = self._require_player()
            if not assist:
                self.hold(["shoot"] if keep_shooting else [])
                spent += self.advance(1)
                continue

            fx = 0.0
            fy = 0.0
            for bx, by in state.bullets:
                dx, dy = px - bx, py - by
                dist2 = dx * dx + dy * dy
                if dist2 > radius * radius:
                    continue
                dist = math.sqrt(dist2) or 1.0
                weight = (radius - dist) / radius / dist
                fx += dx * weight
                fy += dy * weight

            # 轻微回归安全位, 避免贴着板边
            home_scale = 96.0
            fx += (home_x - px) * home_weight / home_scale
            fy += (home_y - py) * home_weight / home_scale

            magnitude = math.hypot(fx, fy)
            buttons: list[str] = []
            if magnitude > 1e-3:
                if fx > 0.12 * magnitude:
                    buttons.append("right")
                elif fx < -0.12 * magnitude:
                    buttons.append("left")
                if fy > 0.12 * magnitude:
                    buttons.append("down")
                elif fy < -0.12 * magnitude:
                    buttons.append("up")
            if keep_shooting:
                buttons.append("shoot")
            self.hold(buttons)
            spent += self.advance(1)
        return ExecutionReport(action.name, spent, notes)

    def _macro_wait_in_game(self, action: Action, notes: list[str]) -> ExecutionReport:
        timeout = int(action.params.get("timeout_frames") or action.frames or 900)
        press_confirm_every = int(action.params.get("press_confirm_every", 90))
        chunk = int(action.params.get("chunk", 15))
        spent = 0
        since_press = 0
        while spent < timeout:
            if self.state().in_game:
                notes.append(f"已进入游戏局内 (用了 {spent} 帧)")
                return ExecutionReport(action.name, spent, notes)
            if press_confirm_every and since_press >= press_confirm_every:
                self.tap(["confirm"], hold=4, after=0)
                since_press = 0
            elif not press_confirm_every:
                pass
            spent += self.advance(chunk)
            since_press += chunk
        notes.append(f"等待超时 ({timeout} 帧) 仍未进入游戏")
        return ExecutionReport(action.name, spent, notes)

    # ------------------------------------------------------------------
    def run_menu_sequence(self, steps: list[dict[str, Any]], values: dict[str, int]) -> list[dict[str, Any]]:
        """执行 offsets json 里的 ``menu_sequence`` (菜单导航到"开始游戏")."""
        log: list[dict[str, Any]] = []
        self.focus()
        for raw_step in steps:
            step: dict[str, Any]
            if isinstance(raw_step, dict):
                step = raw_step
            else:
                step = {"type": "press", "buttons": [raw_step]}
            kind = str(step.get("type", "press")).lower()
            if kind == "press":
                buttons = [resolve_button(b) for b in (step.get("buttons") or ["confirm"])]
                hold = int(step.get("hold", 6))
                after = int(step.get("after", 30))
                self.tap(buttons, hold=hold, after=after)
                log.append({"step": "press", "buttons": buttons, "frames": hold + after})
            elif kind == "wait":
                frames = int(step.get("frames", 60))
                self.advance(frames)
                log.append({"step": "wait", "frames": frames})
            elif kind == "select":
                direction = resolve_button(str(step.get("direction", "down")))
                anchor = resolve_button(str(step.get("anchor", "up")))
                anchor_times = int(step.get("anchor_times", 0))
                hold = int(step.get("hold", 4))
                after = int(step.get("after", 4))
                source = step.get("index_from")
                index = int(values.get(str(source), step.get("index", 0))) if source else int(step.get("index", 0))
                # 先"归零"到列表一端, 再按 index 次
                for _ in range(max(0, anchor_times)):
                    self.tap([anchor], hold=hold, after=after)
                for _ in range(max(0, index)):
                    self.tap([direction], hold=hold, after=after)
                log.append({"step": "select", "from": source, "index": index, "anchor_times": anchor_times})
            elif kind == "wait_in_game":
                report = self._macro_wait_in_game(
                    Action(
                        kind="macro",
                        name="wait_in_game",
                        frames=int(step.get("timeout_frames", 900)),
                        params={"press_confirm_every": int(step.get("press_confirm_every", 90))},
                    ),
                    [],
                )
                log.append({"step": "wait_in_game", "frames": report.frames, "notes": report.notes})
            else:
                log.append({"step": kind, "skipped": True})
        return log


# ----------------------------------------------------------------------
def sleep_until(deadline: float) -> None:
    """忙等 + sleep 混合, 到 ``deadline`` 才返回 (时间以 ``time.perf_counter`` 计)."""
    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            return
        if remaining > 0.002:
            time.sleep(remaining - 0.001)
        else:
            time.sleep(0)
