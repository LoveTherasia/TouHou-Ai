"""环境: 把后端 + 奖励 + 统计 组合成标准 RL 接口.

对外的语义就是最经典的 ``reset / step / observe``::

    env = TouhouEnv(config)
    env.start()
    result = env.reset(difficulty=1)         # 开始新一局
    result = env.step("right+shoot", frames=8)  # 执行动作 -> state/reward/done
"""

from __future__ import annotations

import functools
import threading
from pathlib import Path
from typing import Any

from PIL import Image

from .backends import BACKENDS, Backend
from .capture.visualize import Trail, render_state
from .config import Config
from .errors import BackendError, NoActiveEpisodeError, OffsetUnavailableError
from .games.registry import GameRegistry, GameSpec
from .inputs.actions import Action, parse
from .inputs.bombguard import BombGuard
from .inputs.dialogue import DialogueAssist
from .memory.bullettrack import BulletTracker
from .rl.reward import RewardModel
from .rl.stats import EpisodeStats, StatsTracker
from .types import GameState, StepResult


def _synchronized(method: Any) -> Any:
    """用 ``self.lock`` 串行化环境操作.

    LLM Agent 通过 MCP 观测/调参与后台训练线程 (``Trainer``) 会并发访问同一个
    :class:`TouhouEnv`; 一把可重入锁让两边安全共存 —— Agent 的 ``th_configure``
    可以插在训练的两步之间生效。
    """

    @functools.wraps(method)
    def wrapper(self: "TouhouEnv", *args: Any, **kwargs: Any) -> Any:
        lock = getattr(self, "lock", None)
        if lock is None:  # pragma: no cover - 理论上 __init__ 一定建了锁
            return method(self, *args, **kwargs)
        with lock:
            return method(self, *args, **kwargs)

    return wrapper


class TouhouEnv:
    """东方正作 RL 环境."""

    def __init__(
        self,
        config: Config | None = None,
        *,
        registry: GameRegistry | None = None,
        backend: str | None = None,
        game: str | None = None,
    ) -> None:
        self.config = config or Config.load()
        self.registry = registry or GameRegistry()

        #: 串行化环境操作: MCP 调参 / 观测 与后台训练线程共用这一把可重入锁
        self.lock = threading.RLock()

        backend_name = backend or str(self.config.get("backend", "mock"))
        game_key = game or str(self.config.get("game", "th11"))
        if backend_name not in BACKENDS:
            raise BackendError(
                f"未知后端 {backend_name!r}; 可用: {', '.join(sorted(BACKENDS))}"
            )
        self.backend_name = backend_name
        self.spec: GameSpec = self.registry.get(game_key)
        self.backend: Backend = BACKENDS[backend_name](self.spec, self.config)

        self.reward_model = RewardModel(dict(self.config.section("reward")))
        self.stats = StatsTracker()
        self.trail = Trail(int(self.config.get("visualize.trail_length", 180)))

        self._last_state: GameState | None = None
        self._episode: EpisodeStats | None = None
        self._episode_active = False
        self._episode_frames = 0
        self._started = False
        self._dialogue = DialogueAssist(
            threshold=int(self.config.get("dialogue_threshold_frames", 25) or 25)
        )
        #: 紧急放弹保护 (规则): 只在最近弹幕极近时放弹
        bg = self.config.section("bomb_guard")
        self._bomb_guard = BombGuard(
            enabled=bool(bg.get("enabled", True)),
            radius=float(bg.get("radius", 14.0)),
            min_power=float(bg.get("min_power", 1.0)),
            cooldown_frames=int(bg.get("cooldown_frames", 45)),
            max_per_episode=int(bg.get("max_per_episode", 6)),
        )
        #: 弹幕速度追踪 (按槽位跨帧差分; 只有真机有 active_slots 才有意义)
        self._bullet_tracker = BulletTracker()
        #: 连续出局计数 (in_game 是启发式, 换面/对话会闪断; 需连续多步才判 left_game)
        self._out_of_game_streak = 0
        self._seen_in_game = False

        if self.backend.controller is not None:
            self.backend.controller.on_advance = self._sample_trail

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    @_synchronized
    def start(self) -> dict[str, Any]:
        self.backend.start()
        self._started = True
        return self.status()

    @property
    def started(self) -> bool:
        return self._started

    @_synchronized
    def close(self) -> None:
        if self.backend.controller is not None:
            self.backend.controller.release_all()
        self.backend.close()
        self._started = False

    def __enter__(self) -> "TouhouEnv":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    def _sample_trail(self) -> None:
        if self._last_state is None:
            return
        state = self._last_state
        self.trail.push(state.player_x, state.player_y)

    # ------------------------------------------------------------------
    @_synchronized
    def reset(self, **options: Any) -> StepResult:
        """开始新一局."""
        if not self._started:
            self.start()
        state = self.backend.reset(options=dict(options))
        self._dialogue = DialogueAssist(
            threshold=int(self.config.get("dialogue_threshold_frames", 25) or 25)
        )
        self._bomb_guard.reset()
        self._bullet_tracker.reset()
        self.trail.clear()
        self.trail.push(state.player_x, state.player_y)

        self._episode = EpisodeStats(index=len(self.stats.episodes) + 1)
        self._episode_active = True
        self._episode_frames = 0
        self._out_of_game_streak = 0
        self._seen_in_game = False
        self._last_state = state
        return StepResult(
            state=state,
            reward=0.0,
            done=False,
            truncated=False,
            info={
                "event": "reset",
                "episode": self._episode.index,
                "backend": self.backend_name,
                "game": self.spec.key,
                "options": {k: v for k, v in options.items() if not k.startswith("_")},
            },
        )

    # ------------------------------------------------------------------
    @_synchronized
    def step(
        self,
        action: Any = "noop",
        *,
        frames: int | None = None,
        auto_reset: bool | None = None,
        **params: Any,
    ) -> StepResult:
        """执行一个动作, 推进若干帧, 返回 ``(state, reward, done, info)``."""
        if not self._started:
            self.start()
        if not self._episode_active:
            if auto_reset is None:
                auto_reset = bool(self.config.get("auto_reset", False))
            if auto_reset:
                self.reset()
            else:
                raise NoActiveEpisodeError(
                    "当前没有进行中的一局, 请先调用 start_new_game (或 step 时传 auto_reset=true)。"
                )

        default_frames = int(frames if frames is not None else self.config.get("step_frames", 4))
        max_frames = int(self.config.get("max_step_frames", 600))
        default_frames = max(1, min(default_frames, max_frames))

        parsed = parse(action, default_frames=default_frames, extra_params=params)
        if parsed.frames is None:
            parsed.frames = default_frames
        parsed.frames = max(1, min(int(parsed.frames), max_frames))

        prev = self._last_state
        in_dialogue = self._dialogue.update(prev, parsed.frames)
        # 对话/静默期: **连续按住射击快进**, 且保留策略本步的移动意图 (不覆盖动作)
        assist = (self.backend_name == "memory" and self.config.get("dialogue_assist", True)
                  and in_dialogue and not parsed.is_macro)
        # 紧急放弹: 只在迫不得已(最近弹幕极近)时, 于本动作前补一次放弹
        bomb_guard = False
        if self.backend_name == "memory" and prev is not None:
            bomb_guard = self._bomb_guard.update(prev, parsed.frames)
        bg_frames = 0
        if bomb_guard:
            controller = self.backend.controller
            controller.release_all()
            bg_frames = controller.tap(["bomb"], hold=4, after=4)
        if assist:
            # 对话/静默期: **连续点按 Z** 快进 —— 东方是边沿触发, 长按 Z 不会连跳。
            # 每次点按后重新读一次状态, 一旦敌人/弹幕回来就立刻停手回到战斗。
            controller = self.backend.controller
            keep = [b for b in parsed.buttons if b in {"up", "down", "left", "right", "focus"}]
            controller.release_all()
            spent, taps = controller.tap_repeat(
                [*keep, "confirm"],
                keep=keep,
                hold=int(self.config.get("dialogue_tap_hold", 2) or 2),
                after=int(self.config.get("dialogue_tap_after", 6) or 6),
                max_taps=int(self.config.get("dialogue_max_taps", 12) or 12),
                stop=lambda: not self._dialogue.is_empty(self.backend.read_state()),
            )
            controller.release_all()
            report = {"frames": spent, "dialogue_taps": taps}
        else:
            report = self.backend.apply(parsed)
        if bg_frames:
            report["frames"] = int(report.get("frames", 0)) + bg_frames
        state = self.backend.read_state()
        self._bullet_tracker.update(state)
        state.raw["dialogue_assist"] = bool(assist)
        self._last_state = state

        advanced = int(report.get("frames", parsed.frames) or 0)
        self._episode_frames += advanced

        game_over, cleared, reason = self._terminal_flags(prev, state)
        done = bool(game_over or cleared)
        # max_episode_frames <= 0 表示不限长 (只有真正 game over / clear 才结束)
        max_frames = int(self.config.get("max_episode_frames", 1e9) or 0)
        truncated = max_frames > 0 and self._episode_frames > max_frames

        reward, parts = self.reward_model.compute(
            prev, state, done=done, cleared=cleared, game_over=game_over
        )

        info: dict[str, Any] = {
            "event": "step",
            "action": parsed.name,
            "action_detail": parsed.to_dict(),
            "dialogue_assist": bool(assist),
            "bomb_guard": bool(bomb_guard),
            "frames": advanced,
            "episode_frames": self._episode_frames,
            "backend_frames": report,
            "reward_parts": parts,
            "done_reason": reason,
            "episode": self._episode.index if self._episode else None,
        }
        if report.get("notes"):
            info["notes"] = report["notes"]

        if self._episode is not None:
            self._episode.update(
                StepResult(state=state, reward=reward, done=done, truncated=truncated, info=info),
                prev,
                parsed.name,
            )
            if done or truncated:
                self._episode.cleared = cleared
                self._episode.game_over = game_over
                self._episode.truncated = truncated
                self.stats.add(self._episode)
                self._episode_active = False
                info["episode_summary"] = self._episode.to_dict()

        return StepResult(state=state, reward=reward, done=done, truncated=truncated, info=info)

    # ------------------------------------------------------------------
    def _terminal_flags(self, prev: GameState | None, state: GameState) -> tuple[bool, bool, str]:
        """结束判定。

        真机 (memory 后端) 没有可靠的 "game_over/cleared" 标志, 唯一可信的信号是
        **生命值**: lives <= game_over_lives (th11 实测 -1) 才算 game over。

        特别注意: ``in_game`` 是**启发式**(由 stage/lives/power/score 区间推断),
        换面转场 / 剧情对话 / 读取抖动时会假性为 False 一段时间。若据此判 left_game,
        训练就会在换面时把好局误判成结束并 reset (restart_process 时还会杀进程重启)。
        因此 left_game 默认**关闭** (left_game_patience=0), 需要时才显式开启。
        """
        raw = state.raw or {}
        # mock 后端会显式给出这两个标志; 真机为 None, 不会走到这里
        if raw.get("cleared"):
            return False, True, "cleared"
        if raw.get("game_over"):
            return True, False, "game_over"

        threshold = int(self.config.get("game_over_lives", -1))
        if state.lives is not None and state.lives <= threshold:
            return True, False, "game_over"

        final_stage = int(self.config.get("final_stage", 6))
        if state.stage is not None and state.stage > final_stage:
            return False, True, "cleared"

        # 可选: 连续 N 步不在局内才判 left_game (默认关闭, 见上)
        patience = max(0, int(self.config.get("left_game_patience", 0)))
        if patience > 0:
            if state.in_game:
                self._out_of_game_streak = 0
                self._seen_in_game = True
            else:
                self._out_of_game_streak += 1
            if self._seen_in_game and self._out_of_game_streak >= patience:
                return True, False, "left_game"
        return False, False, ""

    # ------------------------------------------------------------------
    @_synchronized
    def get_state(self) -> GameState:
        if not self._started:
            self.start()
        state = self.backend.read_state()
        self._bullet_tracker.update(state)
        self._last_state = state
        return state

    # ------------------------------------------------------------------
    @_synchronized
    def screenshot(self) -> tuple[Image.Image, str]:
        if not self._started:
            self.start()
        return self.backend.screenshot()

    @_synchronized
    def capture_frame(self, *, mode: str = "auto") -> Image.Image:
        """训练可视化用: 取一张"当前画面".

        * ``auto``  —— 真实后端截图 (游戏窗口), 模拟器后端渲染状态图
        * ``screenshot`` —— 强制截图
        * ``visualize``  —— 强制状态可视化 (带坐标网格/判定点, 更适合看清局面)
        """
        if mode == "visualize":
            return self.visualize()
        if mode == "screenshot" or (mode == "auto" and self.backend_name == "memory"):
            return self.screenshot()[0]
        return self.visualize()

    @_synchronized
    def visualize(self, *, playfield_only: bool = True, **overrides: Any) -> Image.Image:
        state = self.get_state()
        options = dict(self.config.section("visualize"))
        options.update({k: v for k, v in overrides.items() if v is not None})
        title = f"{self.spec.name_zh} {self.spec.name_en} [{self.backend_name}]"
        image = render_state(
            state,
            self.spec,
            scale=float(options.get("scale", 1.0)),
            show_bullets=bool(options.get("show_bullets", True)),
            show_grid=bool(options.get("show_grid", True)),
            show_hud=bool(options.get("show_hud", True)),
            trail=list(self.trail.points) if options.get("trail_length", 1) else None,
            title=title,
        )
        return image

    # ------------------------------------------------------------------
    @_synchronized
    def configure(self, patch: dict[str, Any]) -> dict[str, Any]:
        """在线改配置 (奖励权重等); 返回更新后的配置."""
        updated = self.config.patch(patch)
        if "reward" in patch:
            self.reward_model.update_weights(self.config.section("reward"))
        if "visualize" in patch:
            self.trail = Trail(int(self.config.get("visualize.trail_length", 180)))
        return updated

    # ------------------------------------------------------------------
    @_synchronized
    def status(self) -> dict[str, Any]:
        base = self.backend.status()
        base.update(
            {
                "started": self._started,
                "spec": {
                    "key": self.spec.key,
                    "name_zh": self.spec.name_zh,
                    "name_en": self.spec.name_en,
                    "exes": self.spec.exes,
                    "calibrated": self.spec.calibrated,
                },
                "episode_active": self._episode_active,
                "episode": self._episode.to_dict() if self._episode else None,
            }
        )
        return base
