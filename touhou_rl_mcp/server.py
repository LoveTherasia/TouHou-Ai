"""MCP 服务器 —— 对外暴露的工具集合.

工具命名统一加 ``th_`` 前缀, 便于和其它 MCP server 共存。

核心闭环::

    th_start_new_game()  ->  初始 state
    th_step("right+shoot")  ->  {state, reward, done, truncated, info}
    th_get_state()          ->  当前 state
    th_screenshot()         ->  游戏画面 (图片)
    th_visualize_state()    ->  状态可视化 (图片, 含坐标网格/判定点/弹幕/HUD)
    th_configure({...})     ->  在线调奖励权重等参数 (agentic RL 调参入口)
"""

from __future__ import annotations

import functools
import json
import os
import threading
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.mcpserver.utilities.types import Image as MCPImage

from . import __version__
from .capture.screenshot import crop_playfield, fit_width, image_to_png_bytes, save_png
from .capture.visualize import image_to_png_bytes as viz_png_bytes
from .capture.visualize import save_png as save_viz_png
from .config import Config, default_config_path
from .env import TouhouEnv
from .errors import TouhouError
from .games.calibration import apply_calibration
from .games.registry import GameRegistry
from .inputs.actions import DIRECTION_SETS, MACROS, VERBS, parse
from .training import TrainConfig, Trainer

INSTRUCTIONS = """\
东方 Project (Touhou) 正作 RL 环境。把这台机器上的东方游戏 (th06~th14) 变成
一个可交互的强化学习环境, 供 LLM Agent 或 RL 训练循环使用。

典型用法:
  1. `th_list_games` 看支持哪些作品 / 本机装了哪些 / 哪些内存偏移已标定。
  2. `th_attach` 选择后端与作品:
       backend="mock"   —— 内置模拟器, 不需要游戏本体, 可复现, 适合先跑通流程;
       backend="memory" —— 真实游戏 (读进程内存 + 注入按键 + 窗口截图)。
  3. `th_start_new_game` 开始新一局, 之后循环:
       `th_step` 执行动作 -> 拿到 {state, reward, done, info}
       直到 done=true (game_over / cleared) 或 truncated=true。
  4. `th_screenshot` / `th_visualize_state` 取画面 (返回图片)。
  5. `th_configure` 在线修改奖励权重等超参 —— 这就是 agentic RL 里
     "让 Agent 自己调参" 的接口; `th_episode_stats` 观察训练曲线。

动作 (th_step 的 action 参数) 可以是:
  * 按键组合: "right" / "right+shoot" / "up,left,focus" ("+" "," 空格都行)
  * 简单动词: noop, shoot, focus, focus_shoot, bomb, confirm, cancel, pause
  * 宏动作 (内部 60Hz 闭环执行, 适合低频决策的 LLM):
      "move_to"  -> 配合 x / y 参数把自机开到指定坐标 (需要 player_x/y 偏移)
      "dodge"    -> 内置弹幕躲避辅助, 按 frames 帧自动躲 (需要 player_x/y + 弹幕)
      "wait_in_game" -> 推进到真正进入游戏局内 (跳过开场对话)

真实游戏为实时 60fps, 建议先用 `dodge` / `move_to` 这类宏动作 (低频决策),
或先用 mock 后端把策略/奖励调好再上真机。

内存偏移未标定时的符号: `th_memory_scan` / `th_memory_filter` / `th_memory_read`
是给标定用的工具 (类似 Cheat Engine 的首扫/再扫)。

训练本地智能体 (推荐玩法 —— LLM 反应慢, 不适合亲自打, 而是当"教练"):
  6. `th_train_start` 在后台线程用最高速度训练本地策略 (不占用 MCP 往返)。
  7. 循环观察 `th_train_status` / `th_train_dashboard` (仪表盘图片) -> 看学习曲线。
  8. 依据曲线用 `th_train_configure` 调超参、`th_configure` 调奖励 (下个 episode 生效),
     以此加速收敛; 需要时 `th_train_pause` / `th_train_stop` / `th_train_policy` 保存策略。
"""

mcp: MCPServer = MCPServer(
    name="touhou-rl-mcp",
    title="Touhou RL MCP",
    version=__version__,
    instructions=INSTRUCTIONS,
)


def _safe(fn: Any) -> Any:
    """把"预期内的失败"转成 MCP 的 ``ToolError`` (客户端看到一条清晰错误, 而不是堆栈)."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except ToolError:
            raise
        except TouhouError as exc:
            raise ToolError(str(exc)) from None
        except (ValueError, KeyError, TypeError) as exc:
            raise ToolError(f"{type(exc).__name__}: {exc}") from None

    return wrapper


def tool(fn: Any) -> Any:
    """注册一个 MCP 工具 (统一关闭 structured output, 并加错误保护)."""
    return mcp.tool(structured_output=False)(_safe(fn))

# ---------------------------------------------------------------- 全局状态
_ENV: TouhouEnv | None = None
_TRAINER: Trainer | None = None
_TRAINER_GUARD = threading.RLock()
_REGISTRY = GameRegistry()
_DIR_CACHE: dict[str, str | None] = {}


def _json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def get_env() -> TouhouEnv:
    global _ENV
    if _ENV is None:
        _ENV = TouhouEnv(Config.load(), registry=_REGISTRY)
        _ENV.start()
    return _ENV


def get_trainer() -> Trainer:
    """拿到(必要时创建)训练器 —— 它和 MCP 共用同一个 :class:`TouhouEnv`.

    训练循环在后台线程里以最高速度跑(不经过 MCP), 而 MCP 只负责启动/停止/
    观测/调参; 两者通过 ``env.lock`` 串行化, 见 :mod:`touhou_rl_mcp.env`。
    """
    global _TRAINER
    with _TRAINER_GUARD:
        if _TRAINER is None:
            env = get_env()
            _TRAINER = Trainer(env, TrainConfig(), env_lock=env.lock)
        return _TRAINER


def _discard_trainer(*, save: bool = False) -> None:
    """环境被重建/关闭前必须停掉训练线程, 否则它会继续驱动一个失效的 env."""
    global _TRAINER
    with _TRAINER_GUARD:
        if _TRAINER is None:
            return
        try:
            _TRAINER.stop(save=save)
        except Exception:  # pragma: no cover - 关闭路径尽力而为
            pass
        _TRAINER = None


def _detect_dir(game: str) -> str | None:
    if game in _DIR_CACHE:
        return _DIR_CACHE[game]
    config = Config.load()
    try:
        found = _REGISTRY.detect_dir(
            game,
            list(config.get("search_roots") or []),
            int(config.get("search_depth", 2)),
        )
    except TouhouError:
        found = None
    _DIR_CACHE[game] = found
    return found


# ================================================================ 元信息
@tool
def th_list_games(detect: bool = True) -> str:
    """列出支持的游戏作品、本机安装检测结果、以及各作品内存偏移的标定情况。

    Args:
        detect: 是否扫描磁盘查找安装目录 (第一次会花几秒)。
    """
    games = []
    for spec in _REGISTRY.all():
        calibrated = sorted(k for k, v in spec.fields.items() if v.calibrated)
        entry: dict[str, Any] = {
            "key": spec.key,
            "name": f"{spec.name_zh} / {spec.name_en}",
            "exes": spec.exes,
            "calibrated": spec.calibrated,
            "calibrated_fields": calibrated,
            "uncalibrated_fields": spec.uncalibrated(),
        }
        if detect:
            entry["detected_dir"] = _detect_dir(spec.key)
        games.append(entry)
    return _json(
        {
            "games": games,
            "note": (
                "backend='mock' 可直接用 (推荐先用它跑通奖励与策略); "
                "backend='memory' 需要游戏窗口化运行且对应作品的内存偏移已标定。"
            ),
        }
    )


@tool
def th_env_status() -> str:
    """查看当前环境状态: 后端、作品、进程/窗口、已标定字段、当前这一局的进度。"""
    return _json(get_env().status())


@tool
def th_get_config() -> str:
    """查看当前完整配置 (含奖励权重 reward、控制参数 control、可视化参数 visualize)。"""
    env = get_env()
    return _json(
        {
            "config_path": str(default_config_path()),
            "config": env.config.as_dict(),
            "reward_docs": env.reward_model.describe()["docs"],
            "actions": {
                "directions": sorted(DIRECTION_SETS),
                "verbs": sorted(VERBS),
                "macros": sorted(MACROS),
            },
        }
    )


@tool
def th_configure(patch: dict[str, Any], save: bool = False) -> str:
    """在线修改配置 —— agentic RL 的调参入口。

    常用: 调整奖励权重以改变学习信号, 例如
    ``{"reward": {"miss_penalty": -40, "graze_scale": 0.05, "shaping_scale": 0.5}}``

    Args:
        patch: 需要深合并进配置的对象, 例如 {"reward": {...}} / {"step_frames": 8}。
        save: 是否把结果写回 config.json (默认只在内存里生效)。
    """
    env = get_env()
    updated = env.configure(patch)
    saved = None
    if save:
        env.config.save()
        saved = str(default_config_path())
    return _json({"ok": True, "config": updated, "saved_to": saved})


# ================================================================ 连接
@tool
def th_attach(
    backend: str | None = None,
    game: str | None = None,
    game_dir: str = "",
    restart_process: bool = False,
    save: bool = False,
) -> str:
    """切换/重建环境 (后端与作品), 并连接游戏。

    Args:
        backend: "mock" 内置模拟器 | "memory" 真实游戏。
        game: 作品 key, 例如 th06 / th10 / th11 / th12 (见 th_list_games)。
        game_dir: 游戏安装目录; 留空则自动在 config.search_roots 里查找。
        restart_process: 每次 start_new_game 时是否先杀进程再重启 (保证全新一局)。
        save: 是否把 backend/game/game_dir 写回 config.json。
    """
    global _ENV
    patch: dict[str, Any] = {}
    if backend:
        patch["backend"] = backend
    if game:
        patch["game"] = game
    if game_dir:
        patch["game_dir"] = game_dir
    patch["restart_process"] = bool(restart_process)

    config = Config.load()
    config.patch(patch)

    _discard_trainer()
    if _ENV is not None:
        try:
            _ENV.close()
        except Exception:
            pass
        _ENV = None

    _ENV = TouhouEnv(config, registry=_REGISTRY)
    status = _ENV.start()
    if save:
        config.save()
    return _json({"ok": True, "status": status, "config_path": str(default_config_path())})


@tool
def th_close_env() -> str:
    """关闭环境: 松开所有按住的键, 断开内存连接 (不会关掉游戏进程)。"""
    global _ENV
    _discard_trainer()
    if _ENV is None:
        return _json({"ok": True, "note": "环境本来就没启动"})
    _ENV.close()
    _ENV = None
    return _json({"ok": True})


# ================================================================ 核心 RL 闭环
@tool
def th_start_new_game(
    difficulty: int = 1,
    character: int = 0,
    shot: int = 0,
    menu_index: int = 0,
    seed: int | None = None,
    restart_process: bool = False,
    skip_if_in_game: bool = False,
    start_stage: int = 1,
    include_bullets: bool = False,
) -> str:
    """开始新一局 (RL 的 reset)。返回初始状态。

    Args:
        difficulty: 0=Easy 1=Normal 2=Hard 3=Lunatic (memory 后端用于菜单导航)。
        character: 自机编号 (th11: 0=灵梦 1=魔理沙)。
        shot: 机体/装备编号 (0=A 1=B 2=C)。
        menu_index: 主菜单项编号 (0=Start 1=ExtraStart 2=PracticeStart)。
        seed: 仅 mock 后端: 固定随机种子 => 完全可复现的一局。
        restart_process: 杀掉游戏进程重开。**memory 后端想要"必定全新的一局"就设 True** ——
            游戏在标题画面闲置约 45 秒会自己播 Demo, Demo 期间也会被判定为"局内",
            只有重启进程才能在 Demo 开始前完成菜单导航。
        skip_if_in_game: memory 后端: 若判定已在局内就跳做菜单导航 (默认 False)。
        start_stage: mock 后端: 课程学习, 从第几面开局 (默认 1)。
        include_bullets: 返回的状态里是否包含弹幕坐标列表 (数据量大)。
    """
    env = get_env()
    result = env.reset(
        difficulty=difficulty,
        character=character,
        shot=shot,
        menu_index=menu_index,
        seed=seed,
        restart_process=restart_process,
        skip_if_in_game=skip_if_in_game,
        start_stage=start_stage,
    )
    return _json(result.to_dict(include_bullets=include_bullets))


@tool
def th_step(
    action: str = "noop",
    frames: int = 0,
    auto_reset: bool = False,
    x: float | None = None,
    y: float | None = None,
    radius: float = 34.0,
    timeout_frames: int = 0,
    include_bullets: bool = False,
) -> str:
    """执行一个动作并推进若干帧, 返回 {state, reward, done, truncated, info}。

    Args:
        action: 按键组合 ("right+shoot"、"up,left,focus") 或动词 (noop/shoot/focus/bomb)
                或宏动作 ("move_to" / "dodge" / "wait_in_game")。
        frames: 本动作持续帧数 (60fps)。0 表示用配置里的 step_frames。
        auto_reset: 若上一局已结束, 是否自动开始新一局。
        x / y: move_to 宏的目标坐标 (游戏场地坐标: 384x448, 原点左上)。
        radius: dodge 宏的避弹感知半径 (像素)。
        timeout_frames: move_to 的超时帧数。
        include_bullets: 状态里是否附上弹幕坐标列表。
    """
    env = get_env()
    params: dict[str, Any] = {}
    if x is not None:
        params["x"] = x
    if y is not None:
        params["y"] = y
    if action in {"dodge", "escape_bullets", "autopilot"}:
        params["radius"] = radius
    if action == "move_to" and timeout_frames:
        params["timeout_frames"] = timeout_frames
    result = env.step(
        action,
        frames=(frames or None),
        auto_reset=auto_reset,
        **params,
    )
    return _json(result.to_dict(include_bullets=include_bullets))


@tool
def th_get_state(include_bullets: bool = False) -> str:
    """查询当前状态 (不推进时间)。

    Args:
        include_bullets: 是否包含弹幕坐标列表 [[x, y], ...] (最多 256 个)。
    """
    env = get_env()
    return _json(env.get_state().to_dict(include_bullets=include_bullets))


@tool
def th_send_keys(steps: list[dict[str, Any]] | None = None, save: bool = False) -> str:
    """发送一串原始按键 (菜单导航 / 调试用)。

    Args:
        steps: 形如 ``[{"buttons": ["confirm"], "frames": 6, "after": 30}, {"buttons": ["down"]}]``
        save: 是否把截图/记录保存 (此处保存动作日志到 .runtime)。
    """
    env = get_env()
    controller = env.backend.controller
    if controller is None:
        env.start()
        controller = env.backend.controller
    assert controller is not None
    log = controller.run_menu_sequence(list(steps or []), {})
    return _json({"ok": True, "log": log, "total_frames": controller.total_frames})


@tool
def th_episode_stats(reset: bool = False, window: int = 50) -> str:
    """查看训练统计 (跨局滚动窗口): 平均回报 / 到达面数 / 通关率 / 擦弹 / 掉残。

    Args:
        reset: 是否清空已有统计。
        window: 滚动窗口大小。
    """
    env = get_env()
    if reset:
        env.stats.reset()
    env.stats.window = max(1, int(window))
    return _json(
        {
            "summary": env.stats.summary(),
            "current_episode": env.status().get("episode"),
            "episodes_tail": env.stats.episodes[-3:],
        }
    )


# ================================================================ 观测 (图片)
def _image_result(image_bytes: bytes, text: str, path: str | None) -> list[Any]:
    parts: list[Any] = [text if not path else f"{text}\n saved: {path}"]
    parts.append(MCPImage(data=image_bytes, format="png"))
    return parts


@tool
def th_screenshot(
    playfield_only: bool = False,
    max_width: int = 960,
    save: bool = True,
    quiet: bool = False,
) -> list[Any]:
    """截取当前画面 (真实后端截游戏窗口, 模拟器后端渲染合成画面)。

    Args:
        playfield_only: 只裁剪出游戏场地 (384x448) 区域。
        max_width: 返回前等比缩放的最大宽度 (0 表示不缩放)。
        save: 是否把 PNG 存到 .runtime/shots 下。
        quiet: 是否不返回图片 (只返回路径与说明)。
    """
    env = get_env()
    image, method = env.screenshot()
    if playfield_only:
        image = crop_playfield(image, env.spec.playfield, env.spec.window_size)
    image = fit_width(image, int(max_width))
    path = None
    if save:
        path = save_png(image, env.config.get("capture.out_dir") or None, prefix=f"{env.spec.key}-shot")
    text = f"截图完成: {image.width}x{image.height}, 方式={method}"
    if quiet:
        return [text]
    return _image_result(image_to_png_bytes(image), text, str(path) if path else None)


@tool
def th_visualize_state(
    scale: float = 1.0,
    show_bullets: bool = True,
    show_grid: bool = True,
    show_hud: bool = True,
    save: bool = True,
    quiet: bool = False,
) -> list[Any]:
    """把当前状态画成一张"结构化"的图 (坐标网格 + 判定点 + 弹幕 + Boss血条 + 数值 HUD)。

    比原始截图更适合让 LLM 理解局面; 真实后端若已标定 player_x/y 与弹幕偏移, 图上也会画出来。

    Args:
        scale: 画布缩放。
        show_bullets / show_grid / show_hud: 各图层开关。
        save: 是否存 PNG 到 .runtime/viz。
        quiet: 是否只返回文字说明 (不返回图片)。
    """
    env = get_env()
    image = env.visualize(
        scale=scale,
        show_bullets=show_bullets,
        show_grid=show_grid,
        show_hud=show_hud,
    )
    image = fit_width(image, int(env.config.get("capture.max_width", 960)))
    path = None
    if save:
        path = save_viz_png(image, env.config.get("capture.out_dir") or None, prefix=f"{env.spec.key}-state")
    text = f"状态可视化: {image.width}x{image.height}"
    if quiet:
        return [text]
    return _image_result(viz_png_bytes(image), text, str(path) if path else None)


# ================================================================ 内存标定
def _memory_scanner():
    env = get_env()
    if env.backend_name != "memory":
        raise TouhouError(
            "内存扫描只对真实游戏后端有效。先用 th_attach(backend='memory', game='th11') 连接游戏。"
        )
    backend = env.backend
    return backend.scanner(), backend  # type: ignore[attr-defined]


@tool
def th_memory_scan(
    value: float,
    type: str = "i32",
    align: int = 4,
    limit: int = 500,
    include_images: bool = False,
) -> str:
    """首次扫描: 在游戏进程可读内存里找等于 ``value`` 的地址 (类似 Cheat Engine 首扫)。

    典型流程 (标定自机 X 坐标):
      1. 进入游戏, 记住自机当前 x (场地坐标, 用 th_visualize_state 或估计)。
      2. 保持不动, ``th_memory_scan(值, type="f32")`` —— f32 扫描需要知道精确坐标, 较难;
         更稳的做法是扫描"整数型"的量先 (分数/擦弹), 再回到坐标。

    Args:
        value: 要查找的数值。
        type: i8/u8/i16/u16/i32/u32/f32/f64。
        align: 只保留地址能被该值整除的候选 (4 最常用)。
        limit: 最多返回多少候选。
        include_images: 是否扫描模块映像区 (游戏全局数据常在这里, 但更慢)。
    """
    scanner, backend = _memory_scanner()
    candidates = scanner.scan(
        value, type_name=type, align=int(align), limit=int(limit), include_images=include_images
    )
    return _json(
        {
            "pid": backend.pid,
            "count": len(candidates),
            "candidates": [c.to_dict() for c in candidates[:limit]],
            "note": "让游戏里的数值变化, 再用 th_memory_filter(新值) 缩小范围。",
        }
    )


@tool
def th_memory_filter(value: float, type: str = "i32", limit: int = 500) -> str:
    """再次扫描: 在候选地址中只保留当前值等于 ``value`` 的 (类似 CE 的"再次扫描")。

    Args:
        value: 目标当前值。
        type: 与首次扫描一致的类型。
        limit: 最多返回多少候选。
    """
    scanner, _backend = _memory_scanner()
    candidates = scanner.load()
    if not candidates:
        raise TouhouError("没有候选列表, 请先调用 th_memory_scan 做首次扫描。")
    kept = scanner.filter(candidates, value, type_name=type)
    return _json({"count": len(kept), "candidates": [c.to_dict() for c in kept[:limit]]})


@tool
def th_memory_filter_changed(type: str = "i32", expect: str = "changed", limit: int = 500) -> str:
    """在候选地址里筛出"值发生了变化"或"值没变"的地址。

    Args:
        type: 类型。
        expect: "changed" (变化了) | "unchanged" (没变) | "any"。
        limit: 最多返回多少候选。
    """
    scanner, _backend = _memory_scanner()
    candidates = scanner.load()
    if not candidates:
        raise TouhouError("没有候选列表, 请先调用 th_memory_scan。")
    kept = scanner.filter_changed(candidates, type_name=type, expect=expect)
    return _json({"count": len(kept), "candidates": [c.to_dict() for c in kept[:limit]]})


@tool
def th_memory_read(address: str, type: str = "i32", offsets: list[str] | None = None) -> str:
    """读取指定地址的值; 也可以按指针链解引用。

    Args:
        address: 支持三种写法: "0x1234ABCD" / "th11c.exe+0xA56E4" / 十进制整数。
        type: i8/u8/i16/u16/i32/u32/f32/f64。
        offsets: 指针链偏移 (十六进制字符串), 例如 ["0x18", "0x24"] 表示
            ``[[address]+0x18]+0x24``。最后一项只相加不解引用。
    """
    env = get_env()
    scanner, backend = _memory_scanner()
    mem = backend.mem  # type: ignore[attr-defined]
    assert mem is not None
    if isinstance(address, str) and "+" in address:
        module, _, offset = address.partition("+")
        base = mem.module_base(module.strip())
        resolved = base + int(offset.strip(), 0)
    else:
        resolved = int(str(address), 0)

    payload: dict[str, Any] = {"address": hex(resolved), "type": type}
    try:
        name, base, size = mem.main_module
        if base <= resolved < base + size:
            payload["as_offset"] = f"{name}+0x{resolved - base:X}"
    except Exception:
        pass

    if offsets:
        parsed = [int(str(o), 0) for o in offsets]
        # 与 offsets json 的指针链语义一致 (CE 记法 [[address]+o0]+o1):
        #   先解引用 address 本身, 再逐级解引用 offsets[:-1], 最后一项只相加。
        chain = int(mem.read_value(resolved, "u32"))
        steps = [hex(chain)]
        for off in parsed[:-1]:
            chain = int(mem.read_value(chain + off, "u32"))
            steps.append(hex(chain))
        resolved = chain + parsed[-1]
        payload["chain"] = steps
        payload["resolved"] = hex(resolved)
    payload["value"] = mem.read_value(resolved, type)
    return _json(payload)


@tool
def th_memory_candidates(limit: int = 200) -> str:
    """查看当前保存的候选地址列表 (首次/再次扫描的结果)。"""
    scanner, _backend = _memory_scanner()
    candidates = scanner.load()
    return _json({"count": len(candidates), "candidates": [c.to_dict() for c in candidates[:limit]]})


@tool
def th_memory_clear_candidates() -> str:
    """清空候选地址列表。"""
    scanner, _backend = _memory_scanner()
    scanner.clear()
    return _json({"ok": True})


def _memory_backend() -> tuple[TouhouEnv, Any]:
    env = get_env()
    if env.backend_name != "memory":
        raise TouhouError(
            "内存标定只对真实游戏后端有效。先用 th_attach(backend='memory', game='th11') 连接游戏。"
        )
    return env, env.backend


def _press_factory(env: TouhouEnv):
    """把 (按键组合, 帧数) 翻译成对后端的动作 —— 供自动标定驱动游戏用。"""

    def press(buttons: list[str], frames: int) -> None:
        spec = "+".join(buttons) if buttons else "noop"
        env.backend.apply(parse(spec, default_frames=int(frames)))

    return press


def _field_entry(candidate: Any, *, type_name: str, delta: int = 0) -> dict[str, Any]:
    module = getattr(candidate, "module", None)
    module_offset = getattr(candidate, "module_offset", None)
    if module and module_offset is not None:
        return {"offset": f"0x{module_offset + delta:X}", "type": type_name}
    return {"address": f"0x{candidate.address + delta:X}", "type": type_name}


@tool
def th_memory_find_player(
    settle_frames: int = 150,
    hold_frames: int = 50,
    move_frames: int = 90,
    min_value: float = 1.0,
    max_value: float = 448.0,
    write: bool = False,
    game: str = "",
) -> str:
    """全自动定位**自机坐标** (撞墙法): 返回 player_x 候选, player_y 通常是其 +4。

    比手动 scan 可靠得多 —— 让自机分别撞左右墙, 取"墙边不变、移动期会变"的地址。

    Args:
        settle_frames: 撞墙需要按住的帧数 (卡住就调大)。
        hold_frames: 撞墙后再按住多少帧测稳定性。
        move_frames: 离开墙移动的帧数 (筛掉静态数据)。
        min_value / max_value: 坐标合理区间 (场地是 384x448)。
        write: 是否直接把结果写进 offsets json 的 player_x / player_y。
        game: 作品 key (默认当前环境)。
    """
    env, backend = _memory_backend()
    scanner = backend.scanner()
    candidates = scanner.find_player_coordinates(
        _press_factory(env),
        settle_frames=int(settle_frames),
        hold_frames=int(hold_frames),
        move_frames=int(move_frames),
        low=float(min_value),
        high=float(max_value),
    )
    payload: dict[str, Any] = {
        "count": len(candidates),
        "candidates": [c.to_dict() for c in candidates[:50]],
        "note": "取第一个候选作 player_x, player_y 一般是它的 +4 (两个相邻 f32)。",
    }
    if candidates:
        best = candidates[0]
        payload["snippet"] = {
            "player_x": _field_entry(best, type_name="f32"),
            "player_y": _field_entry(best, type_name="f32", delta=4),
        }
        if write:
            payload["written"] = apply_calibration(
                game or env.spec.key,
                {
                    "player_x": _field_entry(best, type_name="f32"),
                    "player_y": _field_entry(best, type_name="f32", delta=4),
                },
            )
            _REGISTRY.reload()
    return _json(payload)


@tool
def th_memory_find_frame(
    frames: int = 60,
    repeats: int = 2,
    tolerance: float = 0.25,
    write: bool = False,
    game: str = "",
) -> str:
    """全自动定位**帧计数器** frame (每帧 +1 的整数) —— 用于严格帧同步。

    Args:
        frames: 每次推进的帧数。
        repeats: 重复取交集的次数 (越多越干净)。
        tolerance: 帧数误差容限 (墙钟推进会有抖动)。
        write: 是否写进 offsets json 的 frame 字段。
        game: 作品 key (默认当前环境)。
    """
    env, backend = _memory_backend()
    scanner = backend.scanner()
    candidates = scanner.find_frame_counter(
        _press_factory(env),
        frames=int(frames),
        tolerance=float(tolerance),
        repeats=int(repeats),
    )
    payload: dict[str, Any] = {
        "count": len(candidates),
        "candidates": [c.to_dict() for c in candidates[:50]],
    }
    if candidates:
        payload["snippet"] = {"frame": _field_entry(candidates[0], type_name="i32")}
        if write:
            payload["written"] = apply_calibration(
                game or env.spec.key,
                {"frame": _field_entry(candidates[0], type_name="i32")},
            )
            _REGISTRY.reload()
    return _json(payload)


@tool
def th_offsets_write(fields: dict[str, Any], game: str = "", backup: bool = True) -> str:
    """把标定结果写进 offsets json (手动标定后的收尾; 下次 attach 生效)。

    Args:
        fields: ``{"graze": {"offset": "0xA571C", "type": "i32"}}``; 也接受
            简写 ``{"graze": "0xA571C"}`` (默认 i32)。
        game: 作品 key (默认当前环境)。
        backup: 覆盖前是否留一份 .json.bak。
    """
    env = get_env()
    result = apply_calibration(game or env.spec.key, dict(fields or {}), backup=bool(backup))
    _REGISTRY.reload()
    return _json({"ok": True, **result})


@tool
def th_set_game_speed(factor: float = 1.0) -> str:
    """设置真实游戏的**时间缩放** (真机训练提速 / 减速观察)。

    factor > 1 加速 —— 真机只有 60fps, 加速能直接提高训练吞吐;
    factor < 1 减速; = 1 恢复原速。需要 memory 后端且已构建 native/speedhack 钩子。

    Args:
        factor: 速度倍数 (0.01~100, 建议 2~6)。
    """
    env = get_env()
    if env.backend_name != "memory":
        raise TouhouError("时间缩放只对真实游戏后端有效 (mock 后端无需缩放)。")
    if getattr(env.backend, "pid", None) is None:
        env.start()
    from .memory.slowdown import SpeedHook

    hook = SpeedHook()
    if not hook.available():
        raise TouhouError(
            "时间缩放钩子尚未构建: "
            + ", ".join(hook.missing())
            + " (运行 native/speedhack/build.ps1)"
        )
    result = hook.apply(env.backend.pid, float(factor))
    env.configure({"slowdown": {"enabled": float(factor) != 1.0, "factor": float(factor)}})
    return _json({"ok": True, **result})


# ================================================================ 训练 (agentic RL 控制面)
# 分工: 本地智能体 (MLPPolicy) 在后台线程以最高速度打游戏, 不经过 MCP;
# Agent (LLM) 只负责 启动/停止/观测/调参, 用奖励与超参"当教练"来加速训练。
@tool
def th_train_start(
    config: dict[str, Any] | None = None,
    load: str = "",
    reset_history: bool = False,
) -> str:
    """启动/继续本地智能体训练 (后台线程, 不占用 MCP 往返)。

    训练循环直接在进程内驱动同一个 env, 所以速度远高于让 LLM 一步一调用 MCP。
    Agent 的角色是"教练": 用 th_train_status / th_train_dashboard 观察学习曲线,
    用 th_train_configure 调超参、th_configure 调奖励, 用本工具重启训练。

    Args:
        config: 训练超参补丁, 例如
            ``{"lr": 1e-3, "batch_episodes": 16, "entropy_coef": 0.02}``
            (键见 th_train_status 返回的 config)。
        load: 载入已保存策略的路径 (可选, 形如 .runtime/training/policy)。
        reset_history: 是否清空历史曲线后重新开始。
    """
    trainer = get_trainer()
    loaded = None
    if config:
        trainer.reconfigure(config)
    if load:
        loaded = trainer.load(load)
    if reset_history:
        trainer.clear_history()
    status = trainer.start()
    return _json({"ok": True, "loaded": loaded, "status": status})


@tool
def th_train_stop(save: bool = True) -> str:
    """停止训练 (默认保存策略), 释放后台线程。

    Args:
        save: 是否把当前策略落盘到 out_dir/policy.npz。
    """
    if _TRAINER is None:
        return _json({"ok": True, "note": "训练本来就没启动"})
    status = _TRAINER.stop(save=save)
    return _json({"ok": True, "status": status})


@tool
def th_train_pause(paused: bool = True) -> str:
    """暂停/恢复训练 (不释放线程, 便于 Agent 观察后再继续)。

    Args:
        paused: True 暂停, False 恢复。
    """
    trainer = get_trainer()
    if paused:
        trainer.pause()
    else:
        trainer.resume()
    return _json({"ok": True, "running": trainer.running, "paused": trainer.paused})


@tool
def th_train_status(tail: int = 5) -> str:
    """查看训练进度: 运行状态、学习曲线摘要、最近若干局、策略结构。

    Agent 依据这个决定下一步怎么调参 (回报不涨就加熵/降 lr; 活不久就调奖励)。

    Args:
        tail: 返回最近多少局的明细。
    """
    if _TRAINER is None:
        return _json({"running": False, "note": "还没有训练器; 先用 th_train_start 启动"})
    return _json(_TRAINER.status(tail=tail))


@tool
def th_train_dashboard(save: bool = True, quiet: bool = False) -> list[Any]:
    """训练仪表盘 (图片): 当前游戏画面 + 学习曲线 + 实时统计。

    比一长串数字更适合让 LLM 一眼看出"训练在不在进步"。

    Args:
        save: 是否把 PNG 存到训练输出目录。
        quiet: 只返回文字说明 (不返回图片)。
    """
    trainer = get_trainer()
    image = trainer.dashboard_image()
    image = fit_width(image, int(get_env().config.get("capture.max_width", 960)))
    path = None
    if save:
        from .training.visualize import save_png as save_train_png

        path = save_train_png(image, trainer.out_dir, name="dashboard_mcp.png")
    text = f"训练仪表盘: {image.width}x{image.height}"
    if quiet:
        return [text]
    return _image_result(viz_png_bytes(image), text, str(path) if path else None)


@tool
def th_train_configure(patch: dict[str, Any], save: bool = False) -> str:
    """在线修改训练超参 —— "教练"调参入口, 下一个 episode 即生效。

    可同时改奖励: 在 patch 里放一个 ``reward`` 子对象即等价于 th_configure。
    例如 ``{"lr": 1e-3, "reward": {"danger_scale": 1.5, "shaping_scale": 1.0}}``

    Args:
        patch: 训练超参 (lr/gamma/entropy_coef/hidden/batch_episodes/step_frames/
            max_steps/difficulty/k_bullets/use_focus/episodes ...), 可含 ``reward``。
        save: 是否把奖励配置写回 config.json (训练超参不落盘)。
    """
    trainer = get_trainer()
    patch = dict(patch or {})
    reward_patch = patch.pop("reward", None)
    out: dict[str, Any] = {"ok": True}
    if patch:
        out.update(trainer.reconfigure(patch))
    if reward_patch:
        out["reward"] = get_env().configure({"reward": reward_patch})["reward"]
    if save:
        get_env().config.save()
        out["saved_to"] = str(default_config_path())
    return _json(out)


@tool
def th_train_policy(action: str = "save", path: str = "") -> str:
    """策略的保存 / 载入 / 查看。

    Args:
        action: "save" | "load" | "describe"。
        path: 策略文件路径 (省略则用 out_dir/policy)。
    """
    trainer = get_trainer()
    action = action.strip().lower()
    if action == "save":
        saved = trainer.save(path or None)
        return _json({"ok": True, "action": "save", "path": str(saved)})
    if action == "load":
        return _json({"ok": True, "action": "load", "policy": trainer.load(path or None)})
    if action in {"describe", "info"}:
        return _json(
            {
                "ok": True,
                "action": "describe",
                "policy": trainer.policy.describe(),
                "path": str(trainer.policy_path()),
            }
        )
    raise TouhouError(f"未知的 action {action!r}; 可用: save / load / describe")


def run(transport: str = "stdio") -> None:
    """启动 MCP 服务器."""
    mcp.run(transport)  # type: ignore[arg-type]


if __name__ == "__main__":  # pragma: no cover
    run(os.environ.get("TOUHOU_MCP_TRANSPORT", "stdio"))
