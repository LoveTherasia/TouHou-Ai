"""配置加载 / 保存.

配置优先级:
    1. 环境变量 ``TOUHOU_MCP_CONFIG`` 指向的 json 文件
    2. 项目根目录下的 ``config.json``
    3. 内置默认值 :data:`DEFAULT_CONFIG`

配置里的 ``reward`` 段是给 agentic RL 调参用的: Agent 可以通过 MCP 的
``configure`` 工具在线修改奖励权重, 从而"调整参数"式地训练。
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

from .errors import ConfigError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.json"
RUNTIME_DIR = PROJECT_ROOT / ".runtime"

#: 内置默认配置 (会与用户配置做深合并)
DEFAULT_CONFIG: dict[str, Any] = {
    # ---------------- 后端 ----------------
    # "mock"   内置模拟器, 不需要游戏
    # "memory" 真实游戏: 读内存 + 注入按键
    "backend": "mock",
    # 默认游戏 key, 见 touhou_rl_mcp/games/offsets/*.json
    "game": "th11",
    # 游戏安装目录 (留空则自动在 search_roots 中查找)
    "game_dir": "",
    # 每次 start_new_game 是否先杀掉再重启进程
    "restart_process": False,
    "dialogue_assist": True,
    # 对话快进: 东方是边沿触发, 需"连续点按 Z"; 每次按住 dialogue_tap_hold 帧,
    # 松开 dialogue_tap_after 帧, 一次 step 最多点 dialogue_max_taps 次。
    "dialogue_tap_hold": 2,
    "dialogue_tap_after": 3,
    "dialogue_max_taps": 30,
    # 连续多少帧"空场"判定进入对话/静默期 (越小越快触发快进; 默认 12)
    "dialogue_threshold_frames": 12,
    # 空场容忍的敌人上限: TH11 对话时剧本会留 1~2 个不攻击的演员,
    # 真机实测关底对话窗口内 ``enemies_alive`` 是 1~2 而不是 0。
    "dialogue_max_enemies": 2,
    # ---------------- 环境语义 ----------------
    # 一次 step 默认推进多少帧 (60fps)
    "step_frames": 4,
    # 一次 step 最多允许多少帧
    "max_step_frames": 600,
    # 单局最大帧数, 超过则 truncated=True
    "max_episode_frames": 60 * 60 * 12,  # 12 分钟
    # 一局结束后再 step 时是否自动开新局 (RL 循环里方便)
    "auto_reset": False,
    # 只在"生命耗尽"时结束一局; in_game 启发式在换面/对话时会假性为 False,
    # 绝不能据此结束 (那会导致训练中途无故 reset)。0 = 关闭 left_game 判定。
    "left_game_patience": 0,
    # 判定 game over 的生命阈值: lives <= 该值即结束 (th11 实测生命耗尽时 lives=-1;
    # 0 表示"还剩最后一条命", 仍可继续)。
    "game_over_lives": -1,
    # 达到该关卡且通关则视为 clear
    "final_stage": 6,
    # 课程学习: mock 从第几面开局 (1 = 从第一面开始; 用于逐面推进)
    "start_stage": 1,
    # 真机弹幕脚本 (可选): mock 用它代替内置弹型, 复刻真实东方弹幕
    #   由 tools/record_danmaku.py 录制; 例 ".runtime/danmaku/th11_stage1.json"
    "danmaku_script": "",
    # 弹幕脚本起始帧偏移 (从回放的第几帧开始; 用来直接练某一面, 如第 2 面)
    "danmaku_offset": 0,
    # 弹幕类型库 (backends/patterns.py): 按弹型分别训练, 例 "dense_ring" / "aimed_trail"
    #   可选: dense_ring / expanding_homing_ring / aimed_trail / static_swarm /
    #         spiral / fan / wall_gap / random_spray / flower / aimed_spread
    "danmaku_pattern": "",
    # 紧急放弹保护 (规则, 不是策略动作): 只在最近弹幕极近时放弹, 避免乱放。
    "bomb_guard": {
        "enabled": True,
        "radius": 14.0,          # 最近弹幕距离(px)小于它才放
        "min_power": 1.0,        # 灵力不足不放 (th11 放弹消耗 1.0)
        "cooldown_frames": 45,   # 两次规则放弹的最小间隔
        "max_per_episode": 6,    # 每局上限
    },
    # 打开游戏后等待多久 (秒) 认为已进入标题画面
    "launch_wait_seconds": 6.0,
    # ---------------- 奖励 ----------------
    # 唯一来源: config.json 的 reward 段直接覆盖这里 (工具/训练脚本不要再自带一套)
    "reward": {
        "score_scale": 3e-2,  # Δscore * scale —— 攻击意图的主信号 (见 rl/reward.py "attack")
        "graze_scale": 0.02,  # Δgraze * scale
        "power_scale": 0.1,  # Δpower * scale
        "life_bonus": 5.0,  # 每多一残机
        "miss_penalty": -200.0,  # 掉残: 大幅加重 -> 把"不被弹"放在第一位
        "death_penalty": -600.0,  # Game Over
        "stage_clear_bonus": 1500.0,  # 过一关
        "game_clear_bonus": 6000.0,  # 通关
        "spell_capture_bonus": 20.0,  # 确认获得符卡 Bonus 的额外奖励
        "spell_clear_bonus": 150.0,  # 击破一张符卡 (mock spell_cleared)
        "boss_damage_scale": 0.1,  # -Δboss_hp * scale (需要 boss_hp)
        "invincible_output_scale": 0.0,  # 无敌期 分数/伤害/存活 缩放 (0=封堵被弹刷输出)
        "invincible_frames": 150,  # 回退推断无敌窗口 (真机未标定 invincible 时)
        # 危险度只做"微调": 实测 4.0/50px 时一局扣 -3306, 比其它所有项加起来还大 5 倍,
        # 于是策略唯一的学习信号变成"离弹幕越远越好"(密集弹幕里 = 角落), 不再攻击。
        "danger_scale": 0.8,  # 危险度惩罚 (需要 player_x/y + 弹幕)
        "danger_radius": 32.0,  # 危险度感知半径(像素)
        # 缺口余量的势函数塑形 (φ' − φ, 不改变最优策略); 0 = 关闭。
        # 它是 WEIGHT_DOCS 里的一项, 必须在这里也有默认值, 否则 patch 校验会拒绝它。
        "gap_margin_scale": 0.0,
        "alive_bonus": 0.3,  # 每 step 存活奖励
        "time_penalty": -0.01,  # 每 step 的固定惩罚, 鼓励速通
        "clip": 8000.0,  # 单步奖励裁剪 (None 表示不裁剪)
        "shaping_scale": 1.0,  # 整体缩放, 方便 Agent 在线调节
    },
    # ---------------- 控制 ----------------
    "control": {
        "tick_hz": 60,  # 控制线程频率, 应等于游戏帧率
        "focus_window": True,  # 发按键前把游戏窗口置前
        "key_hold_min_frames": 1,
        # 虚拟按键 -> VK / 扫描码, 一般不用改
        "keymap": {},
    },
    # ---------------- 截图 / 可视化 ----------------
    "capture": {
        "out_dir": "",  # 留空 -> <项目>/.runtime/shots
        "max_width": 960,  # 返回前等比缩放到的最大宽度
        "jpeg_quality": 90,
        "prefer_printwindow": True,  # 先试 PrintWindow, 全黑再回退到屏幕截取
    },
    # 状态可视化画布尺寸
    "visualize": {
        "scale": 1.0,  # 画布缩放, 1.0 => 384x448 的游戏场地
        "show_bullets": True,
        "show_grid": True,
        "show_hud": True,
        "trail_length": 180,  # 玩家轨迹保留帧数
    },
    # ---------------- 视觉兜底 (capture/bullets.py) ----------------
    # 没有弹幕偏移的作品 (th06~th14 中除 th11 之外) 可以打开它: 内存读不到弹幕时,
    # 从**截图**里检测小亮块当作弹幕, 坐标换算成 384x448 场地坐标。
    # 代价是每次 read_state 多一次截图 (毫秒级), 所以默认关闭; 它拿不到速度与半径。
    "vision": {
        "enabled": False,
        "bright_thr": 170,  # 亮度阈值 (0~255)
        "sat_thr": 40,      # 饱和度阈值, 用来排除灰白背景/文字
        "min_area": 2,      # 连通域最小面积(像素)
        "max_area": 300,    # 连通域最大面积, 用来排除背景花纹/敌机
        "max_dim": 40,      # 连通域最大边长
    },
    # ---------------- 自动查找游戏 ----------------
    "search_roots": ["F:\\", "E:\\", "D:\\", "C:\\"],
    "search_depth": 2,
    # ---------------- 内存扫描 ----------------
    "scanner": {
        "max_region_mb": 64,
        "align": 4,
        "max_candidates": 20000,
    },
    # ---------------- 真机时间缩放 (可选, 需先构建 native/speedhack) ----------------
    # factor > 1 加速(提高真机训练吞吐), < 1 减速(便于观察); 1.0 = 不注入
    "slowdown": {
        "enabled": False,
        "factor": 1.0,
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


#: 允许任意键名的配置子树 (键名由使用方自己解释)。目前只有按键映射:
#: ``control.keymap`` 的键是虚拟动作名, 无法穷举校验。
FREEFORM_PATHS = frozenset({"control.keymap"})


def _unknown_paths(patch: Any, schema: Any, prefix: str = "") -> list[str]:
    """递归找出 ``patch`` 里 ``schema`` 不认识的键, 返回点分路径列表."""
    if not isinstance(patch, dict) or not isinstance(schema, dict):
        return []
    found: list[str] = []
    for key, value in patch.items():
        path = f"{prefix}{key}"
        if key not in schema:
            found.append(path)
            continue
        if path in FREEFORM_PATHS:
            continue
        sub = schema[key]
        if isinstance(sub, dict) and isinstance(value, dict):
            found.extend(_unknown_paths(value, sub, f"{path}."))
    return found


def _check_shapes(patch: Any, schema: Any, prefix: str = "") -> None:
    """结构校验: 配置段必须是对象 (不能让 ``patch({"reward": null})`` 把整段抹掉)."""
    if not isinstance(patch, dict) or not isinstance(schema, dict):
        return
    for key, value in patch.items():
        if key not in schema:
            continue
        path = f"{prefix}{key}"
        expected = schema[key]
        if not isinstance(expected, dict) or path in FREEFORM_PATHS:
            continue
        if not isinstance(value, dict):
            raise ConfigError(
                f"配置项 {path} 需要一个 JSON 对象, 收到 {type(value).__name__}"
            )
        _check_shapes(value, expected, f"{path}.")


class Config:
    """带落盘能力的配置对象."""

    def __init__(self, data: dict[str, Any] | None = None, path: Path | None = None) -> None:
        self.path = Path(path) if path else default_config_path()
        self._data = _deep_merge(DEFAULT_CONFIG, data or {})

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        p = Path(path) if path else default_config_path()
        if p.exists():
            try:
                raw = json.loads(p.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise ConfigError(f"配置文件 {p} 不是合法 JSON: {exc}") from exc
            if not isinstance(raw, dict):
                raise ConfigError(f"配置文件 {p} 顶层必须是对象")
        else:
            raw = {}
        return cls(raw, path=p)

    # ------------------------------------------------------------------
    def as_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)

    def get(self, dotted: str, default: Any = None) -> Any:
        """``config.get("reward.score_scale")``"""
        node: Any = self._data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def section(self, name: str) -> dict[str, Any]:
        value = self._data.get(name, {})
        return value if isinstance(value, dict) else {}

    # ------------------------------------------------------------------
    def patch(self, changes: dict[str, Any]) -> dict[str, Any]:
        """深合并式更新, 返回更新后的完整配置.

        校验是**递归**的: 未知的顶层项、未知的嵌套项、以及"段被写成非对象"
        都会被拒绝。否则 ``{"reward": {"miss_penelty": -5}}`` 会先写进内存
        (``save=True`` 时还会落盘), 直到下一帧 ``RewardModel.update_weights``
        才抛错 —— 那时坏键已经存下来了。
        """
        if not isinstance(changes, dict):
            raise ConfigError("patch 需要一个 JSON 对象")
        unknown = _unknown_paths(changes, DEFAULT_CONFIG)
        if unknown:
            raise ConfigError(f"未知的配置项: {', '.join(sorted(unknown))}")
        _check_shapes(changes, DEFAULT_CONFIG)
        self._data = _deep_merge(self._data, changes)
        return self.as_dict()

    def save(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return self.path


def default_config_path() -> Path:
    env = os.environ.get("TOUHOU_MCP_CONFIG")
    if env:
        return Path(env).expanduser()
    return DEFAULT_CONFIG_PATH


def runtime_dir() -> Path:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    return RUNTIME_DIR
