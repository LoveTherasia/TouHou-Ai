"""训练器: 在进程内以最高速度跑 RL 循环。

架构定位 (这是整个 agentic RL 的关键):

    LLM Agent ──MCP──▶ Trainer            外层: 调奖励权重、看曲线、决定重启
                         │ (进程内, 不经过 MCP)
                         ▼
                     TouhouEnv ──▶ 真实游戏 / 模拟器   内层: 60fps 高频闭环

**训练循环绝不能走 MCP 工具调用** —— 每次 JSON-RPC 往返是毫秒级开销, 而 RL 需要
每秒上千步。所以训练器在 Python 进程内直接驱动 TouhouEnv, MCP 只负责
"启动/停止/观测/调参"。

奖励来自 env.step() 的返回值, 而奖励权重随时可以用 MCP 的 th_configure 改 ——
**下一个 episode 就生效**, 这就是 "Agent 调奖励" 的闭环。"""

from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..config import runtime_dir
from ..types import GameState
from .features import encode_observation, feature_dim, gap_spec_from, observation_dim
from .gap_label import update_action_age
from .journal import Journal
from .policy import MLPPolicy

#: 9 个方向 (空字符串 = 原地不动)
DIRECTIONS: list[str] = [
    "",
    "up",
    "down",
    "left",
    "right",
    "up_left",
    "up_right",
    "down_left",
    "down_right",
]


def build_action_table(*, use_focus: bool = False, use_bomb: bool = False) -> list[str]:
    """构造动作表. 东方打法里"一直按着射击"是常态, 所以默认始终 shoot。
    use_bomb=True 时追加 9 个放炸弹动作 (方向+shoot+bomb), 这样策略**才可能**
    学会主动放弹。注意: 炸弹是按**边沿**触发; 连续两步都选同一 bomb 动作时, 按键
    一直被按住不会再次触发 —— 需要多次放弹时策略得穿插非 bomb 动作。
    """
    table: list[str] = []
    for direction in DIRECTIONS:
        parts = [direction] if direction else []
        parts.append("shoot")
        table.append("+".join(parts))
    if use_focus:
        for direction in DIRECTIONS:
            parts = [direction] if direction else []
            parts.extend(["shoot", "focus"])
            table.append("+".join(parts))
    if use_bomb:
        for direction in DIRECTIONS:
            parts = [direction] if direction else []
            parts.extend(["shoot", "bomb"])
            table.append("+".join(parts))
        if use_focus:
            for direction in DIRECTIONS:
                parts = [direction] if direction else []
                parts.extend(["shoot", "focus", "bomb"])
                table.append("+".join(parts))
    return table


def apply_scenario(env: Any, scenario: dict[str, float] | None) -> None:
    """把基准场景参数写进 mock 的弹型对象 (例如 ``wall_gate`` 的缺口宽度/漂移)。

    为什么需要它: ``MockBackend.reset`` 每次都会**重建** ``MockGame``, 于是弹型对象
    也是新的 —— 训练器如果在外面 setattr 一次, 第二局就失效了。所以场景必须跟着
    ``TrainConfig`` 走, 在每次 reset 之后重新应用。

    值按**属性原本的类型**强制转换 (55 -> 95.0 会变成 95), 免得 ``frame % interval``
    拿到浮点数。
    """
    if not scenario:
        return
    backend = getattr(env, "backend", None)
    pattern = getattr(getattr(backend, "game", None), "pattern", None)
    if pattern is None:
        raise ValueError("scenario 需要 danmaku_pattern 弹型 (当前环境没有)")
    for key, value in scenario.items():
        if not hasattr(pattern, key):
            raise KeyError(f"弹型 {type(pattern).__name__} 没有参数 {key!r}")
        current = getattr(pattern, key)
        setattr(pattern, key, type(current)(value))


@dataclass
class TrainConfig:
    """训练超参 —— 全部可以通过 MCP 在线修改。"""

    algorithm: str = "ppo"  # "ppo" | "reinforce_baseline"
    episodes: int = 300  # 0 = 无限局
    max_steps: int = 1500  # 每个 episode 最多多少 step
    step_frames: int = 4  # 每个动作持续多少帧
    #: 累积多少局再更新一次参数。REINFORCE 单局梯度方差极大, 攒几局一起更新
    #: 是让它真正能学到东西的关键 (batch=1 时基本学不动)。
    batch_episodes: int = 8
    lr: float = 3e-3
    gamma: float = 0.99
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    hidden: tuple[int, ...] = (64, 64)
    k_bullets: int = 16
    #: 弹幕特征是否包含速度 (每颗弹 +vx,+vy; 观测 74 -> 106)。真机才有效。
    bullet_velocity: bool = False
    #: 敌机特征个数 (0=只用敌机数量; 8=追加最近 8 架敌机的相对位置)
    k_enemies: int = 0
    use_focus: bool = True
    #: 动作空间是否包含"放炸弹" (动作数 9 -> 18); 默认关 (旧模型是 9 动作)
    use_bomb: bool = False
    #: 观测是否包含无敌帧 (让策略知道"现在是否无敌"); 打开会改变 obs_dim
    use_invincible: bool = True
    #: 观测是否包含 Boss 相对位置 (会移动的 Boss 必须给位置)
    use_boss: bool = True
    #: 观测是否包含"危险梯度" (最近弹幕斥力合成) —— 让反应式躲避更易被策略表示
    use_danger_grad: bool = False
    #: 观测改用"危险图" (以自机为中心) 而不是"最近 K 颗弹的一维向量"
    use_grid: bool = False
    #: 观测改用"未来占用栅格 + 几何缺口" (见 training/occupancy.py)。
    #: 这是密集弹幕下唯一能让策略提前横向占位的表示; 需要弹速 (mock 直接提供)。
    use_gap: bool = False
    #: use_gap 时拼接哪几块 (时间分层栅格 / TTA 剖面 / 8 方向安全余量)
    gap_grid: bool = True
    gap_geo: bool = True
    gap_clearance: bool = True
    #: use_gap 时把上一动作 one-hot 拼进观测 (宏动作有承诺期, 策略要知道自己的动量)
    gap_prev_action: bool = True
    #: TTA 剖面层数。2 = 再拼一层"下一堵墙什么时候到", 大漂移基准上必须开
    #: (第 1 层被眼前这堵墙遮死, 看不到下一堵墙的缺口在哪)
    gap_layers: int = 1
    #: 缺口老师的标签口径: now / reach / future / slack /
    #: reach_hold / reach_hyst (后两个有状态, 见 training/gap_label.py)
    gap_label: str = "reach"
    #: 有状态口径的承诺时长 (帧); 0 = 关
    gap_commit_frames: int = 0
    #: 有状态口径的滞回阈值 (帧): 新目标必须好这么多才切换; 0 = 关
    gap_hysteresis: float = 0.0
    #: 观测里加入"连续同向决策步数" (1 维), 让策略能打破等分缺口的平局
    gap_action_age: bool = False
    gap_action_age_cap: int = 12
    #: 基准场景参数 (写进 mock 的弹型对象), 例如 wall_gate 的
    #: ``{"interval": 95, "gap_width": 46, "speed": 2.05, "gap_drift": 0.34}``
    scenario: dict[str, float] = field(default_factory=dict)
    #: 每局在这些区间里**重新采样**的场景参数 (domain randomization)。
    #: 例如 ``{"gap_phase": [0.0, 1.0]}`` —— 不加它, wall_gate 每局的缺口相位都一样,
    #: 训练集退化成"同一个局面重复 N 次", 策略只会背下那一个相位。
    scenario_random: dict[str, list[float]] = field(default_factory=dict)
    #: 真机弹幕脚本的起始位置区间 (占脚本长度的比例), 例如 ``[0.0, 1.0]``。
    #: 脚本会循环, 所以这是脚本模式下训练多样性的**唯一**来源; 空 = 固定在 0。
    script_offset_range: list[float] = field(default_factory=list)
    #: 危险图边长 (格数); 观测维度 = grid_size^2 * (1 或 3) + 全局特征
    grid_size: int = 16
    #: 危险图半径 (像素): 覆盖自机周围 ±grid_span
    grid_span: float = 110.0
    #: 多尺度危险图: "size:span,size:span" (如 "12:90,12:330" = 近场细 + 远场粗)。
    #: 非空时忽略 grid_size/grid_span, 各尺度依次拼接。
    grid_scales: str = ""
    #: 课程学习: mock 从第几面开局 (1=第一面; 用于逐面推进)
    start_stage: int = 1
    #: PPO 超参 (algorithm="ppo" 时生效)
    ppo_clip: float = 0.2
    ppo_epochs: int = 4
    ppo_minibatch: int = 256
    gae_lambda: float = 0.95
    #: PPO KL 早停阈值 (0=关闭): 单批 KL 超过它就停止本批剩余 epoch, 防止策略退化
    target_kl: float = 0.05
    #: RL 阶段到"监督结束时那份参数"的 KL 锚系数 (0=关闭)。
    #: 不开它时, PPO 会把 DAgger 学到的缺口感知冲掉 (实测 1764 -> 1592);
    #: 0.5 ~ 2 之间既能跟奖励走, 又不会忘。
    kl_anchor: float = 0.0
    #: 模仿学习每个 batch 重复几次梯度 (BC/DAgger 收敛更快)
    teacher_epochs: int = 8
    #: 危险状态加权: 离最近弹幕越近, 模仿损失权重越大 (0=不加权)
    danger_weight: float = 0.0
    #: 加权时的"危险半径"(像素)
    danger_weight_radius: float = 60.0
    #: 每步后 sleep 多少秒 (多线程时给真机线程让出 GIL; mock 线程建议 0.003)
    step_sleep: float = 0.0
    #: 是否逐步写 steps.jsonl 日志 (真机约 6 步/秒, 开销可接受; mock 会很大, 慎用)
    log_steps: bool = False
    #: 老师: ""=纯 REINFORCE; "guard"=用 TacticalGuard 选动作并做行为克隆(模仿学习)
    teacher: str = ""
    #: DAgger 模式下每步用老师动作接管的比例 (0=全策略驱动, 1=等价纯模仿)
    dagger_beta: float = 0.2
    seed: int = 20261017
    eval_every: int = 25
    eval_episodes: int = 3
    save_every: int = 50
    #: 每多少局刷新一次 progress.png (存活步数/面数/回报/掉残; 0 = 关)
    plot_every: int = 10
    #: 每多少局抓一帧画面 (供可视化; 0 = 不抓)
    capture_every: int = 5
    difficulty: int = 1
    character: int = 0
    shot: int = 0
    rotate_loadouts: bool = False
    out_dir: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["hidden"] = list(self.hidden)
        return data

    @classmethod
    def from_dict(cls, patch: dict[str, Any]) -> "TrainConfig":
        known = {f for f in cls.__dataclass_fields__}
        unknown = [k for k in patch if k not in known]
        if unknown:
            raise KeyError(f"未知的训练参数:  {', '.join(sorted(unknown))}")
        data = dict(patch)
        if "hidden" in data:
            value = data["hidden"]
            if isinstance(value, str):
                data["hidden"] = tuple(int(x) for x in value.replace(" ", "").split(",") if x)
            else:
                data["hidden"] = tuple(int(x) for x in value)
        return cls(**data)


@dataclass
class EpisodeRecord:
    index: int
    steps: int
    frames: int
    return_: float
    max_stage: int | None = None
    max_score: int | None = None
    max_power: float | None = None
    total_graze: int | None = None
    misses: int = 0
    bombs: int = 0
    cleared: bool = False
    game_over: bool = False
    truncated: bool = False
    duration_s: float = 0.0
    reward_parts: dict[str, float] = field(default_factory=dict)
    update_stats: dict[str, float] = field(default_factory=dict)
    eval: bool = False
    character: int = 0
    shot: int = 0
    #: 本局各动作被选中的次数 (便于看它爱用什么动作 / 有没有放弹)
    actions: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode": self.index,
            "character": self.character,
            "shot": self.shot,
            "steps": self.steps,
            "frames": self.frames,
            "return": round(self.return_, 3),
            "max_stage": self.max_stage,
            "max_score": self.max_score,
            "max_power": self.max_power,
            "total_graze": self.total_graze,
            "misses": self.misses,
            "bombs": self.bombs,
            "cleared": self.cleared,
            "game_over": self.game_over,
            "truncated": self.truncated,
            "duration_s": round(self.duration_s, 3),
            "steps_per_sec": round(self.steps / self.duration_s, 1) if self.duration_s > 0 else None,
            "reward_parts": {k: round(v, 4) for k, v in self.reward_parts.items()},
            "update_stats": self.update_stats,
            "actions": self.actions,
            "eval": self.eval,
        }


@dataclass
class Trajectory:
    """一局的决策轨迹 (供 REINFORCE / PPO 更新)."""

    observations: list[np.ndarray]
    actions: list[int]
    logps: list[float]
    values: list[float]
    rewards: list[float]
    dones: list[bool]
    #: 未真正终止(截断/手动停止)时用于 bootstrap 的末状态价值
    bootstrap_value: float = 0.0
    #: 每个决策的样本权重 (危险状态加权; 空 = 等权)
    weights: list[float] = field(default_factory=list)


class Trainer:
    """后台线程里跑 RL 训练, 对外暴露状态查询 / 在线调参 / 保存加载."""

    def __init__(
        self,
        env: Any,
        config: TrainConfig | None = None,
        *,
        env_lock: threading.RLock | None = None,
        out_dir: str | Path | None = None,
        danmaku_dir: str | Path | None = None,
        policy: Any = None,
        policy_lock: threading.RLock | None = None,
    ) -> None:
        self.env = env
        self.config = config or TrainConfig()
        #: RL 阶段 KL 锚用的参考参数 (监督阶段结束时冻结)
        self._rl_ref: list[Any] | None = None
        if self.config.rotate_loadouts and getattr(env.spec, "key", None) != "th11":
            raise ValueError("六机体轮换目前仅支持 TH11")
        self.env_lock = env_lock or threading.RLock()
        #: 多线程共享策略时用同一把锁保护 forward/backward (真机线程 + mock 线程)
        self.policy_lock = policy_lock or threading.RLock()
        target_dir = out_dir or self.config.out_dir
        self.out_dir = Path(target_dir) if target_dir else runtime_dir() / "training"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        #: 若设置, 训练过程中同时把真机弹幕录成脚本 (按难度/自机/机体区分命名)
        self.danmaku_dir = Path(danmaku_dir) if danmaku_dir else None

        self.action_table = build_action_table(use_focus=self.config.use_focus,
                                               use_bomb=self.config.use_bomb)
        self._action_counts: dict[str, int] = {}
        self.journal = Journal(self.out_dir, log_steps=self.config.log_steps)
        self.obs_dim = observation_dim(self.config)
        if policy is not None:
            # 多线程共享同一个策略 (真机线程 + mock 线程 一起更新它)
            if policy.obs_dim != self.obs_dim or policy.n_actions != len(self.action_table):
                raise ValueError(
                    f"注入的策略维度不匹配: policy={policy.obs_dim}/{policy.n_actions} "
                    f"期望={self.obs_dim}/{len(self.action_table)}")
            self.policy = policy
        else:
            self.policy = MLPPolicy(
                self.obs_dim,
                len(self.action_table),
                hidden=self.config.hidden,
                seed=self.config.seed,
                lr=self.config.lr,
                entropy_coef=self.config.entropy_coef,
            )

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._pause = threading.Event()
        self._state_lock = threading.RLock()
        self._rng = np.random.default_rng(self.config.seed)

        self.episodes: list[EpisodeRecord] = []
        self.evaluations: list[dict[str, Any]] = []
        self.total_steps = 0
        self.total_frames = 0
        self.best_return = float("-inf")
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self.error: str | None = None
        self.current_episode: dict[str, Any] = {}
        self._last_update: dict[str, float] = {}
        #: 最近一次抓到的画面 (PIL Image), 供监视窗口 / MCP 仪表盘使用
        self.latest_frame: Any = None
        self.latest_frame_episode: int = 0
        self._last_frame_save = 0.0
        self._last_curve_save = 0.0
        self.artifacts: dict[str, str] = {}

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive() and not self._stop.is_set()

    @property
    def paused(self) -> bool:
        return self._pause.is_set()

    def start(self) -> dict[str, Any]:
        if self.running:
            return self.status()
        self._stop.clear()
        self._pause.clear()
        self.error = None
        self.started_at = time.perf_counter()
        self.finished_at = None
        self._thread = threading.Thread(target=self._loop, name="touhou-trainer", daemon=True)
        self._thread.start()
        return self.status()

    def stop(self, *, save: bool = True, timeout: float = 30.0) -> dict[str, Any]:
        self._stop.set()
        self._pause.clear()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
        self._thread = None
        if save:
            try:
                self.save()
            except Exception as exc:  # pragma: no cover
                self.error = f"保存策略失败: {exc}"
        self._refresh_curve(force=True)
        self._refresh_progress(force=True)
        return self.status()

    def pause(self) -> None:
        self._pause.set()

    def resume(self) -> None:
        self._pause.clear()

    # ------------------------------------------------------------------
    # 在线调参
    # ------------------------------------------------------------------
    def reconfigure(self, patch: dict[str, Any]) -> dict[str, Any]:
        """修改训练超参; 需要重建网络的项 (hidden/k_bullets/use_focus) 会重建策略。"""
        new_config = TrainConfig.from_dict({**self.config.to_dict(), **patch})
        rebuild = (
            new_config.hidden != self.config.hidden
            or new_config.k_bullets != self.config.k_bullets
            or new_config.use_focus != self.config.use_focus
            or new_config.use_bomb != self.config.use_bomb
            or new_config.bullet_velocity != self.config.bullet_velocity
            or new_config.k_enemies != self.config.k_enemies
            or new_config.use_invincible != self.config.use_invincible
            or new_config.use_boss != self.config.use_boss
            or new_config.use_danger_grad != self.config.use_danger_grad
            or new_config.use_grid != self.config.use_grid
            or new_config.use_gap != self.config.use_gap
            or new_config.gap_grid != self.config.gap_grid
            or new_config.gap_geo != self.config.gap_geo
            or new_config.gap_clearance != self.config.gap_clearance
            or new_config.gap_prev_action != self.config.gap_prev_action
            or new_config.gap_action_age != self.config.gap_action_age
            or new_config.gap_action_age_cap != self.config.gap_action_age_cap
            or new_config.gap_layers != self.config.gap_layers
            or new_config.grid_size != self.config.grid_size
            or new_config.grid_span != self.config.grid_span
            or new_config.grid_scales != self.config.grid_scales
        )
        self.config = new_config
        # 从"有老师"切到"没老师" = 进入 RL 阶段: 此刻冻结参考策略, 给 PPO 一个锚。
        # 必须在这一刻抓, 抓早了是随机的初始策略, 抓晚了已经被 RL 改过。
        if (patch.get("teacher") is not None and not str(new_config.teacher).strip()
                and self.config.kl_anchor > 0):
            with self.policy_lock:
                self._rl_ref = self.policy.snapshot_params()
        self.policy.lr = new_config.lr
        self.policy.entropy_coef = new_config.entropy_coef
        if rebuild:
            self.action_table = build_action_table(use_focus=new_config.use_focus,
                                                   use_bomb=new_config.use_bomb)
            self.obs_dim = observation_dim(new_config)
            self.policy = MLPPolicy(
                self.obs_dim,
                len(self.action_table),
                hidden=new_config.hidden,
                seed=new_config.seed,
                lr=new_config.lr,
                entropy_coef=new_config.entropy_coef,
            )
        return {"config": new_config.to_dict(), "rebuilt_policy": rebuild}

    # ------------------------------------------------------------------
    # 训练主循环
    # ------------------------------------------------------------------
    def _loop(self) -> None:
        try:
            index = len(self.episodes)
            batch: list[Trajectory] = []
            while not self._stop.is_set():
                if self.config.episodes and index >= self.config.episodes:
                    break
                if self._pause.is_set():
                    time.sleep(0.05)
                    continue
                index += 1
                record, trajectory = self._run_episode(index)
                if trajectory is not None:
                    batch.append(trajectory)
                if len(batch) >= max(1, self.config.batch_episodes):
                    stats = self._update_batch(batch)
                    batch = []
                    record.update_stats = stats
                    self._last_update = stats
                with self._state_lock:
                    self.episodes.append(record)
                    if record.return_ > self.best_return:
                        self.best_return = record.return_
                try:
                    self.journal.episode(record.to_dict())
                except Exception:
                    pass
                self._refresh_progress()
                if self.config.capture_every and index % self.config.capture_every == 0:
                    self._capture_frame(index)
                self._refresh_curve()
                if self.config.eval_every and index % self.config.eval_every == 0:
                    self._evaluate(index)
                if self.config.save_every and index % self.config.save_every == 0:
                    try:
                        self.save()
                    except Exception:
                        pass
            if batch:
                self._update_batch(batch)
        except Exception as exc:  # pragma: no cover - 训练线程不能静默死掉
            import traceback

            self.error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        finally:
            try:
                with self.env_lock:
                    if self.env.backend.controller is not None:
                        self.env.backend.controller.release_all()
            except Exception as exc:
                self.error = (self.error or "") + f"\n释放按键失败: {exc}"
            finally:
                self.finished_at = time.perf_counter()

    def _apply_script_offset(self, config: "TrainConfig") -> None:
        """真机弹幕脚本模式: 每局从脚本的随机位置开始。

        脚本是**可以循环**的 (``frame % length``), 所以"从哪个位置开始"就是全部的训练
        多样性来源。固定 offset 的话, 每局都是同一段弹幕重复 —— 和 wall_gate 固定相位
        是同一个坑。偏移通过后端 config 注入, 因为 ``MockBackend.reset`` 每次都读
        ``danmaku_offset`` 重建 MockGame。
        """
        bounds = getattr(config, "script_offset_range", None) or []
        if len(bounds) < 2:
            return
        backend = getattr(self.env, "backend", None)
        script = getattr(backend, "_script", None) or {}
        length = int(script.get("length") or 0)
        cfg = getattr(backend, "config", None)
        if length <= 0 or cfg is None:
            return
        lo, hi = float(bounds[0]), float(bounds[1])
        cfg.patch({"danmaku_offset": int(self._rng.uniform(lo, hi) * length)})

    def _run_episode(
        self, index: int, *, greedy: bool = False
    ) -> tuple[EpisodeRecord, "Trajectory" | None]:
        config = self.config
        character, shot = (divmod((index-1) % 6, 3) if config.rotate_loadouts
                           else (config.character, config.shot))
        with self.env_lock:
            self._apply_script_offset(config)
            result = self.env.reset(seed=config.seed + index, difficulty=config.difficulty,
                                    character=character, shot=shot,
                                    start_stage=config.start_stage)
            apply_scenario(self.env, config.scenario)
            if config.scenario_random:
                apply_scenario(self.env, {
                    key: float(self._rng.uniform(float(bounds[0]), float(bounds[1])))
                    for key, bounds in config.scenario_random.items() if len(bounds) >= 2
                })
            state: GameState = result.state
            if getattr(self.env, "backend_name", None) == "memory" and (state.char_main, state.char_sub) != (character, shot):
                raise ValueError(f"选机失败: 请求 {(character, shot)}, 实际 {(state.char_main, state.char_sub)}")
            obs = encode_observation(state, config, prev_action=-1, action_age=0)

        observations: list[np.ndarray] = []
        actions: list[int] = []
        logps: list[float] = []
        values: list[float] = []
        rewards: list[float] = []
        dones: list[bool] = []
        weights: list[float] = []
        record = EpisodeRecord(index=index, steps=0, frames=0, return_=0.0, eval=greedy)
        recorder = None
        if self.danmaku_dir is not None and not greedy:
            from ..memory.danmaku import DanmakuRecorder

            recorder = DanmakuRecorder()
        record.character, record.shot = character, shot
        started = time.perf_counter()
        act_counts: dict[str, int] = {}
        teacher = (config.teacher or "").strip().lower()
        guard = None
        if teacher and not greedy:
            if teacher in {"guard", "dagger"}:
                from .tactical import TacticalGuard

                guard = TacticalGuard()
            elif teacher in {"rule", "rule_dagger"}:
                from .rule_teacher import RuleTeacher

                guard = RuleTeacher()
            elif teacher in {"gap", "gap_dagger"}:
                from .gap_teacher import GapTeacher

                guard = GapTeacher(gap_spec_from(config), mode=config.gap_label,
                                   commit_frames=config.gap_commit_frames,
                                   hysteresis=config.gap_hysteresis)
        # DAgger: 由**策略自己**驱动游戏(带 beta 概率混入老师动作保证覆盖),
        # 但每步都用老师对其**实际访问到的状态**打标签 -> 修正 BC 的分布漂移。
        # 纯模仿 (guard/rule/gap) 时 beta=1, 完全由老师接管。
        dagger_beta = (float(getattr(config, "dagger_beta", 0.2))
                       if teacher in {"dagger", "rule_dagger", "gap_dagger"} else 1.0)
        # 记录"上一动作"的方向下标; 缺口观测要靠它表达宏动作的承诺方向
        prev_dir = -1
        #: 连续同向决策步数 (给观测里的 action_age 维; 推理时同样可从自己的动作算)
        action_age = 0

        # max_steps <= 0 表示不限长; 一局只在真正 game over / clear 时结束
        step_limit = int(config.max_steps) if config.max_steps and config.max_steps > 0 else None
        step_index = -1
        while True:
            step_index += 1
            if step_limit is not None and step_index >= step_limit:
                break
            if self._stop.is_set():
                break
            if greedy:
                with self.policy_lock:
                    action = self.policy.greedy(obs)
                    value = self.policy.value(obs)
                logp = 0.0
            else:
                with self.policy_lock:
                    action, logp, value = self.policy.sample(obs, self._rng)
            # label = 老师对该状态的动作(用于模仿); exec = 实际执行的动作
            label: int | None = None
            exec_action = action
            if guard is not None:
                try:
                    teacher_action, _tinfo = guard.choose(state, state.frame, self.action_table[action])
                    label = self._teacher_index(teacher_action)
                    if teacher == "guard" or self._rng.random() < dagger_beta:
                        exec_action = label          # 老师接管(beta=1 时等价于纯模仿)
                except Exception:
                    label = None
            action_name = self.action_table[exec_action]
            act_counts[action_name] = act_counts.get(action_name, 0) + 1

            with self.env_lock:
                step = self.env.step(action_name, frames=config.step_frames)
            if config.step_sleep and config.step_sleep > 0:
                time.sleep(config.step_sleep)  # 多线程时给真机线程让出 GIL
            state = step.state
            if recorder is not None:
                recorder.update(state)

            # Dialogue pulses override sampled movement: do not label that
            # outcome as a policy action (would bias the policy gradient).
            if not step.info.get("dialogue_assist") and (guard is None or label is not None):
                observations.append(obs)
                actions.append(label if label is not None else action)
                logps.append(float(logp))
                values.append(value)
                rewards.append(float(step.reward))
                dones.append(bool(step.done))
                weights.append(self._state_weight(state))
            elif rewards:
                rewards[-1] += float(step.reward)
                if step.done:
                    dones[-1] = True

            record.steps += 1
            record.frames += int(step.info.get("frames") or 0)
            record.return_ += float(step.reward)
            for key, value_part in (step.info.get("reward_parts") or {}).items():
                record.reward_parts[key] = round(record.reward_parts.get(key, 0.0) + value_part, 6)

            if state.stage is not None:
                record.max_stage = max(record.max_stage or 0, state.stage)
            if state.score is not None:
                record.max_score = max(record.max_score or 0, state.score)
            if state.power is not None:
                record.max_power = max(record.max_power or 0.0, state.power)
            if state.graze is not None:
                record.total_graze = max(record.total_graze or 0, state.graze)

            reason = step.info.get("done_reason")
            if step.info.get("reward_parts", {}).get("miss"):
                record.misses += 1
            if step.info.get("bomb_guard"):
                record.bombs += 1
            try:
                self.journal.step(
                    {
                        "episode": index,
                        "step": record.steps,
                        "mode": "eval" if greedy else "train",
                        "action": action_name,
                        "reward": round(float(step.reward), 4),
                        "reward_parts": step.info.get("reward_parts") or {},
                        "stage": state.stage,
                        "lives": state.lives,
                        "power": state.power,
                        "graze": state.graze,
                        "bullets": state.bullet_count,
                        "enemies": state.enemies_alive,
                        "player_x": None if state.player_x is None else round(state.player_x, 1),
                        "player_y": None if state.player_y is None else round(state.player_y, 1),
                        "frames": step.info.get("frames"),
                        "done_reason": reason,
                        "dialogue_assist": bool(step.info.get("dialogue_assist")),
                        "bomb_guard": bool(step.info.get("bomb_guard")),
                    }
                )
            except Exception:
                pass
            if step.done or step.truncated:
                record.cleared = reason == "cleared"
                record.game_over = reason in {"game_over", "left_game"}
                record.truncated = bool(step.truncated)
                break

            # 缺口观测需要"上一动作": 动作表是 [9 快速] + [9 低速] + [... 炸弹],
            # 取模 9 就得到方向下标, 与 occupancy.DIRECTIONS 同序。
            next_dir = exec_action % len(DIRECTIONS)
            action_age = update_action_age(prev_dir, action_age, next_dir,
                                          config.gap_action_age_cap)
            prev_dir = next_dir
            obs = encode_observation(state, config, prev_action=prev_dir,
                                     action_age=action_age)

        if step_limit is not None and record.steps >= step_limit and not (record.cleared or record.game_over):
            record.truncated = True
        record.actions = act_counts
        record.duration_s = time.perf_counter() - started
        with self._state_lock:
            self.total_steps += record.steps
            self.total_frames += record.frames
            self.current_episode = {
                "episode": index,
                "steps": record.steps,
                "frames": record.frames,
                "return": round(record.return_, 3),
                "max_stage": record.max_stage,
                "elapsed_s": round(record.duration_s, 2),
            }

        if recorder is not None:
            try:
                path = self.danmaku_dir / (
                    f"{self.env.spec.key}_d{config.difficulty}_c{character}_s{shot}_ep{index}.json")
                recorder.save(path, game=self.env.spec.key, difficulty=config.difficulty,
                              character=character, shot=shot)
            except Exception:
                pass

        if greedy:
            return record, None
        terminal = bool(record.cleared or record.game_over)
        try:
            final_obs = encode_observation(state, config, prev_action=prev_dir,
                                           action_age=action_age)
            if terminal:
                bootstrap = 0.0
            else:
                with self.policy_lock:
                    bootstrap = float(self.policy.value(final_obs))
        except Exception:
            bootstrap = 0.0
        return record, Trajectory(
            observations=observations,
            actions=actions,
            logps=logps,
            values=values,
            rewards=rewards,
            dones=dones,
            bootstrap_value=bootstrap,
            weights=weights,
        )

    def _state_weight(self, state: GameState) -> float:
        """危险状态加权: 离最近弹幕越近, 模仿损失权重越大。

        用于 DAgger/BC 时优先学好"示范里少见但致命"的密集状态。
        """
        w = float(getattr(self.config, "danger_weight", 0.0) or 0.0)
        if w <= 0.0 or state.player_x is None or state.player_y is None or not state.bullets:
            return 1.0
        radius = float(getattr(self.config, "danger_weight_radius", 60.0) or 60.0)
        try:
            arr = np.asarray(state.bullets, dtype=np.float64).reshape(-1, 2)
            d = float(np.hypot(arr[:, 0] - state.player_x, arr[:, 1] - state.player_y).min())
        except Exception:
            return 1.0
        return 1.0 + w * max(0.0, 1.0 - d / max(1e-6, radius))

    def _teacher_index(self, action: str) -> int:
        """把老师动作字符串(如 'up_right+shoot+focus')映射到动作表索引(方向顺序一致)。"""

        head = action.split("+shoot")[0].strip("+ ")
        try:
            direction = DIRECTIONS.index(head) if head else 0
        except ValueError:
            direction = 0
        # 动作表布局: [0..8] 快速, [9..17] 低速(focus); 老师用 focus 时映射到低速块
        if self.config.use_focus and "focus" in action:
            direction += len(DIRECTIONS)
        return direction

    def _update_batch(
        self,
        batch: list[Trajectory],
    ) -> dict[str, float]:
        """把攒下来的几局拼成一个大 batch, 统一算优势并更新一次参数。"""
        algorithm = (self.config.algorithm or "").strip().lower()
        if algorithm == "ppo" and not (self.config.teacher or "").strip():
            return self._update_ppo(batch)

        if (self.config.teacher or "").strip():
            # 模仿学习: 直接交叉熵拟合老师动作, 不用回报/优势
            obs_all: list[np.ndarray] = []
            act_all: list[int] = []
            w_all: list[float] = []
            for traj in batch:
                obs_all.extend(traj.observations)
                act_all.extend(traj.actions)
                if len(traj.weights) == len(traj.observations):
                    w_all.extend(traj.weights)
                else:
                    w_all.extend([1.0] * len(traj.observations))
            if not obs_all:
                return {}
            X = np.stack(obs_all)
            A = np.asarray(act_all, dtype=np.int64)
            W = np.asarray(w_all, dtype=np.float64)
            epochs = max(1, int(getattr(self.config, "teacher_epochs", 8)))
            mb = max(64, int(getattr(self.config, "ppo_minibatch", 256)))
            rng = np.random.default_rng(self.config.seed + len(self.episodes))
            stats: dict[str, float] = {}
            for _ in range(epochs):
                order = rng.permutation(len(X))
                for start in range(0, len(X), mb):
                    idx = order[start:start + mb]
                    with self.policy_lock:
                        stats = self.policy.imitate_batch(X[idx], A[idx], weights=W[idx])
            return stats

        gamma = self.config.gamma
        # 与 _update_ppo 同理: 价值回归的平方损失会冲垮共享隐层, 先按整批回报的量级缩放
        scale = self._value_scale(batch)
        all_obs: list[np.ndarray] = []
        all_actions: list[int] = []
        all_advantages: list[np.ndarray] = []
        all_returns: list[float] = []

        for traj in batch:
            rewards = traj.rewards
            if not rewards:
                continue
            returns = np.zeros(len(rewards), dtype=np.float64)
            running = 0.0
            for i in range(len(rewards) - 1, -1, -1):
                running = rewards[i] + gamma * running
                returns[i] = running
            advantages = returns - np.asarray(traj.values, dtype=np.float64)
            all_obs.extend(traj.observations)
            all_actions.extend(traj.actions)
            all_advantages.append(advantages)
            all_returns.extend((returns / scale).tolist())

        if not all_obs:
            return {}

        advantages = np.concatenate(all_advantages)
        if advantages.size > 1:
            std = float(advantages.std())
            if std > 1e-6:
                advantages = (advantages - float(advantages.mean())) / (std + 1e-8)

        with self.policy_lock:
            return self.policy.update(
                np.stack(all_obs),
                np.asarray(all_actions, dtype=np.int64),
                advantages,
                np.asarray(all_returns, dtype=np.float64),
                value_coef=self.config.value_coef,
            )

    def _update_ppo(self, batch: list[Trajectory]) -> dict[str, float]:
        """PPO: GAE 优势 + clipped surrogate, 可多 epoch / mini-batch 更新。"""
        gamma = float(self.config.gamma)
        lam = float(getattr(self.config, "gae_lambda", 0.95))
        # 先用整批回报的量级把奖励/价值统一缩放, 再做 GAE 与价值回归。
        #
        # 为什么必须缩放: 价值回归的平方损失作用在**共享隐层**上, 而回报量级是几百到
        # 上千 (存活奖励 1/步 × 400 步)。不缩放时 value_err ~ 1e3, 平方损失 ~ 1e6,
        # 一项就盖过被归一化过的策略梯度 —— 实测 value_loss 4618 / grad_norm 2626,
        # 熵在两次更新内掉到 0, DAgger 练出来的 2400 帧策略被打回 624 (= 不动)。
        # 缩放对 GAE 是线性的 (整批同一个 scale), 归一化后的优势完全不变。
        scale = self._value_scale(batch)
        all_obs: list[np.ndarray] = []
        all_actions: list[int] = []
        all_logp: list[float] = []
        all_adv: list[np.ndarray] = []
        all_ret: list[float] = []
        for traj in batch:
            if not traj.rewards:
                continue
            adv, ret = self._gae(traj, gamma, lam, scale=scale)
            all_obs.extend(traj.observations)
            all_actions.extend(traj.actions)
            all_logp.extend(traj.logps)
            all_adv.append(adv)
            all_ret.extend(ret.tolist())
        if not all_obs:
            return {}
        adv = np.concatenate(all_adv)
        if adv.size > 1:
            std = float(adv.std())
            if std > 1e-6:
                adv = (adv - float(adv.mean())) / (std + 1e-8)
        with self.policy_lock:
            stats = self.policy.update_ppo(
                np.stack(all_obs),
                np.asarray(all_actions, dtype=np.int64),
                np.asarray(all_logp, dtype=np.float64),
                adv,
                np.asarray(all_ret, dtype=np.float64),
                clip_ratio=float(getattr(self.config, "ppo_clip", 0.2)),
                epochs=int(getattr(self.config, "ppo_epochs", 4)),
                minibatch=int(getattr(self.config, "ppo_minibatch", 256)),
                value_coef=float(self.config.value_coef),
                entropy_coef=float(self.config.entropy_coef),
                target_kl=float(getattr(self.config, "target_kl", 0.05)),
                ref_params=self._rl_ref,
                kl_coef=float(getattr(self.config, "kl_anchor", 0.0)),
            )
        stats["value_scale"] = round(scale, 4)
        return stats

    @staticmethod
    def _value_scale(batch: list[Trajectory]) -> float:
        """整批回报的量级 (≈ 单局回报绝对值), 用来把价值回归压到 O(1)。"""
        totals = [abs(float(np.sum(tra.rewards))) for tra in batch if tra.rewards]
        return max(1.0, float(np.mean(totals))) if totals else 1.0

    @staticmethod
    def _gae(traj: Trajectory, gamma: float, lam: float,
             *, scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        rewards = np.asarray(traj.rewards, dtype=np.float64) / scale
        values = np.asarray(traj.values, dtype=np.float64) / scale
        dones = traj.dones if len(traj.dones) == len(rewards) else [False] * len(rewards)
        n = len(rewards)
        adv = np.zeros(n, dtype=np.float64)
        next_value = float(traj.bootstrap_value) / scale
        last = 0.0
        for t in range(n - 1, -1, -1):
            nonterminal = 0.0 if dones[t] else 1.0
            delta = rewards[t] + gamma * next_value * nonterminal - values[t]
            last = delta + gamma * lam * nonterminal * last
            adv[t] = last
            next_value = values[t]
        returns = adv + values
        return adv, returns

    def _evaluate(self, index: int) -> dict[str, Any]:
        results = []
        for i in range(self.config.eval_episodes):
            record, _trajectory = self._run_episode(index * 1000 + i, greedy=True)
            results.append(record)
        summary = {
            "at_episode": index,
            "episodes": len(results),
            "mean_return": round(sum(r.return_ for r in results) / max(1, len(results)), 3),
            "mean_steps": round(sum(r.steps for r in results) / max(1, len(results)), 1),
            "best_stage": max((r.max_stage or 0) for r in results),
            "clears": sum(1 for r in results if r.cleared),
        }
        with self._state_lock:
            self.evaluations.append(summary)
        return summary

    def _capture_frame(self, index: int) -> None:
        """抓一帧当前画面 (供可视化 / 监视窗口 / MCP 仪表盘)."""
        try:
            with self.env_lock:
                frame = self.env.capture_frame()
        except Exception:
            return
        self.latest_frame = frame
        self.latest_frame_episode = index
        now = time.perf_counter()
        if now - self._last_frame_save < 1.0:
            return
        self._last_frame_save = now
        try:
            from .visualize import save_png

            path = save_png(frame, self.out_dir / "frames", name="latest.png")
            self.artifacts["frame"] = str(path)
            # 同时刷新仪表盘 (画面 + 曲线 + 统计), 训练过程中随时可以打开。
            dashboard = self.dashboard_image()
            dash_path = save_png(dashboard, self.out_dir, name="dashboard.png")
            self.artifacts["dashboard"] = str(dash_path)
        except Exception:
            pass

    def _refresh_curve(self, *, force: bool = False) -> None:
        """把学习曲线画出来落盘 (节流: 最多 2 秒一次)."""
        now = time.perf_counter()
        if not force and now - self._last_curve_save < 2.0:
            return
        self._last_curve_save = now
        try:
            from .visualize import draw_learning_curve, save_png

            curve = draw_learning_curve(self.episodes, self.evaluations, title="学习曲线")
            path = save_png(curve, self.out_dir, name="curve.png")
            self.artifacts["curve"] = str(path)
        except Exception:
            pass

    def _refresh_progress(self, *, force: bool = False) -> None:
        """??plot_every 局 (??force) 刷新 progress.png: 存活步数/面数/回报/掉残."""
        every = int(self.config.plot_every or 0)
        n = len(self.episodes)
        if not force:
            if every <= 0 or n == 0 or n % every != 0:
                return
        try:
            from .visualize import draw_progress, save_png

            image = draw_progress(self.episodes, title="训练进度 (存活步数 / 面数 / 回报 / 掉残)")
            path = save_png(image, self.out_dir, name="progress.png")
            self.artifacts["progress"] = str(path)
        except Exception:
            pass

    def progress_image(self):
        """存活步数/面数/回报/掉残 四联图 (供外部调用)."""
        from .visualize import draw_progress

        return draw_progress(self.episodes, title="训练进度 (存活步数 / 面数 / 回报 / 掉残)")

    def dashboard_image(self, *, title: str | None = None):
        """当前画面 + 学习曲线 + 实时统计 拼成一张图."""
        from .visualize import draw_dashboard

        return draw_dashboard(
            self.latest_frame,
            self.episodes,
            self.status(tail=4),
            title=title or f"Touhou RL 训练仪表盘 · {self.config.algorithm}",
        )

    def curve_image(self):
        from .visualize import draw_learning_curve

        return draw_learning_curve(self.episodes, self.evaluations, title="学习曲线")

    # ------------------------------------------------------------------
    # 观测
    # ------------------------------------------------------------------
    def status(self, *, tail: int = 5) -> dict[str, Any]:
        with self._state_lock:
            recent = self.episodes[-tail:]
            recent_returns = [r.return_ for r in self.episodes[-50:]]
            stages = [r.max_stage or 0 for r in self.episodes[-50:]]
            elapsed = (self.finished_at or time.perf_counter()) - (self.started_at or time.perf_counter())
            return {
                "running": self.running,
                "paused": self.paused,
                "error": self.error,
                "config": self.config.to_dict(),
                "policy": self.policy.describe(),
                "actions": self.action_table,
                "episodes_done": len(self.episodes),
                "total_steps": self.total_steps,
                "total_frames": self.total_frames,
                "elapsed_s": round(elapsed, 1),
                "steps_per_sec": round(self.total_steps / elapsed, 1) if elapsed > 0 else None,
                "best_return": None if self.best_return == float("-inf") else round(self.best_return, 3),
                "mean_return_last50": round(sum(recent_returns) / len(recent_returns), 3) if recent_returns else None,
                "mean_stage_last50": round(sum(stages) / len(stages), 3) if stages else None,
                "best_stage": max(stages) if stages else None,
                "last_update": self._last_update,
                "current_episode": self.current_episode or None,
                "last_evaluation": self.evaluations[-1] if self.evaluations else None,
                "evaluations": self.evaluations[-40:],
                "recent_episodes": [r.to_dict() for r in recent],
                "policy_path": str(self.policy_path()),
                "artifacts": dict(self.artifacts),
                "has_frame": self.latest_frame is not None,
                "latest_frame_episode": self.latest_frame_episode,
            }

    def policy_path(self) -> Path:
        return self.out_dir / "policy"

    def save(self, path: str | Path | None = None) -> Path:
        target = Path(path) if path else self.policy_path()
        saved = self.policy.save(target)
        meta = {
            "config": self.config.to_dict(),
            "episodes_done": len(self.episodes),
            "total_steps": self.total_steps,
            "best_return": None if self.best_return == float("-inf") else self.best_return,
        }
        import json

        saved.with_suffix(".json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return saved

    def load(self, path: str | Path | None = None) -> dict[str, Any]:
        target = Path(path) if path else self.policy_path()
        npz = target.with_suffix(".npz")
        if not npz.exists():
            raise FileNotFoundError(f"找不到策略文件: {npz}")
        loaded = MLPPolicy.load(npz)
        if loaded.obs_dim != self.obs_dim or loaded.n_actions != len(self.action_table):
            raise ValueError("Checkpoint observation/action dimensions do not match training config")
        self.policy = loaded
        self.config.hidden = self.policy.hidden
        self.action_table = build_action_table(use_focus=self.config.use_focus,
                                              use_bomb=self.config.use_bomb)
        return self.policy.describe()

    def clear_history(self) -> None:
        with self._state_lock:
            self.episodes.clear()
            self.evaluations.clear()
            self.total_steps = 0
            self.total_frames = 0
            self.best_return = float("-inf")
