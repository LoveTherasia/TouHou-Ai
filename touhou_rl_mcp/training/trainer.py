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
from ..types import PLAYFIELD_HEIGHT, PLAYFIELD_WIDTH, GameState
from .features import encode_observation, feature_dim, gap_spec_from, observation_dim
from .gap_label import update_action_age
from .journal import Journal
from .policy import MLPPolicy

#: 判定"自机在 Boss 火力线上"的横向容差 (像素)。
#: mock 的伤害窗口是 22px, 真机正作的散射更宽; 这里只作为**诊断指标**,
#: 不参与奖励计算, 所以取一个略宽的 24px。
#: 诊断用: 自机 y 小于它就记为"贴顶" (与 mpc_top_margin 默认值一致)
TOP_ZONE_DIAG = 42.0

ALIGN_TOLERANCE = 24.0

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
    #: 观测末尾追加 12 维"墙特征" (覆盖率/缺口中心/缺口宽度/到达帧数/8 列占用)。
    #: 需要弹速; 只能与默认观测同用 (不与 use_gap / use_grid 混用)。
    use_wall_feat: bool = False
    #: 载入旧 checkpoint 时**零初始化扩维**输入层 (新列权重 0 => 函数不变)。
    #: 用途: 给已有策略加上"墙特征"再继续训, 而不是从头学。
    expand_input: bool = False
    #: 观测改用"未来占用栅格 + 几何缺口" (见 training/occupancy.py)。
    #: 这是密集弹幕下唯一能让策略提前横向占位的表示; 需要弹速 (mock 直接提供)。
    #: Deep Sets 观测: 不定长弹列表 -> 逐弹 φ 投影 + mean/max 池化 (置换不变)
    use_set: bool = False
    #: φ 的嵌入维度 (固定随机投影)
    set_embed: int = 32
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
    #: 对**无标签步**的 KL 锚系数 (>0 时只要求"这些地方别动")。
    #: 分层老师只在墙段出标签, 不加锚时共享隐层会被墙段标签整体拉走:
    #: 实测平均存活 1852 -> 901 步, 常规弹型对准率 90% -> 5%。
    kl_base: float = 0.0
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
    # ---- score / score_dagger 老师 (按奖励排序打标签) ----
    #: 安全阈值(px): 未来 H 步内最小弹幕间距低于它的方向直接出局
    score_safety_margin: float = 12.0
    #: 对齐窗口(px): |自机 x − 目标 x| 小于它算"在火力线上"
    score_align_tol: float = 20.0
    #: 滞回: 新方向要比保持当前方向好这么多才切换 (减少左右互搏)
    score_commit_margin: float = 0.0
    # ---- hybrid / hybrid_dagger 老师 (力场先验 + 分数重排) ----
    #: 力场先验权重: 与"满对准收益(约 11)"同量级才起作用; 越大越像 rule 老师
    hybrid_prior_weight: float = 2.5
    #: 软安全屏障权重: clearance 归零时的惩罚。25 已经大过满对准收益(11.4),
    #: 再大(100)会把人往角落赶, 实测存活反而掉一半。
    hybrid_barrier_weight: float = 25.0
    #: 对准收益的估计窗口(帧), 默认与安全窗口同长; 拉长会牺牲存活 (见 sweep 记录)
    hybrid_align_horizon: float = 22.0
    #: 直线弹幕墙的前瞻帧数 (0=关闭)。60 能让 wall_gate 从 546 撑到 1320 步,
    #: 代价是 stage1 从 4770 掉到 3813 —— 属于"专用"开关, 不要默认开
    hybrid_wall_horizon: float = 0.0
    # ---- viability / 可行域规划器老师 ----
    #: 规划视野(帧): 必须 >= 墙从生成到压到自机高度的时间(约 150)
    planner_horizon: int = 180
    #: 规划器每层几帧 (2 => 网格 4.4px, 与自机速度精确对齐)
    planner_dt: int = 2
    #: 规划器的对准偏好: 在所有同样安全的位置里挑离 Boss 那列最近的。
    #: 实测 0.15 把对准率从 0~20% 提到 61~92%, 存活一步不掉 (见 VIABILITY_PLANNER.md)
    planner_aim_weight: float = 0.15
    # ---- MPC 老师 (docs/PLANNER_V2.md §8) ----
    #: MPC 视界(帧)。24 = 反应式; 72+ 才能开始为墙让位 (但反应式弹型会退化)
    mpc_horizon: int = 24
    #: MPC 每个控制步推进的帧数 (= 承诺期)
    mpc_dt: int = 3
    #: CEM 采样数 / 迭代轮数
    mpc_samples: int = 48
    mpc_iters: int = 3
    #: 「缺口锚点」权重: 把终点往最近一堵墙的缺口列拉 (墙/闭合环需要 >0)
    mpc_gap: float = 0.0
    #: 「正上方弹幕柱」权重 (隐式缺口项, 早期做法, 与 mpc_gap 二选一)
    mpc_column: float = 0.0
    #: 按 WallDetector 切长/短视界 (短视界 24 + 长视界 72)
    mpc_adaptive: bool = False
    #: 规划一次执行几步 (1 = 每步重规划; 2~3 更稳也更快)
    mpc_plan_every: int = 1
    #: 「并列动作承诺」: 选定方向后 N 帧内不换向（除非新方向明显更便宜）。
    #: 这是把老师标签噪声从源头掐掉的主开关 —— 实测承诺前相邻帧换向率 50~77%。
    mpc_commit: int = 0
    #: 承诺期内允许换向所需的代价优势 (px/代价单位)
    mpc_switch: float = 0.0
    #: focus 动作的代价优势阈值: 慢速要比全速便宜这么多才按 shift
    mpc_focus_margin: float = 0.0
    #: 老师确定化: 每帧用 `seed + frame` 重播种 CEM 噪声 (同一状态同一答案)
    mpc_deterministic: bool = True
    #: 「平滑安全余量」权重: 让代价随"离弹多远"连续变化, 而不是过了 margin 一律 0。
    #: 代价向量要拿去回归 (cost-to-go) 时必须 >0, 否则 70% 的帧都是并列最优。
    mpc_smooth: float = 0.0
    mpc_smooth_sigma: float = 20.0
    #: 承诺期（**游戏帧**，内部按 step_frames 折算成决策步）。实测 8~15 帧是舒服区间：
    #: 太短等于没承诺，太长会在弹幕变向时锁死错误方向。
    mpc_commit_frames: int = 0
    #: 「持续贴边」惩罚：允许上穿躲弹，但待在天花板/墙角要付租金。
    #: 只罚终点站位拦不住"贴顶"——规划器会先上去躲一堵墙、然后一直贴着。
    mpc_top_w: float = 0.0
    mpc_top_margin: float = 42.0
    mpc_bottom_w: float = 0.0
    mpc_bottom_margin: float = 30.0
    mpc_edge_w: float = 0.0
    mpc_edge_margin: float = 26.0
    #: 用**软标签**蒸馏 (老师在每个动作上的代价 -> 目标分布), 而不是 argmax
    soft_labels: bool = False
    #: 软标签的温度: 代价差多少算"同样好"。代价量级 ~10~200, 默认 20
    soft_temp: float = 20.0
    #: 用 **cost-to-go 回归**蒸馏: 学生输出层直接回归 `-相对代价/T`（等价于把
    #: MPC 的价值函数蒸出来），再由学生自己 argmax。与 soft_labels 二选一或并用。
    cost_to_go: bool = False
    #: cost-to-go 回归时对相对代价的上限截断（除以 soft_temp 后等于 logits 下限）
    cost_clip: float = 8.0
    #: 用**并列感知的成对排序损失**蒸馏 cost-to-go（`MLPPolicy.fit_rank`）。
    #: 为什么不用回归：老师代价向量 74~78% 的帧最优与次优完全并列、其余全部撞截断，
    #: 回归目标实际只有 {0, −clip} 两个值，学不出排序（§10.6）。
    rank_ties: bool = False
    #: 用 **tie-aware ListMLE** 蒸馏（与 ``rank_ties`` 二选一，优先级更高）。
    #: 相比"并列组 CE"多用了组外的完整次序，样本效率更高；候选动作采样见下。
    listmle: bool = False
    #: ListMLE 每个状态只随机取 K 个候选动作（必含 argmin）；0 = 用全部动作。
    #: 采样越少算得越快、负例越多样，但 K 太小会漏掉"几乎一样好"的动作。
    rank_candidates: int = 0
    #: 学生侧的**位置正则**：贴顶/贴底/贴边时，给"继续往外走"的动作加惩罚。
    #: 老师那边已有位置偏好，这一项是兜底（学生只看到 32 颗弹，未必推得出"上面没路"）。
    position_reg: bool = False
    position_reg_weight: float = 2.0
    #: 进入"贴边区"的深度阈值（px）：距顶/底/边小于它就算贴边
    position_reg_zone: float = 70.0
    #: 「回中场」吸引力（相对"别再往外"的倍数）。0 = 只罚不奖（实测贴顶 21%/78%）,
    #: 0.8 = 同时奖励往回走，才是"允许上穿但不常驻天花板"的正确语义。
    position_reg_pull: float = 0.8
    #: 排序损失里"多大的代价差才算真的有差别"（代价单位；差更小的对视为并列，不产生梯度）
    rank_margin: float = 20.0

    # ---- 分层切换 (策略 <-> 规划器) ----
    #: 评测/演示时开分层切换: 检测到"整幅横墙 / 多向收网"就改问规划器
    hier_switch: bool = False
    #: 墙段内沿用上次规划方向的步数 (规划器一次 25~40ms, 逐帧算训练会慢到不可用)
    hier_replan_every: int = 1
    #: 每局轮换的弹型 (空=用 config.danmaku_pattern)。写在 TrainConfig 里是因为
    #: "只蒸墙段"必须固定打墙型, 而"不遗忘"又必须混入常规弹型 —— 两者只能靠轮换。
    pattern_cycle: list[str] = field(default_factory=list)
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
    #: 本局起始分数; ``score_gain = max_score - start_score``
    start_score: int = 0
    #: 对 Boss 造成的总伤害 (符卡切换回血不算; 用来判断"有没有在打")
    boss_damage: int = 0
    #: 与 Boss 同列 (|Δx| <= ALIGN_TOLERANCE) 的步数, 以及 Boss 在场的步数
    aligned_steps: int = 0
    boss_steps: int = 0
    #: 分层切换: 有多少步交给了规划器, 以及最后一次触发的原因 (curtain/encircle)
    planner_steps: int = 0
    #: 「贴顶」诊断: 上穿本身是合法手段, 但**一直贴在天花板上**是退化行为
    #: (min_y 常驻 8~20px、几乎不回中场)。这里记每局的贴顶步数与 y 的分布。
    top_steps: int = 0
    y_steps: int = 0
    min_y: float | None = None
    y_sum: float = 0.0
    switch_reason: str = ""

    @property
    def score_gain(self) -> int | None:
        """本局净增分数 —— "打中敌人"在 mock 里直接体现在分数上。"""
        if self.max_score is None:
            return None
        return self.max_score - self.start_score

    @property
    def align_ratio(self) -> float | None:
        """站在 Boss 火力线上的步数占比 (无 Boss 时为 None)."""
        if not self.boss_steps:
            return None
        return round(self.aligned_steps / self.boss_steps, 3)

    @property
    def top_ratio(self) -> float:
        """贴顶步数占比 (y < ``TOP_ZONE_DIAG``)。越低越好。"""
        return round(self.top_steps / self.y_steps, 3) if self.y_steps else 0.0

    @property
    def mean_y(self) -> float:
        return round(self.y_sum / self.y_steps, 1) if self.y_steps else 0.0

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
            "score_gain": self.score_gain,
            "max_power": self.max_power,
            "total_graze": self.total_graze,
            "misses": self.misses,
            "bombs": self.bombs,
            "cleared": self.cleared,
            "game_over": self.game_over,
            "truncated": self.truncated,
            "boss_damage": self.boss_damage,
            "align_ratio": self.align_ratio,
            "planner_steps": self.planner_steps,
            "switch_reason": self.switch_reason,
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
    #: 软标签 (``(n_actions,)`` 目标分布; None = 用硬标签)。
    #: 规划器老师"多个动作同样安全"时 argmax 会随机跳 (实测自一致率 ~25%),
    #: 用代价分布代替 argmax 才能让学生学到真正的东西。
    soft: list[Any] = field(default_factory=list)
    #: 老师的逐动作 cost-to-go (``(n_actions,)``; None = 该步没有代价标签)。
    #: 这是 cost-to-go 蒸馏的监督信号: 学生回归它, 再自己 argmax。
    cost: list[Any] = field(default_factory=list)
    #: 逐动作**位置正则** (``(n_actions,)`` 非负权重)。贴顶时对"继续向上"给正值，
    #: 蒸馏损失里加 ``w·Σ p_a·reg_a``，压住"越贴越贴"的退化。
    posreg: list[Any] = field(default_factory=list)


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
        #: KL 锚的参考参数 (监督阶段第一次更新前冻结; load() 之后也正确)
        self._base_params: list[Any] | None = None
        #: 零初始化扩维前的观测维度 (None = 没有扩维)
        self.expanded_from: int | None = None
        #: 分层切换 (评测/演示): 检测到墙段时把决策交给可行域规划器。
        #: 规划器只在墙段被调用, 所以平均开销很低 (墙段外 0 成本)。
        self._hier_detector = None
        self._hier_planner = None
        #: 墙段的"承诺期"状态: 沿用上次方向 hier_replan_every 步再重算
        self._hier_active = False
        self._hier_age = 0
        self._hier_dir = 0
        if self.config.hier_switch:
            from .hierarchical import WallDetector
            from .viability import ViabilityPlanner

            self._hier_detector = WallDetector()
            self._hier_planner = ViabilityPlanner(
                horizon=int(self.config.planner_horizon),
                dt=int(self.config.planner_dt),
                aim_weight=float(self.config.planner_aim_weight),
            )
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
        # 直接以"无老师 + KL 锚"启动 (常见于: 载入蒸馏好的策略再来 PPO 微调) 时,
        # 不会发生 teacher -> "" 的切换, 必须在开跑前冻结参考策略, 否则 kl_anchor
        # 静默失效 —— 这正是"PPO 把 DAgger 打崩"的一个隐蔽原因。
        self.apply_hyperparams()
        self._capture_rl_reference()
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
    def _capture_rl_reference(self) -> None:
        """冻结 RL 阶段的参考策略 (KL 锚的锚点).

        触发时机只有两个: ``start()`` 时就已经没有老师 (直接 PPO 微调已蒸馏的
        策略), 或者 ``reconfigure`` 把老师摘掉的那一刻。抓早了是随机初始策略,
        抓晚了已经被 RL 改过 —— 两种都会让 ``kl_anchor`` 悄悄失效。
        """
        if self._rl_ref is not None:
            return
        if float(getattr(self.config, "kl_anchor", 0.0)) <= 0.0:
            return
        if str(getattr(self.config, "teacher", "") or "").strip():
            return
        with self.policy_lock:
            self._rl_ref = self.policy.snapshot_params()

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
        if patch.get("teacher") is not None:
            self._capture_rl_reference()
        self.policy.lr = new_config.lr
        self.policy.entropy_coef = new_config.entropy_coef
        if rebuild:
            # 观测/动作维度变了 -> 参数形状变了, 旧锚必须作废, 否则形状不匹配
            self._rl_ref = None
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

    def _apply_pattern(self, index: int, config: "TrainConfig") -> None:
        """每局按 ``pattern_cycle`` 轮换弹型 (mock 后端)。

        为什么必须轮换: 只蒸 ``wall_gate`` 会把通才策略的常规躲避冲掉
        (灾难性遗忘), 而混着练又需要"这一局是哪一种"可控。``MockBackend.reset``
        每次都读 ``config.danmaku_pattern`` 重建弹型, 所以这里改配置就够了。
        """
        cycle = [str(p) for p in (getattr(config, "pattern_cycle", None) or []) if str(p)]
        if not cycle:
            return
        cfg = getattr(self.env, "config", None)
        if cfg is None:
            return
        name = cycle[index % len(cycle)]
        if str(cfg.get("danmaku_pattern") or "") != name:
            cfg.patch({"danmaku_pattern": name})

    def _run_episode(
        self, index: int, *, greedy: bool = False
    ) -> tuple[EpisodeRecord, "Trajectory" | None]:
        config = self.config
        character, shot = (divmod((index-1) % 6, 3) if config.rotate_loadouts
                           else (config.character, config.shot))
        with self.env_lock:
            self._apply_pattern(index, config)
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
        actions: list[Any] = []
        softs: list[Any] = []
        posregs: list[Any] = []
        costs: list[Any] = []
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
        #: 攻击意图的诊断量: 净增分数 / Boss 伤害 / 对准率 (见 EpisodeRecord)
        record.start_score = int(state.score) if state.score is not None else 0
        hp_prev = state.boss_hp
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
            elif teacher in {"score", "score_dagger"}:
                from .score_teacher import ScoreTeacher

                # 把**当前奖励权重**交给老师: score_scale 变了, 标签就会变。
                reward_weights = dict(self.env.config.section("reward"))
                guard = ScoreTeacher(
                    weights=reward_weights,
                    safety_margin=float(getattr(config, "score_safety_margin", 12.0)),
                    align_tol=float(getattr(config, "score_align_tol", 20.0)),
                    commit_margin=float(getattr(config, "score_commit_margin", 0.0)),
                )
            elif teacher in {"planner", "viability"}:
                from .viability import ViabilityPlanner

                # 注意: 每次决策 20~40ms, 只适合"小样本示范/离线打标签",
                # 不适合几千步的 DAgger (会慢到不可用)。
                guard = ViabilityPlanner(
                    horizon=int(getattr(config, "planner_horizon", 180)),
                    dt=int(getattr(config, "planner_dt", 2)),
                    aim_weight=float(getattr(config, "planner_aim_weight", 0.15)),
                )
            elif teacher in {"hier", "hier_dagger"}:
                from .hierarchical import HierarchicalTeacher
                from .viability import ViabilityPlanner

                # 分层: 只在"整幅横墙 / 多向收网"的步打标签, 其余弃权。
                # 弃权 = 不产生梯度, 策略在常规局面保持原样 (避免灾难性遗忘)。
                guard = HierarchicalTeacher(
                    planner=ViabilityPlanner(
                        horizon=int(getattr(config, "planner_horizon", 180)),
                        dt=int(getattr(config, "planner_dt", 2)),
                        aim_weight=float(getattr(config, "planner_aim_weight", 0.15)),
                    ),
                    replan_every=int(getattr(config, "hier_replan_every", 1)),
                )
            elif teacher in {"mpc", "mpc_dagger"}:
                from .mpc_teacher import MPCTeacher

                # 只在"规划器能做的事"上打标签: 上穿/占位/长程站位。
                # 每步 1~15ms, 比 viability(25~40ms) 便宜, 几千步的 DAgger 可承受。
                guard = MPCTeacher(
                    horizon=int(getattr(config, "mpc_horizon", 24)),
                    dt=int(getattr(config, "mpc_dt", 3)),
                    n_samples=int(getattr(config, "mpc_samples", 48)),
                    iters=int(getattr(config, "mpc_iters", 3)),
                    gap_weight=float(getattr(config, "mpc_gap", 0.0)),
                    column_weight=float(getattr(config, "mpc_column", 0.0)),
                    adaptive=bool(getattr(config, "mpc_adaptive", False)),
                    plan_every=int(getattr(config, "mpc_plan_every", 1)),
                    step_frames=int(config.step_frames),
                    use_focus=config.use_focus,
                    commit_steps=int(getattr(config, "mpc_commit", 0)),
                    switch_margin=float(getattr(config, "mpc_switch", 0.0)),
                    focus_margin=float(getattr(config, "mpc_focus_margin", 0.0)),
                    deterministic=bool(getattr(config, "mpc_deterministic", True)),
                    smooth_clear=float(getattr(config, "mpc_smooth", 0.0)),
                    smooth_sigma=float(getattr(config, "mpc_smooth_sigma", 20.0)),
                    commit_frames=int(getattr(config, "mpc_commit_frames", 0)),
                    top_weight=float(getattr(config, "mpc_top_w", 0.0)),
                    top_margin=float(getattr(config, "mpc_top_margin", 42.0)),
                    bottom_weight=float(getattr(config, "mpc_bottom_w", 0.0)),
                    bottom_margin=float(getattr(config, "mpc_bottom_margin", 30.0)),
                    edge_weight=float(getattr(config, "mpc_edge_w", 0.0)),
                    edge_margin=float(getattr(config, "mpc_edge_margin", 26.0)),
                )
            elif teacher in {"hybrid", "hybrid_dagger"}:
                from .hybrid_teacher import HybridTeacher

                # 力场先验 (与 rule 同源) + 分数重排 (吃 reward 权重) + 软安全屏障
                guard = HybridTeacher(
                    weights=dict(self.env.config.section("reward")),
                    safety_margin=float(getattr(config, "score_safety_margin", 12.0)),
                    align_tol=float(getattr(config, "score_align_tol", 20.0)),
                    commit_margin=float(getattr(config, "score_commit_margin", 0.0)),
                    prior_weight=float(getattr(config, "hybrid_prior_weight", 2.5)),
                    barrier_weight=float(getattr(config, "hybrid_barrier_weight", 25.0)),
                    align_horizon=float(getattr(config, "hybrid_align_horizon", 22.0)),
                    wall_horizon=float(getattr(config, "hybrid_wall_horizon", 0.0)),
                )
        # DAgger: 由**策略自己**驱动游戏(带 beta 概率混入老师动作保证覆盖),
        # 但每步都用老师对其**实际访问到的状态**打标签 -> 修正 BC 的分布漂移。
        # 纯模仿 (guard/rule/gap) 时 beta=1, 完全由老师接管。
        dagger_beta = (float(getattr(config, "dagger_beta", 0.2))
                       if teacher in {"dagger", "rule_dagger", "gap_dagger",
                                      "score_dagger", "hybrid_dagger", "hier_dagger",
                                      "mpc_dagger"} else 1.0)
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
                obs_for_action = obs
                if self._hier_detector is not None and self._hier_planner is not None:
                    metrics = self._hier_detector.detect(state)
                    if metrics.switched:
                        self._hier_age += 1
                        if (not self._hier_active
                                or self._hier_age > max(1, int(self.config.hier_replan_every))):
                            info = self._hier_planner.plan(state)
                            self._hier_dir = int(info.direction)
                            self._hier_age = 1
                        self._hier_active = True
                        action = min(self._hier_dir, len(self.action_table) - 1)
                        record.planner_steps += 1
                        record.switch_reason = metrics.reason
                        with self.policy_lock:
                            value = self.policy.value(obs_for_action)
                        logp = 0.0
                        obs_for_action = None
                    else:
                        self._hier_active = False
                        self._hier_age = 0
                if obs_for_action is not None:
                    with self.policy_lock:
                        action = self.policy.greedy(obs_for_action)
                        value = self.policy.value(obs_for_action)
                    logp = 0.0
            else:
                with self.policy_lock:
                    action, logp, value = self.policy.sample(obs, self._rng)
            # label = 老师对该状态的动作(用于模仿); exec = 实际执行的动作
            label: int | None = None
            exec_action = action
            soft_vec = None
            if (guard is not None
                    and bool(getattr(config, "soft_labels", False))
                    and hasattr(guard, "soft_labels")):
                try:
                    soft_vec = guard.soft_labels(
                        state, float(getattr(config, "soft_temp", 20.0)))
                except Exception:
                    soft_vec = None
            cost_vec = None
            if (guard is not None
                    and (bool(getattr(config, "cost_to_go", False))
                         or bool(getattr(config, "rank_ties", False))
                         or bool(getattr(config, "listmle", False)))
                    and hasattr(guard, "cost_vector")):
                try:
                    cost_vec = guard.cost_vector(state)
                except Exception:
                    cost_vec = None
            pos_reg = (self._position_reg(state)
                       if (guard is not None and bool(getattr(config, "position_reg", False)))
                       else None)
            if guard is not None:
                try:
                    teacher_action, _tinfo = guard.choose(state, state.frame, self.action_table[action])
                    # 老师返回空动作 = **弃权** (分层老师只在墙段出标签)。此时必须
                    # 保持 label=None: 若把"策略自己的动作"当标签, 交叉熵等于
                    # "让它对自己更自信", 会压掉熵并固化坏习惯。
                    if teacher_action:
                        label = self._teacher_index(teacher_action)
                        if teacher == "guard" or self._rng.random() < dagger_beta:
                            exec_action = label      # 老师接管(beta=1 时等价于纯模仿)
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
            # kl_base > 0 时, 分层老师弃权的步也要留下来 —— 它们不进交叉熵,
            # 只用来把策略"钉"在基座分布上。
            keep_unlabeled = (guard is not None and label is None
                              and float(getattr(config, "kl_base", 0.0)) > 0.0)
            if not step.info.get("dialogue_assist") and (
                    guard is None or label is not None or keep_unlabeled):
                observations.append(obs)
                if label is not None:
                    actions.append(label)
                elif guard is None:
                    actions.append(action)
                else:
                    actions.append(-1)      # 无标签: 只参与 KL 锚
                softs.append(soft_vec)
                costs.append(cost_vec)
                posregs.append(pos_reg)
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
            # 攻击意图: 打掉的血 / 是否站在 Boss 火力线上 (与 reward 无关, 只做诊断)
            if state.boss_hp is not None and hp_prev is not None:
                hurt = hp_prev - state.boss_hp
                if 0 < hurt < 500:  # 符卡切换/换面会把 hp 抬高, 不算伤害
                    record.boss_damage += hurt
            hp_prev = state.boss_hp
            boss_pos = (state.raw or {}).get("boss_position")
            if state.boss_hp is not None and state.player_x is not None and boss_pos:
                record.boss_steps += 1
                try:
                    if abs(float(state.player_x) - float(boss_pos[0])) <= ALIGN_TOLERANCE:
                        record.aligned_steps += 1
                except (TypeError, ValueError):
                    pass
            if state.player_y is not None:
                try:
                    py = float(state.player_y)
                    record.y_steps += 1
                    record.y_sum += py
                    record.min_y = py if record.min_y is None else min(record.min_y, py)
                    if py < TOP_ZONE_DIAG:
                        record.top_steps += 1
                except (TypeError, ValueError):
                    pass
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
            soft=softs,
            cost=costs,
            posreg=posregs,
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

    def _action_dirs(self) -> np.ndarray:
        """``(n_actions, 2)`` 每个动作的单位方向（从动作名解析，含 focus 块）。"""
        cached = getattr(self, "_dir_cache", None)
        if cached is not None and cached.shape[0] == len(self.action_table):
            return cached
        dirs = np.zeros((len(self.action_table), 2), dtype=np.float64)
        for i, name in enumerate(self.action_table):
            if "left" in name:
                dirs[i, 0] = -1.0
            elif "right" in name:
                dirs[i, 0] = 1.0
            if "up" in name:
                dirs[i, 1] = -1.0
            elif "down" in name:
                dirs[i, 1] = 1.0
        self._dir_cache = dirs
        return dirs

    def _position_reg(self, state: Any) -> np.ndarray | None:
        """贴顶/贴底/贴边时，给"继续往外走"的动作一个非负权重。

        为什么学生侧还要再加这一项：老师的代价里已经有位置偏好（``mpc_top_w``），
        但学生**只看到最近 32 颗弹**——它未必能推出"上面已经没有路了"。这项正则
        不依赖观测，只依赖自机位置，是纯几何的兜底，专门压住"上穿之后一直贴顶"。

        权重按越界深度线性给（贴得越深罚得越重），到 ``zone`` 边界降为 0，
        所以在边界附近不会突然"抖"一下。
        """
        if not bool(getattr(self.config, "position_reg", False)):
            return None
        px, py = getattr(state, "player_x", None), getattr(state, "player_y", None)
        if px is None or py is None:
            return None
        zone = max(1.0, float(getattr(self.config, "position_reg_zone", 70.0)))
        # 「回中场」的吸引力: 只罚"继续往外"是不够的 —— 实测贴顶 21%、ring_gate
        # 更是 78%（学生学会上穿之后就一直挂在天花板上）。原因是"不往上"≠"往下"：
        # 在顶部时"停住/横移"同样不产生惩罚，于是最省事的选择就是继续贴着。
        # 这里给"往回走"**负权重**（乘 p_a 后等于奖励），形成"贴顶 -> 回中场"的闭环。
        pull = float(getattr(self.config, "position_reg_pull", 0.8))
        dirs = self._action_dirs()
        reg = np.zeros(dirs.shape[0], dtype=np.float64)
        # 顶部: 罚继续向上, 同时奖励往下
        if py < zone:
            depth = (zone - float(py)) / zone
            reg += depth * (np.maximum(0.0, -dirs[:, 1]) - pull * np.maximum(0.0, dirs[:, 1]))
        # 底部: 罚继续向下, 奖励往上
        if py > PLAYFIELD_HEIGHT - zone:
            depth = (float(py) - (PLAYFIELD_HEIGHT - zone)) / zone
            reg += depth * (np.maximum(0.0, dirs[:, 1]) - pull * np.maximum(0.0, -dirs[:, 1]))
        # 左右: 贴边时罚往外、奖励往内
        if px < zone:
            depth = (zone - float(px)) / zone
            reg += depth * (np.maximum(0.0, -dirs[:, 0]) - pull * np.maximum(0.0, dirs[:, 0]))
        elif px > PLAYFIELD_WIDTH - zone:
            depth = ((float(px) - (PLAYFIELD_WIDTH - zone)) / zone)
            reg += depth * (np.maximum(0.0, dirs[:, 0]) - pull * np.maximum(0.0, -dirs[:, 0]))
        return reg if np.any(np.abs(reg) > 1e-9) else None

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
            act_all: list[Any] = []
            w_all: list[float] = []
            has_soft = any(getattr(t, "soft", None) for t in batch)
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
            has_cost = any(getattr(t, "cost", None) for t in batch)
            if has_cost:
                # cost-to-go 蒸馏: 目标是老师**逐动作的代价向量**, 不是他挑的那个动作。
                # 并列动作天然免疫 (代价接近 -> 目标 logits 接近), 而硬标签会把
                # "这帧老师随机翻了左边"当成真理教给学生。
                n_act = len(self.action_table)
                C = np.full((len(act_all), n_act), np.nan, dtype=np.float64)
                PR = np.zeros((len(act_all), n_act), dtype=np.float64)
                use_posreg = (bool(getattr(self.config, "position_reg", False))
                              and float(getattr(self.config, "position_reg_weight", 0.0)) > 0.0)
                pos = 0
                for traj in batch:
                    vecs = list(getattr(traj, "cost", []) or [])
                    regs = list(getattr(traj, "posreg", []) or [])
                    for i in range(len(traj.actions)):
                        vec = vecs[i] if i < len(vecs) else None
                        if vec is not None:
                            arr = np.asarray(vec, dtype=np.float64).reshape(-1)
                            if arr.shape[0] == n_act:
                                C[pos] = arr
                        if use_posreg:
                            r = regs[i] if i < len(regs) else None
                            if r is not None:
                                ra = np.asarray(r, dtype=np.float64).reshape(-1)
                                if ra.shape[0] == n_act:
                                    # 允许负权重: 负值 = 奖励"回到中场"
                                    PR[pos] = ra
                        pos += 1
                if not bool(np.isfinite(C).any()):
                    return {}
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
                            if bool(getattr(self.config, "listmle", False)):
                                stats = self.policy.fit_listmle(
                                    X[idx], C[idx], weights=W[idx],
                                    margin=float(getattr(self.config, "rank_margin", 20.0)),
                                    candidates=int(getattr(self.config, "rank_candidates", 0)),
                                    # 种子随更新次数走: 每轮换一批负例, 否则采到的
                                    # 那几个动作会被系统性强化
                                    seed=int(self.config.seed) + self.policy._t,
                                    posreg=PR[idx] if use_posreg else None,
                                    posreg_weight=float(
                                        getattr(self.config, "position_reg_weight", 0.0)),
                                )
                            elif bool(getattr(self.config, "rank_ties", False)):
                                # 并列感知排序: 只从"代价差 > rank_margin"的动作对里学
                                stats = self.policy.fit_rank(
                                    X[idx], C[idx], weights=W[idx],
                                    margin=float(getattr(self.config, "rank_margin", 20.0)),
                                )
                            else:
                                stats = self.policy.fit_cost(
                                    X[idx], C[idx], weights=W[idx],
                                    temp=float(getattr(self.config, "soft_temp", 20.0)),
                                    clip=float(getattr(self.config, "cost_clip", 8.0)),
                                )
                return stats
            if has_soft:
                # 每行变成目标分布: 软标签优先, 硬标签转 one-hot, -1 行全 0 (只进 KL 锚)
                n_act = len(self.action_table)
                P = np.zeros((len(act_all), n_act), dtype=np.float64)
                pos = 0
                for traj in batch:
                    vecs = list(getattr(traj, "soft", []) or [])
                    for i, a_lab in enumerate(traj.actions):
                        vec = vecs[i] if i < len(vecs) else None
                        if vec is not None:
                            v = np.asarray(vec, dtype=np.float64).reshape(-1)
                            if v.shape[0] == n_act:
                                P[pos] = np.maximum(v, 0.0)
                        elif a_lab is not None and a_lab >= 0:
                            P[pos, int(a_lab)] = 1.0
                        pos += 1
                A = P
            else:
                A = np.asarray(act_all, dtype=np.int64)
            W = np.asarray(w_all, dtype=np.float64)
            epochs = max(1, int(getattr(self.config, "teacher_epochs", 8)))
            mb = max(64, int(getattr(self.config, "ppo_minibatch", 256)))
            rng = np.random.default_rng(self.config.seed + len(self.episodes))
            kl_coef = float(getattr(self.config, "kl_base", 0.0) or 0.0)
            ref = None
            if kl_coef > 0.0 and bool((A < 0).any()):
                # 第一次更新前冻结参考参数: 此时还没被这批梯度改过
                if self._base_params is None:
                    self._base_params = self.policy.snapshot_params()
                ref = self._base_params
            stats: dict[str, float] = {}
            for _ in range(epochs):
                order = rng.permutation(len(X))
                for start in range(0, len(X), mb):
                    idx = order[start:start + mb]
                    with self.policy_lock:
                        stats = self.policy.imitate_batch(
                            X[idx], A[idx], weights=W[idx],
                            ref_params=ref, kl_coef=kl_coef,
                        )
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
        # 价值头学的是**缩放后**的回报 (见 _ppo_step: 目标 ret 也来自这里), 所以
        # values / bootstrap **不能**再除一次 scale。
        #
        # 历史 bug: 这里把 values 也除以 scale, 于是基准被压到 ~1/scale (≈0),
        # 优势退化成"逐步即时奖励", 策略只贪当前一步 —— 表现为"躲开了眼前的弹、
        # 却不站位输出、很快被后续弹幕打死", 正是 PPO 微调必崩的直接原因之一。
        values = np.asarray(traj.values, dtype=np.float64)
        dones = traj.dones if len(traj.dones) == len(rewards) else [False] * len(rewards)
        n = len(rewards)
        adv = np.zeros(n, dtype=np.float64)
        next_value = float(traj.bootstrap_value)
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
        expand = bool(self.config.expand_input)
        if (expand and loaded.obs_dim < self.obs_dim
                and loaded.n_actions == len(self.action_table)):
            # 零初始化扩维: 新特征必须在观测**末尾** (use_wall_feat 就是这么接的)。
            # 扩展后函数与原策略逐位相同, 所以"加上墙特征"不会先破坏已有能力。
            old = loaded.obs_dim
            loaded.widen_input(self.obs_dim)
            self.expanded_from = old
        if loaded.obs_dim != self.obs_dim or loaded.n_actions != len(self.action_table):
            hint = ""
            if loaded.obs_dim != self.obs_dim:
                hint = (f" (checkpoint={loaded.obs_dim}, 本配置={self.obs_dim}; "
                        f"若加了 use_wall_feat 请加 --expand-input, "
                        f"否则请让 use_wall_feat/危险梯度等开关与 checkpoint 一致)")
            raise ValueError("Checkpoint observation/action dimensions do not match "
                            f"training config{hint}")
        self.policy = loaded
        self.config.hidden = self.policy.hidden
        self.action_table = build_action_table(use_focus=self.config.use_focus,
                                              use_bomb=self.config.use_bomb)
        # 用**本次 run** 的超参覆盖 checkpoint 里自带的 lr / entropy。
        # 不覆盖会出现"命令行写 --lr 3e-5, 实际按 checkpoint 的 1e-3 更新"的静默 bug ——
        # 实测它让所有 PPO 微调都以 10~30 倍学习率跑, 三组"调小 lr"的实验完全没生效,
        # 结果全都把 DAgger 的策略打崩。
        self.apply_hyperparams()
        return self.policy.describe()

    def apply_hyperparams(self) -> None:
        """把当前 config 的 lr / entropy 同步到策略对象上 (load / start 时都要)."""
        self.policy.lr = float(self.config.lr)
        self.policy.entropy_coef = float(self.config.entropy_coef)

    def clear_history(self) -> None:
        with self._state_lock:
            self.episodes.clear()
            self.evaluations.clear()
            self.total_steps = 0
            self.total_frames = 0
            self.best_return = float("-inf")
