"""未来占用栅格 + 几何缺口观测.

为什么需要"未来"
----------------
60Hz 下, 决定"现在往哪走"的信息往往不在当前帧: 一堵弹幕墙 40 帧后才到自机,
但**横向占位现在就得开始**。只看当前帧, 策略只能学到"哪颗弹离我最近" -> 反应式
躲避 -> 被逼到墙角。

这里把弹幕按已知弹道 (位置 + 速度) 外推到若干未来时刻, 得到两类测量:

1. **时间分层占用栅格** (``grid``): 以自机为中心, 每一层 t 记录"t 帧后哪些格子
   有弹"。层与层之间的差异直接编码了弹幕的移动方向与缺口开合。
2. **每列最早到达时间** (``tta``, time-to-arrival): 把栅格沿自机所在高度压成一条
   一维剖面 —— 第 j 列在多少帧后会被弹幕占据。**这条剖面的最大值就是几何缺口**,
   对横向弹幕墙、环形弹、扇形弹都成立, 不需要为每种弹型写规则。

设计原则 (关键)
--------------
* **只输出测量, 不输出决策**。"缺口中心 x / 宽度 / 最早到达时间"是观测量;
  "该往左还是往右"必须由策略自己学。把决策写进特征 = 把人工控制器塞进输入,
  策略就退化成"复读机", 换个弹型立刻失效。
* **横向覆盖整个场地, 纵向以自机为原点**。场宽固定 384px, 所以横向用绝对车道;
  纵向关心的是"我上方的弹什么时候到", 用相对窗口更省维度。纯相对窗口的问题是
  一堵全屏宽的弹墙可能在窗口外留缝, 策略于是"看不见"半个屏幕的缺口。
* 全部向量化, 单帧开销亚毫秒级, 真机 60Hz 也扛得住。

用法::

    enc = GapObservation(GapSpec(use_grid=True, use_geo=True))
    obs = enc(state, prev_action=3)          # -> np.ndarray, 定长
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..types import PLAYFIELD_HEIGHT, PLAYFIELD_WIDTH, GameState

#: 8 个方向 (与 training/planner.py 的宏动作顺序一致; 第 0 个是原地)
DIRECTIONS: tuple[tuple[float, float], ...] = (
    (0.0, 0.0), (0.0, -1.0), (0.0, 1.0), (-1.0, 0.0), (1.0, 0.0),
    (-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0),
)
#: 自机速度 (px/帧, mock 与正作接近)
PLAYER_SPEED = 4.4
#: 弹幕半径的粗略上界 (px), 用于把"距离"换算成"安全余量"
BULLET_RADIUS = 3.0
HITBOX_RADIUS = 2.0


@dataclass(frozen=True)
class GapSpec:
    """观测规格 (可序列化到策略元数据里, 保证评估时编码一致)。"""

    #: 占用栅格的未来时刻 (帧)
    horizons: tuple[int, ...] = (0, 12, 26)
    #: 栅格: 横向覆盖**整个场地** (384px / cols), 纵向以自机为原点
    rows: int = 10
    cols: int = 24
    span_up: float = 300.0
    span_down: float = 56.0
    #: TTA 剖面: 全场地 ``tta_cols`` 条车道 (12px/条), 纵向带宽 tta_band
    tta_cols: int = 32
    tta_band: float = 26.0
    tta_horizon: int = 240
    tta_step: int = 3
    #: TTA 剖面的层数。1 = 只看"最早被威胁"; 2 = 再拼一层"这一列**下一堵墙**什么时候
    #: 到"。间隔小于"缺口横移所需时间"时 (v3), 必须提前为下一堵墙占位, 而最早到达
    #: 时间被前一堵墙完全遮住 —— 两层剖面是唯一能看到下一堵墙缺口的几何量。
    tta_layers: int = 1
    #: 是否拼接两类观测
    use_grid: bool = True
    use_geo: bool = True
    #: 是否拼接"8 方向 6 帧后的安全余量" (局部几何)
    use_clearance: bool = True
    #: 是否拼接上一动作 one-hot (9 维) —— 策略需要知道自己的动量
    use_prev_action: bool = True
    #: 是否拼接"连续同向决策步数" (1 维, 0~1)。给策略一个因果可见的承诺记忆:
    #: 抖动发生在两个几乎等价的缺口之间, 仅有 prev_action 不足以打破平局。
    use_action_age: bool = False
    #: 动作年龄的归一化上限 (步); 12 步 = 72 帧
    action_age_cap: int = 12


def _bullets(state: GameState) -> tuple[np.ndarray, np.ndarray]:
    """返回 (位置 Nx2, 速度 Nx2); 速度缺失时补 0。"""
    if not state.bullets:
        return np.zeros((0, 2)), np.zeros((0, 2))
    pos = np.asarray(state.bullets, dtype=np.float64).reshape(-1, 2)
    raw = (state.raw or {}).get("bullet_velocity") or []
    try:
        vel = np.asarray(raw, dtype=np.float64).reshape(-1, 2)
        if vel.shape[0] != pos.shape[0]:
            raise ValueError("length mismatch")
    except Exception:
        vel = np.zeros_like(pos)
    return pos, vel


def _grid_layers(px: float, py: float, pos: np.ndarray, vel: np.ndarray,
                 spec: GapSpec, out: np.ndarray) -> None:
    """把各未来时刻的弹幕位置写进占用栅格 (层, rows, cols)。

    横向覆盖整个场地而不是"自机附近": 一堵全屏宽的弹墙, 缺口可能在任何一侧, 只看
    ±200px 会直接看不见它 —— 那策再怎么学也躲不开。
    """
    out.fill(0.0)
    if pos.shape[0] == 0:
        return
    width = float(PLAYFIELD_WIDTH)
    height = spec.span_up + spec.span_down
    x0 = 0.0
    y0 = py - spec.span_up
    for li, t in enumerate(spec.horizons):
        xs = pos[:, 0] + vel[:, 0] * t
        ys = pos[:, 1] + vel[:, 1] * t
        ix = ((xs - x0) / width * spec.cols).astype(np.intp)
        iy = ((ys - y0) / height * spec.rows).astype(np.intp)
        ok = (ix >= 0) & (ix < spec.cols) & (iy >= 0) & (iy < spec.rows)
        if np.any(ok):
            out[li, iy[ok], ix[ok]] = 1.0


def _arrival_times(py: float, pos: np.ndarray, vel: np.ndarray,
                   spec: GapSpec, *, min_gap: float = 12.0
                   ) -> tuple[np.ndarray, np.ndarray]:
    """每条车道**前两次**被弹幕威胁的时刻 (帧), 越大越安全。

    返回 ``(t1, t2)``:

    * ``t1``: 第一次到达时间 —— 就是 TTA 剖面本身;
    * ``t2``: 第二次到达时间 (与第一次相隔至少 ``min_gap`` 帧), 用来回答"站在这里
      不动的话, 我还有多久可以挪窝"。

    只有第一次到达是不够的: 一条车道可能 30 帧后被打穿、60 帧后又空出来, 也可能
    30 帧后被打穿、下一堵墙马上又盖住它。"待得住多久"必须看第二次 —— 这就是
    ``slack`` 标签比 ``reach`` 多出来的那一维。
    """
    cols = spec.tta_cols
    centers = _column_centers(spec)
    reach = HITBOX_RADIUS + BULLET_RADIUS
    never = float(spec.tta_horizon + spec.tta_step)
    t1 = np.full(cols, never)
    t2 = np.full(cols, never)

    # 场地外 / 自机到不了的列: 直接判为"立刻被占", 不给出虚假的安全感
    reachable = (centers >= 8.0) & (centers <= PLAYFIELD_WIDTH - 8.0)
    t1[~reachable] = 0.0
    t2[~reachable] = 0.0
    if pos.shape[0] == 0:
        return t1, t2

    ts = np.arange(0, spec.tta_horizon + 1, spec.tta_step, dtype=np.float64)
    xs = pos[:, 0][None, :] + vel[:, 0][None, :] * ts[:, None]
    ys = pos[:, 1][None, :] + vel[:, 1][None, :] * ts[:, None]
    hot = (np.abs(ys - py) <= spec.tta_band) & (xs >= -reach) & \
          (xs <= PLAYFIELD_WIDTH + reach)
    if not np.any(hot):
        return t1, t2

    valid = hot.any(axis=0)
    if not np.any(valid):
        return t1, t2
    idx = np.flatnonzero(valid)
    first = np.argmax(hot, axis=0)[idx]     # 每颗弹首次进入带宽的时刻下标
    t_first = ts[first]
    x_first = xs[first, idx]
    # **按车道区间与危险区是否相交**来判定, 而不是"车道中心是否落在危险区内"。
    #
    # 这条是踩过的坑: 车道 12px、判定半径 5px, 中心只覆盖 10px。—— 一颗落在两个
    # 中心正中间 (比如 x=12, 中心在 6 和 18) 的弹, 两颗中心都离它 6px > 5px, 于是
    # **整颗弹在剖面里凭空消失**。9px 间距的墙大约每 4 颗就有 1 颗落在这种 2px 盲区
    # 里, 观测于是把"其实封死"的车道报成安全 —— 策略学到的是一个不存在的缺口。
    cell = float(PLAYFIELD_WIDTH) / max(1, cols)
    lo = np.clip(np.floor((x_first - reach) / cell).astype(np.intp), 0, cols - 1)
    hi = np.clip(np.floor((x_first + reach) / cell).astype(np.intp), 0, cols - 1)
    for i in range(idx.shape[0]):
        np.minimum(t1[lo[i]:hi[i] + 1], t_first[i], out=t1[lo[i]:hi[i] + 1])

    # 第二次到达: 展开成 (弹, 车道) 对, 过滤掉与第一次太近的, 再按车道取最小
    lengths = hi - lo + 1
    total = int(lengths.sum())
    if total:
        starts = np.repeat(lo, lengths)
        offsets = np.arange(total) - np.repeat(np.cumsum(lengths) - lengths, lengths)
        pair_cols = starts + offsets
        pair_times = np.repeat(t_first, lengths)
        mask = pair_times > t1[pair_cols] + min_gap
        if np.any(mask):
            np.minimum.at(t2, pair_cols[mask], pair_times[mask])
    return t1, t2


def _tta_profile(py: float, pos: np.ndarray, vel: np.ndarray,
                 spec: GapSpec) -> np.ndarray:
    """每列"最早被弹幕威胁"的时间 (帧), 归一化到 [0, 1], 越接近 1 越安全。

    两个容易写错的细节:

    1. **必须按命中半径判定**。判"某列安全"不能只看弹是不是正好落在这一列,
       而要问"自机站在这列时会不会被擦到" —— 即 ``|x_bullet - x_col| <= 命中半径``。
       少了这一步, 缺口宽度会被高估 (少算 2*5px), 策略会去走实际走不通的窄缝。
    2. **场地外的列不是安全区**。自机被夹在 [8, 376] 内, 那些永远不会有弹的
       "场地外列"如果按 tta=∞ 处理, 会被误判成最安全的缺口, 策略于是学会贴边。
       贴边恰恰是被弹的经典死法。

    单调性同样重要: ``1/(1+t)`` 是递减的, 会把缺口变成最低点, 语义正好相反。
    """
    t1, _t2 = _arrival_times(py, pos, vel, spec)
    return np.clip(t1 / float(max(1, spec.tta_horizon)), 0.0, 1.0)


def _tta_profiles(py: float, pos: np.ndarray, vel: np.ndarray,
                  spec: GapSpec) -> np.ndarray:
    """多层 TTA 剖面, 形状 ``(layers*tta_cols,)``, 每层归一化到 [0, 1]。

    第 2 层是"这一列**第二堵墙**什么时候到"。为什么需要它: 多层到达时间取 min 之后,
    后一堵墙的信息被前一堵墙彻底遮住 —— 间隔 75 帧、缺口横移 324px 时, 自机必须
    在**上一堵墙还没落下**的时候就开始为下一堵墙挪位, 而只看第 1 层的剖面在那段
    时间里是"整片都被封死"的同一个图案, 策略无从判断该往哪边准备。

    第 2 层里"值高"的车道 = 前一堵墙打穿它、但下一堵墙留了缺口 —— 也就是下一堵墙
    的缺口位置。这正是 v3 那种大漂移场景唯一需要的先见量。

    目前最多 2 层 (``_arrival_times`` 只给前两次到达), 更多层没有对应的物理量。
    """
    cols = spec.tta_cols
    layers = max(1, int(spec.tta_layers))
    horizon = float(max(1, spec.tta_horizon))
    t1, t2 = _arrival_times(py, pos, vel, spec)
    if layers == 1:
        return np.clip(t1 / horizon, 0.0, 1.0)
    return np.clip(np.concatenate([t1, t2]) / horizon, 0.0, 1.0)[: layers * cols]


def _column_centers(spec: GapSpec) -> np.ndarray:
    """每条车道的中心 x (场地绝对坐标, 与自机位置无关)。"""
    cell = float(PLAYFIELD_WIDTH) / spec.tta_cols
    return (np.arange(spec.tta_cols) + 0.5) * cell


def _best_column(profile: np.ndarray, px: float, spec: GapSpec) -> int:
    """剖面最大值中**离自机最近**的那条车道。

    不能直接用 ``np.argmax``: "整片都安全"时所有车道都被截到 1.0, argmax 会返回
    第 0 条, 于是"缺口"永远在最左边 —— 自机于是持续贴左墙, 而这恰恰是最容易被弹的
    死法。改成在并列最优里挑最近的车道, 语义变成"没必要动就别动"。
    """
    centers = _column_centers(spec)
    top = float(profile.max()) if profile.size else 0.0
    tied = np.flatnonzero(profile >= top - 1e-9)
    if not tied.size:
        return profile.size // 2
    return int(tied[np.argmin(np.abs(centers[tied] - px))])


def _gap_features(profile: np.ndarray, px: float, spec: GapSpec) -> np.ndarray:
    """从 TTA 剖面提取"缺口"的几何量 (4 维, 全是测量)。

    * 缺口中心的横向位置 (相对自机, 用 sin/cos 避免 ±π 跳变)
    * 缺口的安全余量 (归一化到 [0,1])
    * 缺口宽度 (占屏宽比例)

    注意这里的中心是**相对位置**: 传进来的 profile 以自机为中心, 所以中点是自机
    自己。整片安全时中心应当落在 0 附近 (无缺口可追), 而不是某个远端列。
    """
    n = profile.shape[0]
    k = _best_column(profile, px, spec)
    # 相邻的"较安全"连续段宽度
    thresh = max(0.55 * float(profile[k]), 0.02)
    width = 1
    for direction in (-1, 1):
        j = k
        while 0 <= j + direction < n and profile[j + direction] >= thresh:
            j += direction
            width += 1
    dx = (_column_centers(spec)[k] - px) / (PLAYFIELD_WIDTH / 2.0)   # 约 [-1, 1]
    dx = float(max(-1.0, min(1.0, dx)))
    return np.array([
        math.sin(dx * math.pi / 2.0),                  # 缺口在左/右 (平滑, 无跳变)
        1.0 - abs(dx),                                 # 缺口在正前方的程度
        float(profile[k]),
        width / n,
    ], dtype=np.float64)


def _clearances(px: float, py: float, pos: np.ndarray, vel: np.ndarray,
                frames: int = 6) -> np.ndarray:
    """8 个方向走 ``frames`` 帧后的安全余量 (越小越危险)。"""
    if pos.shape[0] == 0:
        return np.ones(len(DIRECTIONS), dtype=np.float64)
    xs = pos[:, 0] + vel[:, 0] * frames
    ys = pos[:, 1] + vel[:, 1] * frames
    out = np.empty(len(DIRECTIONS), dtype=np.float64)
    reach = HITBOX_RADIUS + BULLET_RADIUS
    for i, (ux, uy) in enumerate(DIRECTIONS):
        n = math.hypot(ux, uy)
        dx, dy = (ux / n, uy / n) if n else (0.0, 0.0)
        nx = min(PLAYFIELD_WIDTH - 8.0, max(8.0, px + dx * PLAYER_SPEED * frames))
        ny = min(PLAYFIELD_HEIGHT - 16.0, max(32.0, py + dy * PLAYER_SPEED * frames))
        d = np.hypot(xs - nx, ys - ny).min()
        out[i] = min(1.0, max(0.0, (d - reach) / 40.0))
    return out


def _globals(state: GameState, px: float, py: float) -> np.ndarray:
    return np.array([
        px / PLAYFIELD_WIDTH,
        py / PLAYFIELD_HEIGHT,
        min(1.0, (px - 8.0) / 60.0),
        min(1.0, (PLAYFIELD_WIDTH - 8.0 - px) / 60.0),
        min(1.0, (py - 32.0) / 60.0),
        min(1.0, (PLAYFIELD_HEIGHT - 16.0 - py) / 60.0),
        min(1.0, (state.lives or 0) / 5.0),
    ], dtype=np.float64)


def gap_observation_dim(spec: GapSpec) -> int:
    dim = 0
    if spec.use_grid:
        dim += len(spec.horizons) * spec.rows * spec.cols
    if spec.use_geo:
        dim += spec.tta_cols * max(1, int(spec.tta_layers)) + 4
    if spec.use_clearance:
        dim += len(DIRECTIONS) * 2          # 6 帧 + 12 帧
    dim += _globals(GameState(), 0.0, 0.0).shape[0]
    if spec.use_prev_action:
        dim += len(DIRECTIONS)
    if spec.use_action_age:
        dim += 1
    return int(dim)


class GapObservation:
    """把 :class:`GameState` 编码成"未来占用栅格 + 几何缺口"向量。

    每次调用**都返回一块新数组** (除非显式传 ``out``)。

    这里曾经复用内部缓冲区, 结果 ``rollout`` 里 ``obs_list.append(obs)`` 拿到的是同一个
    对象 —— 整条轨迹堆叠出来全是最后一帧的观测。BC 因为显式 ``.copy()`` 躲过了,
    RL 却一直在"用终局状态重复 N 次"更新策略, 这足以单独解释"RL 一微调就崩"。
    亚毫秒的分配换一个不会静默出错的契约, 值得。
    """

    def __init__(self, spec: GapSpec | None = None) -> None:
        self.spec = spec or GapSpec()
        self.dim = gap_observation_dim(self.spec)

    # ------------------------------------------------------------------
    def __call__(self, state: GameState, prev_action: int = -1,
                 out: np.ndarray | None = None, action_age: int = 0) -> np.ndarray:
        spec = self.spec
        vec = np.empty(self.dim, dtype=np.float64) if out is None else out
        px = float(state.player_x) if state.player_x is not None else PLAYFIELD_WIDTH / 2
        py = float(state.player_y) if state.player_y is not None else PLAYFIELD_HEIGHT - 32
        pos, vel = _bullets(state)
        off = 0
        if spec.use_grid:
            cells = len(spec.horizons) * spec.rows * spec.cols
            grid = vec[off:off + cells].reshape(len(spec.horizons), spec.rows, spec.cols)
            _grid_layers(px, py, pos, vel, spec, grid)
            off += cells
        if spec.use_geo:
            cols = spec.tta_cols
            layers = max(1, int(spec.tta_layers))
            profiles = _tta_profiles(py, pos, vel, spec)
            vec[off:off + cols * layers] = profiles
            profile = profiles[:cols]          # 几何特征只看第 1 层 (眼前这堵墙)
            off += cols * layers
            vec[off:off + 4] = _gap_features(profile, px, spec)
            off += 4
        if spec.use_clearance:
            c6 = _clearances(px, py, pos, vel, 6)
            c12 = _clearances(px, py, pos, vel, 12)
            vec[off:off + 9] = c6
            vec[off + 9:off + 18] = c12
            off += 18
        g = _globals(state, px, py)
        vec[off:off + g.shape[0]] = g
        off += g.shape[0]
        if spec.use_prev_action:
            vec[off:off + 9] = 0.0
            if 0 <= prev_action < 9:
                vec[off + prev_action] = 1.0
            off += 9
        if spec.use_action_age:
            cap = max(1, int(spec.action_age_cap))
            vec[off] = min(1.0, max(0.0, float(action_age)) / cap)
            off += 1
        return vec

    # ------------------------------------------------------------------
    def describe(self) -> dict[str, float | int | bool | list[int]]:
        s = self.spec
        return {
            "dim": self.dim,
            "horizons": list(s.horizons),
            "grid": [s.rows, s.cols] if s.use_grid else [],
            "tta_cols": s.tta_cols if s.use_geo else 0,
            "use_grid": s.use_grid,
            "use_geo": s.use_geo,
            "use_clearance": s.use_clearance,
            "use_action_age": s.use_action_age,
            "action_age_cap": s.action_age_cap,
        }


def _column_index(x: float, spec: GapSpec) -> int:
    """横坐标 -> 车道下标 (与 :func:`_column_centers` 同一套切分)。"""
    cell = float(PLAYFIELD_WIDTH) / max(1, spec.tta_cols)
    return int(min(spec.tta_cols - 1, max(0, x / cell)))


def lane_safety(state: GameState, spec: GapSpec | None = None) -> float:
    """自机**当前所在车道**的安全余量, 归一化到 [0, 1] (1 = 很久以后才被威胁)。

    这是纯测量量 ("我站的地方还撑多久"), 不含"该往哪走"。作为**势函数**做奖励塑形
    ``r += β·(φ' − φ)``, 既不改变最优策略, 又给了"站着不动"一个密集的负梯度。

    注意这里必须按**自机所在车道**取值。之前实验里写成 ``profile[tta_cols // 2]``,
    那是场地正中间 (x=192) 的车道, 不是自机 —— 于是塑形项一直在奖励"回到屏幕中心",
    和"别贴墙角"的半条规矩正好打架。
    """
    spec = spec or GapSpec()
    pos, vel = _bullets(state)
    py = float(state.player_y) if state.player_y is not None else PLAYFIELD_HEIGHT - 32
    px = float(state.player_x) if state.player_x is not None else PLAYFIELD_WIDTH / 2
    profile = _tta_profile(py, pos, vel, spec)
    if profile.size == 0:
        return 0.0
    return float(profile[_column_index(px, spec)])


@dataclass
class GapProbe:
    """诊断用: 打印某一帧的 TTA 剖面与识别到的缺口 (不参与训练)。"""

    spec: GapSpec = field(default_factory=GapSpec)

    def profile(self, state: GameState) -> np.ndarray:
        pos, vel = _bullets(state)
        py = float(state.player_y or PLAYFIELD_HEIGHT - 32)
        return _tta_profile(py, pos, vel, self.spec)
