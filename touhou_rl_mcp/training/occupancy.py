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
* **以自机为中心**, 保证平移不变性 (换个站位仍然有效)。
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
    rows: int = 10
    cols: int = 13
    #: 栅格覆盖范围: 横向 ±span_x, 纵向上方 span_up / 下方 span_down
    span_x: float = 208.0
    span_up: float = 300.0
    span_down: float = 56.0
    #: 每列最早到达时间剖面的列数与纵向带宽
    tta_cols: int = 24
    tta_band: float = 26.0
    tta_horizon: int = 240
    tta_step: int = 3
    #: 是否拼接两类观测
    use_grid: bool = True
    use_geo: bool = True
    #: 是否拼接"8 方向 6 帧后的安全余量" (局部几何)
    use_clearance: bool = True
    #: 是否拼接上一动作 one-hot (9 维) —— 策略需要知道自己的动量
    use_prev_action: bool = True


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
    """把各未来时刻的弹幕位置写进占用栅格 (层, rows, cols)。"""
    out.fill(0.0)
    if pos.shape[0] == 0:
        return
    width = 2.0 * spec.span_x
    height = spec.span_up + spec.span_down
    x0 = px - spec.span_x
    y0 = py - spec.span_up
    for li, t in enumerate(spec.horizons):
        xs = pos[:, 0] + vel[:, 0] * t
        ys = pos[:, 1] + vel[:, 1] * t
        ix = ((xs - x0) / width * spec.cols).astype(np.intp)
        iy = ((ys - y0) / height * spec.rows).astype(np.intp)
        ok = (ix >= 0) & (ix < spec.cols) & (iy >= 0) & (iy < spec.rows)
        if np.any(ok):
            out[li, iy[ok], ix[ok]] = 1.0


def _tta_profile(px: float, py: float, pos: np.ndarray, vel: np.ndarray,
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
    cols = spec.tta_cols
    width = 2.0 * spec.span_x
    cell = width / cols
    x0 = px - spec.span_x
    centers = x0 + (np.arange(cols) + 0.5) * cell
    reach = HITBOX_RADIUS + BULLET_RADIUS
    never = float(spec.tta_horizon + spec.tta_step)
    tta = np.full(cols, never)

    # 场地外 / 自机到不了的列: 直接判为"立刻被占", 不给出虚假的安全感
    reachable = (centers >= 8.0) & (centers <= PLAYFIELD_WIDTH - 8.0)
    tta[~reachable] = 0.0

    if pos.shape[0] == 0:
        return np.clip(tta / spec.tta_horizon, 0.0, 1.0)
    ts = np.arange(0, spec.tta_horizon + 1, spec.tta_step, dtype=np.float64)
    xs = pos[:, 0][None, :] + vel[:, 0][None, :] * ts[:, None]
    ys = pos[:, 1][None, :] + vel[:, 1][None, :] * ts[:, None]
    hot = (np.abs(ys - py) <= spec.tta_band) & (np.abs(xs - px) <= spec.span_x + reach)
    if not np.any(hot):
        return np.clip(tta / spec.tta_horizon, 0.0, 1.0)
    first = np.argmax(hot, axis=0)              # 每颗弹首次进入带宽的时刻下标
    valid = hot.any(axis=0)
    if np.any(valid):
        t_first = ts[first[valid]]
        x_first = xs[first[valid], np.flatnonzero(valid)]
        lo = np.searchsorted(centers, x_first - reach, side="left")
        hi = np.searchsorted(centers, x_first + reach, side="right")
        for i in range(t_first.shape[0]):
            if hi[i] > lo[i]:
                np.minimum(tta[lo[i]:hi[i]], t_first[i], out=tta[lo[i]:hi[i]])
    return np.clip(tta / spec.tta_horizon, 0.0, 1.0)


def _gap_features(profile: np.ndarray) -> np.ndarray:
    """从 TTA 剖面提取"缺口"的几何量 (4 维, 全是测量)。

    * 缺口中心的横向位置 (相对自机, 用 sin/cos 避免 ±π 跳变)
    * 缺口的安全余量 (归一化到 [0,1])
    * 缺口宽度 (占屏宽比例)
    """
    n = profile.shape[0]
    k = int(np.argmax(profile))
    # 环形相邻的"较安全"连续段宽度
    thresh = max(0.55 * float(profile[k]), 0.02)
    width = 1
    for direction in (-1, 1):
        j = k
        while 0 <= j + direction < n and profile[j + direction] >= thresh:
            j += direction
            width += 1
    rel = (k + 0.5) / n - 0.5                       # [-0.5, 0.5)
    ang = rel * 2.0 * math.pi
    return np.array([
        math.sin(ang), math.cos(ang),
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
        dim += spec.tta_cols + 4
    if spec.use_clearance:
        dim += len(DIRECTIONS) * 2          # 6 帧 + 12 帧
    dim += _globals(GameState(), 0.0, 0.0).shape[0]
    if spec.use_prev_action:
        dim += len(DIRECTIONS)
    return int(dim)


class GapObservation:
    """把 :class:`GameState` 编码成"未来占用栅格 + 几何缺口"向量。"""

    def __init__(self, spec: GapSpec | None = None) -> None:
        self.spec = spec or GapSpec()
        self.dim = gap_observation_dim(self.spec)
        self._buf = np.empty(self.dim, dtype=np.float64)

    # ------------------------------------------------------------------
    def __call__(self, state: GameState, prev_action: int = -1,
                 out: np.ndarray | None = None) -> np.ndarray:
        spec = self.spec
        vec = self._buf if out is None else out
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
            profile = _tta_profile(px, py, pos, vel, spec)
            vec[off:off + spec.tta_cols] = profile
            off += spec.tta_cols
            vec[off:off + 4] = _gap_features(profile)
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
        }


@dataclass
class GapProbe:
    """诊断用: 打印某一帧的 TTA 剖面与识别到的缺口 (不参与训练)。"""

    spec: GapSpec = field(default_factory=GapSpec)

    def profile(self, state: GameState) -> np.ndarray:
        pos, vel = _bullets(state)
        px = float(state.player_x or PLAYFIELD_WIDTH / 2)
        py = float(state.player_y or PLAYFIELD_HEIGHT - 32)
        return _tta_profile(px, py, pos, vel, self.spec)
