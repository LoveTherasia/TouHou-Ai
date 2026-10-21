"""状态 -> 特征向量.

RL 智能体的输入。设计原则:
  * **以自机为中心**: 弹幕只提供"相对自机的位移", 这样策略天然具有平移不变性,
    换个位置也能用, 学得快得多。
  * **按距离取最近的 K 个弹**, 并按距离排序 —— 对策略来说, "最近的那几颗弹在哪"
    比"第 17 颗弹在哪"重要得多。
  * 维度固定 (不足补 0 并给 valid=0), 方便接 MLP / SB3。

真实后端必须有有效自机和实体观测，否则明确报错，不再静默填零训练。
敌人位置由状态接口提供；74 维策略只编码敌人数量。

三种观测口径 (都由 ``config`` 决定, 训练器与所有工具统一走
:func:`encode_observation`, 不要各自手拼参数):

* ``use_gap=True``  —— 未来占用栅格 + 几何缺口剖面
  (:mod:`~touhou_rl_mcp.training.occupancy`)。密集弹幕下唯一能让策略
  "提前横向占位"的表示, 需要弹速。
* ``use_grid=True`` —— 以自机为中心的危险图栅格 (静态, 不推演未来)。
* 默认 —— 最近 ``k_bullets`` 颗弹的 (dx, dy, dist, valid) 一维向量。
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..types import PLAYFIELD_HEIGHT, PLAYFIELD_WIDTH, GameState
from ..observation import require_entities
from .occupancy import GapObservation, GapSpec, gap_observation_dim

#: 每个弹幕占的特征数: (dx, dy, 距离, 是否有效)
BULLET_FEATURE_DIM = 4
#: 每个弹幕额外增加的"速度"特征数: (vx, vy), 单位 像素/游戏帧
BULLET_VELOCITY_DIM = 2
#: 每架敌机占的特征数: (dx, dy, 距离, 是否有效)
ENEMY_FEATURE_DIM = 4
#: 自机/全局状态占的特征数
GLOBAL_FEATURE_DIM = 10
#: 速度归一化尺度 (弹速约 1~6 px/帧)
VELOCITY_SCALE = 8.0
#: "墙特征"的维度: 覆盖率 / 缺口中心 / 缺口宽度 / 到达帧数 + 8 个列带占用
WALL_FEATURE_DIM = 12
#: 墙特征的列带数 (384px / 8 = 48px 一带, 比判别器的 16 列粗, 对 MLP 更好学)
WALL_BANDS = 8
#: 墙特征的前瞻帧数 (与规划器视野一致: 180 帧后才压到自机的墙不算)
WALL_LOOKAHEAD = 180.0
#: 竖直方向多近算"同一排" (px)
WALL_ROW_BAND = 26.0


def bullet_feature_dim(use_velocity: bool = False) -> int:
    return BULLET_FEATURE_DIM + (BULLET_VELOCITY_DIM if use_velocity else 0)


def global_feature_dim(use_invincible: bool = False, use_boss: bool = False,
                       use_danger_grad: bool = False) -> int:
    """全局特征维度 (可选追加 无敌帧 1 维 / Boss 位置 2 维 / 危险梯度 2 维)。"""
    return (GLOBAL_FEATURE_DIM + (1 if use_invincible else 0)
            + (2 if use_boss else 0) + (2 if use_danger_grad else 0))


def feature_dim(k_bullets: int = 16, use_velocity: bool = False, k_enemies: int = 0,
                use_invincible: bool = False, use_boss: bool = False,
                use_danger_grad: bool = False, use_wall_feat: bool = False) -> int:
    if k_bullets < 0 or k_enemies < 0:
        raise ValueError("k_bullets/k_enemies must be nonnegative")
    return (global_feature_dim(use_invincible, use_boss, use_danger_grad)
            + k_bullets * bullet_feature_dim(use_velocity) + k_enemies * ENEMY_FEATURE_DIM
            + (WALL_FEATURE_DIM if use_wall_feat else 0))


#: 危险梯度感知半径 (与规则老师的斥力核一致)
DANGER_GRAD_RADIUS = 30.0
DANGER_GRAD_SCALE = 3.0


def _danger_grad(state: GameState, px: float, py: float,
                 radius: float = DANGER_GRAD_RADIUS) -> tuple[float, float]:
    """最近弹幕的"斥力合成"向量 (物理启发的危险梯度)。

    这正是规则老师躲弹幕时最核心的量: 把半径内每颗弹的斥力按 1/距离 加权求和。
    作为特征给策略, 可以让"反应式躲避"这种函数更容易被 MLP 表示。
    """
    fx = fy = 0.0
    for bx, by in state.bullets:
        dx, dy = px - bx, py - by
        d2 = dx * dx + dy * dy
        if d2 > radius * radius:
            continue
        d = math.sqrt(d2) or 1.0
        w = (radius - d) / radius / d
        fx += dx * w
        fy += dy * w
    return fx, fy


def _boss_xy(state: GameState) -> tuple[float | None, float | None]:
    pos = (state.raw or {}).get("boss_position")
    if isinstance(pos, (list, tuple)) and len(pos) >= 2:
        try:
            return float(pos[0]), float(pos[1])
        except (TypeError, ValueError):
            return None, None
    return None, None


def _wall_row(state: GameState, px: float, py: float) -> dict[str, Any] | None:
    """找"最近一堵正在下压的弹幕墙"的几何 (连续口径, 不设开关阈值)。

    与 :class:`~touhou_rl_mcp.training.hierarchical.WallDetector` 的区别: 判别器
    要回答"切不切", 所以必须有硬阈值; 这里是**给策略看的连续量**, 不设阈值 ——
    边界让策略自己学。

    做法: 只看"匀速下落"的弹 (``vy >= 0.7`` 且 ``|vx| <= 0.9``), 取**最快到达自机**
    的那一颗所在的横带 (``|Δy| <= 13``), 在带内算横向占用。
    """
    if not state.bullets:
        return None
    pos = np.asarray(state.bullets, dtype=np.float64).reshape(-1, 2)
    raw = (state.raw or {}).get("bullet_velocity") or []
    try:
        vel = np.asarray(raw, dtype=np.float64).reshape(-1, 2)
        if vel.shape[0] != pos.shape[0]:
            raise ValueError
    except (TypeError, ValueError):
        return None
    falling = (vel[:, 1] >= 0.7) & (np.abs(vel[:, 0]) <= 0.9)
    if not falling.any():
        return None
    idx = np.flatnonzero(falling)
    eta = (py - pos[idx, 1]) / np.maximum(vel[idx, 1], 1e-6)
    near = (eta <= WALL_LOOKAHEAD) & (eta >= -12.0)
    idx, eta = idx[near], eta[near]
    if len(idx) < 4:
        return None
    head = int(np.argmin(eta))
    y0 = float(pos[idx[head], 1])
    row = idx[np.abs(pos[idx, 1] - y0) <= WALL_ROW_BAND / 2.0]
    if len(row) < 4:
        return None

    xs = pos[row, 0]
    band = np.clip((xs / PLAYFIELD_WIDTH * WALL_BANDS).astype(np.int64), 0, WALL_BANDS - 1)
    occ = np.zeros(WALL_BANDS, dtype=np.float64)
    occ[np.unique(band)] = 1.0
    covered = float(occ.mean())

    # 最大连续空带 = 缺口; 全占满时缺口宽度记 0
    best_len = best_start = 0
    run = 0
    for i in range(WALL_BANDS):
        if occ[i] < 0.5:
            run += 1
            if run > best_len:
                best_len, best_start = run, i - run + 1
        else:
            run = 0
    band_w = PLAYFIELD_WIDTH / WALL_BANDS
    gap_w = best_len * band_w
    gap_cx = (best_start + best_len / 2.0) * band_w if best_len else px
    return {
        "covered": covered,
        "occ": occ,
        "gap_cx": float(gap_cx),
        "gap_w": float(gap_w),
        "eta": float(eta[head]),
        "count": int(len(row)),
    }


def wall_features(state: GameState, px: float | None = None,
                  py: float | None = None) -> np.ndarray:
    """12 维"墙特征", **追加在观测向量末尾** (见 :data:`WALL_FEATURE_DIM`)。

    为什么需要它: 143 维观测里只有"最近 16 颗弹", 覆盖不到 ±70px 以外 ——
    一堵 384px 宽、36 颗弹的墙在观测里只表现为"身边有十几颗朝下飞的弹",
    **缺口在哪完全不可见**。所以无论怎么在墙段做监督, 策略都只能学到"别乱动"
    (实测 `wall_gate` 552~657 步, 比基线 814 更差)。

    这 12 维是"把看不见的东西写进观测":

    ====== ====================================================
    0      这一排的横向覆盖率 (0~1): 越高越"封死"
    1      缺口中心相对自机的横向偏移 / 场地宽 (-1~1)
    2      缺口宽度 / 场地宽
    3      到达自机高度的帧数 / 180 (1 = 立刻到)
    4..11  8 个列带 (48px) 的占用 (0/1)
    ====== ====================================================

    没有墙时全 0。**必须放在末尾**: checkpoint 的输入层可以零初始化扩维,
    新列权重为 0 => 旧策略的函数完全不变, 再从零学这几列。
    """
    px = float(px) if px is not None else (
        float(state.player_x) if state.player_x is not None else PLAYFIELD_WIDTH / 2)
    py = float(py) if py is not None else (
        float(state.player_y) if state.player_y is not None else PLAYFIELD_HEIGHT - 32)
    out = np.zeros(WALL_FEATURE_DIM, dtype=np.float64)
    row = _wall_row(state, px, py)
    if row is None:
        return out
    out[0] = row["covered"]
    out[1] = float(np.clip((row["gap_cx"] - px) / PLAYFIELD_WIDTH, -1.0, 1.0))
    out[2] = row["gap_w"] / PLAYFIELD_WIDTH
    out[3] = float(np.clip(row["eta"] / WALL_LOOKAHEAD, 0.0, 1.0))
    out[4:] = row["occ"]
    return out


def encode_state(
    state: GameState,
    *,
    k_bullets: int = 16,
    use_velocity: bool = False,
    k_enemies: int = 0,
    use_invincible: bool = False,
    use_boss: bool = False,
    use_danger_grad: bool = False,
    use_wall_feat: bool = False,
    out: np.ndarray | None = None,
    validate: bool = True,
) -> np.ndarray:
    """把一帧状态编码成定长向量 (dtype=float64)。

    ``use_velocity=True`` 时每颗弹多给 (vx, vy) —— 来自 ``state.raw["bullet_velocity"]``
    (由 :class:`~touhou_rl_mcp.memory.bullettrack.BulletTracker` 按槽位跨帧差分得到)。
    ``k_enemies>0`` 时追加**最近的 k 架敌机**的相对位移 (dx, dy, dist, valid) ——
    老师的瞄准/站位决策依赖敌机位置, 不给策略就学不出来。
    """
    if validate:
        require_entities(state)
    per_bullet = bullet_feature_dim(use_velocity)
    bullet_block = k_bullets * per_bullet
    gdim = global_feature_dim(use_invincible, use_boss, use_danger_grad)
    px = float(state.player_x) if state.player_x is not None else PLAYFIELD_WIDTH / 2
    py = float(state.player_y) if state.player_y is not None else PLAYFIELD_HEIGHT - 32

    boss_ratio = 0.0
    if state.boss_hp is not None and state.boss_max_hp:
        boss_ratio = max(0.0, min(1.0, state.boss_hp / max(1.0, state.boss_max_hp)))

    dim = feature_dim(k_bullets, use_velocity, k_enemies, use_invincible, use_boss,
                      use_danger_grad, use_wall_feat)
    vec = np.empty(dim, dtype=np.float64) if out is None else out
    vec[:GLOBAL_FEATURE_DIM] = (
        px / PLAYFIELD_WIDTH,                                   # 自机 x
        py / PLAYFIELD_HEIGHT,                                   # 自机 y
        min(1.0, (state.stage or 1) / 6.0),                      # 关卡进度
        min(1.0, (state.lives or 0) / 9.0),                      # 残机
        min(1.0, (state.power or 0.0) / 4.0),                    # 灵力
        min(1.0, (state.graze or 0) / 2000.0),                   # 擦弹
        boss_ratio,                                              # Boss 血量比
        min(1.0, (state.bullet_count or len(state.bullets)) / 150.0),  # 弹幕密度
        1.0 if state.in_game else 0.0,                           # 是否在局内
        min(1.0, (state.enemies_alive or 0) / 10.0),             # 敌机数量
    )
    if use_invincible:
        # 无敌帧 (归一化到 150 帧): 让策略知道"现在打也没用/可以安全调整站位"
        inv = state.invincible
        vec[GLOBAL_FEATURE_DIM] = (
            0.0 if inv is None else min(1.0, max(0.0, float(inv) / 150.0))
        )
    if use_boss:
        # Boss 相对位置 (会移动的 Boss 必须给位置, 否则策略无法对准输出)
        off = GLOBAL_FEATURE_DIM + (1 if use_invincible else 0)
        bx, by = _boss_xy(state)
        vec[off] = 0.0 if bx is None else (bx - px) / PLAYFIELD_WIDTH
        vec[off + 1] = 0.0 if by is None else (by - py) / PLAYFIELD_HEIGHT
    if use_danger_grad:
        # 危险梯度 (最近弹幕的斥力合成): 让"反应式躲避"更容易被策略表示
        off = GLOBAL_FEATURE_DIM + (1 if use_invincible else 0) + (2 if use_boss else 0)
        gx, gy = _danger_grad(state, px, py)
        vec[off] = float(np.clip(gx / DANGER_GRAD_SCALE, -1.0, 1.0))
        vec[off + 1] = float(np.clip(gy / DANGER_GRAD_SCALE, -1.0, 1.0))

    base = gdim
    vec[base:] = 0.0
    bullets = state.bullets
    if bullets and k_bullets:
        arr = np.asarray(bullets, dtype=np.float64).reshape(-1, 2)
        dx = arr[:, 0] - px
        dy = arr[:, 1] - py
        dist2 = dx * dx + dy * dy
        count = min(k_bullets, dx.shape[0])
        # 取最近的 count 个 (argpartition 比 argsort 快很多)
        if dx.shape[0] > count:
            idx = np.argpartition(dist2, count - 1)[:count]
            idx = idx[np.argsort(dist2[idx])]
        else:
            idx = np.argsort(dist2)
        dxs = dx[idx]
        dys = dy[idx]
        dist = np.sqrt(dxs * dxs + dys * dys)
        block = vec[base : base + count * per_bullet].reshape(count, per_bullet)
        block[:, 0] = dxs / PLAYFIELD_WIDTH
        block[:, 1] = dys / PLAYFIELD_HEIGHT
        block[:, 2] = dist / 320.0
        block[:, 3] = 1.0
        if use_velocity:
            velocities = (state.raw or {}).get("bullet_velocity") or []
            try:
                vel = np.asarray(velocities, dtype=np.float64).reshape(-1, 2)
                if vel.shape[0] != arr.shape[0]:
                    raise ValueError("velocity/bullet count mismatch")
                block[:, 4] = vel[idx, 0] / VELOCITY_SCALE
                block[:, 5] = vel[idx, 1] / VELOCITY_SCALE
            except Exception:
                block[:, 4] = 0.0
                block[:, 5] = 0.0

    # 敌机: 追加最近 k_enemies 架 (dx, dy, dist, valid) —— 老师的瞄准/站位需要它
    if k_enemies:
        ebase = base + bullet_block
        enemies = state.enemy_positions
        if enemies:
            earr = np.asarray(enemies, dtype=np.float64).reshape(-1, 2)
            edx = earr[:, 0] - px
            edy = earr[:, 1] - py
            ed2 = edx * edx + edy * edy
            ecount = min(k_enemies, edx.shape[0])
            if edx.shape[0] > ecount:
                eidx = np.argpartition(ed2, ecount - 1)[:ecount]
                eidx = eidx[np.argsort(ed2[eidx])]
            else:
                eidx = np.argsort(ed2)
            eblock = vec[ebase : ebase + ecount * ENEMY_FEATURE_DIM].reshape(ecount, ENEMY_FEATURE_DIM)
            eblock[:, 0] = edx[eidx] / PLAYFIELD_WIDTH
            eblock[:, 1] = edy[eidx] / PLAYFIELD_HEIGHT
            eblock[:, 2] = np.sqrt(ed2[eidx]) / 320.0
            eblock[:, 3] = 1.0

    # 墙特征: **末尾** 12 维 (checkpoint 零初始化扩维依赖这个位置)
    if use_wall_feat:
        vec[dim - WALL_FEATURE_DIM:] = wall_features(state, px, py)
    return vec


def encode_frames(
    states: list[GameState],
    *,
    k_bullets: int = 16,
    use_velocity: bool = False,
    k_enemies: int = 0,
    use_invincible: bool = False,
    use_boss: bool = False,
    use_danger_grad: bool = False,
    use_wall_feat: bool = False,
) -> np.ndarray:
    out = np.empty((len(states), feature_dim(k_bullets, use_velocity, k_enemies,
                                             use_invincible, use_boss, use_danger_grad,
                                             use_wall_feat)),
                   dtype=np.float64)
    for i, state in enumerate(states):
        encode_state(state, k_bullets=k_bullets, use_velocity=use_velocity,
                     k_enemies=k_enemies, use_invincible=use_invincible,
                     use_boss=use_boss, use_danger_grad=use_danger_grad,
                     use_wall_feat=use_wall_feat, out=out[i])
    return out


# ---------------------------------------------------------------------------
# 空间表示: "危险图" (以自机为中心的弹幕位置栅格)
# ---------------------------------------------------------------------------
# 与"最近 K 颗弹的一维向量"不同, 危险图保留了**弹幕的空间分布**:
# 把自机周围 ±span 像素的区域切成 size×size 的格子, 每格记录
#   * 弹幕密度 (双线性 splat, 再饱和压缩)
#   * 平均速度 (vx, vy) —— 能看出"这团弹往哪走"
# 这样策略看到的是一张"局部地图", 而不是被排序截断的若干标量。

#: 默认栅格边长 (格数)
GRID_DEFAULT_SIZE = 16
#: 默认栅格半径 (像素): 覆盖自机周围 ±span
GRID_DEFAULT_SPAN = 110.0


def grid_channels(use_velocity: bool = False) -> int:
    """每格的通道数: 密度 1 + (可选) 速度 2。"""
    return 1 + (BULLET_VELOCITY_DIM if use_velocity else 0)


def grid_feature_dim(size: int = GRID_DEFAULT_SIZE, use_velocity: bool = False) -> int:
    if size < 2:
        raise ValueError("grid size must be >= 2")
    return size * size * grid_channels(use_velocity)


def grid_scales_from(spec: str | None, *, default_size: int = GRID_DEFAULT_SIZE,
                     default_span: float = GRID_DEFAULT_SPAN) -> list[tuple[int, float]]:
    """解析多尺度网格规格 ``"12:90,12:330"`` (= 近场 12x12/±90px + 远场 12x12/±330px)。

    空串 -> 退化成单尺度 ``[(default_size, default_span)]``, 保持向后兼容。
    """
    scales: list[tuple[int, float]] = []
    for part in str(spec or "").replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        size_s, _, span_s = part.partition(":")
        size = int(size_s)
        if size < 2:
            raise ValueError(f"grid size must be >= 2: {part!r}")
        scales.append((size, float(span_s)))
    return scales or [(int(default_size), float(default_span))]


def grid_feature_dim_scales(scales: list[tuple[int, float]],
                            use_velocity: bool = False) -> int:
    """多尺度栅格的总维度 (各尺度相加)。"""
    channels = grid_channels(use_velocity)
    return sum(int(size) * int(size) * channels for size, _ in scales)


def grid_observation_dim(size: int = GRID_DEFAULT_SIZE, use_velocity: bool = False,
                         use_invincible: bool = False, use_boss: bool = False,
                         use_danger_grad: bool = False,
                         scales: list[tuple[int, float]] | None = None,
                         use_wall_feat: bool = False) -> int:
    """危险图观测维度 = 栅格(可多尺度) + 全局特征 (自机/Boss/残机/无敌帧...)。

    ``use_wall_feat`` 时再在末尾追加 12 维墙特征 (与默认观测同一套布局)。
    """
    grid_dim = (grid_feature_dim_scales(scales, use_velocity) if scales
                else grid_feature_dim(size, use_velocity))
    dim = grid_dim + global_feature_dim(use_invincible, use_boss, use_danger_grad)
    return dim + (WALL_FEATURE_DIM if use_wall_feat else 0)


def _encode_globals(vec: np.ndarray, state: GameState, px: float, py: float,
                    use_invincible: bool, use_boss: bool,
                    use_danger_grad: bool) -> None:
    """把全局特征写进 ``vec`` 的前 ``global_feature_dim(...)`` 维。"""
    boss_ratio = 0.0
    if state.boss_hp is not None and state.boss_max_hp:
        boss_ratio = max(0.0, min(1.0, state.boss_hp / max(1.0, state.boss_max_hp)))
    vec[:GLOBAL_FEATURE_DIM] = (
        px / PLAYFIELD_WIDTH,
        py / PLAYFIELD_HEIGHT,
        min(1.0, (state.stage or 1) / 6.0),
        min(1.0, (state.lives or 0) / 9.0),
        min(1.0, (state.power or 0.0) / 4.0),
        min(1.0, (state.graze or 0) / 2000.0),
        boss_ratio,
        min(1.0, (state.bullet_count or len(state.bullets)) / 150.0),
        1.0 if state.in_game else 0.0,
        min(1.0, (state.enemies_alive or 0) / 10.0),
    )
    if use_invincible:
        inv = state.invincible
        vec[GLOBAL_FEATURE_DIM] = (
            0.0 if inv is None else min(1.0, max(0.0, float(inv) / 150.0))
        )
    if use_boss:
        off = GLOBAL_FEATURE_DIM + (1 if use_invincible else 0)
        bx, by = _boss_xy(state)
        vec[off] = 0.0 if bx is None else (bx - px) / PLAYFIELD_WIDTH
        vec[off + 1] = 0.0 if by is None else (by - py) / PLAYFIELD_HEIGHT
    if use_danger_grad:
        off = GLOBAL_FEATURE_DIM + (1 if use_invincible else 0) + (2 if use_boss else 0)
        gx, gy = _danger_grad(state, px, py)
        vec[off] = float(np.clip(gx / DANGER_GRAD_SCALE, -1.0, 1.0))
        vec[off + 1] = float(np.clip(gy / DANGER_GRAD_SCALE, -1.0, 1.0))


def _splat_scale(state: GameState, px: float, py: float, size: int, span: float,
                 use_velocity: bool, grid: np.ndarray, arr: np.ndarray) -> None:
    """把 ``±span`` 窗口内的弹幕双线性 splat 进 ``grid`` (size,size,channels)。"""
    dx = arr[:, 0] - px
    dy = arr[:, 1] - py
    inside = (np.abs(dx) <= span) & (np.abs(dy) <= span)
    if not np.any(inside):
        return
    dxs = dx[inside]
    dys = dy[inside]
    gx = (dxs + span) / (2.0 * span) * (size - 1)
    gy = (dys + span) / (2.0 * span) * (size - 1)
    ix = np.clip(np.floor(gx).astype(np.intp), 0, size - 2)
    iy = np.clip(np.floor(gy).astype(np.intp), 0, size - 2)
    fx = gx - ix
    fy = gy - iy
    vel = None
    if use_velocity:
        velocities = (state.raw or {}).get("bullet_velocity") or []
        try:
            all_vel = np.asarray(velocities, dtype=np.float64).reshape(-1, 2)
            if all_vel.shape[0] != arr.shape[0]:
                raise ValueError("velocity/bullet count mismatch")
            vel = all_vel[inside]
        except Exception:
            vel = None
    dens = np.zeros((size, size), dtype=np.float64)
    vsum = np.zeros((size, size, 2), dtype=np.float64) if vel is not None else None
    for ox, oy in ((0, 0), (1, 0), (0, 1), (1, 1)):
        wx = fx if ox else (1.0 - fx)
        wy = fy if oy else (1.0 - fy)
        w = wx * wy
        np.add.at(dens, (iy + oy, ix + ox), w)
        if vsum is not None:
            np.add.at(vsum, (iy + oy, ix + ox, 0), w * vel[:, 0])
            np.add.at(vsum, (iy + oy, ix + ox, 1), w * vel[:, 1])
    grid[:, :, 0] = dens / (1.0 + dens)  # 饱和压缩, 避免密集处爆量
    if vsum is not None:
        mean = vsum / np.maximum(dens, 1e-9)[:, :, None]
        grid[:, :, 1] = np.clip(mean[:, :, 0] / VELOCITY_SCALE, -1.0, 1.0)
        grid[:, :, 2] = np.clip(mean[:, :, 1] / VELOCITY_SCALE, -1.0, 1.0)


def encode_state_grid(
    state: GameState,
    *,
    size: int = GRID_DEFAULT_SIZE,
    span: float = GRID_DEFAULT_SPAN,
    scales: list[tuple[int, float]] | None = None,
    use_velocity: bool = True,
    use_invincible: bool = False,
    use_boss: bool = False,
    use_danger_grad: bool = False,
    use_wall_feat: bool = False,
    out: np.ndarray | None = None,
    validate: bool = True,
) -> np.ndarray:
    """把状态编码成"危险图 + 全局特征"的定长向量 (栅格在前, 全局在后)。

    单尺度: 以自机为中心覆盖 ``[-span, +span]`` 像素的一张 ``size×size`` 栅格。

    **多尺度** (``scales=[(12, 90), (12, 330)]``): 同时给出"近场细图"和"远场粗图",
    各尺度依次拼接。真机弹幕统计显示: 最近弹距离中位数 ~118px, 单一 ±110px 窗口
    **有 53% 的时间是空的** —— 远场必须单独给一层, 否则策略看不见正在飞来的弹。

    双线性 splat 让弹幕位置在格子间平滑过渡, 避免"弹幕跨格时特征跳变"。
    """
    if validate:
        require_entities(state)
    resolved = scales or [(int(size), float(span))]
    channels = grid_channels(use_velocity)
    grid_dim = grid_feature_dim_scales(resolved, use_velocity)
    gdim = global_feature_dim(use_invincible, use_boss, use_danger_grad)
    wall_dim = WALL_FEATURE_DIM if use_wall_feat else 0
    vec = (np.empty(grid_dim + gdim + wall_dim, dtype=np.float64)
           if out is None else out)
    px = float(state.player_x) if state.player_x is not None else PLAYFIELD_WIDTH / 2
    py = float(state.player_y) if state.player_y is not None else PLAYFIELD_HEIGHT - 32
    _encode_globals(vec[grid_dim:grid_dim + gdim], state, px, py,
                    use_invincible, use_boss, use_danger_grad)

    vec[:grid_dim] = 0.0
    bullets = state.bullets
    if bullets:
        arr = np.asarray(bullets, dtype=np.float64).reshape(-1, 2)
        off = 0
        for gsize, gspan in resolved:
            cells = gsize * gsize * channels
            _splat_scale(state, px, py, gsize, gspan, use_velocity,
                         vec[off : off + cells].reshape(gsize, gsize, channels), arr)
            off += cells
    if use_wall_feat:
        vec[-WALL_FEATURE_DIM:] = wall_features(state, px, py)
    return vec


# ---------------------------------------------------------------------------
# 统一入口: 训练器/评估脚本只需调用下面两个函数, 就能在
#   * "最近 K 颗弹的一维向量" (旧)
#   * "危险图栅格" (config.use_grid=True)
#   * "未来占用栅格 + 几何缺口" (config.use_gap=True)
# 之间切换, 而不用改任何训练逻辑。
# ---------------------------------------------------------------------------


def gap_spec_from(config: Any) -> GapSpec:
    """按 ``TrainConfig`` 的缺口开关构造 :class:`GapSpec`。

    这些开关会进 ``policy.json`` 的 ``config`` 段, 所以离线评估/真机播放能自动
    还原出**训练时**的编码方式 —— 编码不一致是这类实验最隐蔽的坑。
    """
    return GapSpec(
        use_grid=bool(getattr(config, "gap_grid", True)),
        use_geo=bool(getattr(config, "gap_geo", True)),
        use_clearance=bool(getattr(config, "gap_clearance", True)),
        use_prev_action=bool(getattr(config, "gap_prev_action", True)),
        use_action_age=bool(getattr(config, "gap_action_age", False)),
        action_age_cap=int(getattr(config, "gap_action_age_cap", 12) or 12),
        tta_layers=int(getattr(config, "gap_layers", 1) or 1),
    )


#: 缺口编码器缓存: ``GapObservation`` 无状态 (每次返回新数组), 按 spec 复用即可
_GAP_ENCODERS: dict[GapSpec, GapObservation] = {}


def gap_encoder(config: Any) -> GapObservation:
    spec = gap_spec_from(config)
    encoder = _GAP_ENCODERS.get(spec)
    if encoder is None:
        encoder = _GAP_ENCODERS[spec] = GapObservation(spec)
    return encoder


def observation_dim(config: Any) -> int:
    """按 :class:`TrainConfig` 算出观测维度 (缺口 / 危险图 / 最近K弹三选一)。"""
    wall = WALL_FEATURE_DIM if getattr(config, "use_wall_feat", False) else 0
    if getattr(config, "use_gap", False):
        return gap_observation_dim(gap_spec_from(config)) + wall
    if getattr(config, "use_grid", False):
        size = getattr(config, "grid_size", GRID_DEFAULT_SIZE)
        span = getattr(config, "grid_span", GRID_DEFAULT_SPAN)
        scales = grid_scales_from(getattr(config, "grid_scales", ""),
                                  default_size=size, default_span=span)
        return grid_observation_dim(size, config.bullet_velocity, config.use_invincible,
                                    config.use_boss, config.use_danger_grad, scales=scales,
                                    use_wall_feat=bool(wall))
    return feature_dim(config.k_bullets, config.bullet_velocity, config.k_enemies,
                       config.use_invincible, config.use_boss, config.use_danger_grad,
                       bool(wall))


class ObservationEncoder:
    """把"观测配置"打包成一个可调用对象, 供**离线工具**使用。

    工具 (demo / 真机播放 / 评估) 只要从保存的 ``policy_*.json`` 里读出 config,
    就能自动得到正确的编码方式 —— 不用再手写一堆 ``use_boss=...`` 参数,
    也不会在"缺口策略"上错用旧特征。::

        enc = ObservationEncoder.from_meta(meta)   # meta = json["config"]
        obs = enc(state, prev_action=3)            # -> np.ndarray
        assert policy.obs_dim == enc.dim
    """

    def __init__(self, *, use_grid: bool = False,
                 grid_size: int = GRID_DEFAULT_SIZE, grid_span: float = GRID_DEFAULT_SPAN,
                 grid_scales: str = "",
                 k_bullets: int = 16, bullet_velocity: bool = False, k_enemies: int = 0,
                 use_invincible: bool = False, use_boss: bool = False,
                 use_danger_grad: bool = False, use_wall_feat: bool = False,
                 use_gap: bool = False, gap_grid: bool = True, gap_geo: bool = True,
                 gap_clearance: bool = True, gap_prev_action: bool = True,
                 gap_action_age: bool = False, gap_action_age_cap: int = 12,
                 gap_layers: int = 1) -> None:
        self.use_grid = bool(use_grid)
        self.grid_size = int(grid_size)
        self.grid_span = float(grid_span)
        self.grid_scales = str(grid_scales or "")
        self.k_bullets = int(k_bullets)
        self.bullet_velocity = bool(bullet_velocity)
        self.k_enemies = int(k_enemies)
        self.use_invincible = bool(use_invincible)
        self.use_boss = bool(use_boss)
        self.use_danger_grad = bool(use_danger_grad)
        self.use_wall_feat = bool(use_wall_feat)
        self.use_gap = bool(use_gap)
        self.gap_grid = bool(gap_grid)
        self.gap_geo = bool(gap_geo)
        self.gap_clearance = bool(gap_clearance)
        self.gap_prev_action = bool(gap_prev_action)
        self.gap_action_age = bool(gap_action_age)
        self.gap_action_age_cap = int(gap_action_age_cap)
        self.gap_layers = int(gap_layers)

    @classmethod
    def from_meta(cls, meta: dict | None = None) -> "ObservationEncoder":
        meta = dict(meta or {})
        return cls(
            use_grid=bool(meta.get("use_grid", False)),
            grid_size=int(meta.get("grid_size", GRID_DEFAULT_SIZE)),
            grid_span=float(meta.get("grid_span", GRID_DEFAULT_SPAN)),
            grid_scales=str(meta.get("grid_scales", "") or ""),
            k_bullets=int(meta.get("k_bullets", 16)),
            bullet_velocity=bool(meta.get("bullet_velocity", False)),
            k_enemies=int(meta.get("k_enemies", 0)),
            use_invincible=bool(meta.get("use_invincible", False)),
            use_boss=bool(meta.get("use_boss", False)),
            use_danger_grad=bool(meta.get("use_danger_grad", False)),
            use_wall_feat=bool(meta.get("use_wall_feat", False)),
            use_gap=bool(meta.get("use_gap", False)),
            gap_grid=bool(meta.get("gap_grid", True)),
            gap_geo=bool(meta.get("gap_geo", True)),
            gap_clearance=bool(meta.get("gap_clearance", True)),
            gap_prev_action=bool(meta.get("gap_prev_action", True)),
            gap_action_age=bool(meta.get("gap_action_age", False)),
            gap_action_age_cap=int(meta.get("gap_action_age_cap", 12) or 12),
            gap_layers=int(meta.get("gap_layers", 1) or 1),
        )

    @property
    def scales(self) -> list[tuple[int, float]]:
        return grid_scales_from(self.grid_scales, default_size=self.grid_size,
                                default_span=self.grid_span)

    @property
    def gap_spec(self) -> GapSpec:
        return gap_spec_from(self)

    @property
    def dim(self) -> int:
        return observation_dim(self)

    def __call__(self, state: GameState, out: np.ndarray | None = None,
                 prev_action: int = -1, action_age: int = 0) -> np.ndarray:
        return encode_observation(state, self, out=out, prev_action=prev_action,
                                  action_age=action_age)

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        if self.use_gap:
            spec = self.gap_spec
            return (f"ObservationEncoder(gap grid={spec.use_grid} geo={spec.use_geo} "
                    f"clearance={spec.use_clearance} age={spec.use_action_age} dim={self.dim})")
        if self.use_grid:
            scales = ",".join(f"{s}x{s}/±{sp:g}" for s, sp in self.scales)
            return (f"ObservationEncoder(grid[{scales}] vel={self.bullet_velocity} "
                    f"dim={self.dim})")
        wall = " wall" if self.use_wall_feat else ""
        return (f"ObservationEncoder(k_bullets={self.k_bullets} vel={self.bullet_velocity} "
                f"k_enemies={self.k_enemies}{wall} dim={self.dim})")


def encode_observation(state: GameState, config: Any,
                       out: np.ndarray | None = None,
                       prev_action: int = -1,
                       action_age: int = 0) -> np.ndarray:
    """按 :class:`TrainConfig` 编码一帧 (缺口 / 危险图 / 最近K弹)。

    ``prev_action`` 只对 ``use_gap`` 有效 (方向下标 0~8, -1=未知), 用于把"上一动作"
    的 one-hot 拼进观测 —— 宏动作有 6 帧承诺, 策略必须知道自己在往哪走。
    ``action_age`` 是"连续同向决策了多少步", 同样只对 ``use_gap`` 有效。
    """
    wall = bool(getattr(config, "use_wall_feat", False))
    if getattr(config, "use_gap", False):
        base = gap_encoder(config)(state, prev_action=prev_action,
                                   action_age=action_age)
        if not wall:
            if out is not None:
                out[...] = base
                return out
            return base
        dim = base.shape[0] + WALL_FEATURE_DIM
        vec = np.empty(dim, dtype=np.float64) if out is None else out
        vec[:base.shape[0]] = base
        px = float(state.player_x) if state.player_x is not None else PLAYFIELD_WIDTH / 2
        py = float(state.player_y) if state.player_y is not None else PLAYFIELD_HEIGHT - 32
        vec[-WALL_FEATURE_DIM:] = wall_features(state, px, py)
        return vec
    if getattr(config, "use_grid", False):
        return encode_state_grid(
            state,
            size=getattr(config, "grid_size", GRID_DEFAULT_SIZE),
            span=getattr(config, "grid_span", GRID_DEFAULT_SPAN),
            scales=grid_scales_from(getattr(config, "grid_scales", ""),
                                    default_size=getattr(config, "grid_size", GRID_DEFAULT_SIZE),
                                    default_span=getattr(config, "grid_span", GRID_DEFAULT_SPAN)),
            use_velocity=config.bullet_velocity,
            use_invincible=config.use_invincible,
            use_boss=config.use_boss,
            use_danger_grad=config.use_danger_grad,
            use_wall_feat=wall,
            out=out,
        )
    return encode_state(
        state,
        k_bullets=config.k_bullets,
        use_velocity=config.bullet_velocity,
        k_enemies=config.k_enemies,
        use_invincible=config.use_invincible,
        use_boss=config.use_boss,
        use_danger_grad=config.use_danger_grad,
        use_wall_feat=getattr(config, "use_wall_feat", False),
        out=out,
    )
