"""MPC / CEM 规划器：在**学习到的动力学预测**上做滚动优化.

与 ``viability.py`` 的区别
--------------------------
* ``viability`` 是**可达图 DP**：把"未来占用"离散成格子反向递推，适合一次性算
  安全区域，但每次 25~40ms，且只优化"别死"，没有偏好。
* 这里是**采样式 MPC**：对候选速度序列打分（安全 + 对准 + 站位），CEM 迭代逼近。
  可以用任意代价函数、可以接动力学模型，单次预算目标 < 10ms。

每步只执行首动作（receding horizon），这是 MPC 的标准做法：规划 24 帧、只走 3 帧。
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..types import PLAYFIELD_HEIGHT, PLAYFIELD_WIDTH, GameState
from .dynamics import ConstantVelocity
from .occupancy import BULLET_RADIUS, DIRECTIONS, HITBOX_RADIUS, PLAYER_SPEED

#: 9 个动作名（与 ``DIRECTIONS`` 同序，用 "+" 连接以便后端的按键解析）
ACTION_NAMES: tuple[str, ...] = (
    "shoot",
    "up+shoot",
    "down+shoot",
    "left+shoot",
    "right+shoot",
    "up+left+shoot",
    "up+right+shoot",
    "down+left+shoot",
    "down+right+shoot",
)
#: 加 focus 的版本（慢速移动，密弹里必须能半步走）
ACTION_NAMES_FOCUS: tuple[str, ...] = tuple(
    name.replace("+shoot", "+shoot+focus") for name in ACTION_NAMES
)
#: 按住 focus（Shift）时的速度比例。mock: 2.0 / 4.4；真机正作同量级。
FOCUS_SPEED_RATIO = 2.0 / PLAYER_SPEED


def _bullet_arrays(state: GameState) -> np.ndarray:
    """``(M, 6)`` = ``(x, y, vx, vy, ax, ay)``；缺速度/加速度时补 0。"""
    bullets = state.bullets or []
    if not bullets:
        return np.zeros((0, 6), dtype=np.float64)
    arr = np.asarray(bullets, dtype=np.float64).reshape(-1, 2)
    vel = (state.raw or {}).get("bullet_velocity") or []
    acc = (state.raw or {}).get("bullet_accel") or []
    out = np.zeros((arr.shape[0], 6), dtype=np.float64)
    out[:, :2] = arr
    try:
        v = np.asarray(vel, dtype=np.float64).reshape(-1, 2)
        if v.shape[0] == arr.shape[0]:
            out[:, 2:4] = v
    except Exception:
        pass
    try:
        a = np.asarray(acc, dtype=np.float64).reshape(-1, 2)
        if a.shape[0] == arr.shape[0]:
            out[:, 4:6] = a
    except Exception:
        pass
    return out


class MPCPlanner:
    """CEM-MPC 规划器.

    参数
    ----
    horizon : int
        规划视界（游戏帧）。24 帧 ≈ 0.4 秒，够覆盖"现在不动、两拍后被扫到"的风险。
    dt : int
        每个控制步推进多少帧（也是"承诺期"）。
    n_samples / n_elite / iters : int
        CEM 的采样数 / 精英数 / 迭代轮数。48/8/3 在 mock 上约 3~6 ms/次。
    clearance_margin : float
        期望的最小安全余量（px）。低于它开始扣分，低于 0 = 撞上。
    """

    def __init__(self, *, horizon: int = 24, dt: int = 3, n_samples: int = 48,
                 n_elite: int = 8, iters: int = 3, seed: int = 0,
                 speed: float = PLAYER_SPEED, focus_speed: float | None = None,
                 clearance_margin: float = 10.0, aim_weight: float = 0.35,
                 y_weight: float = 0.25, y_target: float = 0.85,
                 column_weight: float = 0.0, column_sigma_x: float = 18.0,
                 column_sigma_y: float = 60.0,
                 gap_weight: float = 0.0, gap_radius: float = 13.0,
                 gap_cluster: float = 14.0, gap_min_bullets: int = 8,
                 gap_min_coverage: float = 0.5, exact_clearance: bool = True,
                 warm_start: bool = True, init_std: float = 0.5,
                 dynamics: Any | None = None,
                 use_focus: bool = False, focus_distance: float = 0.0,
                 commit_steps: int = 0, switch_margin: float = 0.0,
                 focus_margin: float = 0.0, deterministic: bool = False,
                 smooth_clear: float = 0.0, smooth_sigma: float = 20.0,
                 top_weight: float = 0.0, top_margin: float = 42.0,
                 bottom_weight: float = 0.0, bottom_margin: float = 30.0,
                 edge_weight: float = 0.0, edge_margin: float = 26.0) -> None:
        self.horizon = max(1, int(horizon))
        self.dt = max(1, int(dt))
        self.n_steps = max(1, self.horizon // self.dt)
        self.n_samples = max(2, int(n_samples))
        self.n_elite = max(2, min(int(n_elite), self.n_samples))
        self.iters = max(1, int(iters))
        self.rng = np.random.default_rng(seed)
        self.speed = float(speed)
        # 按住 focus 的速度比 = 2.0 / 4.4（mock 实测；正作同量级）。用**比例**而不是
        # 绝对值：老师在不同 step_frames 下会把 speed 乘 k，focus 必须跟着乘 k。
        self.focus_speed = (self.speed * FOCUS_SPEED_RATIO if focus_speed is None
                            else float(focus_speed))
        self.clearance_margin = float(clearance_margin)
        self.aim_weight = float(aim_weight)
        self.y_weight = float(y_weight)
        self.y_target = float(y_target)
        # 「正上方弹幕柱」惩罚: 只优化"当前距离"时, 规划器会站在一堵还远、但必
        # 定从头顶压下来的墙下面 (wall_gate 实测 548 步)。这一项奖励"站在缺口
        # 下方", 是墙/闭合环能不能过的关键。
        self.column_weight = float(column_weight)
        self.column_sigma_x = float(column_sigma_x)
        self.column_sigma_y = float(column_sigma_y)
        # 「缺口锚点项」: 弹幕柱只能隐式表达"别站在墙下面", 但 `wall_gate` 真正
        # 需要的是**提前站到缺口列** —— 墙从 y=108 落到自机高度要 ~146 帧, 远长于
        # 72 帧视界, 视界内它"还没到", 任何终点看起来都安全 (实测地形代价完全
        # 平坦, 加了也没用)。所以这里显式找出"最近一堵墙的缺口中心", 把终点往
        # 那一列拉: 代价 = |x_end − x_gap| / 场地宽。
        #
        # 只在"确实是一堵墙"时生效: 同一时刻的穿越事件 >= gap_min_bullets 颗,
        # 且这些弹封住了 >= gap_min_coverage 的列。稀疏射击不会触发 (否则会把
        # 自机拽去一个毫无意义的"最宽空档")。
        self.gap_weight = float(gap_weight)
        self.gap_radius = float(gap_radius)
        self.gap_cluster = float(gap_cluster)
        self.gap_min_bullets = int(gap_min_bullets)
        self.gap_min_coverage = float(gap_min_coverage)
        # 「连续时间安全余量」: 只在采样时刻算距离会**穿透** —— dt=12 时一颗弹
        # 每步走 1.9*12≈23px, 判定直径只有 10px, 采样点全落在"擦过去"的空隙里,
        # 于是长视界规划器以为自己安全 (实测 h180 只有 648 步, 比短视界还差)。
        # 这里改成对每段相对运动解析求最近接近距离, 与 dt 无关。
        self.exact_clearance = bool(exact_clearance)
        # 「热启动」: CEM 的均值不要每次从 0 开始, 而是用上一步解**平移一位**。
        # 不热启动时相邻两帧的解几乎无关 (实测老师"自一致率" 21%、77% 的帧会换动作),
        # 这样的标签拿去模仿等于教学生学噪声。滚动优化本来就该热启动。
        self.warm_start = bool(warm_start)
        self.init_std = float(init_std)
        self._last_mean: np.ndarray | None = None
        self.dynamics = dynamics or ConstantVelocity()
        self.use_focus = bool(use_focus)
        self.focus_distance = float(focus_distance)
        # 「并列动作承诺」：CEM 在代价平坦处每一帧都可能翻到另一个同样安全的方向，
        # 实测相邻帧换向率 50~77%。这既是老师标签的噪声源（学生学到的就是"乱翻"），
        # 也是真机上"手抖"。承诺期 = 选定方向后 N 帧内不换，除非新方向明显更便宜
        # （差价 > switch_margin）。这是"从源头掐掉标签噪声"的第一步。
        self.commit_steps = max(0, int(commit_steps))
        self.switch_margin = float(switch_margin)
        # focus（半速）与全速的取舍阈值：慢速只有在明显更安全时才按 shift
        self.focus_margin = float(focus_margin)
        # 「确定性」：每帧用 `seed + frame` 重新播种 CEM 噪声，同一状态无论第几次
        # 被问到都得到同一个解（不再依赖调用历史）。测量"老师对自己是否一致"必须
        # 先排除 RNG 序列漂移这一项。
        self.deterministic = bool(deterministic)
        self.seed = int(seed)
        # 「平滑安全余量」: 硬阈值代价 (低于 margin 才扣分) 让"离弹 12px"和"离弹 60px"
        # 代价完全相同 —— 于是 70%+ 的帧里好几个动作并列最优, 代价向量退化成带并列的
        # one-hot, cost-to-go 回归就失去意义 (见 PLANNER_V2 §10.6)。
        # 加一项 w·exp(−clear/σ) 让余量连续可分; 默认 0 = 关闭, 不影响既有行为。
        self.smooth_clear = float(smooth_clear)
        self.smooth_sigma = max(1.0, float(smooth_sigma))
        # 「持续贴边惩罚」: 只罚**终点站位**（上面的 y_weight）拦不住"贴顶" ——
        # 规划器完全可以先上穿躲开一堵墙，然后在顶端待满整个视界（终点 y 也不算太离谱，
        # 但整段都在天花板上）。实测生成的对局里 min_y 常驻 8~20px，就是这种退化。
        # 这里对**整条轨迹**逐步算"超出安全区多少"，取平均（与视界长度无关），再用
        # 二次函数拉开"贴 5px"与"贴 30px"的差距。
        # 语义：**允许上去**（上穿是规划器唯一能过某些墙的手段），但待在上面要付租金。
        self.top_weight = float(top_weight)
        self.top_margin = float(top_margin)
        self.bottom_weight = float(bottom_weight)
        self.bottom_margin = float(bottom_margin)
        self.edge_weight = float(edge_weight)
        self.edge_margin = float(edge_margin)
        self._committed_idx: int | None = None
        self._commit_left = 0
        #: 同一帧只算一次 CEM（``choose`` 与 ``action_costs`` 共享）
        self._plan_state: GameState | None = None
        self._plan_seq: np.ndarray | None = None
        #: 最近一次 ``action_costs`` 的结果（诊断 / 蒸馏取 cost-to-go 用）
        self.last_costs: np.ndarray | None = None
        self.last_cost: float | None = None

    # ------------------------------------------------------------------
    def _bullet_positions(self, bullets: np.ndarray, t: float) -> np.ndarray:
        """预测 ``t`` 帧后每颗弹的位置 ``(M, 2)``。"""
        if bullets.shape[0] == 0:
            return np.zeros((0, 2))
        base = bullets[:, :2] + bullets[:, 2:4] * t
        if isinstance(self.dynamics, ConstantVelocity):
            return base
        resid = self.dynamics.predict_batch(bullets, float(t))
        return base + resid

    # ------------------------------------------------------------------
    def _crossing_profile(self, bullets: np.ndarray, py: float) -> tuple[np.ndarray, np.ndarray]:
        """每颗弹"穿过自机高度"的时刻 ``t`` 与那时的横向位置 ``x``。

        只保留 ``vy>0``（朝下走）且在视界内会到达自机高度的弹。``t`` 已经减去
        30px 的安全提前量（弹到自机上方 30px 就算这一列要封了）。
        """
        if bullets.shape[0] == 0:
            return np.zeros(0), np.zeros(0)
        vy = bullets[:, 3]
        ok = vy > 0.05
        if not np.any(ok):
            return np.zeros(0), np.zeros(0)
        b = bullets[ok]
        t = (py - 30.0 - b[:, 1]) / b[:, 3]
        keep = (t > 0) & (t <= self.horizon * 2.5)
        if not np.any(keep):
            return np.zeros(0), np.zeros(0)
        return t[keep], b[keep, 0] + b[keep, 2] * t[keep]

    # ------------------------------------------------------------------
    def gap_anchor(self, bullets: np.ndarray, py: float) -> float | None:
        """最近一堵墙的**缺口中心 x**；不是墙就返回 ``None``。

        做法: 取最早一批"穿过自机高度"的弹 (``gap_cluster`` 帧内算同一批),
        把它们的横坐标按 ``gap_radius`` 摊成占用列, 找出最宽的连续空档, 取其中点。
        """
        t_cross, x_cross = self._crossing_profile(bullets, py)
        if t_cross.size < self.gap_min_bullets:
            return None
        # 只瞄准**视界之外**最近的一堵墙: 视界内的墙由 min_clear 逐时刻处理,
        # 视界外的墙才是"必须现在就开始站位"的那个。若瞄准最早一堵 (含视界内),
        # 每次墙穿过自机高度后目标会突跳 61px, 自机永远差一拍 (实测 1144 步)。
        beyond = t_cross[t_cross > self.horizon]
        if beyond.size >= self.gap_min_bullets:
            t_cross, x_cross = t_cross[t_cross > self.horizon], x_cross[t_cross > self.horizon]
        first = float(t_cross.min())
        group = np.abs(t_cross - first) <= self.gap_cluster
        xs = np.sort(x_cross[group])
        if xs.size < self.gap_min_bullets:
            return None
        cols = np.arange(8.0, PLAYFIELD_WIDTH - 8.0, 2.0)
        blocked = (np.abs(cols[:, None] - xs[None, :]) < self.gap_radius).any(axis=1)
        if float(blocked.mean()) < self.gap_min_coverage:
            return None
        best_len, best_center, run = 0.0, None, None
        for i, free in enumerate(blocked):
            if free:
                run = None if run is None else run
                continue
            if run is None:
                run = i
            if i == len(blocked) - 1 or blocked[i + 1]:
                length = float(cols[i] - cols[run])
                if length > best_len:
                    best_len, best_center = length, float((cols[i] + cols[run]) / 2.0)
                run = None
        if best_center is None:
            return None
        return best_center

    def _gap_cost(self, xs_end: np.ndarray, anchor: float | None) -> np.ndarray:
        if anchor is None:
            return np.zeros(xs_end.shape[0])
        return self.gap_weight * np.abs(xs_end - anchor) / PLAYFIELD_WIDTH

    def _rollout(self, state: GameState, bullets: np.ndarray,
                 seq: np.ndarray, speed: Any | None = None) -> tuple[np.ndarray, np.ndarray]:
        """对 ``(K, n_steps, 2)`` 的单位方向序列打分，返回 ``(cost, end_xy)``。

        ``speed`` 可以是标量或 ``(K,)`` —— 后者用于**同一批 rollout 里混入 focus
        动作**：按住 shift 时自机只有全速的 ~45%，如果还按全速算，规划器会以为
        自己一步能走 8.8px，实际只走了 4px，算出来的"安全位置"根本到不了。
        """
        k = seq.shape[0]
        if speed is None:
            sp = np.full((k, self.n_steps), self.speed)
        else:
            sp = np.asarray(speed, dtype=np.float64)
            if sp.ndim == 0 or sp.size == 1:
                sp = np.full((k, self.n_steps), float(sp.reshape(-1)[0]))
            elif sp.ndim == 1:
                sp = np.repeat(sp.reshape(-1, 1), self.n_steps, axis=1)
            if sp.shape[0] == 1 and k > 1:
                sp = np.repeat(sp, k, axis=0)
        px = float(state.player_x if state.player_x is not None else PLAYFIELD_WIDTH / 2)
        py = float(state.player_y if state.player_y is not None else PLAYFIELD_HEIGHT - 32)
        step_px = sp * self.dt
        xs = np.clip(px + np.cumsum(seq[:, :, 0] * step_px, axis=1),
                     8.0, PLAYFIELD_WIDTH - 8.0)
        ys = np.clip(py + np.cumsum(seq[:, :, 1] * step_px, axis=1),
                     8.0, PLAYFIELD_HEIGHT - 8.0)

        hit_r = BULLET_RADIUS + HITBOX_RADIUS
        min_clear = np.full(k, 1e9)
        if bullets.shape[0]:
            vel = bullets[:, 2:4]
            prev_x = np.concatenate([np.full((k, 1), px), xs[:, :-1]], axis=1)
            prev_y = np.concatenate([np.full((k, 1), py), ys[:, :-1]], axis=1)
            for step in range(self.n_steps):
                if self.exact_clearance:
                    # 段起点 (t=step*dt) 的弹位置 + 段内相对运动, 解析求最近点
                    pos0 = self._bullet_positions(bullets, step * self.dt)
                    r0x = prev_x[:, step][:, None] - pos0[None, :, 0]
                    r0y = prev_y[:, step][:, None] - pos0[None, :, 1]
                    rvx = (xs[:, step] - prev_x[:, step])[:, None] / self.dt - vel[None, :, 0]
                    rvy = (ys[:, step] - prev_y[:, step])[:, None] / self.dt - vel[None, :, 1]
                    rv2 = np.maximum(1e-6, rvx * rvx + rvy * rvy)
                    tau = np.clip(-(r0x * rvx + r0y * rvy) / rv2, 0.0, self.dt)
                    cx = r0x + rvx * tau
                    cy = r0y + rvy * tau
                    d = np.sqrt(cx * cx + cy * cy)
                else:
                    pos = self._bullet_positions(bullets, (step + 1) * self.dt)
                    pts = np.stack([xs[:, step], ys[:, step]], axis=1)
                    d = np.linalg.norm(pts[:, None, :] - pos[None, :, :], axis=2)
                min_clear = np.minimum(min_clear, d.min(axis=1) - hit_r)
        else:
            min_clear[:] = 1e3

        # 代价: 安全余量是主项 (低于 margin 二次扣分), 再加对准与站位
        deficit = np.maximum(0.0, self.clearance_margin - min_clear)
        cost = 8.0 * deficit ** 2
        cost += np.where(min_clear < 0, 200.0 + 50.0 * (-min_clear), 0.0)
        if self.smooth_clear:
            cost += self.smooth_clear * np.exp(
                -np.clip(min_clear, -20.0, 400.0) / self.smooth_sigma)
        boss = (state.raw or {}).get("boss_position")
        if boss:
            cost += self.aim_weight * np.abs(xs[:, -1] - float(boss[0])) / PLAYFIELD_WIDTH * 10.0
        cost += self.y_weight * np.abs(ys[:, -1] / PLAYFIELD_HEIGHT - self.y_target) * 10.0
        # 持续贴边: 逐步算超出量, 平均后二次加权。用 mean 而不是 sum 是为了让
        # 代价与视界长度无关 (否则 72 帧视界会自动比 24 帧重 3 倍, 长/短视界不可比)。
        if self.top_weight or self.bottom_weight or self.edge_weight:
            zone = np.zeros_like(ys)
            if self.top_weight:
                zone += self.top_weight * np.maximum(0.0, self.top_margin - ys) ** 2
            if self.bottom_weight:
                low = PLAYFIELD_HEIGHT - self.bottom_margin
                zone += self.bottom_weight * np.maximum(0.0, ys - low) ** 2
            if self.edge_weight:
                right = PLAYFIELD_WIDTH - self.edge_margin
                zone += self.edge_weight * (np.maximum(0.0, self.edge_margin - xs) ** 2
                                            + np.maximum(0.0, xs - right) ** 2)
            cost += zone.mean(axis=1)
        if self.column_weight and bullets.shape[0]:
            pos = self._bullet_positions(bullets, float(self.horizon))
            dx = (pos[None, :, 0] - xs[:, -1][:, None]) / self.column_sigma_x
            dy = (pos[None, :, 1] - ys[:, -1][:, None]) / self.column_sigma_y
            # 只惩罚"还在上面、但会压到自机高度"的弹
            above = (pos[None, :, 1] <= ys[:, -1][:, None]).astype(np.float64)
            cost += self.column_weight * (np.exp(-0.5 * (dx ** 2 + dy ** 2)) * above).sum(axis=1)
        if self.gap_weight:
            cost += self._gap_cost(xs[:, -1], self.gap_anchor(bullets, py))
        return cost, np.stack([xs[:, -1], ys[:, -1]], axis=1)

    # ------------------------------------------------------------------
    def _plan(self, state: GameState) -> np.ndarray:
        """同一帧只做一次 CEM：``choose`` 与 ``action_costs`` 共用这份解."""
        cached = self._plan_seq
        if cached is not None and self._plan_state is state:
            return cached
        seq = self.choose_sequence(state)
        self._plan_state = state
        self._plan_seq = seq
        return seq

    def action_costs(self, state: GameState, *, focus: bool | None = None,
                     mode: str = "plan") -> np.ndarray:
        """逐动作 cost-to-go —— 软标签 / cost-to-go 蒸馏的监督信号。

        ``mode="plan"``（默认，也是蒸馏该用的）
            对每个动作算 **"先走这一步，然后接着走 CEM 规划出来的剩余路径"** 的代价。
            这正是滚动优化下"执行某个动作的真实代价"（一步偏离 + 贪心续接），
            所以 ``argmin`` 就等于规划器自己会选的动作 —— 学生学的是老师的**价值
            函数**，学完自己 argmin，而不是被塞一个"老师这一帧选了谁"。
        ``mode="hold"``
            老口径：假设该动作一直按满整个视界。代价差被放大、但系统性偏保守
            （"一直往右"和"先往右再绕回"完全是两回事），argmin 与老师实际动作
            只有约一半一致。

        ``focus=True`` 时返回 **18 维**（前 9 全速、后 9 按住 shift 半速），与
        ``trainer.build_action_table(use_focus=True)`` 的顺序一致 —— 否则学生
        会被教"什么时候按 shift"，而老师给的代价却是按全速算的，物理对不上。
        """
        bullets = _bullet_arrays(state)
        dirs = np.asarray(DIRECTIONS, dtype=np.float64)
        n = dirs.shape[0]
        use_focus = self.use_focus if focus is None else bool(focus)
        if mode == "plan" and self.n_steps >= 2:
            plan = self._plan(state)
            seqs = np.repeat(plan[None, :, :], n, axis=0)
            seqs[:, 0, :] = dirs
            cost, _ = self._rollout(state, bullets, seqs)
            out = [cost]
            if use_focus:
                # focus 只改**第一步**的速度：按住 shift 走半步，之后照样按规划走
                sp = np.full((n, self.n_steps), self.speed)
                sp[:, 0] = self.focus_speed
                cost_slow, _ = self._rollout(state, bullets, seqs, speed=sp)
                out.append(cost_slow)
            out = np.concatenate(out)
        else:
            seq = np.repeat(dirs[:, None, :], self.n_steps, axis=1)
            cost, _ = self._rollout(state, bullets, seq)
            out = [cost]
            if use_focus:
                cost_slow, _ = self._rollout(state, bullets, seq, speed=self.focus_speed)
                out.append(cost_slow)
            out = np.concatenate(out)
        self.last_costs = out
        return out

    def cost_vector(self, state: GameState, *, focus: bool | None = None,
                    mode: str = "plan") -> np.ndarray:
        """逐动作 cost-to-go（与 ``action_costs`` 同义，供蒸馏显式调用）。"""
        return self.action_costs(state, focus=focus, mode=mode)

    def soft_labels(self, state: GameState, temperature: float = 20.0,
                    mode: str = "plan") -> np.ndarray:
        """``(n_actions,)`` 的动作目标分布 (与动作表同序)。"""
        cost = np.asarray(self.action_costs(state, mode=mode), dtype=np.float64)
        z = -(cost - cost.min()) / max(1e-6, float(temperature))
        z -= z.max()
        p = np.exp(z)
        return p / max(1e-12, p.sum())

    # ------------------------------------------------------------------
    def choose_sequence(self, state: GameState) -> np.ndarray:
        """CEM 搜索最优的 ``(n_steps, 2)`` 单位方向序列。"""
        if self.deterministic:
            # 同一帧号 -> 同一串噪声。与"调用了几次"无关，测量老师自身一致性时
            # 才不会把 RNG 序列漂移误当成规划不稳定。
            frame = int(getattr(state, "frame", 0) or 0)
            self.rng = np.random.default_rng(self.seed * 100003 + frame)
        bullets = _bullet_arrays(state)
        if self.warm_start and self._last_mean is not None and self._last_mean.shape[0] == self.n_steps:
            mean = self._last_mean.copy()
        else:
            mean = np.zeros((self.n_steps, 2))
        std = np.full((self.n_steps, 2), self.init_std)
        best_seq = mean.copy()
        best_cost = float("inf")
        for _ in range(self.iters):
            noise = self.rng.standard_normal((self.n_samples, self.n_steps, 2)) * std
            seqs = np.clip(mean + noise, -1.0, 1.0)
            seqs[0] = mean                        # 保留一个"当前最优"样本, 防退化
            cost, _ = self._rollout(state, bullets, seqs)
            order = np.argsort(cost)
            elite = seqs[order[:self.n_elite]]
            if float(cost[order[0]]) < best_cost:
                best_cost = float(cost[order[0]])
                best_seq = seqs[order[0]].copy()
            mean = elite.mean(axis=0)
            std = np.maximum(elite.std(axis=0), 0.05)
        self.last_cost = best_cost
        # 热启动用**精英均值**而不是单个最优样本: 后者对噪声最敏感
        if mean.shape[0] == self.n_steps:
            self._last_mean = mean
        return best_seq

    # ------------------------------------------------------------------
    @staticmethod
    def _quantize(vec: np.ndarray) -> int:
        """把单位方向量化到 8 方向下标（阈值 0.38 ≈ 偏离主轴 21° 以外才算斜向）。"""
        ux = 0 if abs(vec[0]) < 0.38 else (1 if vec[0] > 0 else -1)
        uy = 0 if abs(vec[1]) < 0.38 else (1 if vec[1] > 0 else -1)
        for i, (dx, dy) in enumerate(DIRECTIONS):
            if (int(dx), int(dy)) == (ux, uy):
                return i
        return 0

    @staticmethod
    def _nearest(state: GameState) -> float:
        bullets = state.bullets or []
        if not bullets or state.player_x is None:
            return 1e9
        try:
            arr = np.asarray(bullets, dtype=np.float64).reshape(-1, 2)
            return float(np.hypot(arr[:, 0] - state.player_x,
                                  arr[:, 1] - state.player_y).min())
        except Exception:
            return 1e9

    def _apply_commit(self, costs: np.ndarray, idx: int) -> int:
        """并列动作承诺：承诺期内不换向，除非新方向明显更便宜。

        ``costs`` 是 ``action_costs`` 的结果（9 或 18 维，前 9 个是全速方向）。
        因为只比较"方向"，focus 块不参与。
        """
        if self.commit_steps <= 0:
            return idx
        prev = self._committed_idx
        if prev is not None and self._commit_left > 0 and prev != idx:
            if float(costs[prev]) <= float(costs[idx]) + self.switch_margin:
                idx = prev
        if idx != prev:
            self._commit_left = self.commit_steps
        else:
            self._commit_left = max(0, self._commit_left - 1)
        self._committed_idx = idx
        return idx

    def choose(self, state: GameState) -> str:
        """返回动作名。动作直接取 **逐动作 cost-to-go 的 argmin**（与蒸馏同一口径）。

        为什么不再用 CEM 序列量化：规划器自己的答案要与"教给学生的代价向量"
        一致，学生 argmax 才等于老师在做的事。CEM 的连续解只用来提供**续接路径**
        （见 ``action_costs(mode="plan")``）。
        """
        if state.player_x is None or state.player_y is None:
            return ACTION_NAMES[0]
        costs = self.action_costs(state)
        n = len(DIRECTIONS)
        if not self.use_focus:
            idx = self._apply_commit(costs, int(np.argmin(costs)))
            return ACTION_NAMES[idx]
        # 18 个动作一起 argmin：慢速动作有自己的代价（半步走），不需要再用
        # "离弹多近就按 shift" 这种硬阈值 —— 那会让老师的动作和它给出的代价
        # 对不上，学生学的是代价、做出来却是阈值。
        best = int(np.argmin(costs))
        idx, slow = best % n, best >= n
        idx = self._apply_commit(costs, idx)
        if idx != best % n:
            # 承诺把方向改回去了: 在承诺方向上重新比快/慢
            slow = bool(costs[n + idx] + self.focus_margin < costs[idx])
        if not slow and self.focus_distance > 0:
            slow = (self._nearest(state) < self.focus_distance
                    and costs[n + idx] <= costs[idx] + self.focus_margin)
        return (ACTION_NAMES_FOCUS if slow else ACTION_NAMES)[idx]

    def describe(self) -> dict[str, Any]:
        return {"name": "mpc", "horizon": self.horizon, "dt": self.dt,
                "column_weight": self.column_weight, "gap_weight": self.gap_weight,
                "samples": self.n_samples, "iters": self.iters,
                "commit_steps": self.commit_steps, "switch_margin": self.switch_margin,
                "focus": self.use_focus, "deterministic": self.deterministic,
                "top_weight": self.top_weight, "top_margin": self.top_margin,
                "edge_weight": self.edge_weight, "bottom_weight": self.bottom_weight,
                "dynamics": self.dynamics.describe().get("name", "?")}


class AdaptiveMPCPlanner:
    """按局面切换视界的 MPC.

    实测长短视界是**互相伤害**的:

    * 短视界 (24 帧) —— 反应式弹型全 2000 步, 但 `wall_gate` 只有 548;
    * 长视界 (72 帧 + 缺口项) —— `wall_gate` 提到 1152, 但 `aimed_spread` 掉到 922、
      `dense_ring` 1219、`closing_ring` 862。

    原因很直接: 长视界把"远处的墙"也算进代价, 于是自机开始为 1 秒后的墙让位,
    反而在原地该微调的时候僵住。所以按 ``WallDetector`` 判别"是不是墙/收网局面",
    分别用长/短视界 —— 与 ``hierarchical.py`` 的分层是同一个思路, 只是这里切的是
    **同一个规划器的参数**, 不需要两个不同的决策器。
    """

    def __init__(self, *, short_horizon: int = 24, short_dt: int = 3,
                 long_horizon: int = 72, long_dt: int = 6,
                 long_column: float = 2.0, long_gap: float = 0.0,
                 n_samples: int = 48, iters: int = 3,
                 seed: int = 0, use_focus: bool = False,
                 dynamics: Any | None = None,
                 detector: Any | None = None,
                 commit_steps: int = 0, switch_margin: float = 0.0,
                 focus_margin: float = 0.0, deterministic: bool = False,
                 smooth_clear: float = 0.0, smooth_sigma: float = 20.0,
                 top_weight: float = 0.0, top_margin: float = 42.0,
                 bottom_weight: float = 0.0, bottom_margin: float = 30.0,
                 edge_weight: float = 0.0, edge_margin: float = 26.0) -> None:
        common = dict(n_samples=n_samples, iters=iters, seed=seed,
                      use_focus=use_focus, dynamics=dynamics,
                      commit_steps=commit_steps, switch_margin=switch_margin,
                      focus_margin=focus_margin, deterministic=deterministic,
                      smooth_clear=smooth_clear, smooth_sigma=smooth_sigma,
                      top_weight=top_weight, top_margin=top_margin,
                      bottom_weight=bottom_weight, bottom_margin=bottom_margin,
                      edge_weight=edge_weight, edge_margin=edge_margin)
        self.short = MPCPlanner(horizon=short_horizon, dt=short_dt, **common)
        self.long = MPCPlanner(horizon=long_horizon, dt=long_dt,
                               column_weight=long_column, gap_weight=long_gap, **common)
        if detector is None:
            from .hierarchical import WallDetector

            detector = WallDetector()
        self.detector = detector
        self.last_mode = "short"

    # ------------------------------------------------------------------
    def _pick(self, state: GameState) -> MPCPlanner:
        """按 ``WallDetector`` 选出这一步该用哪个视界 (与 ``choose`` 完全一致)。"""
        try:
            metrics = self.detector.detect(state)
            wall = bool(getattr(metrics, "switched", False))
        except Exception:  # pragma: no cover - 检测失败时退化为短视界
            wall = False
        self.last_mode = "long" if wall else "short"
        return self.long if wall else self.short

    def cost_vector(self, state: GameState, *, focus: bool | None = None) -> np.ndarray:
        """逐动作 cost-to-go（与 ``choose`` 用同一个子规划器）。"""
        return self._pick(state).action_costs(state, focus=focus)

    def choose(self, state: GameState) -> str:
        return self._pick(state).choose(state)

    def describe(self) -> dict[str, Any]:
        return {"name": "adaptive-mpc", "short": self.short.describe(),
                "long": self.long.describe(), "last_mode": self.last_mode}
