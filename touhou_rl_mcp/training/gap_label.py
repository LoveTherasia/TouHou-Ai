"""缺口任务的弱监督标签 (只回答"往哪走", 不回答"该不该躲")。

为什么标签要住在包里
--------------------
它原先长在 ``tools/research/gap_rl.py`` 里, 于是只有那个实验脚本能用 —— 真实训练
路径 (:class:`~touhou_rl_mcp.training.trainer.Trainer` 的 BC/DAgger) 反而拿不到。
标签是"教什么"的定义, 属于训练代码, 不属于某个脚本。

四种口径 (全部只用几何测量, 不含手写代价函数)
--------------------------------------------
``now``
    当前剖面里最空的那条车道。**问题**: 不管自机来不来得及, 而且整片安全时 argmax
    会退化成固定的一条车道 (已用"并列取最近"修掉)。
``reach``
    在 ``now`` 上加入可达性: 目标车道必须"赶过去以后还有余量"。已经站对时标签是
    原地, 不会为了 3 个像素的偏差来回摆。**默认口径**。
``future``
    承诺语义: 沿同一方向推 4 步 (24 帧) 取最好的时刻。实测它偏好"苟住", 单独当老师
    只有 806 帧 (v2)。
``slack``
    两堵墙的 maximin: 不只问"第一堵墙我还有多少余量", 还问"站在这里第二堵墙之前我
    还剩多少时间可以挪窝"。把"停在一个马上又要封死的位置"降权 —— 这是 ``reach``
    唯一的短视之处: 它只看第一次威胁的到达时刻。

三个实测过的坑 (写在这里, 免得再踩)
----------------------------------
* 只算一步落点 -> 所有方向都不如原地 (还要付 6 帧赶路), 标签退化成"永远别动";
* 把纵向位移按弹速折算成"少了几帧余量" -> 下移看起来永远划算, 标签退化成"一直按着下";
* 在 ``now`` 之前先算 ``reach`` 的目标再提前返回 -> "now" 分支永远走不到, 两种口径
  输出 100% 相同 (v2 上实测过)。现在每个口径各自独立。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..types import PLAYFIELD_WIDTH, GameState
from .occupancy import (
    DIRECTIONS, PLAYER_SPEED, GapSpec,
    _arrival_times, _best_column, _bullets, _column_centers,
)

#: 宏动作承诺帧数 (60Hz 下的 0.1 秒); 必须与训练时的 ``step_frames`` 一致
COMMIT_FRAMES = 6

#: 判定半径 (自机 2 + 弹 3) 取整后的量化档位: 差 6px 以内的余量视为同一档,
#: 避免"差 1 帧"就换动作的抖动
REACH_PAD = 6.0

#: 缺口任务的宏动作表 (8 方向 + 原地, 全部常按射击)。
#: 顺序必须与 :data:`touhou_rl_mcp.training.occupancy.DIRECTIONS` 一致 ——
#: 标签输出的是下标, 训练器按同名方向映射回自己的动作表。
MACRO_ACTIONS: tuple[str, ...] = (
    "shoot", "up+shoot", "down+shoot", "left+shoot", "right+shoot",
    "up_left+shoot", "up_right+shoot", "down_left+shoot", "down_right+shoot",
)

#: 同一车道被两次弹幕覆盖的最小间隔 (帧); 小于它的算同一次威胁
SECOND_GAP_FRAMES = 12.0

LABEL_MODES: tuple[str, ...] = ("now", "reach", "future", "slack")

#: 有状态口径: 需要一个 :class:`GapCommitment` 记住"当前目标列"。
#: 它们不能当成逐帧函数调用 —— 同一帧在不同的承诺状态下标签不同。
HOLD_MODES: tuple[str, ...] = ("reach_hold", "reach_hyst")

#: 训练/看板可选的完整口径 (逐帧口径 + 有状态口径)
TEACHER_MODES: tuple[str, ...] = LABEL_MODES + HOLD_MODES

#: 方向 -> 横向符号 (用于"同向连续决策了多少步"这一维观测)
_LEFT_ACTIONS = frozenset({3, 5, 7})
_RIGHT_ACTIONS = frozenset({4, 6, 8})


def direction_sign(action: int) -> int:
    """宏动作的横向符号: -1 左 / +1 右 / 0 原地或上下。"""
    index = int(action) % len(DIRECTIONS)
    if index in _LEFT_ACTIONS:
        return -1
    if index in _RIGHT_ACTIONS:
        return 1
    return 0


def update_action_age(prev_action: int, age: int, action: int,
                      cap: int = 12) -> int:
    """更新"连续同向决策步数"。

    这是给策略的**因果记忆**: 两次决策之间只看 ``prev_action`` 不够, 策略还需要
    知道"我已经往左走了几步"。抖动发生在两个几乎等价的缺口之间, 这个计数是打破
    平局所需的唯一历史量, 而且推理时能从自己的动作里算出来, 不需要老师。
    """
    sign = direction_sign(action)
    if sign == 0:
        return 0
    if sign == direction_sign(prev_action):
        return min(max(1, int(cap)), max(0, int(age)) + 1)
    return 1



def tta_frames(state: GameState, spec: GapSpec | None = None) -> np.ndarray:
    """每车道"还有多少帧被占" (帧, 越大越安全)。"""
    spec = spec or GapSpec()
    pos, vel = _bullets(state)
    py = _player_y(state)
    return _arrival_times(py, pos, vel, spec)[0]


def _player_x(state: GameState) -> float:
    return float(state.player_x) if state.player_x is not None else PLAYFIELD_WIDTH / 2


def _player_y(state: GameState) -> float:
    from ..types import PLAYFIELD_HEIGHT

    return float(state.player_y) if state.player_y is not None else PLAYFIELD_HEIGHT - 32


def macro_x(x: float, action: int, frames: int = COMMIT_FRAMES) -> float:
    """朝某方向承诺 ``frames`` 帧后的横坐标 (斜向按 45° 归一化)。"""
    ux, uy = DIRECTIONS[action]
    norm = math.hypot(ux, uy)
    dx = (ux / norm) if norm else 0.0
    return min(PLAYFIELD_WIDTH - 8.0, max(8.0, x + dx * PLAYER_SPEED * frames))


def reach_target(state: GameState, spec: GapSpec | None = None) -> float | None:
    """考虑"能不能及时赶到"之后, 值得去的目标车道中心 (绝对 x)。

    ``可用余量 - 赶路时间``: 站得住且来得及的车道才值得去, 已经站对了就不动
    (距离代价为 0, 天然占优)。这是下面三个口径共用的"目的地"。
    """
    spec = spec or GapSpec()
    tta = tta_frames(state, spec)
    if not np.any(tta > 0):
        return None
    px = _player_x(state)
    centers = _column_centers(spec)
    travel = np.abs(centers - px) / PLAYER_SPEED
    slack = tta - travel                      # 赶到之后还剩多少帧安全余量
    padded = np.floor(slack / REACH_PAD) * REACH_PAD
    best = float(padded.max())
    tied = np.flatnonzero(padded >= best - 1e-9)
    k = int(tied[np.argmin(np.abs(centers[tied] - px))])
    return float(centers[k])


def _direction_to(target: float | None, px: float, *, dead_zone: float = 10.0) -> int:
    if target is None:
        return 0
    delta = target - px
    if abs(delta) < dead_zone:
        return 0
    return 4 if delta > 0 else 3


def now_label(state: GameState, spec: GapSpec | None = None) -> int:
    """当前剖面里最空的一条车道 (不管赶不赶得到)。"""
    spec = spec or GapSpec()
    tta = tta_frames(state, spec)
    if not np.any(tta > 0):
        return 0
    px = _player_x(state)
    k = _best_column(tta / float(max(1, spec.tta_horizon)), px, spec)
    return _direction_to(float(_column_centers(spec)[k]), px)


def reach_label(state: GameState, spec: GapSpec | None = None) -> int:
    return _direction_to(reach_target(state, spec), _player_x(state))


def future_label(state: GameState, spec: GapSpec | None = None,
                 lookahead: int = 4) -> int:
    """前瞻标签: 朝某个方向连续走 ``lookahead`` 个宏动作, 取这条直线上最好的时刻。

    ``now`` / ``reach`` 都只看"此刻哪条车道最空", 缺口在 100px 外时给不出"现在就
    该往那边挪"的信号 —— 而每步只能走 26px。
    """
    spec = spec or GapSpec()
    tta = tta_frames(state, spec)
    if not np.any(tta > 0):
        return 0
    px = _player_x(state)
    cols = spec.tta_cols
    cell = float(PLAYFIELD_WIDTH) / cols

    def column(x: float) -> int:
        return int(min(cols - 1, max(0, x / cell)))

    best: tuple[tuple[float, float], int] | None = None
    for action in range(len(DIRECTIONS)):
        x = px
        score = -1e18
        for step in range(1, lookahead + 1):
            x = macro_x(x, action)
            score = max(score, float(tta[column(x)]) - step * COMMIT_FRAMES)
        quantised = math.floor(score / REACH_PAD) * REACH_PAD
        key = (-quantised, abs(macro_x(px, action) - px))
        if best is None or key < best[0]:
            best = (key, action)
    return 0 if best is None else best[1]


def slack_label(state: GameState, spec: GapSpec | None = None) -> int:
    """两堵墙的 maximin 余量。

    ``reach`` 只问"第一堵墙到达时我还有多少余量", 于是会把人往"现在最空、但下一堵
    墙马上又要封死"的位置送。这里把两件事取最小:

    * ``t1 - travel``: 赶过去之后, 第一堵墙还剩多少余量;
    * ``t2 - t1``: 站在那儿不动, 到第二堵墙之前还剩多少时间可以挪窝。

    第二项其实是"这地方让我待多久", 是第一项看不到的维度。仍然只是测量, 不含
    "该往哪走"。
    """
    spec = spec or GapSpec()
    pos, vel = _bullets(state)
    px = _player_x(state)
    t1, t2 = _arrival_times(_player_y(state), pos, vel, spec,
                            min_gap=SECOND_GAP_FRAMES)
    if not np.any(t1 > 0):
        return 0
    centers = _column_centers(spec)
    travel = np.abs(centers - px) / PLAYER_SPEED
    # "永远不会有第二次威胁" 是**最好的**情况, 不能算成 dwell=0 (那就是最差)。
    never = float(spec.tta_horizon + spec.tta_step)
    dwell = np.where(t2 >= never, float(spec.tta_horizon), t2 - t1)
    dwell = np.clip(dwell, 0.0, float(spec.tta_horizon))
    slack = np.minimum(t1 - travel, dwell)
    padded = np.floor(slack / REACH_PAD) * REACH_PAD
    best = float(padded.max())
    tied = np.flatnonzero(padded >= best - 1e-9)
    k = int(tied[np.argmin(np.abs(centers[tied] - px))])
    return _direction_to(float(centers[k]), px)


@dataclass
class GapCommitment:
    """有状态的"目标列承诺"。

    为什么必须加这一层
    ------------------
    ``reach`` 是单步贪心的: 每一帧都重新问"哪列最空"。当两个缺口的分数只差几帧时,
    自机往一边挪 6 帧后另一边就反超 —— 标签于是每 6 帧左右互搏一次。实测 18% 的决策
    在切换动作, 净位移约等于 0。策略忠实地学会了这个抖动 (策略=老师, 逐帧相同),
    所以修在标签侧。

    两种修法
    --------
    * ``reach_hold``: 目标列选定后, ``commit_frames`` 帧内不重算; 只有当前目标
      快要被弹幕覆盖 (``slack <= safety_margin``) 时才紧急换目标。
    * ``reach_hyst``: 不放固定时长, 而是要求新目标比当前目标好 ``hysteresis`` 帧
      才切换 (滞回)。分数按 ``REACH_PAD`` 量化, 避免 1 帧噪声触发换边。

    这是"决策状态", 不是游戏状态: 同一个 ``GameState`` 在不同的承诺状态下会得到
    不同标签。评估时必须每局新建/重置一个实例, 否则跨局状态会泄漏。
    """

    spec: GapSpec
    mode: str = "reach_hold"
    commit_frames: int = 18
    hysteresis: float = 10.0
    safety_margin: float = 0.0
    target: float | None = None
    hold_until: int = -1
    side: int = 0
    switches: int = 0
    decisions: int = 0

    def reset(self) -> None:
        self.target = None
        self.hold_until = -1
        self.side = 0
        self.switches = 0
        self.decisions = 0

    # ------------------------------------------------------------------
    def _slack(self, state: GameState) -> tuple[np.ndarray, np.ndarray, float]:
        tta = tta_frames(state, self.spec)
        centers = _column_centers(self.spec)
        px = _player_x(state)
        travel = np.abs(centers - px) / PLAYER_SPEED
        return centers, tta - travel, px

    @staticmethod
    def _nearest(centers: np.ndarray, x: float) -> int:
        return int(np.argmin(np.abs(centers - x)))

    def decide(self, state: GameState, frame: int | None = None) -> tuple[int, dict]:
        """返回 ``(动作下标, 诊断信息)``。"""
        centers, slack, px = self._slack(state)
        if slack.size == 0 or not np.any(slack > 0):
            return 0, {"target": None, "switched": False, "reason": "no_bullets"}

        padded = np.floor(slack / REACH_PAD) * REACH_PAD
        best = float(padded.max())
        tied = np.flatnonzero(padded >= best - 1e-9)
        best_idx = int(tied[np.argmin(np.abs(centers[tied] - px))])

        tick = int(frame) if frame is not None else self.decisions
        self.decisions += 1
        switched = False
        reason = "hold"

        if self.target is None:
            self.target = float(centers[best_idx])
            self.hold_until = tick + max(0, int(self.commit_frames))
            self.side = (1 if self.target > px else -1 if self.target < px else 0)
            switched = True
            reason = "init"
        else:
            current_idx = self._nearest(centers, self.target)
            current = float(slack[current_idx])
            unsafe = current <= float(self.safety_margin)
            if unsafe:
                new_idx = best_idx
                reason = "unsafe"
            elif self.mode == "reach_hyst":
                better = best > float(padded[current_idx]) + float(self.hysteresis)
                new_idx = best_idx if better else current_idx
                reason = "hysteresis" if better else "hold"
            elif tick < self.hold_until:
                new_idx = current_idx
                reason = "hold"
            else:
                new_idx = best_idx
                reason = "expired"

            if new_idx != current_idx:
                self.target = float(centers[new_idx])
                self.hold_until = tick + max(0, int(self.commit_frames))
                self.side = (1 if self.target > px else -1 if self.target < px else 0)
                switched = True
                self.switches += 1
            # 目标没变时**不重置**承诺计时: 承诺窗口从"上次换目标"开始算,
            # 否则每帧顺延会变成永远不换 (滑窗), 等于没有承诺。

        action = _direction_to(self.target, px)
        # 已经走到/越过目标列就停下, 不要为了回正而反向 —— 6 帧一步走 26px,
        # 不处理越界会在目标两侧来回摆, 把"承诺"变成新的抖动来源。
        delta = float(self.target) - px
        if self.side and delta * self.side <= 10.0:
            action = 0
        return action, {
            "target": self.target,
            "side": self.side,
            "switched": switched,
            "reason": reason,
            "best_score": best,
            "hold_until": self.hold_until,
            "decisions": self.decisions,
            "switches": self.switches,
        }


#: 口径 -> 实现 (``gap_label`` 的分发表; 加口径只改这里)
_LABELS = {
    "now": now_label,
    "reach": reach_label,
    "future": future_label,
    "slack": slack_label,
}


def gap_label(state: GameState, spec: GapSpec | None = None,
              mode: str = "reach") -> int:
    """把一个状态打上宏动作标签 (0~8, 与 :data:`MACRO_ACTIONS` 同序)。

    ``reach_hold`` / ``reach_hyst`` 是**有状态**口径, 同一帧在不同承诺状态下标签
    不同, 不能逐帧调用 —— 请用 :class:`GapCommitment` / ``GapTeacher``。
    """
    try:
        fn = _LABELS[mode]
    except KeyError:
        raise KeyError(
            f"未知的逐帧标签口径: {mode!r}; 可用: {', '.join(LABEL_MODES)}; "
            f"有状态口径 {', '.join(HOLD_MODES)} 请用 GapCommitment"
        ) from None
    return int(fn(state, spec))


__all__ = [
    "COMMIT_FRAMES",
    "GapCommitment",
    "HOLD_MODES",
    "LABEL_MODES",
    "MACRO_ACTIONS",
    "REACH_PAD",
    "SECOND_GAP_FRAMES",
    "TEACHER_MODES",
    "direction_sign",
    "future_label",
    "gap_label",
    "macro_x",
    "now_label",
    "reach_label",
    "reach_target",
    "slack_label",
    "tta_frames",
    "update_action_age",
]
