"""弹幕速度追踪: 用**槽位号**作为弹幕身份, 跨帧差分得到 (vx, vy)。

弹幕数组每个活动槽位对应一颗弹; 读取器已给出 ``active_slots``(与 ``state.bullets``
同序), 所以同一槽位在两帧间就是同一颗弹, 直接相减再除以帧差即可, 不需要匹配。

    v = (pos_now - pos_prev) / (frame_now - frame_prev)     单位: 像素/游戏帧

结果写进 ``state.raw["bullet_velocity"]``, 与 ``state.bullets`` **同序对齐**,
供 ``training.features.encode_state(use_velocity=True)`` 使用。

防护:
  * dt 不合理 (<=0 或过大) -> 该帧速度置 0;
  * 槽位被回收/换弹导致位移过大 -> 判为不可信, 置 0。
"""
from __future__ import annotations

from typing import Any


class BulletTracker:
    def __init__(self, *, max_dt: int = 120, max_speed: float = 15.0) -> None:
        self.max_dt = int(max_dt)
        self.max_speed = float(max_speed)
        self._prev: dict[int, tuple[float, float]] = {}
        self._prev_frame: int | None = None

    def reset(self) -> None:
        self._prev = {}
        self._prev_frame = None

    def update(self, state: Any) -> Any:
        """更新速度并把 ``state.raw["bullet_velocity"]`` 设为与 bullets 同序的速度列表。"""
        entities = (getattr(state, "raw", None) or {}).get("entities", {})
        detail = entities.get("bullets", {}) if isinstance(entities, dict) else {}
        slots = detail.get("active_slots") or []
        bullets = list(getattr(state, "bullets", []) or [])
        frame = int(getattr(state, "frame", 0) or 0)
        dt = (frame - self._prev_frame) if self._prev_frame is not None else 0

        velocities: list[tuple[float, float]] = []
        for i, (x, y) in enumerate(bullets):
            slot = slots[i] if i < len(slots) else None
            vx = vy = 0.0
            if slot is not None and 0 < dt <= self.max_dt and slot in self._prev:
                px, py = self._prev[slot]
                vx, vy = (float(x) - px) / dt, (float(y) - py) / dt
                if (vx * vx + vy * vy) ** 0.5 > self.max_speed:
                    vx = vy = 0.0  # 槽位回收/换弹, 不可信
            velocities.append((vx, vy))

        # 后端**自带**弹速时 (mock 的 Bullet 有 vx/vy, 但没有 active_slots),
        # 必须原样保留 —— 否则差分得到的全 0 会覆盖掉真实速度,
        # 让所有"弹道外推"类观测 (占用栅格 / 规划器 / 危险梯度) 全部失效。
        if not slots:
            provided = None
            if isinstance(getattr(state, "raw", None), dict):
                provided = state.raw.get("bullet_velocity")
            if isinstance(provided, (list, tuple)) and len(provided) == len(bullets):
                self._prev = {i: (float(x), float(y)) for i, (x, y) in enumerate(bullets)}
                self._prev_frame = frame
                return state

        # 记录本帧 (键用槽位; 没有槽位信息时退化为按序号)
        if slots:
            self._prev = {int(slot): (float(x), float(y))
                          for slot, (x, y) in zip(slots, bullets)}
        else:
            self._prev = {i: (float(x), float(y)) for i, (x, y) in enumerate(bullets)}
        self._prev_frame = frame

        if isinstance(getattr(state, "raw", None), dict):
            state.raw["bullet_velocity"] = velocities
        return state
