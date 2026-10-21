"""视觉弹幕检测 (兜底方案, 不依赖内存偏移, 对任意作品可用).

思路:
  1. 把游戏场地裁出来 (capture.screenshot.crop_playfield);
  2. 弹幕 = 场地里**又亮又饱和的小色块**; 背景花纹/敌机是大连通域, 被尺寸过滤掉;
  3. 用"游程 + 并查集"做连通域 (不依赖 scipy), 取质心;
  4. 质心像素 -> 游戏内部坐标: internal = pixel / crop_scale + origin。

注意: 视觉拿不到速度, 也可能把"道具(P/B/F 方块)"误当成弹幕; 需要时用颜色/形状再过滤。

接入点: ``backends/memory.py`` 在 ``vision.enabled=true`` 且内存里读不到弹幕
(该作品没有弹幕偏移, 或实体读取报错) 时调用 :func:`bullets_from_frame`。
默认关闭 —— 每次读数多一次截图, 真机 60fps 训练时开销明显。
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable

import numpy as np
from PIL import Image, ImageDraw

if TYPE_CHECKING:  # pragma: no cover - 只为类型提示, 避免运行时循环导入
    from ..games.registry import GameSpec

__all__ = [
    "annotate",
    "bullets_from_frame",
    "connected_components",
    "detect_blobs",
    "detect_bullets",
    "find_hitbox",
]

#: th11 内部坐标 = 场地内像素 + (32,32) (由 player_x/y 标定反推, 可被 hitbox 标定覆盖)
DEFAULT_ORIGIN = (32.0, 32.0)


def _row_runs(row: np.ndarray) -> list[tuple[int, int]]:
    d = np.diff(row.astype(np.int8))
    starts = (np.flatnonzero(d == 1) + 1).tolist()
    ends = (np.flatnonzero(d == -1) + 1).tolist()
    if row.size and row[0]:
        starts.insert(0, 0)
    if row.size and row[-1]:
        ends.append(int(row.size))
    return list(zip(starts, ends))


def connected_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """游程 + 并查集的连通域标记 (纯 numpy/python)。返回 (labels, count)。"""
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    parent: list[int] = [0]

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    nxt = 0
    prev_runs: list[tuple[int, int, int]] = []  # (x0, x1, label)
    for y in range(h):
        runs = _row_runs(mask[y])
        cur: list[tuple[int, int, int]] = []
        for x0, x1 in runs:
            overlapping = [lab for (p0, p1, lab) in prev_runs if p0 < x1 and x0 < p1]
            if overlapping:
                lab = min(overlapping)
                for other in overlapping:
                    union(lab, other)
            else:
                nxt += 1
                lab = nxt
                parent.append(lab)
            labels[y, x0:x1] = lab
            cur.append((x0, x1, lab))
        prev_runs = cur
    if nxt:
        roots = np.array([find(i) for i in range(nxt + 1)], dtype=np.int32)
        labels = roots[labels]
    return labels, nxt


def detect_blobs(
    image: Image.Image,
    *,
    bright_thr: int = 170,
    sat_thr: int = 40,
    min_area: int = 2,
    max_area: int = 300,
    max_dim: int = 40,
) -> list[dict[str, Any]]:
    """返回场地图里的小亮块: [{x, y, area, w, h, r, g, b}] (像素坐标, 质心)。"""
    arr = np.asarray(image.convert("RGB"), dtype=np.int16)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    mx = arr.max(axis=2)
    mn = arr.min(axis=2)
    sat = mx - mn
    mask = (mx >= bright_thr) & (sat >= sat_thr)
    if not mask.any():
        return []
    labels, n = connected_components(mask)
    if n == 0:
        return []
    ys, xs = np.nonzero(mask)
    lab = labels[ys, xs]
    order = np.argsort(lab)
    lab_s = lab[order]
    ys_s = ys[order]
    xs_s = xs[order]
    bounds = np.searchsorted(lab_s, np.arange(1, n + 1))
    bounds = np.append(bounds, len(lab_s))
    out: list[dict[str, Any]] = []
    for i in range(n):
        s, e = bounds[i], bounds[i + 1]
        area = e - s
        if area < min_area or area > max_area:
            continue
        yy = ys_s[s:e]
        xx = xs_s[s:e]
        h = int(yy.max() - yy.min() + 1)
        w = int(xx.max() - xx.min() + 1)
        if w > max_dim or h > max_dim:
            continue
        out.append(
            {
                "x": float(xx.mean()),
                "y": float(yy.mean()),
                "area": int(area),
                "w": w,
                "h": h,
                "r": int(r[yy, xx].mean()),
                "g": int(g[yy, xx].mean()),
                "b": int(b[yy, xx].mean()),
            }
        )
    return out


def detect_bullets(
    image: Image.Image,
    *,
    crop_scale: float = 1.0,
    origin: tuple[float, float] = DEFAULT_ORIGIN,
    **kwargs: Any,
) -> list[tuple[float, float]]:
    """检测**已裁剪的场地**并换算成游戏内部坐标 [(x, y), ...].

    ``origin`` 是"场地在窗口内的偏移", 输出是**窗口系**坐标 (东方正作内部坐标系)。
    要和 ``memory/entities.py`` 的实体坐标 (384x448 场地系) 混用, 请走
    :func:`bullets_from_frame` —— 它负责裁剪并把原点归零。
    """
    blobs = detect_blobs(image, **kwargs)
    ox, oy = origin
    return [(bl["x"] / crop_scale + ox, bl["y"] / crop_scale + oy) for bl in blobs]


def bullets_from_frame(
    image: Image.Image,
    spec: "GameSpec",
    *,
    playfield: list[int] | None = None,
    window_size: list[int] | None = None,
    **kwargs: Any,
) -> list[tuple[float, float]]:
    """从**整窗截图**里检测弹幕, 返回 384x448 场地坐标.

    与 :func:`detect_bullets` 的区别: 先按 ``spec.playfield`` 裁出场地, 再做检测。
    裁剪后原点就是场地左上角, 所以 ``origin=(0, 0)``、``crop_scale = 裁剪宽 / 384``。
    这样得到的坐标与 ``memory/entities.py`` 的实体坐标在同一个坐标系里,
    ``dodge`` 宏 / 危险度奖励 / 可视化都能直接混用。
    """
    from ..types import PLAYFIELD_WIDTH
    from .screenshot import crop_playfield

    box = playfield if playfield is not None else list(getattr(spec, "playfield", []) or [])
    size = window_size if window_size is not None else getattr(spec, "window_size", None)
    cropped = crop_playfield(image, box or [32, 16, 416, 464], size)
    if cropped.width <= 0:
        return []
    scale = cropped.width / PLAYFIELD_WIDTH
    return detect_bullets(cropped, crop_scale=scale or 1.0, origin=(0.0, 0.0), **kwargs)


def find_hitbox(image: Image.Image, *, bright: int = 150) -> tuple[float, float] | None:
    """找自机判定点 (按住 focus 时的红点): 返回像素坐标质心。"""
    arr = np.asarray(image.convert("RGB"), dtype=np.int16)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    mask = (r >= bright) & (g <= 90) & (b <= 90) & (r - g >= 80) & (r - b >= 80)
    if not mask.any():
        return None
    ys, xs = np.nonzero(mask)
    return float(xs.mean()), float(ys.mean())


def annotate(image: Image.Image, points: Iterable[tuple[float, float]], origin=(0.0, 0.0), scale=1.0) -> Image.Image:
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    ox, oy = origin
    for x, y in points:
        px = (x - ox) * scale
        py = (y - oy) * scale
        draw.ellipse([px - 6, py - 6, px + 6, py + 6], outline=(0, 255, 0), width=2)
    return out
