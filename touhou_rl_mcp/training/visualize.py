"""训练可视化 —— 让"训练过程"看得见.

产出三类图:
  * :func:`draw_learning_curve`  学习曲线 (回报 + 到达面数)
  * :func:`draw_dashboard`       仪表盘: 当前游戏画面 + 学习曲线 + 实时统计
  * :func:`draw_frame_strip`     一局内若干帧的拼图 (看策略到底在干什么)

它们既是 :mod:`tools.watch_training` 实时监视窗口的内容, 也是 MCP 工具
``th_train_dashboard`` 的返回值 —— Agent 和人都能"看到"训练进展。
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

from PIL import Image, ImageDraw

from ..capture.visualize import _font as load_font

BG = (16, 18, 28)
PANEL = (24, 27, 40)
GRID = (44, 50, 70)
AXIS = (110, 125, 160)
FG = (226, 232, 245)
DIM = (140, 155, 180)
RAW = (90, 150, 220)
SMOOTH = (110, 235, 150)
STAGE = (255, 190, 90)
EVAL = (255, 120, 200)
WARN = (255, 120, 120)


def _moving_average(values: Sequence[float], window: int) -> list[float]:
    if not values:
        return []
    window = max(1, int(window))
    out: list[float] = []
    running = 0.0
    for i, value in enumerate(values):
        running += value
        if i >= window:
            running -= values[i - window]
        out.append(running / min(i + 1, window))
    return out


def _nice_range(low: float, high: float, *, pad: float = 0.08) -> tuple[float, float]:
    if not math.isfinite(low) or not math.isfinite(high):
        return -1.0, 1.0
    if high - low < 1e-9:
        high = low + 1.0
    span = high - low
    return low - span * pad, high + span * pad


def _axis_ticks(low: float, high: float, count: int = 4) -> list[float]:
    """生成整齐的刻度值 (整数范围就用整数刻度, 避免出现重复标签)."""
    span = high - low
    if span <= 0:
        return [low]
    if span <= 10 and float(span).is_integer():
        step = max(1, int(round(span / count)))
        ticks = []
        value = float(int(low))
        while value <= high + 1e-9:
            ticks.append(value)
            value += step
        return ticks
    return [low + span * i / count for i in range(count + 1)]


def draw_learning_curve(
    episodes: Sequence[Any],
    evaluations: Sequence[dict[str, Any]] | None = None,
    *,
    width: int = 960,
    height: int = 560,
    smooth_window: int = 20,
    title: str = "学习曲线",
) -> Image.Image:
    """画三张子图: 每局回报 / 每局步数(存活时长) / 到达面数.

    "每局步数"是早期最灵敏的进步指标 —— 策略学会躲弹幕后, 第一件事就是活得
    更久, 而回报数字可能因为危险度惩罚的尺度变化而看不出来。
    """
    canvas = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(canvas)
    font_small = load_font(12)
    font_mid = load_font(14)

    returns = [float(getattr(e, "return_", 0.0)) for e in episodes]
    steps = [float(getattr(e, "steps", 0) or 0) for e in episodes]
    stages = [float(getattr(e, "max_stage", 0) or 0) for e in episodes]

    left, right = 70, width - 24
    header = 30
    footer = 30
    panel_gap = 26
    panel_height = (height - header - footer - panel_gap * 2) // 3

    draw.text((left, 8), title, font=load_font(16), fill=FG)
    if not returns:
        draw.text((left, header + 20), "还没有数据 (训练尚未开始)", font=font_mid, fill=DIM)
        return canvas

    panels = [
        ("每局回报", returns, RAW, SMOOTH, "", 2),
        ("每局步数 (存活时长)", steps, (120, 200, 255), (170, 230, 255), "", 0),
        ("到达面数", stages, STAGE, None, " 面", 0),
    ]

    for panel_index, (label, values, dot_color, line_color, unit, digits) in enumerate(panels):
        top = header + panel_index * (panel_height + panel_gap)
        bottom = top + panel_height
        low, high = _nice_range(min(values), max(values))

        def to_y(value: float, top=top, bottom=bottom, low=low, high=high) -> float:
            return bottom - (value - low) / (high - low or 1.0) * (bottom - top)

        def to_x(index: int) -> float:
            return left + (right - left) * (index / max(1, len(values) - 1))

        for tick in _axis_ticks(low, high):
            y = to_y(tick)
            if not (top - 1 <= y <= bottom + 1):
                continue
            draw.line([(left, y), (right, y)], fill=GRID)
            text = f"{tick:.{digits}f}{unit}"
            draw.text((6, y - 7), text, font=font_small, fill=AXIS)

        if len(values) <= 400:
            for i, value in enumerate(values):
                x, y = to_x(i), to_y(value)
                draw.ellipse([x - 1.5, y - 1.5, x + 1.5, y + 1.5], fill=dot_color)
        else:
            draw.line([(to_x(i), to_y(v)) for i, v in enumerate(values)], fill=dot_color)

        smoothed = _moving_average(values, smooth_window)
        if line_color and len(smoothed) >= 2:
            draw.line([(to_x(i), to_y(v)) for i, v in enumerate(smoothed)], fill=line_color, width=2)

        for evaluation in evaluations or []:
            index = int(evaluation.get("at_episode", 0))
            if 0 <= index < len(values):
                x = to_x(index)
                draw.line([(x, top), (x, bottom)], fill=EVAL)

        draw.rectangle([left, top, right, bottom], outline=AXIS)
        draw.text((left, top - 18), label, font=font_mid, fill=line_color or dot_color)

    legend = [
        (RAW, f"每局回报 (共 {len(returns)} 局)"),
        (SMOOTH, f"滑动平均 ({smooth_window} 局)"),
        (STAGE, "到达面数"),
        (EVAL, "贪心评估"),
    ]
    x = left
    for color, text in legend:
        draw.line([(x, height - 16), (x + 18, height - 16)], fill=color, width=3)
        draw.text((x + 24, height - 24), text, font=font_small, fill=DIM)
        x += 30 + int(draw.textlength(text, font=font_small)) + 18
    return canvas


def draw_frame_strip(
    frames: Sequence[Image.Image],
    labels: Sequence[str] | None = None,
    *,
    target_height: int = 220,
    title: str = "",
) -> Image.Image:
    """把一局里的若干帧横向拼起来 (用来肉眼检查策略行为)."""
    if not frames:
        return Image.new("RGB", (320, 120), BG)
    scaled = []
    for frame in frames:
        ratio = target_height / frame.height
        scaled.append(frame.resize((max(1, int(frame.width * ratio)), target_height), Image.LANCZOS))
    gap = 6
    width = sum(f.width for f in scaled) + gap * (len(scaled) - 1)
    header = 26 if title else 0
    canvas = Image.new("RGB", (width, target_height + header), BG)
    x = 0
    for i, frame in enumerate(scaled):
        canvas.paste(frame, (x, header))
        x += frame.width + gap
    draw = ImageDraw.Draw(canvas)
    if title:
        draw.text((6, 5), title, font=load_font(15), fill=FG)
    if labels:
        x = 0
        font = load_font(11)
        for i, frame in enumerate(scaled):
            if i < len(labels):
                draw.text((x + 4, header + 4), labels[i], font=font, fill=(255, 255, 255))
            x += frame.width + gap
    return canvas


def draw_dashboard(
    frame: Image.Image | None,
    episodes: Sequence[Any],
    status: dict[str, Any],
    *,
    width: int = 1440,
    frame_width: int = 520,
    title: str = "",
) -> Image.Image:
    """仪表盘: 左边是当前游戏画面, 右边是学习曲线, 上面一行是实时统计."""
    header_h = 74
    curve_h = 560
    height = header_h + max(curve_h, 0) + 34

    canvas = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(canvas)
    font_mid = load_font(14)
    font_big = load_font(17)

    # ---- 顶部统计 ----
    draw.rectangle([0, 0, width, header_h - 8], fill=PANEL)
    running = "运行中" if status.get("running") else ("已暂停" if status.get("paused") else "已停止")
    draw.text((14, 8), f"{title or '训练仪表盘'}   [{running}]", font=font_big, fill=FG)

    def fmt(value: Any, digits: int = 2, suffix: str = "") -> str:
        if value is None:
            return "-"
        if isinstance(value, float):
            return f"{value:.{digits}f}{suffix}"
        return f"{value}{suffix}"

    cells = [
        ("局数", fmt(status.get("episodes_done"), suffix="")),
        ("总步数", fmt(status.get("total_steps"))),
        ("近50局回报", fmt(status.get("mean_return_last50"))),
        ("最好回报", fmt(status.get("best_return"))),
        ("平均面数", fmt(status.get("mean_stage_last50"))),
        ("最好面数", fmt(status.get("best_stage"))),
        ("步/秒", fmt(status.get("steps_per_sec"), 0)),
        ("已用", fmt(status.get("elapsed_s"), 0, " s")),
    ]
    x = 14
    for name, value in cells:
        draw.text((x, 34), name, font=font_mid, fill=DIM)
        draw.text((x, 50), value, font=font_mid, fill=FG)
        x += 128
    error = status.get("error")
    if error:
        draw.text((14, 50), "训练出错, 详见日志", font=font_mid, fill=WARN)

    # ---- 左侧: 当前画面 ----
    if frame is not None:
        ratio = frame_width / frame.width
        preview = frame.resize((frame_width, max(1, int(frame.height * ratio))), Image.LANCZOS)
        canvas.paste(preview, (12, header_h))
        draw.rectangle([12, header_h, 12 + preview.width, header_h + preview.height], outline=AXIS)
        draw.text((16, header_h + preview.height + 6), "当前画面 (策略正在打的这一帧)", font=font_mid, fill=DIM)
    else:
        draw.text((16, header_h + 20), "尚未抓到画面", font=font_mid, fill=DIM)

    # ---- 右侧: 学习曲线 ----
    curve = draw_learning_curve(
        episodes,
        status.get("evaluations"),
        width=width - frame_width - 40,
        height=curve_h,
        title="学习曲线",
    )
    canvas.paste(curve, (frame_width + 26, header_h))

    # ---- 底部: 最近几局 ----
    recent = status.get("recent_episodes") or []
    if recent:
        y = header_h + curve_h + 8
        line = "   ".join(
            f"ep{r['episode']}: 回报{r['return']:.0f} 面{r.get('max_stage')} 步{r['steps']}"
            + (" ★通关" if r.get("cleared") else "")
            for r in recent[-4:]
        )
        draw.text((14, min(y, height - 20)), line, font=font_mid, fill=DIM)
    return canvas


def draw_progress(
    episodes: Sequence[Any],
    *,
    width: int = 960,
    height: int = 720,
    smooth_window: int = 20,
    title: str = "训练进度",
) -> Image.Image:
    """四张子图: 存活步数 / 到达面数 / 每局回报 / 掉残次数。

    训练日志 + 这张图就是"看智能体哪里不行、据此调参"的主要依据:
      * 存活步数不涨 -> 它还没学会活;
      * 面数不涨但存活在涨 -> 火力/推进不足 (奖励里过面权重要加大);
      * 掉残次数高 -> 危险度惩罚 / 存活奖励要调;
      * 回报和存活背离 -> 奖励尺度问题 (看 train.jsonl 的 reward_parts)。
    """
    canvas = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(canvas)
    font_small = load_font(12)
    font_mid = load_font(14)

    steps = [float(getattr(e, "steps", 0) or 0) for e in episodes]
    stages = [float(getattr(e, "max_stage", 0) or 0) for e in episodes]
    returns = [float(getattr(e, "return_", 0.0)) for e in episodes]
    misses = [float(getattr(e, "misses", 0) or 0) for e in episodes]

    left, right = 74, width - 24
    header = 30
    footer = 28
    gap = 18
    panel_height = (height - header - footer - gap * 3) // 4

    draw.text((left, 8), title, font=load_font(16), fill=FG)
    if not steps:
        draw.text((left, header + 20), "还没有数据 (训练尚未开始)", font=font_mid, fill=DIM)
        return canvas

    panels = [
        ("存活步数 (每局)", steps, (120, 200, 255), (170, 230, 255), "", 0),
        ("到达面数", stages, STAGE, None, " 面", 0),
        ("每局回报", returns, RAW, SMOOTH, "", 1),
        ("掉残次数 (越少越好)", misses, WARN, (255, 180, 120), "", 0),
    ]
    for pi, (label, values, dot, line, unit, digits) in enumerate(panels):
        top = header + pi * (panel_height + gap)
        bottom = top + panel_height
        low, high = _nice_range(min(values), max(values))

        def to_y(v, top=top, bottom=bottom, low=low, high=high):
            return bottom - (v - low) / (high - low or 1.0) * (bottom - top)

        def to_x(i, n=len(values)):
            return left + (right - left) * (i / max(1, n - 1))

        for tick in _axis_ticks(low, high):
            y = to_y(tick)
            if top - 1 <= y <= bottom + 1:
                draw.line([(left, y), (right, y)], fill=GRID)
                draw.text((6, y - 7), f"{tick:.{digits}f}{unit}", font=font_small, fill=AXIS)

        if len(values) <= 400:
            for i, v in enumerate(values):
                x, y = to_x(i), to_y(v)
                draw.ellipse([x - 1.5, y - 1.5, x + 1.5, y + 1.5], fill=dot)
        else:
            draw.line([(to_x(i), to_y(v)) for i, v in enumerate(values)], fill=dot)
        smoothed = _moving_average(values, smooth_window)
        if line and len(smoothed) >= 2:
            draw.line([(to_x(i), to_y(v)) for i, v in enumerate(smoothed)], fill=line, width=2)

        draw.rectangle([left, top, right, bottom], outline=AXIS)
        draw.text((left, top - 16), label, font=font_mid, fill=line or dot)

    draw.text((left, height - 20), f"共 {len(steps)} 局 | 平均存活 "
              f"{sum(steps)/len(steps):.1f} 步 | 最远 {int(max(stages))} 面 | "
              f"平均掉残 {sum(misses)/len(misses):.2f}", font=font_small, fill=DIM)
    return canvas


def save_png(image: Image.Image, out_dir: str | Path | None, *, name: str) -> Path:
    from ..config import runtime_dir

    directory = Path(out_dir) if out_dir else runtime_dir() / "training"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    image.save(path)
    return path
