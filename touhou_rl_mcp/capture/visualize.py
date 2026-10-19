"""状态可视化 —— 把游戏状态画成一张图.

用途:
  * 给 LLM Agent 一张"结构化"的画面 (比原始截图更容易理解: 有坐标网格、
    自机位置、判定点、弹幕、Boss 血条、数值 HUD)
  * 训练时快速肉眼 debug

游戏中坐标约定: 场地 384x448, 原点在场地左上角, y 轴向下。
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Iterable, Sequence

from PIL import Image, ImageDraw, ImageFont

from ..games.registry import GameSpec
from ..types import GameState, PLAYFIELD_HEIGHT, PLAYFIELD_WIDTH

# 调色板
BG = (12, 12, 24)
GRID = (34, 40, 64)
PLAYFIELD_BORDER = (90, 110, 170)
BULLET = (255, 96, 96)
BULLET_CORE = (255, 235, 235)
ENEMY = (120, 220, 255)
BOSS = (255, 180, 80)
PLAYER = (150, 255, 180)
HITBOX = (255, 60, 60)
TRAIL = (60, 160, 110)
HUD_BG = (18, 18, 30)
HUD_FG = (220, 230, 245)
HUD_DIM = (140, 155, 180)

_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
    r"C:\Windows\Fonts\segoeui.ttf",
]


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in _FONT_CANDIDATES:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


class Trail:
    """自机轨迹缓存 (可视化用)."""

    def __init__(self, length: int = 180) -> None:
        self.points: deque[tuple[float, float]] = deque(maxlen=max(2, length))

    def push(self, x: float | None, y: float | None) -> None:
        if x is not None and y is not None:
            self.points.append((float(x), float(y)))

    def clear(self) -> None:
        self.points.clear()


def render_state(
    state: GameState,
    spec: GameSpec | None = None,
    *,
    scale: float = 1.0,
    show_bullets: bool = True,
    show_grid: bool = True,
    show_hud: bool = True,
    trail: Iterable[tuple[float, float]] | None = None,
    title: str = "",
) -> Image.Image:
    """把一帧状态画成图片 (含坐标网格 + HUD)."""
    if spec is not None and spec.playfield and spec.playfield[2] - spec.playfield[0] > 0:
        field_w = float(spec.playfield[2] - spec.playfield[0])
        field_h = float(spec.playfield[3] - spec.playfield[1])
    else:
        field_w, field_h = PLAYFIELD_WIDTH, PLAYFIELD_HEIGHT

    scale = max(0.25, float(scale))
    pw, ph = int(field_w * scale), int(field_h * scale)
    hud_w = int(190 * scale) if show_hud else 0
    canvas = Image.new("RGB", (pw + hud_w, ph), BG)
    draw = ImageDraw.Draw(canvas, "RGBA")

    # 场地背景
    draw.rectangle([0, 0, pw, ph], fill=(10, 10, 20))

    if show_grid:
        step = int(48 * scale)
        for x in range(step, pw, step):
            draw.line([(x, 0), (x, ph)], fill=GRID, width=1)
        for y in range(step, ph, step):
            draw.line([(0, y), (pw, y)], fill=GRID, width=1)
        # 安全参考线: 常见停驻高度 y≈400
        sweet = int(400 * scale)
        draw.line([(0, sweet), (pw, sweet)], fill=(50, 70, 60), width=1)

    def px(x: float) -> float:
        return x * scale

    def py(y: float) -> float:
        return y * scale

    # Boss 血条
    if state.boss_hp is not None and state.boss_max_hp:
        ratio = max(0.0, min(1.0, state.boss_hp / max(1, state.boss_max_hp)))
        bar_w = pw - int(40 * scale)
        x0, y0 = int(20 * scale), int(8 * scale)
        draw.rectangle([x0, y0, x0 + bar_w, y0 + max(3, int(5 * scale))], fill=(50, 30, 30))
        draw.rectangle([x0, y0, x0 + int(bar_w * ratio), y0 + max(3, int(5 * scale))], fill=(230, 90, 90))
    if state.boss_hp is not None:
        bx, by = px(field_w / 2), py(70)
        radius = 11 * scale
        draw.ellipse([bx - radius, by - radius, bx + radius, by + radius], fill=(*BOSS, 210))
        draw.ellipse([bx - 3 * scale, by - 3 * scale, bx + 3 * scale, by + 3 * scale], fill=(255, 255, 255))

    # 敌机
    for ex, ey in state.enemy_positions:
        size = 5 * scale
        draw.rectangle(
            [px(ex) - size, py(ey) - size, px(ex) + size, py(ey) + size],
            fill=(*ENEMY, 220),
        )

    # 弹幕
    if show_bullets and state.bullets:
        radius = max(1.5, 3.0 * scale)
        for bx, by in state.bullets:
            cx, cy = px(bx), py(by)
            draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=(*BULLET, 200))
            if scale >= 0.9:
                draw.point((cx, cy), fill=BULLET_CORE)

    # 自机轨迹
    if trail is not None:
        points = [(px(x), py(y)) for x, y in trail]
        if len(points) >= 2:
            draw.line(points, fill=(*TRAIL, 160), width=max(1, int(1.5 * scale)))

    # 自机
    if state.player_x is not None and state.player_y is not None:
        cx, cy = px(state.player_x), py(state.player_y)
        body = 7.0 * scale
        draw.polygon(
            [(cx, cy - body), (cx - body * 0.8, cy + body * 0.7), (cx + body * 0.8, cy + body * 0.7)],
            fill=(*PLAYER, 220),
        )
        dot = max(1.0, 2.0 * scale)
        draw.ellipse([cx - dot, cy - dot, cx + dot, cy + dot], fill=HITBOX)

    draw.rectangle([0, 0, pw - 1, ph - 1], outline=PLAYFIELD_BORDER, width=1)

    # ---------------- HUD ----------------
    if show_hud:
        draw.rectangle([pw, 0, pw + hud_w - 1, ph - 1], fill=HUD_BG)
        draw.line([(pw, 0), (pw, ph)], fill=PLAYFIELD_BORDER, width=1)
        lines = _hud_lines(state, title)
        y = int(8 * scale)
        for text, color, size in lines:
            font = _font(max(9, int(size * scale)))
            draw.multiline_text((pw + int(8 * scale), y), text, font=font, fill=color)
            y += int((size + 5) * scale) * (text.count("\n") + 1)
        draw.text(
            (pw + int(8 * scale), ph - int(18 * scale)),
            "384x448 field",
            font=_font(max(9, int(10 * scale))),
            fill=HUD_DIM,
        )
    return canvas


def _hud_lines(state: GameState, title: str) -> list[tuple[str, tuple[int, int, int], int]]:
    def fmt(value: Any, digits: int = 0) -> str:
        if value is None:
            return "-"
        if isinstance(value, float):
            return f"{value:.{digits}f}"
        return str(value)

    lines: list[tuple[str, tuple[int, int, int], int]] = []
    if title:
        lines.append((_wrap(title, 20), HUD_FG, 12))
    lines.append((f"帧    {state.frame}", HUD_DIM, 11))
    state_label = "游戏中" if state.in_game else "非游戏中"
    lines.append((f"状态  {state_label}", HUD_FG, 12))
    lines.append((f"关卡  {fmt(state.stage)}  面", HUD_FG, 13))
    lines.append((f"残机  {fmt(state.lives)}", HUD_FG, 13))
    lines.append((f"灵力  {fmt(state.power, 2)}", HUD_FG, 13))
    lines.append((f"分数  {fmt(state.score)}", HUD_FG, 13))
    lines.append((f"擦弹  {fmt(state.graze)}", HUD_FG, 12))
    if state.char_name:
        lines.append((f"自机  {state.char_name}", HUD_FG, 12))
    if state.spell_name:
        lines.append(("符卡", HUD_DIM, 11))
        lines.append((_wrap(state.spell_name, 11), HUD_FG, 12))
    if state.player_x is not None:
        lines.append((f"坐标  ({fmt(state.player_x, 1)}, {fmt(state.player_y, 1)})", HUD_DIM, 11))
    lines.append((f"弹数  {fmt(state.bullet_count)}", HUD_DIM, 11))
    if state.boss_hp is not None:
        lines.append((f"Boss  {fmt(state.boss_hp)}", HUD_DIM, 11))
    return lines


def _wrap(text: str, width: int) -> str:
    return "\n".join(text[i : i + width] for i in range(0, len(text), width))


def render_window(
    state: GameState,
    spec: GameSpec | None = None,
    *,
    options: dict[str, Any] | None = None,
    trail: Iterable[tuple[float, float]] | None = None,
    title: str = "",
) -> Image.Image:
    """合成一张"仿游戏窗口"的 640x480 画面 (模拟器后端的截图就是它)."""
    opts = dict(options or {})
    scale = float(opts.get("scale", 1.0))
    field = render_state(
        state,
        spec,
        scale=scale,
        show_bullets=bool(opts.get("show_bullets", True)),
        show_grid=bool(opts.get("show_grid", True)),
        show_hud=bool(opts.get("show_hud", True)),
        trail=trail,
        title=title,
    )
    canvas_w = 640 if scale >= 1.0 else int(640 * scale)
    canvas_h = 480 if scale >= 1.0 else int(480 * scale)
    canvas = Image.new("RGB", (canvas_w, canvas_h), (0, 0, 0))
    # 场地在窗口内的偏移: (32, 16)
    offset_x = int(32 * scale) if scale >= 1.0 else int(field.height / 448 * 32)
    offset_y = int(16 * scale)
    canvas.paste(field, (offset_x, offset_y))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, canvas_w - 1, canvas_h - 1], outline=(70, 70, 90))
    return canvas


def save_png(image: Image.Image, out_dir: str | Path | None, *, prefix: str = "state") -> Path:
    import datetime as _dt

    from ..config import runtime_dir

    directory = Path(out_dir) if out_dir else runtime_dir() / "viz"
    directory.mkdir(parents=True, exist_ok=True)
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
    path = directory / f"{prefix}-{stamp}.png"
    image.save(path)
    return path


def image_to_png_bytes(image: Image.Image) -> bytes:
    import io

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
