"""窗口截图.

东方正作使用 Direct3D 渲染, 普通 ``PrintWindow`` 往往得到全黑画面, 所以这里
按"由好到坏"依次尝试多种抓取方式, 并自动跳过全黑结果:

    1. ``PrintWindow(hwnd, PW_RENDERFULLCONTENT)`` —— Win8.1+ 对 D3D 也有效
    2. ``PrintWindow(hwnd, 0)``
    3. 从窗口 DC ``BitBlt`` 客户区
    4. 从桌面 DC ``BitBlt`` 客户区 (最终兜底)

抓到的都统一**裁剪成客户区** (即游戏真正的渲染面: 640x480, 或按窗口缩放后的
768x576 / 960x720 等), 不含窗口边框和标题栏。

> 建议把游戏设为**窗口化** (游戏内 Options 或 ``thXXconfig.ini``)。独占全屏时
> 截图和按键注入都会变得不可靠。
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Any

from PIL import Image

from ..config import runtime_dir

PW_CLIENTONLY = 0x00000001
PW_RENDERFULLCONTENT = 0x00000002

try:  # pragma: no cover - 仅在 Windows 上可用
    import win32con
    import win32gui
    import win32ui

    _PYWIN32 = True
except Exception:  # pragma: no cover
    _PYWIN32 = False

import ctypes

_user32 = ctypes.WinDLL("user32", use_last_error=True)
#: 注意: pywin32 的 win32gui 并没有导出 PrintWindow, 必须自己绑 ctypes
_user32.PrintWindow.restype = ctypes.c_int
_user32.PrintWindow.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint]


def _print_window(hwnd: int, hdc: int, flags: int) -> bool:
    return bool(_user32.PrintWindow(ctypes.c_void_p(hwnd), ctypes.c_void_p(hdc), flags))


class CaptureError(RuntimeError):
    pass


def _require_pywin32() -> None:
    if not _PYWIN32:
        raise CaptureError(
            "截图功能需要 pywin32 (pip install pywin32); 模拟器后端不需要它"
        )


def client_rect_on_screen(hwnd: int) -> tuple[int, int, int, int]:
    """客户区在屏幕坐标系里的 ``(left, top, right, bottom)``."""
    left, top = win32gui.ClientToScreen(hwnd, (0, 0))
    _, _, width, height = win32gui.GetClientRect(hwnd)
    return int(left), int(top), int(left + width), int(top + height)


def _client_offset(hwnd: int) -> tuple[int, int, int, int, int, int]:
    """返回 ``(window_rect, client_offset_in_window, client_size)``."""
    rect = win32gui.GetWindowRect(hwnd)
    client = win32gui.GetClientRect(hwnd)
    client_size = (client[2] - client[0], client[3] - client[1])
    origin = win32gui.ClientToScreen(hwnd, (0, 0))
    offset = (origin[0] - rect[0], origin[1] - rect[1])
    return rect, offset, client_size


def _looks_blank(image: Image.Image, min_bright_ratio: float = 0.02, sample: tuple[int, int] = (80, 60)) -> bool:
    """判断一张图是否"没抓到画面" (全黑 / 全同色).

    只看灰度的**最亮像素比例**: D3D 抓取失败时整张图是纯黑, 偶尔飘进来的几个
    标题栏文字像素不足以骗过这个判据 (这一点早期版本踩过坑)。
    """
    if image.width == 0 or image.height == 0:
        return True
    raw = image.convert("L").resize(sample).tobytes()
    if not raw:
        return True
    if max(raw) - min(raw) <= 6:
        return True
    bright = sum(1 for value in raw if value > 24)
    return (bright / len(raw)) < min_bright_ratio


def _grab_with_printwindow(hwnd: int, flag: int, offset: tuple[int, int], size: tuple[int, int]) -> Image.Image:
    hwindc = win32gui.GetWindowDC(hwnd)
    try:
        srcdc = win32ui.CreateDCFromHandle(hwindc)
        memdc = srcdc.CreateCompatibleDC()
        bitmap = win32ui.CreateBitmap()
        rect = win32gui.GetWindowRect(hwnd)
        width, height = rect[2] - rect[0], rect[3] - rect[1]
        bitmap.CreateCompatibleBitmap(srcdc, width, height)
        memdc.SelectObject(bitmap)
        try:
            _print_window(hwnd, memdc.GetSafeHdc(), flag)
            info = bitmap.GetInfo()
            full = Image.frombuffer(
                "RGB", (info["bmWidth"], info["bmHeight"]), bitmap.GetBitmapBits(True), "raw", "BGRX", 0, 1
            ).copy()
        finally:
            try:
                win32gui.DeleteObject(bitmap.GetHandle())
            except Exception:
                pass
            memdc.DeleteDC()
    finally:
        win32gui.ReleaseDC(hwnd, hwindc)
    return full.crop((offset[0], offset[1], offset[0] + size[0], offset[1] + size[1]))


def _grab_with_bitblt(source_dc: Any, source_origin: tuple[int, int], size: tuple[int, int]) -> Image.Image:
    width, height = size
    if width <= 0 or height <= 0:
        raise CaptureError(f"无效的抓取尺寸: {size}")
    memdc = source_dc.CreateCompatibleDC()
    bitmap = win32ui.CreateBitmap()
    bitmap.CreateCompatibleBitmap(source_dc, width, height)
    memdc.SelectObject(bitmap)
    try:
        memdc.BitBlt((0, 0), (width, height), source_dc, source_origin, win32con.SRCCOPY)
        info = bitmap.GetInfo()
        return Image.frombuffer(
            "RGB", (info["bmWidth"], info["bmHeight"]), bitmap.GetBitmapBits(True), "raw", "BGRX", 0, 1
        ).copy()
    finally:
        try:
            win32gui.DeleteObject(bitmap.GetHandle())
        except Exception:
            pass
        memdc.DeleteDC()


def grab_window(hwnd: int, *, prefer_printwindow: bool = True) -> tuple[Image.Image, str]:
    """抓取游戏**客户区**画面, 返回 ``(image, method)``."""
    _require_pywin32()
    rect, offset, client_size = _client_offset(hwnd)
    if client_size[0] <= 0 or client_size[1] <= 0:
        raise CaptureError(f"窗口客户区为空: {client_size}")

    best: tuple[Image.Image, str] | None = None

    def consider(image: Image.Image, name: str) -> tuple[Image.Image, str] | None:
        nonlocal best
        if not _looks_blank(image):
            return image, name
        if best is None:
            best = (image, name)
        return None

    if prefer_printwindow:
        # 1) 优先 PrintWindow (DWM 对 D3D 支持最好, 且不怕被遮挡)
        for name, flag in (
            ("printwindow-client", PW_CLIENTONLY | PW_RENDERFULLCONTENT),
            ("printwindow-fullcontent", PW_RENDERFULLCONTENT),
            ("printwindow", 0),
        ):
            try:
                image = _grab_with_printwindow(hwnd, flag, offset, client_size)
            except Exception:
                continue
            hit = consider(image, name)
            if hit:
                return hit

    # 2) 窗口 DC 的客户区 BitBlt (窗口化 D3D8 实测最稳的一条路)
    try:
        hwindc = win32gui.GetWindowDC(hwnd)
        try:
            srcdc = win32ui.CreateDCFromHandle(hwindc)
            hit = consider(_grab_with_bitblt(srcdc, offset, client_size), "window-dc")
            if hit:
                return hit
        finally:
            win32gui.ReleaseDC(hwnd, hwindc)
    except Exception:
        pass

    # 3) 桌面 DC 的客户区 BitBlt (要求窗口可见且未被遮挡)
    try:
        desktop = win32gui.GetDesktopWindow()
        hdeskdc = win32gui.GetWindowDC(desktop)
        try:
            srcdc = win32ui.CreateDCFromHandle(hdeskdc)
            origin = client_rect_on_screen(hwnd)[:2]
            hit = consider(_grab_with_bitblt(srcdc, origin, client_size), "desktop-dc")
            if hit:
                return hit
        finally:
            win32gui.ReleaseDC(desktop, hdeskdc)
    except Exception:
        pass

    if best is not None:
        return best
    raise CaptureError("所有截图方式都失败 (可能是独占全屏或被 DWM 保护挡住)")


def playfield_rect(image_size: tuple[int, int], playfield: list[int], window_size: list[int] | None = None) -> tuple[int, int, int, int]:
    """把 640x480 基准的游戏场地矩形换算到实际画面尺寸."""
    base_w, base_h = (window_size or [640, 480])[:2]
    scale_x = image_size[0] / float(base_w or 640)
    scale_y = image_size[1] / float(base_h or 480)
    left, top, right, bottom = playfield
    return (
        max(0, round(left * scale_x)),
        max(0, round(top * scale_y)),
        min(image_size[0], round(right * scale_x)),
        min(image_size[1], round(bottom * scale_y)),
    )


def crop_playfield(
    image: Image.Image,
    playfield: list[int],
    window_size: list[int] | None = None,
) -> Image.Image:
    """按游戏场地矩形裁剪 (自动适配窗口缩放)."""
    box = playfield_rect(image.size, playfield, window_size)
    if box[2] <= box[0] or box[3] <= box[1]:
        return image
    return image.crop(box)


def fit_width(image: Image.Image, max_width: int) -> Image.Image:
    if max_width <= 0 or image.width <= max_width:
        return image
    ratio = max_width / image.width
    return image.resize((max_width, max(1, int(image.height * ratio))), Image.LANCZOS)


def _timestamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]


def save_png(image: Image.Image, out_dir: str | Path | None, *, prefix: str = "shot") -> Path:
    directory = Path(out_dir) if out_dir else runtime_dir() / "shots"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{prefix}-{_timestamp()}.png"
    image.save(path)
    return path


def image_to_png_bytes(image: Image.Image) -> bytes:
    import io

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
