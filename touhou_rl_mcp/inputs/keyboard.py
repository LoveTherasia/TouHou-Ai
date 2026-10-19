"""向游戏窗口注入键盘输入 (SendInput).

东方正作通过 DirectInput 读键盘, 所以必须用 ``SendInput`` / ``keybd_event``
这类"驱动级"注入 —— 直接 ``PostMessage(WM_KEYDOWN)`` 对 DInput 无效。
发送前通常需要把游戏窗口置前 (``focus_window``)。
"""

from __future__ import annotations

import ctypes
import time
from ctypes import wintypes

_user32 = ctypes.WinDLL("user32", use_last_error=True)

INPUT_KEYBOARD = 1
KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_SCANCODE = 0x0008
KEYEVENTF_UNICODE = 0x0004

SW_RESTORE = 9
SW_SHOW = 5

#: 虚拟按键名 -> VK 码
VK_CODES: dict[str, int] = {
    "up": 0x26,
    "down": 0x28,
    "left": 0x25,
    "right": 0x27,
    "shoot": 0x5A,       # Z
    "confirm": 0x5A,     # Z
    "bomb": 0x58,        # X
    "cancel": 0x58,      # X
    "focus": 0x10,       # Shift
    "pause": 0x1B,       # Esc
    "menu": 0x1B,
    "enter": 0x0D,
    "space": 0x20,
    "ctrl": 0x11,
    "home": 0x24,
    "end": 0x23,
    "pgup": 0x21,
    "pgdn": 0x22,
}

#: 需要 EXTENDEDKEY 标记的键
_EXTENDED = {0x26, 0x28, 0x25, 0x27, 0x24, 0x23, 0x21, 0x22, 0x2D, 0x2E}

#: 别名 (方便 Agent 写动作)
ALIASES: dict[str, str] = {
    "z": "confirm",
    "x": "cancel",
    "shift": "focus",
    "esc": "pause",
    "escape": "pause",
    "shot": "shoot",
    "fire": "shoot",
    "return": "enter",
    "up_arrow": "up",
    "north": "up",
    "south": "down",
    "west": "left",
    "east": "right",
}


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_void_p),
    ]


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_void_p),
    ]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ("uMsg", wintypes.DWORD),
        ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD),
    ]


class _INPUTUNION(ctypes.Union):
    _fields_ = [("ki", KEYBDINPUT), ("mi", MOUSEINPUT), ("hi", HARDWAREINPUT)]


class INPUT(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = [("type", wintypes.DWORD), ("u", _INPUTUNION)]


_user32.SendInput.restype = wintypes.UINT
_user32.SendInput.argtypes = [wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int]

_user32.SetForegroundWindow.restype = wintypes.BOOL
_user32.SetForegroundWindow.argtypes = [wintypes.HWND]
_user32.ShowWindow.restype = wintypes.BOOL
_user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
_user32.IsIconic.restype = wintypes.BOOL
_user32.IsIconic.argtypes = [wintypes.HWND]
_user32.GetForegroundWindow.restype = wintypes.HWND
_user32.GetWindowThreadProcessId.restype = wintypes.DWORD
_user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
_user32.AttachThreadInput.restype = wintypes.BOOL
_user32.AttachThreadInput.argtypes = [wintypes.DWORD, wintypes.DWORD, wintypes.BOOL]
_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
_kernel32.GetCurrentThreadId.restype = wintypes.DWORD
_kernel32.GetCurrentThreadId.argtypes = []
_user32.BringWindowToTop.restype = wintypes.BOOL
_user32.BringWindowToTop.argtypes = [wintypes.HWND]


def resolve_button(name: str) -> str:
    key = name.strip().lower()
    return ALIASES.get(key, key)


def vk_of(name: str) -> int:
    key = resolve_button(name)
    if key in VK_CODES:
        return VK_CODES[key]
    if len(key) == 1:
        return ord(key.upper())
    raise KeyError(f"未知按键: {name!r}")


class Keyboard:
    """SendInput 键盘注入."""

    def __init__(self, hwnd: int | None = None, *, focus: bool = True) -> None:
        self.hwnd = hwnd
        self.focus_enabled = focus
        self._held: set[int] = set()
        self._held_names: set[str] = set()

    # ------------------------------------------------------------------
    def set_window(self, hwnd: int | None) -> None:
        if hwnd != self.hwnd:
            self.release_all()
            self.hwnd = hwnd

    def focus_window(self) -> bool:
        """把目标窗口置前并激活 (DirectInput 只会读前台窗口的按键)."""
        if not self.hwnd:
            return False
        if _user32.IsIconic(self.hwnd):
            _user32.ShowWindow(self.hwnd, SW_RESTORE)
        else:
            _user32.ShowWindow(self.hwnd, SW_SHOW)

        if _user32.GetForegroundWindow() == self.hwnd:
            return True

        # AttachThreadInput 绕过前台窗口锁定
        target_thread = _user32.GetWindowThreadProcessId(self.hwnd, None)
        current_thread = _kernel32.GetCurrentThreadId()
        attached = False
        try:
            if target_thread and target_thread != current_thread:
                attached = bool(_user32.AttachThreadInput(current_thread, target_thread, True))
            _user32.BringWindowToTop(self.hwnd)
            ok = bool(_user32.SetForegroundWindow(self.hwnd))
        finally:
            if attached:
                _user32.AttachThreadInput(current_thread, target_thread, False)
        return ok or _user32.GetForegroundWindow() == self.hwnd

    # ------------------------------------------------------------------
    def _send(self, vk: int, up: bool) -> None:
        flags = KEYEVENTF_KEYUP if up else 0
        if vk in _EXTENDED:
            flags |= KEYEVENTF_EXTENDEDKEY
        item = INPUT()
        item.type = INPUT_KEYBOARD
        item.ki = KEYBDINPUT(wVk=vk, wScan=0, dwFlags=flags, time=0, dwExtraInfo=None)
        sent = _user32.SendInput(1, ctypes.byref(item), ctypes.sizeof(INPUT))
        if sent != 1:
            raise OSError(f"SendInput 失败 (vk=0x{vk:02X}), GetLastError={ctypes.get_last_error()}")

    def key_down(self, name: str) -> None:
        self._held_names.add(resolve_button(name))
        vk = vk_of(name)
        if vk in self._held:
            return
        self._send(vk, up=False)
        self._held.add(vk)

    def key_up(self, name: str) -> None:
        vk = vk_of(name)
        # 只有当没有别的别名仍指向同一个 VK 时才真正抬起
        others = {n for n in self._held_names if n != resolve_button(name) and vk_of(n) == vk}
        self._held_names.discard(resolve_button(name))
        if others or vk not in self._held:
            return
        self._send(vk, up=True)
        self._held.discard(vk)

    def release_all(self) -> None:
        for vk in list(self._held):
            try:
                self._send(vk, up=True)
            except OSError:
                pass
        self._held.clear()
        self._held_names.clear()

    @property
    def held(self) -> set[int]:
        return set(self._held)

    @property
    def held_names(self) -> set[str]:
        return set(self._held_names)

    # ------------------------------------------------------------------
    def hold(self, names: list[str]) -> None:
        """把"当前按住的键"整体设为 ``names``."""
        wanted_names = {resolve_button(n) for n in names}
        wanted_vks = {vk_of(n) for n in wanted_names}
        for vk in list(self._held - wanted_vks):
            try:
                self._send(vk, up=True)
            except OSError:
                pass
            self._held.discard(vk)
        for vk in wanted_vks - self._held:
            self._send(vk, up=False)
            self._held.add(vk)
        self._held_names = wanted_names

    def tap(self, name: str, *, hold_seconds: float = 0.05) -> None:
        self.key_down(name)
        time.sleep(hold_seconds)
        self.key_up(name)
