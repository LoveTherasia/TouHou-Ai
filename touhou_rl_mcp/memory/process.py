"""Windows 进程 / 内存访问 (纯 ctypes, 无第三方依赖).

同时支持:
  * 从 64 位 Python 读取 32 位游戏 (th06..th19 均为 32 位) —— 用
    ``EnumProcessModulesEx(LIST_MODULES_ALL)`` 拿模块基址, 与 pymem 的做法一致。
  * 查找进程主窗口 (用于截图 / 置前发按键)。
"""

from __future__ import annotations

import ctypes
import struct
import time
from ctypes import wintypes
from dataclasses import dataclass
from typing import Iterable

from ..errors import GameNotFoundError, ProcessMemoryError

# ---------------------------------------------------------------- 常量
PROCESS_QUERY_INFORMATION = 0x0400
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
PROCESS_VM_OPERATION = 0x0008
PROCESS_VM_READ = 0x0010
PROCESS_VM_WRITE = 0x0020

TH32CS_SNAPPROCESS = 0x00000002
TH32CS_SNAPMODULE = 0x00000008
TH32CS_SNAPMODULE32 = 0x00000010

MEM_COMMIT = 0x1000
MEM_IMAGE = 0x1000000
MEM_MAPPED = 0x40000
MEM_PRIVATE = 0x20000
PAGE_NOACCESS = 0x01
PAGE_GUARD = 0x100
PAGE_READONLY = 0x02
PAGE_READWRITE = 0x04
PAGE_WRITECOPY = 0x08
PAGE_EXECUTE = 0x10
PAGE_EXECUTE_READ = 0x20
PAGE_EXECUTE_READWRITE = 0x40
PAGE_EXECUTE_WRITECOPY = 0x80

LIST_MODULES_ALL = 0x03

_READABLE = (
    PAGE_READONLY
    | PAGE_READWRITE
    | PAGE_WRITECOPY
    | PAGE_EXECUTE_READ
    | PAGE_EXECUTE_READWRITE
    | PAGE_EXECUTE_WRITECOPY
)

_k32 = ctypes.WinDLL("kernel32", use_last_error=True)
_psapi = ctypes.WinDLL("psapi", use_last_error=True)
_user32 = ctypes.WinDLL("user32", use_last_error=True)

# ---------------------------------------------------------------- 结构体


class PROCESSENTRY32W(ctypes.Structure):
    _fields_ = [
        ("dwSize", wintypes.DWORD),
        ("cntUsage", wintypes.DWORD),
        ("th32ProcessID", wintypes.DWORD),
        ("th32DefaultHeapID", ctypes.c_void_p),
        ("th32ModuleID", wintypes.DWORD),
        ("cntThreads", wintypes.DWORD),
        ("th32ParentProcessID", wintypes.DWORD),
        ("pcPriClassBase", wintypes.LONG),
        ("dwFlags", wintypes.DWORD),
        ("szExeFile", wintypes.WCHAR * 260),
    ]


class MODULEINFO(ctypes.Structure):
    _fields_ = [
        ("lpBaseOfDll", ctypes.c_void_p),
        ("SizeOfImage", wintypes.DWORD),
        ("EntryPoint", ctypes.c_void_p),
    ]


class MEMORY_BASIC_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BaseAddress", ctypes.c_void_p),
        ("AllocationBase", ctypes.c_void_p),
        ("AllocationProtect", wintypes.DWORD),
        ("__alignment1", wintypes.DWORD),
        ("RegionSize", ctypes.c_size_t),
        ("State", wintypes.DWORD),
        ("Protect", wintypes.DWORD),
        ("Type", wintypes.DWORD),
        ("__alignment2", wintypes.DWORD),
    ]


class RECT(ctypes.Structure):
    _fields_ = [
        ("left", wintypes.LONG),
        ("top", wintypes.LONG),
        ("right", wintypes.LONG),
        ("bottom", wintypes.LONG),
    ]


# ---------------------------------------------------------------- argtypes
_k32.OpenProcess.restype = wintypes.HANDLE
_k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
_k32.CloseHandle.restype = wintypes.BOOL
_k32.CloseHandle.argtypes = [wintypes.HANDLE]
_k32.ReadProcessMemory.restype = wintypes.BOOL
_k32.ReadProcessMemory.argtypes = [
    wintypes.HANDLE, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_size_t),
]
_k32.WriteProcessMemory.restype = wintypes.BOOL
_k32.WriteProcessMemory.argtypes = [
    wintypes.HANDLE, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_size_t),
]
_k32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
_k32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
_k32.Process32FirstW.restype = wintypes.BOOL
_k32.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESSENTRY32W)]
_k32.Process32NextW.restype = wintypes.BOOL
_k32.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESSENTRY32W)]
_k32.VirtualQueryEx.restype = ctypes.c_size_t
_k32.VirtualQueryEx.argtypes = [
    wintypes.HANDLE, ctypes.c_void_p, ctypes.POINTER(MEMORY_BASIC_INFORMATION),
    ctypes.c_size_t,
]

_psapi.EnumProcessModulesEx.restype = wintypes.BOOL
_psapi.EnumProcessModulesEx.argtypes = [
    wintypes.HANDLE, ctypes.POINTER(wintypes.HMODULE), wintypes.DWORD,
    ctypes.POINTER(wintypes.DWORD), wintypes.DWORD,
]
_psapi.GetModuleBaseNameW.restype = wintypes.DWORD
_psapi.GetModuleBaseNameW.argtypes = [
    wintypes.HANDLE, wintypes.HMODULE, wintypes.LPWSTR, wintypes.DWORD,
]
_psapi.GetModuleInformation.restype = wintypes.BOOL
_psapi.GetModuleInformation.argtypes = [
    wintypes.HANDLE, wintypes.HMODULE, ctypes.POINTER(MODULEINFO), wintypes.DWORD,
]

_user32.GetForegroundWindow.restype = wintypes.HWND
_user32.FindWindowW.restype = wintypes.HWND
_user32.FindWindowW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR]
_user32.GetWindowThreadProcessId.restype = wintypes.DWORD
_user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
_user32.IsWindowVisible.restype = wintypes.BOOL
_user32.IsWindowVisible.argtypes = [wintypes.HWND]
_user32.GetWindowRect.restype = wintypes.BOOL
_user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(RECT)]
_user32.GetClientRect.restype = wintypes.BOOL
_user32.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(RECT)]
_user32.GetWindowTextLengthW.restype = ctypes.c_int
_user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
_user32.GetWindowTextW.restype = ctypes.c_int
_user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
_user32.SetForegroundWindow.restype = wintypes.BOOL
_user32.SetForegroundWindow.argtypes = [wintypes.HWND]
_user32.ShowWindow.restype = wintypes.BOOL
_user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
_user32.IsIconic.restype = wintypes.BOOL
_user32.IsIconic.argtypes = [wintypes.HWND]
_user32.GetClassNameW.restype = ctypes.c_int
_user32.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]

WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
_user32.EnumWindows.restype = wintypes.BOOL
_user32.EnumWindows.argtypes = [WNDENUMPROC, wintypes.LPARAM]


# ---------------------------------------------------------------- 进程查找
@dataclass
class ProcessInfo:
    pid: int
    exe: str

    def to_dict(self) -> dict[str, object]:
        return {"pid": self.pid, "exe": self.exe}


def list_processes() -> list[ProcessInfo]:
    snap = _k32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
    if not snap or snap == ctypes.c_void_p(-1).value:
        raise ProcessMemoryError("CreateToolhelp32Snapshot 失败")
    entry = PROCESSENTRY32W()
    entry.dwSize = ctypes.sizeof(entry)
    out: list[ProcessInfo] = []
    try:
        ok = _k32.Process32FirstW(snap, ctypes.byref(entry))
        while ok:
            out.append(ProcessInfo(pid=int(entry.th32ProcessID), exe=str(entry.szExeFile)))
            ok = _k32.Process32NextW(snap, ctypes.byref(entry))
    finally:
        _k32.CloseHandle(snap)
    return out


def find_pids(exe_names: Iterable[str]) -> list[int]:
    wanted = {name.lower() for name in exe_names}
    return [p.pid for p in list_processes() if p.exe.lower() in wanted]


def find_first_pid(exe_names: Iterable[str]) -> int | None:
    pids = find_pids(exe_names)
    return pids[0] if pids else None


# ---------------------------------------------------------------- 窗口
@dataclass
class WindowInfo:
    hwnd: int
    title: str
    class_name: str
    pid: int
    rect: tuple[int, int, int, int]
    client: tuple[int, int]

    @property
    def width(self) -> int:
        return self.rect[2] - self.rect[0]

    @property
    def height(self) -> int:
        return self.rect[3] - self.rect[1]

    def to_dict(self) -> dict[str, object]:
        return {
            "hwnd": self.hwnd,
            "title": self.title,
            "class_name": self.class_name,
            "pid": self.pid,
            "rect": list(self.rect),
            "client_size": list(self.client),
        }


def _window_info(hwnd: int) -> WindowInfo:
    length = _user32.GetWindowTextLengthW(hwnd)
    buf = ctypes.create_unicode_buffer(length + 1)
    _user32.GetWindowTextW(hwnd, buf, length + 1)
    cls = ctypes.create_unicode_buffer(256)
    _user32.GetClassNameW(hwnd, cls, 256)
    pid = wintypes.DWORD(0)
    _user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    rect = RECT()
    _user32.GetWindowRect(hwnd, ctypes.byref(rect))
    client = RECT()
    _user32.GetClientRect(hwnd, ctypes.byref(client))
    return WindowInfo(
        hwnd=int(hwnd),
        title=buf.value,
        class_name=cls.value,
        pid=int(pid.value),
        rect=(rect.left, rect.top, rect.right, rect.bottom),
        client=(client.right - client.left, client.bottom - client.top),
    )


def find_windows(pid: int | None = None, *, visible_only: bool = True) -> list[WindowInfo]:
    found: list[WindowInfo] = []

    def _cb(hwnd, _lparam):
        if visible_only and not _user32.IsWindowVisible(hwnd):
            return True
        info = _window_info(hwnd)
        if pid is not None and info.pid != pid:
            return True
        if info.width <= 0 or info.height <= 0:
            return True
        found.append(info)
        return True

    _user32.EnumWindows(WNDENUMPROC(_cb), 0)
    found.sort(key=lambda w: w.width * w.height, reverse=True)
    return found


def find_game_window(pid: int | None = None, title_hints: Iterable[str] = ()) -> WindowInfo | None:
    """找游戏窗口: 优先标题含关键字, 否则取该进程面积最大的可见窗口."""
    hints = [h.lower() for h in title_hints if h]
    windows = find_windows(pid)

    # 东方系列窗口类名通常是 "Touhou Project" 之类, 标题会随机切换(播放 BGM 时)
    def score(w: WindowInfo) -> tuple[int, int]:
        cls_hit = 1 if "touhou" in w.class_name.lower() else 0
        title_hit = 1 if any(h in w.title.lower() for h in hints) else 0
        return (max(cls_hit, title_hit), w.width * w.height)

    if not windows:
        return None
    windows.sort(key=score, reverse=True)
    best = windows[0]
    if best.width < 200 or best.height < 150:
        return None
    return best


# ---------------------------------------------------------------- 内存读写
class ProcessMemory:
    """目标进程的内存读写句柄."""

    def __init__(self, pid: int) -> None:
        self.pid = pid
        access = (
            PROCESS_QUERY_INFORMATION
            | PROCESS_VM_READ
            | PROCESS_VM_WRITE
            | PROCESS_VM_OPERATION
        )
        handle = _k32.OpenProcess(access, False, pid)
        if not handle:
            # 权限不足时退化为只读句柄
            handle = _k32.OpenProcess(
                PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_VM_READ, False, pid
            )
        if not handle:
            raise ProcessMemoryError(
                f"OpenProcess(pid={pid}) 失败, GetLastError={ctypes.get_last_error()}"
            )
        self.handle = handle
        self._modules: dict[str, tuple[int, int]] | None = None

    # ------------------------------------------------------------------
    def close(self) -> None:
        if getattr(self, "handle", None):
            _k32.CloseHandle(self.handle)
            self.handle = None  # type: ignore[assignment]

    def __enter__(self) -> "ProcessMemory":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def is_alive(self) -> bool:
        if not getattr(self, "handle", None):
            return False
        code = wintypes.DWORD()
        if not _k32.GetExitCodeProcess(self.handle, ctypes.byref(code)):
            return False
        return code.value == 259  # STILL_ACTIVE

    # ------------------------------------------------------------------
    def read_bytes(self, address: int, size: int) -> bytes:
        buf = (ctypes.c_ubyte * size)()
        read = ctypes.c_size_t(0)
        ok = _k32.ReadProcessMemory(
            self.handle, ctypes.c_void_p(address), buf, size, ctypes.byref(read)
        )
        if not ok or read.value != size:
            raise ProcessMemoryError(
                f"ReadProcessMemory(0x{address:X}, {size}) 失败, "
                f"读出 {read.value} 字节, GetLastError={ctypes.get_last_error()}"
            )
        return bytes(buf)

    def read_value(self, address: int, type_name: str) -> float:
        fmt = _TYPE_FORMATS.get(type_name)
        if fmt is None:
            raise ProcessMemoryError(f"不支持的内存类型: {type_name}")
        size = struct.calcsize(fmt)
        data = self.read_bytes(address, size)
        value = struct.unpack(fmt, data)[0]
        return value

    def write_bytes(self, address: int, data: bytes) -> None:
        buf = (ctypes.c_ubyte * len(data)).from_buffer_copy(data)
        written = ctypes.c_size_t(0)
        ok = _k32.WriteProcessMemory(
            self.handle, ctypes.c_void_p(address), buf, len(data), ctypes.byref(written)
        )
        if not ok or written.value != len(data):
            raise ProcessMemoryError(
                f"WriteProcessMemory(0x{address:X}) 失败, GetLastError={ctypes.get_last_error()}"
            )

    # ------------------------------------------------------------------
    def modules(self, refresh: bool = False) -> dict[str, tuple[int, int]]:
        """``{模块名小写: (基址, 大小)}``"""
        if self._modules is not None and not refresh:
            return self._modules
        out: dict[str, tuple[int, int]] = {}
        needed = wintypes.DWORD(0)
        capacity = 1024
        while True:
            array = (wintypes.HMODULE * capacity)()
            if not _psapi.EnumProcessModulesEx(
                self.handle, array, ctypes.sizeof(array), ctypes.byref(needed), LIST_MODULES_ALL
            ):
                raise ProcessMemoryError(
                    f"EnumProcessModulesEx 失败, GetLastError={ctypes.get_last_error()}"
                )
            count = needed.value // ctypes.sizeof(wintypes.HMODULE)
            if count < capacity:
                for i in range(count):
                    hmod = array[i]
                    name_buf = ctypes.create_unicode_buffer(260)
                    if not _psapi.GetModuleBaseNameW(self.handle, hmod, name_buf, 260):
                        continue
                    info = MODULEINFO()
                    if not _psapi.GetModuleInformation(
                        self.handle, hmod, ctypes.byref(info), ctypes.sizeof(info)
                    ):
                        continue
                    out[name_buf.value.lower()] = (
                        int(info.lpBaseOfDll or 0),
                        int(info.SizeOfImage),
                    )
                break
            capacity = count + 64
        self._modules = out
        return out

    def module_base(self, name: str) -> int:
        modules = self.modules()
        key = name.lower()
        if key in modules:
            return modules[key][0]
        # 容错: 前缀匹配 (例如传 "th11" 而模块是 "th11c.exe")
        for mod_name, (base, _size) in modules.items():
            if mod_name.startswith(key) or key.startswith(mod_name):
                return base
        available = ", ".join(sorted(modules)) or "(空)"
        raise ProcessMemoryError(f"在进程中找不到模块 {name!r}; 现有模块: {available}")

    @property
    def main_module(self) -> tuple[str, int, int]:
        modules = self.modules()
        if not modules:
            raise ProcessMemoryError("进程没有可枚举的模块")
        name = next((n for n in modules if n.endswith(".exe")), next(iter(modules)))
        base, size = modules[name]
        return name, base, size

    # ------------------------------------------------------------------
    def memory_regions(self) -> list[tuple[int, int, int, int]]:
        """可读的已提交区域 ``[(base, size, protect, type)]``"""
        regions: list[tuple[int, int, int, int]] = []
        address = 0
        mbi = MEMORY_BASIC_INFORMATION()
        max_address = 1 << 47
        while address < max_address:
            result = _k32.VirtualQueryEx(
                self.handle, ctypes.c_void_p(address), ctypes.byref(mbi), ctypes.sizeof(mbi)
            )
            if not result:
                break
            base = int(mbi.BaseAddress or 0)
            size = int(mbi.RegionSize)
            if size == 0:
                break
            if (
                mbi.State == MEM_COMMIT
                and (mbi.Protect & _READABLE)
                and not (mbi.Protect & PAGE_GUARD)
                and not (mbi.Protect & PAGE_NOACCESS)
            ):
                regions.append((base, size, int(mbi.Protect), int(mbi.Type)))
            address = base + size
        return regions


_TYPE_FORMATS: dict[str, str] = {
    "i8": "<b",
    "u8": "<B",
    "i16": "<h",
    "u16": "<H",
    "i32": "<i",
    "u32": "<I",
    "f32": "<f",
    "f64": "<d",
}


# ---------------------------- GetExitCodeProcess 绑定 (放在最后避免遗漏) --
_k32.GetExitCodeProcess.restype = wintypes.BOOL
_k32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]


def wait_for_process(exe_names: Iterable[str], timeout: float = 30.0, interval: float = 0.25) -> int:
    """等待进程出现, 返回 pid."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        pid = find_first_pid(exe_names)
        if pid is not None:
            return pid
        time.sleep(interval)
    raise GameNotFoundError(
        f"等待进程 {', '.join(exe_names)} 出现超时 ({timeout:.0f}s)"
    )


PROCESS_TERMINATE = 0x0001


def terminate_pid(pid: int, *, timeout: float = 8.0) -> bool:
    """终止进程并等待其退出."""
    handle = _k32.OpenProcess(PROCESS_TERMINATE | PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return False
    try:
        if not _k32.TerminateProcess(handle, 0):
            return False
    finally:
        _k32.CloseHandle(handle)

    deadline = time.time() + timeout
    while time.time() < deadline:
        if pid not in {p.pid for p in list_processes()}:
            return True
        time.sleep(0.1)
    return False
