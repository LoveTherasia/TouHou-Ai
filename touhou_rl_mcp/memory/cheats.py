"""内存改写辅助 (锁血等) —— 供标定工作流使用.

东方是 32 位游戏, 残机等全局量是 i32。把某个地址持续写回固定值, 就能实现
"锁血": 自机被弹后残机 -1 会被我们立刻写回, 于是**不会 game over**,
从而有充足时间标定自机位置等其它偏移。
"""

from __future__ import annotations

import struct
import threading
import time
from typing import Any


class MemoryLock:
    """把某个 i32 地址持续写回固定值 (后台线程).

    ``floor`` 不为 None 时: 只在当前值 **<= floor** 时才写回 (这样既能观察数值的
    正常递减, 又能在快见底时兜住) —— 锁残机时用它, 否则会把"掉残"信号盖掉。
    """

    def __init__(
        self, mem: Any, address: int, value: int, *, interval: float = 0.02, floor: int | None = None
    ) -> None:
        self.mem = mem
        self.address = int(address)
        self.value = int(value)
        self.interval = float(interval)
        self.floor = floor
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.writes = 0

    def _run(self) -> None:
        payload = struct.pack("<i", self.value)
        while not self._stop.is_set():
            try:
                if self.floor is None:
                    self.mem.write_bytes(self.address, payload)
                    self.writes += 1
                else:
                    cur = int(self.mem.read_value(self.address, "i32"))
                    if cur <= self.floor:
                        self.mem.write_bytes(self.address, payload)
                        self.writes += 1
            except Exception:
                pass
            time.sleep(self.interval)

    def start(self) -> "MemoryLock":
        if self._thread is None:
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="mem-lock", daemon=True)
            self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def __enter__(self) -> "MemoryLock":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()


class LivesLock(MemoryLock):
    """锁残机: 让自机打不死 (标定自机位置时用).

    默认只在残机降到 ``floor`` (默认 1) 时才写回 ``value`` —— 这样"掉残"仍可观察。
    """

    def __init__(
        self, mem: Any, address: int, *, value: int = 5, interval: float = 0.02, floor: int = 1
    ) -> None:
        super().__init__(mem, address, value, interval=interval, floor=floor)
