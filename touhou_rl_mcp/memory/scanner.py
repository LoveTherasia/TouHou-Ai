"""内存扫描 —— 用来标定缺失的偏移量 (自机坐标 / Boss 血量 / 擦弹 ...).

设计目标是复刻 Cheat-Engine 的 "首次扫描 / 再次扫描" 工作流:
    memory_scan(值)  ->  候选地址列表 (持久化)
    memory_filter(值) ->  在候选里筛出当前值等于新值的地址
    memory_read(地址) ->  直接读某地址, 方便验证
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..config import runtime_dir
from ..errors import ProcessMemoryError
from .process import ProcessMemory


def _region_floats(arrays: list[tuple[int, "np.ndarray"]], base: int) -> "np.ndarray | None":
    """在快照里按 base 找到区域, 返回它的 float32 视图 (零拷贝)."""
    for region_base, array in arrays:
        if region_base == base:
            return array.view("<f4")
    return None


@dataclass
class Candidate:
    address: int
    value: int | float
    module: str | None = None
    module_offset: int | None = None
    region_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"address": f"0x{self.address:X}", "value": self.value}
        if self.module and self.module_offset is not None:
            out["as_offset"] = f"{self.module}+0x{self.module_offset:X}"
        if self.region_type:
            out["region"] = self.region_type
        return out


_TYPE_FORMATS = {
    "i8": "<b", "u8": "<B", "i16": "<h", "u16": "<H",
    "i32": "<i", "u32": "<I", "i32le": "<i", "u32le": "<I",
    "f32": "<f", "f64": "<d",
}
_TYPE_SIZES = {k: struct.calcsize(v) for k, v in _TYPE_FORMATS.items()}


class MemoryScanner:
    """针对某个进程的扫描器 (候选列表按 pid 持久化)."""

    def __init__(self, mem: ProcessMemory, *, max_region_bytes: int = 64 * 1024 * 1024) -> None:
        self.mem = mem
        self.max_region_bytes = max_region_bytes

    # ------------------------------------------------------------------
    def _candidate_path(self) -> Path:
        return runtime_dir() / "scanner" / f"pid{self.mem.pid}.json"

    def _module_map(self) -> list[tuple[str, int, int]]:
        return [(name, base, size) for name, (base, size) in self.mem.modules().items()]

    def _locate_module(self, address: int) -> tuple[str | None, int | None]:
        for name, base, size in self._module_map():
            if base <= address < base + size:
                return name, address - base
        return None, None

    @staticmethod
    def _region_type(memory_type: int) -> str:
        from .process import MEM_IMAGE, MEM_MAPPED, MEM_PRIVATE

        if memory_type & MEM_IMAGE:
            return "image"
        if memory_type & MEM_MAPPED:
            return "mapped"
        if memory_type & MEM_PRIVATE:
            return "private"
        return "other"

    # ------------------------------------------------------------------
    def scan(
        self,
        value: int | float,
        *,
        type_name: str = "i32",
        align: int = 4,
        limit: int = 20000,
        include_images: bool = False,
        neighbour_offset: int | None = None,
        neighbour_min: float | None = None,
        neighbour_max: float | None = None,
    ) -> list[Candidate]:
        """首次扫描.

        ``neighbour_offset`` + ``neighbour_min/max`` 可以再加一个"邻居约束":
        只保留 ``地址 + neighbour_offset`` 处同类型读数的值落在 [min, max] 内的候选。
        这是定位**结构体字段**的关键技巧 —— 例如自机 x 与 y 相邻, 扫 192.0(f32) 时
        要求 +4 处的 f32 落在 [100, 448] 内, 候选会立刻收敛到玩家结构体。
        """
        fmt = _TYPE_FORMATS.get(type_name)
        if fmt is None:
            raise ProcessMemoryError(f"不支持的扫描类型: {type_name}")
        try:
            needle = struct.pack(fmt, value)
        except struct.error as exc:
            raise ProcessMemoryError(f"值 {value!r} 不能用类型 {type_name} 编码: {exc}") from exc

        hits: list[Candidate] = []
        for base, size, protect, memory_type in self.mem.memory_regions():
            if size > self.max_region_bytes:
                continue
            rtype = self._region_type(memory_type)
            if not include_images and rtype == "image":
                continue
            try:
                data = self.mem.read_bytes(base, size)
            except ProcessMemoryError:
                continue
            start = 0
            while len(hits) < limit:
                idx = data.find(needle, start)
                if idx < 0:
                    break
                address = base + idx
                start = idx + 1
                if align and address % align:
                    continue
                if neighbour_offset is not None and (
                    neighbour_min is not None or neighbour_max is not None
                ):
                    try:
                        neighbour = self.mem.read_value(address + neighbour_offset, type_name)
                    except ProcessMemoryError:
                        continue
                    if neighbour_min is not None and neighbour < neighbour_min:
                        continue
                    if neighbour_max is not None and neighbour > neighbour_max:
                        continue
                module, module_offset = self._locate_module(address)
                hits.append(
                    Candidate(
                        address=address,
                        value=value,
                        module=module,
                        module_offset=module_offset,
                        region_type=rtype,
                    )
                )
            if len(hits) >= limit:
                break
        self.save(hits, query=value, type_name=type_name)
        return hits

    def scan_pattern(
        self,
        pattern: str | bytes,
        *,
        align: int = 4,
        limit: int = 20000,
        include_images: bool = False,
    ) -> list[Candidate]:
        """按**字节序列**扫描 (hex 字符串, 例如 ``"0000C843 0000C843"`` 或带 ``??`` 通配).

        用来一次定位整个结构体字段, 例如自机坐标 ``x, y`` 连续存放时, 直接扫
        ``struct.pack("<ff", x, y).hex()`` 一步到位。
        """
        if isinstance(pattern, bytes):
            needle = pattern
        else:
            cleaned = pattern.replace(" ", "").replace("0x", "").replace(",", "")
            if "?" in cleaned:
                return self._scan_pattern_wildcard(cleaned, align=align, limit=limit, include_images=include_images)
            try:
                needle = bytes.fromhex(cleaned)
            except ValueError as exc:
                raise ProcessMemoryError(f"非法 hex 模式: {pattern!r} ({exc})") from exc
        hits: list[Candidate] = []
        for base, size, protect, memory_type in self.mem.memory_regions():
            if size > self.max_region_bytes:
                continue
            rtype = self._region_type(memory_type)
            if not include_images and rtype == "image":
                continue
            try:
                data = self.mem.read_bytes(base, size)
            except ProcessMemoryError:
                continue
            start = 0
            while len(hits) < limit:
                idx = data.find(needle, start)
                if idx < 0:
                    break
                address = base + idx
                start = idx + 1
                if align and address % align:
                    continue
                module, module_offset = self._locate_module(address)
                hits.append(
                    Candidate(
                        address=address,
                        value=None,
                        module=module,
                        module_offset=module_offset,
                        region_type=rtype,
                    )
                )
            if len(hits) >= limit:
                break
        self.save(hits, query=pattern if isinstance(pattern, str) else pattern.hex(), type_name="bytes")
        return hits

    def _scan_pattern_wildcard(
        self, pattern: str, *, align: int, limit: int, include_images: bool
    ) -> list[Candidate]:
        """支持 ``??`` 通配的字节扫描 (实现得很朴素: 逐个字节比对)."""
        mask = []
        for i in range(0, len(pattern), 2):
            token = pattern[i : i + 2]
            if token in {"??", "?"}:
                mask.append(None)
            else:
                mask.append(int(token, 16))
        if not mask:
            raise ProcessMemoryError("空模式")
        hits: list[Candidate] = []
        first = next((i for i, b in enumerate(mask) if b is not None), None)
        if first is None:
            raise ProcessMemoryError("模式全是通配符")
        pivot = bytes([mask[first]])
        for base, size, protect, memory_type in self.mem.memory_regions():
            if size > self.max_region_bytes:
                continue
            rtype = self._region_type(memory_type)
            if not include_images and rtype == "image":
                continue
            try:
                data = self.mem.read_bytes(base, size)
            except ProcessMemoryError:
                continue
            start = 0
            while len(hits) < limit:
                idx = data.find(pivot, start)
                if idx < 0:
                    break
                start = idx + 1
                address = base + idx - first
                if align and address % align:
                    continue
                offset = idx - first
                if offset < 0 or offset + len(mask) > len(data):
                    continue
                if all(
                    expected is None or data[offset + i] == expected
                    for i, expected in enumerate(mask)
                ):
                    module, module_offset = self._locate_module(address)
                    hits.append(
                        Candidate(
                            address=address,
                            value=None,
                            module=module,
                            module_offset=module_offset,
                            region_type=rtype,
                        )
                    )
            if len(hits) >= limit:
                break
        self.save(hits, query=pattern, type_name="bytes")
        return hits

    def scan_values(
        self,
        values: list[float],
        *,
        type_name: str = "f32",
        align: int = 4,
        limit: int = 200,
        include_images: bool = False,
    ) -> list[Candidate]:
        """扫描**连续存放的多个值** (等价于把结构体字段一起扫).

        例如自机坐标是连续两个 float: ``scan_values([192.0, 400.0], type_name="f32")``。
        """
        fmt = _TYPE_FORMATS.get(type_name)
        if fmt is None:
            raise ProcessMemoryError(f"不支持的类型: {type_name}")
        needle = b"".join(struct.pack(fmt, v) for v in values)
        return self.scan_pattern(needle, align=align, limit=limit, include_images=include_images)

    # ------------------------------------------------------------------
    def filter(self, candidates: Iterable[Candidate], value: int | float, *, type_name: str = "i32") -> list[Candidate]:
        """保留当前值 == ``value`` 的候选."""
        kept: list[Candidate] = []
        for cand in candidates:
            try:
                current = self.mem.read_value(cand.address, type_name)
            except ProcessMemoryError:
                continue
            if abs(float(current) - float(value)) < 1e-6:
                kept.append(
                    Candidate(
                        address=cand.address,
                        value=current,
                        module=cand.module,
                        module_offset=cand.module_offset,
                        region_type=cand.region_type,
                    )
                )
        self.save(kept)
        return kept

    def filter_changed(
        self, candidates: Iterable[Candidate], *, type_name: str = "i32", expect: str = "any"
    ) -> list[Candidate]:
        """保留相比上次记录发生变化(或未变化)的候选.

        ``expect``: ``"changed"`` | ``"unchanged"`` | ``"any"``
        """
        kept: list[Candidate] = []
        for cand in candidates:
            try:
                current = self.mem.read_value(cand.address, type_name)
            except ProcessMemoryError:
                continue
            if cand.value is None:
                # 字节模式扫出来的候选没有"上次的值", 无法判断是否变化 —— 保留
                changed = True
            else:
                changed = abs(float(current) - float(cand.value)) > 1e-6
            if expect == "changed" and not changed:
                continue
            if expect == "unchanged" and changed:
                continue
            kept.append(
                Candidate(
                    address=cand.address,
                    value=current,
                    module=cand.module,
                    module_offset=cand.module_offset,
                    region_type=cand.region_type,
                )
            )
        self.save(kept)
        return kept

    # ------------------------------------------------------------------
    # "未知初始值" 扫描 (delta scan)
    # ------------------------------------------------------------------
    def snapshot_path(self) -> Path:
        return runtime_dir() / "scanner" / f"pid{self.mem.pid}.snap"

    def snapshot(
        self,
        *,
        include_images: bool = True,
        max_total_bytes: int = 768 * 1024 * 1024,
        persist: bool = True,
    ) -> dict[str, Any]:
        """给整个可读内存拍一张快照, 供 :meth:`scan_delta` 做差分.

        这是找**坐标**这类"不知道确切数值"的字段的标准做法:
        拍快照 -> 让自机移动 -> 差分出所有变化了的地址 -> 再差分一次收敛。

        ``persist=True`` 会把快照写到 ``.runtime/scanner/pid<pid>.snap``,
        这样跨 MCP 工具调用也能用 (代价是几百 MB 磁盘)。
        """
        regions: list[tuple[int, bytes]] = []
        total = 0
        skipped = 0
        for base, size, protect, memory_type in self.mem.memory_regions():
            if size > self.max_region_bytes:
                skipped += 1
                continue
            if not include_images and self._region_type(memory_type) == "image":
                continue
            if total + size > max_total_bytes:
                skipped += 1
                continue
            try:
                data = self.mem.read_bytes(base, size)
            except ProcessMemoryError:
                skipped += 1
                continue
            regions.append((base, data))
            total += size
        self._snapshot = regions

        if persist:
            path = self.snapshot_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "wb") as fh:
                fh.write(struct.pack("<I", len(regions)))
                for base, data in regions:
                    fh.write(struct.pack("<II", base, len(data)))
                    fh.write(data)

        return {
            "regions": len(regions),
            "bytes": total,
            "skipped_regions": skipped,
            "mb": round(total / 1024 / 1024, 1),
            "path": str(self.snapshot_path()) if persist else None,
        }

    def _load_snapshot(self) -> list[tuple[int, bytes]]:
        cached = getattr(self, "_snapshot", None)
        if cached:
            return cached
        path = self.snapshot_path()
        if not path.exists():
            raise ProcessMemoryError("还没有快照, 请先调用 snapshot()")
        regions: list[tuple[int, bytes]] = []
        with open(path, "rb") as fh:
            count = struct.unpack("<I", fh.read(4))[0]
            for _ in range(count):
                base, size = struct.unpack("<II", fh.read(8))
                regions.append((base, fh.read(size)))
        self._snapshot = regions
        return regions

    def clear_snapshot(self) -> None:
        self._snapshot = []
        path = self.snapshot_path()
        if path.exists():
            path.unlink()

    def scan_delta(
        self,
        *,
        type_name: str = "f32",
        min_value: float | None = None,
        max_value: float | None = None,
        only_previous: bool = True,
        align: int = 4,
        limit: int = 20000,
        unchanged: bool = False,
        neighbour_offset: int | None = None,
        neighbour_min: float | None = None,
        neighbour_max: float | None = None,
        require_neighbour_changed: bool = False,
    ) -> list[Candidate]:
        """与快照做差分: 找出值发生了变化 (或未变化) 的地址.

        Args:
            type_name: 把变化了的字节按这个类型解读 (用来做范围过滤)。
            min_value / max_value: 只保留当前值落在这个区间的候选。
            only_previous: 若已有候选列表, 只在这些地址里筛 (多轮收敛用)。
            unchanged: ``True`` 则找"没变化"的地址。
            neighbour_offset / neighbour_min / neighbour_max: 邻居约束 ——
                要求 ``地址 + neighbour_offset`` 处的同类型读数也落在区间内。
            require_neighbour_changed: 还要求邻居**也发生了变化**。
                找自机坐标时把 (offset=4, 区间=[1,448], 要求变化) 一起打开,
                因为自机 x 与 y 是相邻的两个 float, 会一起变 —— 候选会瞬间收敛。
        """
        snapshot = self._load_snapshot()

        restrict: set[int] | None = None
        if only_previous:
            previous = self.load()
            if previous:
                restrict = {c.address for c in previous}

        fmt = _TYPE_FORMATS.get(type_name)
        size = struct.calcsize(fmt) if fmt else 4
        pattern = struct.Struct(fmt) if fmt else None

        def read_at(buffer: bytes, offset: int) -> tuple[bool, float | None]:
            if offset < 0 or offset + size > len(buffer):
                return False, None
            chunk = buffer[offset : offset + size]
            if pattern is None:
                return True, None
            try:
                return True, float(pattern.unpack(chunk)[0])
            except struct.error:
                return False, None

        hits: list[Candidate] = []
        for base, old in snapshot:
            try:
                new = self.mem.read_bytes(base, len(old))
            except ProcessMemoryError:
                continue
            if new == old and not unchanged:
                continue
            for offset in range(0, len(old) - size + 1, max(1, align)):
                address = base + offset
                if restrict is not None and address not in restrict:
                    continue
                slice_new = new[offset : offset + size]
                changed = slice_new != old[offset : offset + size]
                if changed == unchanged:
                    continue
                value: float | None = None
                if pattern is not None:
                    ok, value = read_at(new, offset)
                    if not ok or value is None:
                        continue
                    if min_value is not None and value < min_value:
                        continue
                    if max_value is not None and value > max_value:
                        continue
                if neighbour_offset is not None:
                    neighbour_at = offset + neighbour_offset
                    ok, neighbour = read_at(new, neighbour_at)
                    if not ok or neighbour is None:
                        continue
                    if neighbour_min is not None and neighbour < neighbour_min:
                        continue
                    if neighbour_max is not None and neighbour > neighbour_max:
                        continue
                    if require_neighbour_changed:
                        old_at = neighbour_at
                        if old_at < 0 or old_at + size > len(old):
                            continue
                        if new[old_at : old_at + size] == old[old_at : old_at + size]:
                            continue
                module, module_offset = self._locate_module(address)
                hits.append(
                    Candidate(
                        address=address,
                        value=value,
                        module=module,
                        module_offset=module_offset,
                        region_type="delta",
                    )
                )
                if len(hits) >= limit:
                    break
            if len(hits) >= limit:
                break
        self.save(hits, query=f"delta({type_name})", type_name=type_name)
        return hits

    # ------------------------------------------------------------------
    # 全自动定位自机坐标 (撞墙法)
    # ------------------------------------------------------------------
    def _capture_arrays(self, include_images: bool = True) -> list[tuple[int, np.ndarray]]:
        """把可读内存抓成 ``[(base, uint32 数组)]`` (numpy 视图, 零拷贝)."""
        import numpy as np

        out: list[tuple[int, np.ndarray]] = []
        for base, size, protect, memory_type in self.mem.memory_regions():
            if size > self.max_region_bytes:
                continue
            if not include_images and self._region_type(memory_type) == "image":
                continue
            try:
                data = self.mem.read_bytes(base, size)
            except ProcessMemoryError:
                continue
            count = len(data) // 4
            if count == 0:
                continue
            out.append((base, np.frombuffer(data, dtype="<u4", count=count)))
        out.sort(key=lambda item: item[0])
        return out

    @staticmethod
    def _unchanged_masks(
        before: list[tuple[int, np.ndarray]],
        after: list[tuple[int, np.ndarray]],
        *,
        changed: bool = False,
    ) -> list[tuple[int, np.ndarray]]:
        """返回 ``[(base, bool 掩码)]``.

        ``changed=False`` -> 掩码为 True 表示该槽**没变**;
        ``changed=True``  -> 掩码为 True 表示该槽**变了**。

        用 numpy 做整块比较: 1600 万个槽只要十几毫秒 (纯 Python 循环要几十秒)。
        """
        after_by_base = dict(after)
        masks: list[tuple[int, np.ndarray]] = []
        for base, old in before:
            new = after_by_base.get(base)
            if new is None or new.shape != old.shape:
                continue
            masks.append((base, (old != new) if changed else (old == new)))
        return masks

    @staticmethod
    def _mask_addresses(masks: list[tuple[int, np.ndarray]]) -> set[int]:
        out: set[int] = set()
        for base, mask in masks:
            for index in np.flatnonzero(mask).tolist():
                out.add(base + 4 * int(index))
        return out

    def find_player_coordinates(
        self,
        press,
        *,
        settle_frames: int = 150,
        hold_frames: int = 50,
        move_frames: int = 90,
        low: float = 1.0,
        high: float = 448.0,
        include_images: bool = True,
        log=None,
    ) -> list[Candidate]:
        """全自动找自机 x 坐标 (返回候选, 取第一个即可; ``+4`` 就是 y).

        **撞墙法** —— 比"时间差分"可靠得多:

        1. 按住 LEFT 很久, 自机会被精确地 **clamp 在左边界** (一个固定浮点数);
        2. 此时再按住 LEFT 一小段时间, 自机 x **不再变化**, 而弹幕还在动
           -> 取"这段时间内完全没变的地址" = 自机 x 的候选 (混着静态数据);
        3. 按住 RIGHT 撞右边界, 同样取"没变的地址";
        4. 两个集合的交集里, 再要求"左墙时的值和右墙时的值**不同**" ——
           静态数据会被这一步全部筛掉, 因为它们的值不会变。

        剩下几乎就只有自机 x 了。
        """
        import numpy as np

        emit = log or (lambda *_: None)

        def capture() -> list[tuple[int, np.ndarray]]:
            return self._capture_arrays(include_images)

        def still_set(before, after) -> dict[int, float]:
            """{地址: 值} —— 在 before->after 期间完全没变的槽, 值取当前 f32."""
            result: dict[int, float] = {}
            for base, mask in self._unchanged_masks(before, after):
                indices = np.flatnonzero(mask)
                if indices.size == 0:
                    continue
                floats = _region_floats(after, base)
                if floats is None:
                    continue
                for index in indices.tolist():
                    result[base + 4 * int(index)] = float(floats[int(index)])
            return result

        # ---- 阶段 1: 撞左墙 ----
        press(["left", "down"], settle_frames)  # 先到左下角 (相对安全)
        press(["left"], settle_frames)
        left_a = capture()
        press(["left"], hold_frames)
        left_b = capture()
        left_still = still_set(left_a, left_b)
        emit(f"左墙稳定期不变地址 {len(left_still)} 个")

        # ---- 阶段 2: 撞右墙 ----
        press(["right"], settle_frames * 2)
        right_a = capture()
        press(["right"], hold_frames)
        right_b = capture()
        right_still = still_set(right_a, right_b)
        emit(f"右墙稳定期不变地址 {len(right_still)} 个")

        common = set(left_still) & set(right_still)
        emit(f"两个稳定期都不变: {len(common)} 个")

        candidates: list[Candidate] = []
        for address in sorted(common):
            v_left = left_still[address]
            v_right = right_still[address]
            if not (low <= v_left <= high and low <= v_right <= high):
                continue
            if abs(v_left - v_right) < 1e-6:
                continue  # 值没变 -> 静态数据, 不是自机 x
            module, module_offset = self._locate_module(address)
            candidates.append(
                Candidate(
                    address=address,
                    value=v_right,
                    module=module,
                    module_offset=module_offset,
                    region_type="player_x",
                )
            )
        emit(f"值在 [{low:g}, {high:g}] 内且左右墙值不同: {len(candidates)} 个")

        # ---- 阶段 3: 离开右墙往左走, 只保留"移动期确实变了"的地址 ----
        # 这一条能筛掉所有静态数据 (它们在墙边和移动期都不变)。
        if candidates and move_frames > 0:
            press(["left"], move_frames)
            after_move = capture()
            moved = self._mask_addresses(self._unchanged_masks(right_b, after_move, changed=True))
            before_count = len(candidates)
            candidates = [c for c in candidates if c.address in moved]
            emit(f"移动期确实变化: {before_count} -> {len(candidates)} 个")

        self.save(candidates, query="find_player", type_name="f32")
        return candidates

    # ------------------------------------------------------------------
    # 全自动定位帧计数器 (frame)
    # ------------------------------------------------------------------
    def find_frame_counter(
        self,
        advance,
        *,
        frames: int = 60,
        tolerance: float = 0.25,
        repeats: int = 2,
        include_images: bool = True,
        log=None,
    ) -> list[Candidate]:
        """找游戏主循环的**帧计数器** (每帧 +1 的整数).

        ``frame`` 偏移标定后, ``advance()`` 可以按帧数严格同步 (比墙钟 sleep 稳得多)。

        做法: 让游戏推进 ``frames`` 帧, 找"这段时间里整数读数恰好增加了 ~frames"
        的地址; 重复 ``repeats`` 次取交集 (静态数据与偶发变化的量会被筛掉)。
        因为推进是墙钟驱动的, 实际帧数会有抖动, 所以用 ``tolerance`` 容差。

        Args:
            advance: 让游戏推进 N 帧的回调 (由调用方用按键/noop 实现)。
            frames: 每次推进的帧数。
            tolerance: 允许的帧数误差比例 (0.25 => ±25%)。
            repeats: 重复几次取交集 (越多越干净, 也越慢)。
            include_images: 是否扫描模块映像区 (帧计数器通常在全局数据里)。
        """
        import numpy as np

        emit = log or (lambda *_: None)
        low = frames * (1.0 - tolerance)
        high = frames * (1.0 + tolerance)

        def capture() -> list[tuple[int, np.ndarray]]:
            return self._capture_arrays(include_images)

        def advanced_addresses(
            before: list[tuple[int, np.ndarray]], after: list[tuple[int, np.ndarray]]
        ) -> set[int]:
            after_by_base = dict(after)
            keep: set[int] = set()
            for base, old in before:
                new = after_by_base.get(base)
                if new is None or new.shape != old.shape:
                    continue
                diff = (new.astype(np.int64) - old.astype(np.int64)) & 0xFFFFFFFF
                diff = np.where(diff >= 0x80000000, diff - 0x100000000, diff)
                mask = (diff >= low) & (diff <= high)
                for index in np.flatnonzero(mask).tolist():
                    keep.add(base + 4 * int(index))
            return keep

        candidates: set[int] | None = None
        previous = capture()
        for round_index in range(max(1, repeats)):
            advance(frames)
            current = capture()
            kept = advanced_addresses(previous, current)
            emit(f"第 {round_index + 1} 轮: 推进 {frames} 帧后 +{frames} 的地址 {len(kept)} 个")
            candidates = kept if candidates is None else (candidates & kept)
            previous = current

        out: list[Candidate] = []
        for address in sorted(candidates or set()):
            module, module_offset = self._locate_module(address)
            out.append(
                Candidate(
                    address=address,
                    value=None,
                    module=module,
                    module_offset=module_offset,
                    region_type="frame",
                )
            )
        emit(f"最终候选: {len(out)} 个")
        self.save(out, query="find_frame", type_name="i32")
        return out

    # ------------------------------------------------------------------
    # 全自动定位残机 (lives)
    # ------------------------------------------------------------------
    def module_i32(self) -> tuple[str, int, "np.ndarray"]:
        """把主模块映像读成 ``(模块名, 基址, i32 数组)`` (残机等全局量都在这里).

        模块映像里有不可读页, 所以**按可读区域分块读**, 读不到的填 0。
        """
        import numpy as np

        name, base, size = self.mem.main_module
        arr = np.zeros(size // 4, dtype="<i4")
        for rbase, rsize, _prot, _mtype in self.mem.memory_regions():
            lo = max(rbase, base)
            hi = min(rbase + rsize, base + size)
            if hi <= lo:
                continue
            try:
                data = self.mem.read_bytes(lo, hi - lo)
            except Exception:
                continue
            count = len(data) // 4
            offset = (lo - base) // 4
            arr[offset : offset + count] = np.frombuffer(data, dtype="<i4", count=count)
        return name, base, arr

    def find_lives(
        self,
        *,
        lo: int = 1,
        hi: int = 9,
        settle: float = 4.0,
        log=None,
    ) -> list[Candidate]:
        """在模块映像里找"数值小、且一段时间内不变"的整数 —— 残机 (lives) 的候选.

        残机是全局量 (模块映像内), 平时不变, 只在**掉残时 -1**。所以先用
        "值在 [lo,hi] + 等待 settle 秒仍不变"筛出少量候选; 再靠"掉残 -1"确认
        (见 :meth:`watch_decrements`)。返回的是模块相对偏移, **跨重启稳定**。

        Args:
            lo / hi: 残机的合理范围 (默认 1..9)。
            settle: 等待多少秒, 期间要求数值不变 (用于剔除一直在变的量)。
        """
        import numpy as np
        import time

        emit = log or (lambda *_: None)
        name, base, arr0 = self.module_i32()
        idx = np.flatnonzero((arr0 >= lo) & (arr0 <= hi))
        emit(f"模块 {name} 内 i32 落在 [{lo},{hi}]: {idx.size} 个")
        time.sleep(settle)
        _, _, arr1 = self.module_i32()
        if arr1.size == arr0.size:
            idx = idx[arr1[idx] == arr0[idx]]
        emit(f"等待 {settle:.0f}s 后仍不变: {idx.size} 个")
        out: list[Candidate] = []
        for i in idx.tolist():
            out.append(
                Candidate(
                    address=base + 4 * int(i),
                    value=int(arr0[i]),
                    module=name,
                    module_offset=4 * int(i),
                    region_type="lives",
                )
            )
        self.save(out, query="find_lives", type_name="i32")
        return out

    def watch_decrements(
        self,
        candidates: list[Candidate],
        *,
        press,
        seconds: float = 40.0,
        poll: float = 0.05,
        log=None,
    ) -> dict[int, list[int]]:
        """让自机主动撞弹 ``seconds`` 秒, 记录每个候选"每次 -1"后的新值序列.

        掉残的那个会得到 ``[1, 0, -1]`` 这种**严格 -1 递减**的序列; 定时器则是密集递减
        或会回绕。只处理主模块内的候选 (残机在模块映像里), 每轮读一次模块映像做差分。
        """
        import time

        emit = log or (lambda *_: None)
        _, base, arr = self.module_i32()
        idxs = {
            c.address: (c.address - base) // 4
            for c in candidates
            if base <= c.address < base + 4 * arr.size
        }
        prev = {a: int(arr[i]) for a, i in idxs.items()}
        events: dict[int, list[int]] = {a: [] for a in idxs}
        deadline = time.time() + seconds
        last_press = 0.0
        while time.time() < deadline:
            if time.time() - last_press > 1.5:
                press(["up"], 20)
                press(["left"], 10)
                press(["right"], 10)
                press(["down"], 10)
                last_press = time.time()
            _, _, arr = self.module_i32()
            for a, i in idxs.items():
                cur = int(arr[i])
                if cur == prev[a] - 1:
                    events[a].append(cur)
                prev[a] = cur
            time.sleep(poll)
        return {a: ev for a, ev in events.items() if ev}

    # ------------------------------------------------------------------
    # 利用"被击中 -> 自机回到固定初始位置"定位自机坐标
    # ------------------------------------------------------------------
    def find_respawn_coordinates(
        self,
        lives_address: int,
        *,
        hits: int = 3,
        press=None,
        timeout: float = 150.0,
        refresh: float = 1.0,
        log=None,
    ) -> list[Candidate]:
        """靠"被击中时自机跳回固定初始位置"来找自机坐标.

        每次 ``lives`` 减少 1 的瞬间抓 f32 快照并与上一次快照差分, 记录所有变化地址的
        **新值**; 反复几次后, 只保留"每次被击中后都跳回**同一个固定值**"的地址 —— 那就是
        自机 x (相邻的 y 同理)。比撞墙法稳得多, 且天然不怕弹幕干扰。

        Args:
            lives_address: 已标定的残机地址 (lives)。
            hits: 需要观察几次被击中。
            press: 推进游戏并(可选)把自机送去撞弹的回调。
            refresh: 无被击中时, 每隔多久刷新一次基准快照。
        """
        import numpy as np
        import time

        emit = log or (lambda *_: None)

        def read_lives() -> int | None:
            try:
                return int(self.mem.read_value(lives_address, "i32"))
            except Exception:
                return None

        def snap() -> dict[int, "np.ndarray"]:
            return dict(self._capture_arrays(include_images=False))

        records: dict[int, list[float]] = {}
        last = snap()
        prev_lives = read_lives()
        got = 0
        deadline = time.time() + timeout
        last_refresh = time.time()
        while got < hits and time.time() < deadline:
            if press is not None:
                press(["noop"], 2)
            cur_lives = read_lives()
            if cur_lives is not None and prev_lives is not None and cur_lives < prev_lives:
                time.sleep(1.2)  # 等死亡动画结束、自机回到固定初始位置
                after = snap()
                for base, a_after in after.items():
                    a_before = last.get(base)
                    if a_before is None or a_before.shape != a_after.shape:
                        continue
                    f_after = a_after.view("<f4")
                    for i in np.flatnonzero(a_after != a_before).tolist():
                        v = float(f_after[i])
                        # 只记录"像坐标"的值 (排除被击中时被清零的字段)
                        if 1.0 <= v <= 448.0:
                            records.setdefault(base + 4 * int(i), []).append(v)
                got += 1
                emit(f"第 {got}/{hits} 次被击中: 累计变化地址 {len(records)}")
                last = after
                last_refresh = time.time()
            elif time.time() - last_refresh > refresh:
                last = snap()
                last_refresh = time.time()
            prev_lives = cur_lives
            time.sleep(0.03)

        out: list[Candidate] = []
        for addr, vals in records.items():
            if len(vals) >= hits and max(vals) - min(vals) < 0.5:
                module, module_offset = self._locate_module(addr)
                out.append(
                    Candidate(
                        address=addr,
                        value=vals[0],
                        module=module,
                        module_offset=module_offset,
                        region_type="respawn",
                    )
                )
        emit(f"每次被击中后都跳回同一固定值的地址: {len(out)} 个")
        self.save(out, query="find_respawn", type_name="f32")
        return out

    # ------------------------------------------------------------------
    # 撞墙法 ∩ "被击中跳回固定初始位置" —— 双重特征定位自机坐标
    # ------------------------------------------------------------------
    def find_player_by_respawn_and_wall(
        self,
        lives_address: int,
        *,
        press,
        hits: int = 3,
        timeout: float = 180.0,
        log=None,
    ) -> list[Candidate]:
        """先用撞墙法得到"像自机 x"的候选, 再用"被击中时跳回固定初始值"确认.

        1. 撞墙法: 墙边不动期间不变 + 移动期会变 + 左右墙值不同 -> 一批候选;
        2. 观察被击中 (lives -1), 记录候选在**被击中后**的值;
        3. 只保留"每次被击中后都跳到**同一固定值**"的 —— 自机被击中会回到固定初始位置,
           弹幕/其它量不会每次都跳到同一个值。
        """
        import struct
        import time

        emit = log or (lambda *_: None)
        cands = self.find_player_coordinates(
            press,
            settle_frames=120,
            hold_frames=45,
            move_frames=80,
            low=1.0,
            high=448.0,
            include_images=False,
            log=emit,
        )
        emit(f"撞墙法候选 {len(cands)} 个")
        if not cands:
            return []
        addrs = [c.address for c in cands]

        def read_i32(a: int) -> int | None:
            try:
                return int(self.mem.read_value(a, "i32"))
            except Exception:
                return None

        def read_f32(a: int) -> float | None:
            try:
                return float(struct.unpack("<f", self.mem.read_bytes(a, 4))[0])
            except Exception:
                return None

        records: dict[int, list[float]] = {a: [] for a in addrs}
        prev_lives = read_i32(lives_address)
        got = 0
        deadline = time.time() + timeout
        while got < hits and time.time() < deadline:
            press(["noop"], 2)
            cur = read_i32(lives_address)
            if cur is not None and prev_lives is not None and cur < prev_lives:
                time.sleep(1.2)  # 等自机回到固定初始位置
                for a in addrs:
                    v = read_f32(a)
                    if v is not None and 1.0 <= v <= 448.0:
                        records[a].append(v)
                got += 1
                emit(f"第 {got}/{hits} 次被击中")
            prev_lives = cur
            time.sleep(0.03)

        out: list[Candidate] = []
        for a in addrs:
            vals = records[a]
            if len(vals) >= hits and max(vals) - min(vals) < 0.5:
                out.append(Candidate(address=a, value=vals[0], region_type="respawn"))
        emit(f"双重特征后候选 {len(out)} 个")
        self.save(out, query="player_combined", type_name="f32")
        return out

    # ------------------------------------------------------------------
    # 全自动定位无敌帧 (invincible)
    # ------------------------------------------------------------------
    def find_invincible(
        self,
        lives_address: int,
        *,
        hits: int = 2,
        press=None,
        timeout: float = 120.0,
        refresh: float = 1.0,
        log=None,
    ) -> list[Candidate]:
        """找"无敌帧"计数: 被击中瞬间由 0 变成正值, 随后递减回 0.

        用与找残机相同的手法 —— 在 lives -1 的瞬间抓 i32 差分, 找"从 0 变成正值"的地址;
        反复几次取交集。锁定它就能让自机**不再被击中**(因此不再回位), 撞墙法随即可用。
        """
        import numpy as np
        import time

        emit = log or (lambda *_: None)

        def read_lives() -> int | None:
            try:
                return int(self.mem.read_value(lives_address, "i32"))
            except Exception:
                return None

        def snap() -> dict[int, "np.ndarray"]:
            return dict(self._capture_arrays(include_images=True))

        records: dict[int, list[int]] = {}
        last = snap()
        prev_lives = read_lives()
        got = 0
        deadline = time.time() + timeout
        last_refresh = time.time()
        while got < hits and time.time() < deadline:
            if press is not None:
                press(["noop"], 2)
            cur_lives = read_lives()
            if cur_lives is not None and prev_lives is not None and cur_lives < prev_lives:
                # 命中后连续采样 ~3s: 无敌计数应"0 -> 正值 -> 回到 0"且单调递减
                time.sleep(0.05)
                seqs = []
                for _ in range(10):
                    seqs.append(snap())
                    time.sleep(0.3)
                for base, a0 in last.items():
                    if a0 is None:
                        continue
                    if any(s.get(base) is None or s[base].shape != a0.shape for s in seqs):
                        continue
                    v0 = a0.view("<i4")
                    vv = [s[base].view("<i4") for s in seqs]
                    with np.errstate(invalid="ignore"):
                        mask = (v0 <= 1) & (vv[0] >= 10) & (vv[0] <= 3000)
                        for k in range(9):
                            mask &= vv[k + 1] <= vv[k]
                        mask &= vv[9] <= 2  # ~3s 内回到 0
                    for i in np.flatnonzero(mask).tolist():
                        records.setdefault(base + 4 * int(i), []).append(int(vv[0][i]))
                got += 1
                emit(f"第 {got}/{hits} 次被击中: 累计候选 {len(records)}")
                last = seqs[-1]
                last_refresh = time.time()
            elif time.time() - last_refresh > refresh:
                last = snap()
                last_refresh = time.time()
            prev_lives = cur_lives
            time.sleep(0.03)

        out: list[Candidate] = []
        for addr, vals in records.items():
            if len(vals) >= hits:
                module, module_offset = self._locate_module(addr)
                out.append(
                    Candidate(
                        address=addr,
                        value=vals[-1],
                        module=module,
                        module_offset=module_offset,
                        region_type="invincible",
                    )
                )
        emit(f"被击中由 0 变正值的地址: {len(out)} 个")
        self.save(out, query="find_invincible", type_name="i32")
        return out

    # ------------------------------------------------------------------
    # 回位法 + 左右验证 -> 自机 x 候选 (少量)
    # ------------------------------------------------------------------
    def find_player_by_respawn_lr(
        self, lives_address: int, *, press, timeout: float = 120.0, log=None
    ) -> list[Candidate]:
        """被击中复位后 x≈中心、y≈底部, 再要求"左移变小、右移变大" —— 得到少量候选."""
        import time

        emit = log or (lambda *_: None)

        def read_i32(a: int) -> int | None:
            try:
                return int(self.mem.read_value(a, "i32"))
            except Exception:
                return None

        def read_f32(a: int) -> float | None:
            try:
                return float(__import__("struct").unpack("<f", self.mem.read_bytes(a, 4))[0])
            except Exception:
                return None

        prev = read_i32(lives_address)
        hit = False
        deadline = time.time() + timeout
        while time.time() < deadline:
            if press is not None:
                press(["up"], 6)
            cur = read_i32(lives_address)
            if cur is not None and prev is not None and cur < prev:
                hit = True
                break
            prev = cur
        if not hit:
            emit("没等到被击中")
            return []
        time.sleep(1.2)  # 等复位
        arrays = dict(self._capture_arrays(include_images=True))
        import numpy as np

        hits: list[int] = []
        for base, arr in arrays.items():
            f = arr.view("<f4")
            if len(f) < 2:
                continue
            with np.errstate(invalid="ignore"):
                m = (f[:-1] >= 150.0) & (f[:-1] <= 240.0) & (f[1:] >= 350.0) & (f[1:] <= 448.0)
            for i in np.flatnonzero(m).tolist():
                hits.append(base + 4 * int(i))
        emit(f"复位后 x∈[150,240] & y∈[350,448]: {len(hits)}")
        if press is not None:
            press(["left"], 110)
        keep = [a for a in hits if (read_f32(a) is not None and 1 <= read_f32(a) <= 60)]
        if press is not None:
            press(["right"], 150)
        final = [a for a in keep if (read_f32(a) is not None and 300 <= read_f32(a) <= 384)]
        emit(f"左右验证后: {len(final)}")
        out = [Candidate(address=a, value=read_f32(a), region_type="respawn_lr") for a in final]
        self.save(out, query="player_respawn_lr", type_name="f32")
        return out

    # ------------------------------------------------------------------
    # 指针链查找 (CE 指针扫描的自动化版)
    # ------------------------------------------------------------------
    def find_pointer_chain(
        self,
        target: int,
        *,
        max_offset: int = 0x1000,
        max_levels: int = 4,
        limit: int = 60,
        arrays: dict[int, "np.ndarray"] | None = None,
        log=None,
    ) -> list[dict[str, Any]]:
        """为地址 ``target`` 找**模块级指针链** (以主模块基址为根).

        指针链语义 (与 reader 一致)::

            addr = *(base + pointer)
            for off in offsets[:-1]: addr = *(addr + off)
            value_addr = addr + offsets[-1]

        做法: 反向 BFS —— 先找"含指针(值+偏移==target)的地址", 是模块内的即为根;
        否则沿堆再上一级, 直到落到模块指针。

        Returns:
            ``[{"pointer": 模块偏移, "module": 名, "offsets": [...], "levels": n}, ...]``
        """
        import numpy as np

        emit = log or (lambda *_: None)
        name, mbase, msize = self.mem.main_module
        if arrays is None:
            arrays = dict(self._capture_arrays(include_images=True))

        def pointers_to(tgt: int) -> list[tuple[int, int]]:
            lo = max(0, tgt - max_offset)
            out: list[tuple[int, int]] = []
            for base, arr in arrays.items():
                a = arr.astype(np.int64)
                mask = (a >= lo) & (a <= tgt) & (a >= 0x10000)
                for i in np.flatnonzero(mask).tolist():
                    out.append((base + 4 * int(i), tgt - int(arr[i])))
            return out

        results: list[dict[str, Any]] = []
        frontier: list[tuple[int, list[int]]] = [(target, [])]
        seen: set[tuple[int, tuple[int, ...]]] = set()
        for level in range(1, max_levels + 1):
            nxt: list[tuple[int, list[int]]] = []
            for tgt, inner in frontier:
                for Q, off in pointers_to(tgt):
                    chain = [off] + inner
                    if mbase <= Q < mbase + msize:
                        results.append(
                            {"pointer": Q - mbase, "module": name, "offsets": chain, "levels": level}
                        )
                        if len(results) >= limit:
                            emit(f"找到 {len(results)} 条链 (上限)")
                            return results
                    else:
                        key = (Q, tuple(chain))
                        if key in seen:
                            continue
                        seen.add(key)
                        nxt.append((Q, chain))
            emit(f"第 {level} 层: 新队列 {len(nxt)}, 累计链 {len(results)}")
            frontier = nxt[:300]  # 限制规模, 防爆炸
            if not frontier:
                break
        return results

    # ------------------------------------------------------------------
    def save(self, candidates: list[Candidate], *, query: Any = None, type_name: str = "i32") -> Path:
        path = self._candidate_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "pid": self.mem.pid,
            "count": len(candidates),
            "query": query,
            "type": type_name,
            "candidates": [c.to_dict() for c in candidates],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def load(self) -> list[Candidate]:
        path = self._candidate_path()
        if not path.exists():
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        out: list[Candidate] = []
        for item in data.get("candidates", []):
            address = item["address"]
            address = int(address, 16) if isinstance(address, str) else int(address)
            module, module_offset = self._locate_module(address)
            out.append(Candidate(address=address, value=item.get("value"), module=module, module_offset=module_offset))
        return out

    def clear(self) -> None:
        path = self._candidate_path()
        if path.exists():
            path.unlink()
