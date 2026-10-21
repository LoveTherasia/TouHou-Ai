"""找 TH11 掉落物 (P点/点道具/星) 的内存布局 —— 只读, 会重启游戏。

分三步走, 每一步都能单独跑:

1. ``globals``: 进关攒一波道具, 转储模块里"管理器指针区" (``base+0xA8C00..0xA8E00``)
   的每个 4 字节槽 —— 弹幕在 ``0xA8D68``、符卡在 ``0xA8D6C``、敌人在 ``0xA8D7C``,
   道具的管理器多半就在同一块里。对每个像指针的值, 读出它指向对象的几个候选计数字段。
2. ``watch 0xA8D80``: 盯住某个槽 20 秒 (一边打一边收), 打印目标对象里
   ``+0x5C/+0x60/+0x64/+0x68/+0x6C/+0x70`` 的变化 —— 道具池的计数会随收道具抖动。
3. ``diff``: 兜底 —— 全内存 float 差分, 找"x 不变、y 在涨、都在场地内"的地址,
   按地址聚类后报常见步长 (道具是固定步长的池子)。

前两步不会改游戏; 第三步只是读内存。都会 ``restart_process`` 重开一局。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from touhou_rl_mcp.backends.memory import MemoryBackend  # noqa: E402
from touhou_rl_mcp.config import Config  # noqa: E402
from touhou_rl_mcp.games.registry import GameRegistry  # noqa: E402
from touhou_rl_mcp.inputs.actions import parse  # noqa: E402
from touhou_rl_mcp.memory.cheats import LivesLock  # noqa: E402

WINDOW = (0xA8A00, 0xA9300)
COUNT_FIELDS = (0x5C, 0x60, 0x64, 0x68, 0x6C, 0x70)


def enter(cfg: Config):
    be = MemoryBackend(GameRegistry().get("th11"), cfg)
    be.start()
    be.reset(options={"restart_process": True, "difficulty": 0})
    print("[items] 已进关, 等自机可操作 ...", flush=True)
    for _ in range(80):
        s = be.read_state()
        if s.in_game and (s.power or 0) > 0.3:
            break
        be.apply(parse("noop", default_frames=20))
    return be


def farm(be, rounds: int = 70) -> None:
    """边打边左右晃: 让杂鱼掉道具, 同时有概率吃到."""
    for k in range(rounds):
        be.apply(parse("shoot+left" if k % 2 else "shoot+right", default_frames=8))
        if not be.read_state().in_game:
            break


def cmd_globals(be) -> int:
    mem = be.mem
    base = mem.main_module[1]

    def u32(a: int) -> int:
        try:
            return int(mem.read_value(a, "u32"))
        except Exception:  # noqa: BLE001
            return 0

    print(f"[items] base=0x{base:08X} 管理器区 0x{WINDOW[0]:X}..0x{WINDOW[1]:X}", flush=True)
    for off in range(WINDOW[0], WINDOW[1], 4):
        p = u32(base + off)
        if not (0x01000000 <= p <= 0x7FFFFFFF):
            continue
        try:
            counts = [u32(p + f) for f in COUNT_FIELDS]
            head = u32(p + 0x64)
            probe = u32(p + 0x68)
            floats = np.frombuffer(mem.read_bytes(p + 0x64, 32), dtype="<f4")
        except Exception:  # noqa: BLE001
            continue
        print(f"  0x{off:X} -> 0x{p:08X}  counts(+5C..+70)={counts} "
              f" [+64]=0x{head:08X} [+68]=0x{probe:08X} "
              f"f32@+64={np.round(floats[:6], 2).tolist()}", flush=True)
    return 0


def cmd_watch(be, slot: str, seconds: float) -> int:
    mem = be.mem
    base = mem.main_module[1]
    off = int(slot, 16)
    p = int(mem.read_value(base + off, "u32"))
    print(f"[items] 监视 0x{off:X} -> 0x{p:08X} 共 {seconds:.0f}s", flush=True)
    if not p:
        print("[items] 该槽为空")
        return 1
    deadline = time.perf_counter() + seconds
    last = None
    k = 0
    while time.perf_counter() < deadline:
        be.apply(parse("shoot+left" if k % 2 else "shoot+right", default_frames=6))
        k += 1
        try:
            cur = [int(mem.read_value(p + f, "u32")) for f in COUNT_FIELDS]
            first = [round(float(v), 1) for v in
                     np.frombuffer(mem.read_bytes(p + 0x64, 24), dtype="<f4")]
        except Exception:  # noqa: BLE001
            print("[items] 读失败 (对象可能被重建)")
            break
        if cur != last:
            print(f"   t={time.perf_counter() % 1000:6.1f} counts={cur} f32@+64={first}", flush=True)
            last = cur
    return 0


def cmd_diff(be, gap: float = 0.05) -> int:
    """逐区域"读 A -> 等 gap -> 读 B", 让每个区域的采样间隔一致.

    整块先拍 A 再拍 B 的间隔不可控 (拍 A 本身就要几秒), 道具的位移会被摊平;
    逐区域配对后间隔固定 2~4 帧, 道具"x 不动 + y 匀速下落 ~1px/帧"才好认。
    """
    mem = be.mem
    regions = [(b, sz) for b, sz, _p, ty in mem.memory_regions()
               if sz <= 64 * 1024 * 1024 and not (ty & 0x1000000)]
    print(f"[items] 待扫区域 {len(regions)} 个, 每个间隔 {gap*1000:.0f}ms", flush=True)

    hits: dict[int, tuple[float, float, float]] = {}
    scanned = 0
    for b, sz in regions:
        try:
            a = np.frombuffer(mem.read_bytes(b, sz), dtype="<f4").copy()
            time.sleep(gap)
            c = np.frombuffer(mem.read_bytes(b, sz), dtype="<f4").copy()
        except Exception:  # noqa: BLE001
            continue
        if a.shape != c.shape or len(a) < 2:
            continue
        scanned += 1
        x0, y0 = a[:-1], a[1:]
        x1, y1 = c[:-1], c[1:]
        ok = ((np.abs(x1 - x0) < 0.01)              # x 完全不动
              & (y1 > y0 + 0.3) & (y1 < y0 + 8.0)  # 2~4 帧的匀速下落
              & (x1 >= 16) & (x1 <= 368)
              & (y1 >= 0) & (y1 <= 460))
        for i in np.flatnonzero(ok).tolist():
            hits[b + 4 * int(i)] = (float(x1[i]), float(y1[i]), float(y1[i] - y0[i]))
    print(f"[items] 扫了 {scanned} 个区域, 'x不动+y下落' 命中 {len(hits)} 个地址", flush=True)
    if not hits:
        return 1

    order = sorted(hits)
    clusters, cur = [], [order[0]]
    for addr in order[1:]:
        if addr - cur[-1] <= 0x1000:
            cur.append(addr)
        else:
            clusters.append(cur)
            cur = [addr]
    clusters.append(cur)
    clusters.sort(key=len, reverse=True)
    for cl in clusters[:6]:
        diffs: dict[int, int] = {}
        for i in range(len(cl) - 1):
            d = cl[i + 1] - cl[i]
            diffs[d] = diffs.get(d, 0) + 1
        top = sorted(diffs.items(), key=lambda kv: -kv[1])[:3]
        print(f"   span 0x{cl[0]:08X}..0x{cl[-1]:08X} 元素={len(cl)} "
              f"步长={[(hex(k), v) for k, v in top]}", flush=True)
        for addr in cl[:4]:
            x, y, d = hits[addr]
            print(f"      0x{addr:08X} x={x:.1f} y={y:.1f} dy={d:.2f}", flush=True)
    return 0


def cmd_dump(be, address: str, length: int = 0x200) -> int:
    """十六进制 + float 双视图转储, 用来人眼找"池指针/计数"."""
    mem = be.mem
    base = mem.main_module[1]
    addr = int(address, 16)
    if addr < 0x200000:  # 小值按"模块内全局槽"解释, 现场解引用 (堆地址每次运行都变)
        ptr = int(mem.read_value(base + addr, "u32"))
        print(f"[items] 全局 0x{addr:X} -> 0x{ptr:08X}", flush=True)
        addr = ptr
    if not addr:
        print("[items] 空指针")
        return 1
    data = mem.read_bytes(addr, length)
    for row in range(0, length, 16):
        chunk = data[row : row + 16]
        hexs = " ".join(f"{b:02x}" for b in chunk)
        u32 = [int.from_bytes(chunk[i : i + 4], "little") for i in range(0, len(chunk), 4)]
        flt = np.frombuffer(chunk, dtype="<f4")
        fstr = " ".join(f"{v:9.2f}" for v in flt)
        ustr = " ".join(f"{v:08X}" for v in u32)
        print(f"  +{row:04X}  {fstr} | {ustr}", flush=True)
    return 0


STRIDES = (0x10, 0x14, 0x18, 0x1C, 0x20, 0x24, 0x28, 0x2C, 0x30, 0x38, 0x40,
           0x48, 0x50, 0x58, 0x60, 0x68, 0x70, 0x80, 0x90, 0xA0, 0xB0, 0xC0,
           0x100, 0x140, 0x180, 0x200, 0x280, 0x300, 0x400)


def cmd_pool(be, max_regions: int = 60) -> int:
    """在堆里找"像道具池"的数组: 少量(3~40)个槽带场地坐标, 且这些坐标挤成一团.

    判据来自道具的生成方式: 一只杂鱼死亡会同时掉 3~8 个道具, 位置几乎重合,
    一起往下落。所以"位置在场地内 + 总数很少 + y 跨度小"同时成立才算候选。
    弹幕池虽然也有大量场地坐标, 但活动数远大于 40, 会被这一条直接排掉。
    """
    mem = be.mem
    regions = [(b, sz) for b, sz, _p, ty in mem.memory_regions()
               if 0x10000 <= sz <= 8 * 1024 * 1024 and not (ty & 0x1000000)]
    regions.sort(key=lambda r: -r[1])
    print(f"[items] 候选堆块 {len(regions)} 个 (扫前 {max_regions} 个)", flush=True)

    found: list[tuple[int, int, int, int, float]] = []
    for b, sz in regions[:max_regions]:
        try:
            arr = np.frombuffer(mem.read_bytes(b, sz), dtype="<f4")
        except Exception:  # noqa: BLE001
            continue
        if len(arr) < 64:
            continue
        x_ok = (arr >= 16) & (arr <= 368)
        y_ok = np.zeros_like(x_ok)
        y_ok[:-1] = (arr[1:] >= 0) & (arr[1:] <= 460)
        starts = np.flatnonzero(x_ok & y_ok)
        if starts.size == 0:
            continue
        # 只拿"前面 3 个槽都还在场地里"的起点当种子, 免得把零散位置全试一遍
        for i in starts[:400].tolist():
            for stride in STRIDES:
                step = stride // 4
                if i + 3 * step >= len(arr):
                    continue
                xs, ys, cnt = [], [], 0
                for k in range(8):
                    j = i + k * step
                    if j + 1 >= len(arr):
                        break
                    x, y = float(arr[j]), float(arr[j + 1])
                    if 16 <= x <= 368 and 0 <= y <= 460:
                        cnt += 1
                        xs.append(x)
                        ys.append(y)
                if cnt >= 4 and np.ptp(ys) < 90.0:
                    found.append((b + 4 * i, stride, cnt, int(np.ptp(ys)), float(np.mean(xs))))

    found.sort(key=lambda t: -t[2])
    print(f"[items] 候选 {len(found)} 个:", flush=True)
    seen = set()
    for addr, stride, cnt, spread, mx in found[:20]:
        key = (addr // 0x1000, stride)
        if key in seen:
            continue
        seen.add(key)
        print(f"    0x{addr:08X} 步长=0x{stride:X} 有效槽={cnt} y跨度={spread} 平均x={mx:.0f}",
              flush=True)
    return 0


def cmd_watch_all(be, seconds: float = 18.0, hz: float = 8.0) -> int:
    """同时盯住管理器指针区每个对象的"小整数计数器", 找随道具出现/消失而抖动的字段.

    思路: 道具池的活动数一定是 **0..40 之间的小整数**, 且会来回变。
    对每个"像指针"的全局槽, 采样其对象 ``+0x00..+0x200`` 的 u32, 只留下
    "始终 <= 64 且至少取过 3 个不同值"的 (槽, 偏移), 这些就是候选计数。
    """
    mem = be.mem
    base = mem.main_module[1]
    slots: dict[int, int] = {}
    for off in range(WINDOW[0], WINDOW[1], 4):
        try:
            p = int(mem.read_value(base + off, "u32"))
        except Exception:  # noqa: BLE001
            continue
        if 0x01000000 <= p <= 0x7FFFFFFF:
            slots[off] = p
    print(f"[items] 监视 {len(slots)} 个对象, {seconds:.0f}s @ {hz:.0f}Hz", flush=True)

    series: dict[tuple[int, int], list[int]] = {}
    deadline = time.perf_counter() + seconds
    k = 0
    while time.perf_counter() < deadline:
        be.apply(parse("shoot+left" if k % 2 else "shoot+right", default_frames=4))
        k += 1
        for off, ptr in slots.items():
            try:
                raw = mem.read_bytes(ptr, 0x200)
            except Exception:  # noqa: BLE001
                continue
            for i in range(0, 0x200, 4):
                v = int.from_bytes(raw[i : i + 4], "little")
                series.setdefault((off, i), []).append(v)
        time.sleep(1.0 / hz)

    hot = []
    for key, vals in series.items():
        if len(vals) < 6:
            continue
        lo, hi = min(vals), max(vals)
        if hi > 64:
            continue
        if len(set(vals)) >= 3:
            hot.append((key, lo, hi, len(set(vals)), vals[:14]))
    hot.sort(key=lambda t: -t[3])
    print(f"[items] 抖动的 0..64 计数器 {len(hot)} 个:", flush=True)
    for (off, i), lo, hi, n, sample in hot[:20]:
        print(f"    槽0x{off:X} +0x{i:03X}: {lo}..{hi} 取值={n} 样本={sample}", flush=True)
    return 0


def _in_field_count(mem, regions) -> dict[int, int]:
    """每个堆块里"像场地坐标的 float 对"的数量."""
    out = {}
    for b, sz in regions:
        try:
            arr = np.frombuffer(mem.read_bytes(b, sz), dtype="<f4")
        except Exception:  # noqa: BLE001
            continue
        if len(arr) < 4:
            continue
        pair = ((arr[:-1] >= 8) & (arr[:-1] <= 384)
                & (arr[1:] >= 0) & (arr[1:] <= 464))
        # 只数"相邻两个 float 都在场地内"的位置 (x,y 相邻是 TH11 实体的写法)
        out[b] = int(pair.sum())
    return out


def cmd_countdiff(be) -> int:
    """比"有道具"与"没道具"两个时刻, 每个堆块里的场地坐标对数量差异.

    道具存在时间很短 (落地即消失), 所以不追求精确的位移, 只看**数量涨落**:
    哪个堆块在道具多的时候坐标对明显更多, 它就是道具池。
    """
    mem = be.mem
    regions = [(b, sz) for b, sz, _p, ty in mem.memory_regions()
               if 0x10000 <= sz <= 8 * 1024 * 1024 and not (ty & 0x1000000)]
    print("[items] 安静期采样 ...", flush=True)
    be.apply(parse("left", default_frames=20))
    be.apply(parse("right", default_frames=20))
    quiet = _in_field_count(mem, regions)
    print(f"[items] 安静期: 总坐标对={sum(quiet.values())}", flush=True)

    print("[items] 打死一波然后再采 ...", flush=True)
    for k in range(6):
        be.apply(parse("shoot+left" if k % 2 else "shoot+right", default_frames=6))
    busy = _in_field_count(mem, regions)
    print(f"[items] 道具期: 总坐标对={sum(busy.values())}", flush=True)

    delta = sorted(((busy[b] - quiet.get(b, 0), b, regions_dict_size(b, regions))
                    for b in busy), key=lambda t: -t[0])
    print("[items] 坐标对增量最大的堆块:", flush=True)
    for d, b, sz in delta[:10]:
        print(f"    0x{b:08X} 大小=0x{sz:X} 增量={d:+d} (安静 {quiet.get(b, 0)} -> {busy[b]})",
              flush=True)
    return 0


def regions_dict_size(base: int, regions) -> int:
    for b, sz in regions:
        if b == base:
            return sz
    return 0


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "globals"
    cfg = Config({"backend": "memory", "game": "th11", "restart_process": True,
                  "dialogue_assist": True, "capture": {"prefer_printwindow": False}})
    be = enter(cfg)
    lock = LivesLock(be.mem, be.reader.address_of("lives"), value=9, floor=1).start()
    try:
        farm(be)
        be.apply(parse("noop", default_frames=10))
        if cmd == "globals":
            return cmd_globals(be)
        if cmd == "watch":
            return cmd_watch(be, sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 20.0)
        if cmd == "diff":
            return cmd_diff(be)
        if cmd == "countdiff":
            return cmd_countdiff(be)
        if cmd == "watchall":
            return cmd_watch_all(be, float(sys.argv[2]) if len(sys.argv) > 2 else 18.0)
        if cmd == "pool":
            return cmd_pool(be)
        if cmd == "dump":
            return cmd_dump(be, sys.argv[2], int(sys.argv[3], 16) if len(sys.argv) > 3 else 0x200)
        print("用法: find_items.py [globals|watch <0xOFF> [秒]|watchall [秒]|diff|pool|dump <0xADDR|0x全局槽> [长度]]")
        return 2
    finally:
        lock.stop()
        be.close()


if __name__ == "__main__":
    raise SystemExit(main())
