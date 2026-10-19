"""真机探针 —— 标定内存偏移用的命令行工具.

它是 MCP 工具的"命令行版", 好处是每一步都是独立进程, 可以一边看着游戏一边
反复执行, 不用重连 MCP。

用法示例::

    python tools/probe.py status
    python tools/probe.py start --screenshots            # 菜单导航进游戏, 每步截图
    python tools/probe.py shot --crop
    python tools/probe.py hold right --frames 60          # 按住右键 60 帧
    python tools/probe.py scan 0 --type i32               # 首扫
    python tools/probe.py filter 11 --type i32            # 再扫
    python tools/probe.py changed --expect changed        # 筛出变化了的候选
    python tools/probe.py cands --limit 20
    python tools/probe.py read th11c.exe+0xA56E4 --type i32
    python tools/probe.py read 0x1234ABCD --type f32 --offsets 0x18,0x24
    python tools/probe.py watch 0x1234ABCD --type f32 --seconds 6
    python tools/probe.py write-json                      # 把标定结果提示打印出来
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from touhou_rl_mcp.capture.screenshot import crop_playfield, grab_window, save_png  # noqa: E402
from touhou_rl_mcp.capture.visualize import save_png as save_viz  # noqa: E402
from touhou_rl_mcp.config import Config, runtime_dir  # noqa: E402
from touhou_rl_mcp.errors import TouhouError  # noqa: E402
from touhou_rl_mcp.games.calibration import apply_calibration  # noqa: E402
from touhou_rl_mcp.games.registry import GameRegistry  # noqa: E402
from touhou_rl_mcp.inputs.actions import parse  # noqa: E402
from touhou_rl_mcp.backends.memory import MemoryBackend  # noqa: E402


def build_backend(args: argparse.Namespace) -> MemoryBackend:
    config = Config.load(args.config)
    # probe 工具永远针对真实游戏后端 (config.json 里可能是 mock)
    config.patch({"backend": "memory", "game": args.game})
    registry = GameRegistry()
    spec = registry.get(args.game)
    backend = MemoryBackend(spec, config)
    backend.start()
    return backend


def fmt_state(state) -> str:
    player = "-" if state.player_x is None else f"({state.player_x:.1f}, {state.player_y:.1f})"
    return (
        f"in_game={state.in_game} stage={state.stage} lives={state.lives} "
        f"power={state.power} score={state.score} graze={state.graze} "
        f"spell={state.spell_id}({state.spell_name}) player={player} "
        f"boss_hp={state.boss_hp} frame={state.frame}"
    )


def cmd_status(args: argparse.Namespace) -> int:
    backend = build_backend(args)
    state = backend.read_state()
    print(f"进程: pid={backend.pid} 窗口={backend.window and backend.window.title!r}")
    print(f"模块基址: 0x{backend.reader.base_address:X}")
    print(f"状态: {fmt_state(state)}")
    print("原始读数:")
    for key, value in state.raw.items():
        spec = backend.spec.field(key)
        mark = "已标定" if spec.calibrated else "未标定"
        addr = backend.reader.address_of(key)
        addr_text = f"@0x{addr:08X}" if addr else ""
        print(f"  {key:<12} {str(value):>14}   [{mark}] {addr_text}")
    return 0


def cmd_start(args: argparse.Namespace) -> int:
    backend = build_backend(args)
    if args.screenshots:
        window = backend.window
        if window:
            image, method = grab_window(window.hwnd)
            path = save_png(image, None, prefix="th11-title")
            print(f"标题画面截图: {path} ({method})")

    if args.keys:
        steps = [{"type": "press", "buttons": [k], "hold": 6, "after": args.after} for k in args.keys]
        log = backend.controller.run_menu_sequence(steps, {})
        print(f"自定义按键: {log}")

    state = backend.reset(
        options={
            "difficulty": args.difficulty,
            "character": args.character,
            "shot": args.shot,
            "menu_index": args.menu_index,
            "menu": args.menu,
            "restart_process": args.restart,
        }
    )
    print(f"reset 后状态: {fmt_state(state)}")
    if state.in_game:
        print("=> 成功进入游戏局内")
    else:
        print("=> 没能确认进入游戏; 请检查菜单序列")
    if args.screenshots:
        window = backend.window
        if window:
            image, method = grab_window(window.hwnd)
            save_png(image, None, prefix="th11-after-start")
            save_viz(backend_state_image(backend), None, prefix="th11-after-start")
            print("已保存 start 之后的截图与状态可视化")
    return 0


def backend_state_image(backend: MemoryBackend):
    from touhou_rl_mcp.capture.visualize import render_state

    state = backend.read_state()
    return render_state(state, backend.spec, title=f"{backend.spec.name_zh} [memory]")


def cmd_shot(args: argparse.Namespace) -> int:
    backend = build_backend(args)
    window = backend.window
    if window is None:
        print("找不到游戏窗口")
        return 1
    image, method = grab_window(window.hwnd)
    if args.crop:
        image = crop_playfield(image, backend.spec.playfield, backend.spec.window_size)
    path = save_png(image, args.out, prefix=f"{backend.spec.key}-shot")
    print(f"{path}  size={image.size} method={method}")
    state = backend.read_state()
    print(f"状态: {fmt_state(state)}")
    return 0


def cmd_viz(args: argparse.Namespace) -> int:
    backend = build_backend(args)
    path = save_viz(backend_state_image(backend), args.out, prefix=f"{backend.spec.key}-state")
    print(f"{path}")
    return 0


def cmd_hold(args: argparse.Namespace) -> int:
    backend = build_backend(args)
    action = parse("+".join(args.keys), default_frames=args.frames)
    before = backend.read_state()
    report = backend.apply(action)
    time.sleep(0.15)
    after = backend.read_state()
    print(f"按住 {args.keys} {args.frames} 帧 -> {report}")
    print(f"前: {fmt_state(before)}")
    print(f"后: {fmt_state(after)}")
    if before.score is not None and after.score is not None and after.score != before.score:
        print(f"!! 分数变化 {before.score} -> {after.score} (说明按键生效了)")
    return 0


def cmd_walk(args: argparse.Namespace) -> int:
    """按住四个方向各走一段, 每段后读一次状态 (用来找坐标偏移的入口)."""
    backend = build_backend(args)
    for key, frames in (("left", args.frames), ("right", args.frames * 2), ("up", args.frames), ("down", args.frames * 2)):
        backend.apply(parse(key, default_frames=frames))
        time.sleep(0.1)
        print(f"{key:<6} x{frames:<4} -> {fmt_state(backend.read_state())}")
    return 0


def _scanner(args: argparse.Namespace):
    backend = build_backend(args)
    return backend.scanner(), backend


def cmd_scan(args: argparse.Namespace) -> int:
    scanner, backend = _scanner(args)
    hits = scanner.scan(
        args.value,
        type_name=args.type,
        align=args.align,
        limit=args.limit,
        include_images=args.images,
        neighbour_offset=args.neighbour_offset,
        neighbour_min=args.neighbour_min,
        neighbour_max=args.neighbour_max,
    )
    print(f"首扫 {args.value} ({args.type}): {len(hits)} 个候选")
    _print_candidates(hits[: args.show])
    return 0


def cmd_find_coord(args: argparse.Namespace) -> int:
    """全自动定位自机坐标: 快照 -> 移动 -> 差分 -> 再差分收敛.

    不需要知道坐标的确切数值 (这才是找坐标的正确姿势)。
    """
    backend = build_backend(args)

    if args.restart:
        state = backend.reset(options={"restart_process": True, "difficulty": args.difficulty})
        print(f"重开一局: {fmt_state(state)}")

    # 注意: scanner 必须在 reset(restart_process=True) 之后取, 否则它绑定的还是旧进程句柄
    scanner = backend.scanner()

    # 等自机被创建 (th11 开局有对话, 对话期间 power=0 且玩家实体可能还没建)
    for _ in range(40):
        state = backend.read_state()
        if state.power is not None and state.power > 0.5:
            print(f"自机已就绪: {fmt_state(state)}")
            break
        backend.apply(parse("noop", default_frames=30))
    else:
        print("警告: 一直没等到 power>0.5, 继续尝试")

    def press(buttons, frames):
        """只按方向, 不按射击 —— 否则自机子弹会被当成"移动期变化"的噪声."""
        spec = "+".join(buttons) if buttons else "noop"
        backend.apply(parse(spec, default_frames=int(frames)))

    print(
        f"开始自动标定 (撞墙法): 撞左墙 {args.settle} 帧 + 稳定 {args.hold} 帧, 再撞右墙"
    )
    candidates = scanner.find_player_coordinates(
        press,
        settle_frames=args.settle,
        hold_frames=args.hold,
        low=args.min,
        high=args.max,
        include_images=args.images,
        log=print,
    )

    mem = backend.mem
    print(f"\n最终候选 {len(candidates)} 个:")
    for cand in candidates[: args.show]:
        parts = [f"0x{cand.address:08X}"]
        for delta in (0, 4, 8, 12):
            try:
                parts.append(f"+{delta}={mem.read_value(cand.address + delta, 'f32'):.3f}")
            except TouhouError:
                parts.append(f"+{delta}=?")
        print("  " + "  ".join(parts))
    if candidates:
        best = candidates[0]
        snippet = (
            f'  "player_x": {{ "address": "{hex(best.address)}", "type": "f32" }},\n'
            f'  "player_y": {{ "address": "{hex(best.address + 4)}", "type": "f32" }}'
        )
        print(
            f"\n写进 touhou_rl_mcp/games/offsets/{backend.spec.key}.json 的 fields:\n{snippet}"
        )
        if getattr(args, "write", False):
            result = apply_calibration(
                backend.spec.key,
                {
                    "player_x": {"address": hex(best.address), "type": "f32"},
                    "player_y": {"address": hex(best.address + 4), "type": "f32"},
                },
            )
            print(f"\n已写入 {result['path']} (备份: {result['backup']})")
    return 0


def cmd_find_frame(args: argparse.Namespace) -> int:
    """全自动定位帧计数器 (每帧 +1 的整数), 用于严格帧同步."""
    backend = build_backend(args)

    def advance(frames: int) -> None:
        backend.apply(parse("noop", default_frames=int(frames)))

    scanner = backend.scanner()
    print(f"自动找帧计数器: 每次推进 {args.frames} 帧 x {args.repeats} 轮 ...")
    candidates = scanner.find_frame_counter(
        advance,
        frames=args.frames,
        tolerance=args.tolerance,
        repeats=args.repeats,
        include_images=args.images,
        log=print,
    )
    print(f"\n最终候选 {len(candidates)} 个:")
    _print_candidates(candidates[: args.show])
    if candidates:
        best = candidates[0]
        entry = (
            {"offset": hex(best.module_offset), "type": "i32"}
            if best.module and best.module_offset is not None
            else {"address": hex(best.address), "type": "i32"}
        )
        print(f'\n写进 offsets 的 fields:  "frame": {json.dumps(entry)}')
        if args.write:
            result = apply_calibration(backend.spec.key, {"frame": entry})
            print(f"\n已写入 {result['path']} (备份: {result['backup']})")
    return 0


def cmd_write_field(args: argparse.Namespace) -> int:
    """把某个字段的标定结果写进 offsets json (手动标定后的收尾)."""
    entry: dict[str, Any] = {"type": args.type}
    value = args.value
    if args.as_offset:
        entry["offset"] = value
    else:
        entry["address"] = value
    result = apply_calibration(args.game, {args.field: entry}, backup=not args.no_backup)
    print(f"已写入 {result['path']}")
    print(f"  {args.field} = {json.dumps(result['written'][args.field], ensure_ascii=False)}")
    if result["backup"]:
        print(f"  备份: {result['backup']}")
    return 0


def cmd_snapshot(args: argparse.Namespace) -> int:
    scanner, _backend = _scanner(args)
    info = scanner.snapshot(include_images=args.images, persist=args.persist)
    print(f"快照完成: {info}")
    return 0


def cmd_delta(args: argparse.Namespace) -> int:
    scanner, backend = _scanner(args)
    if args.move:
        backend.apply(parse(args.move, default_frames=args.frames))
    hits = scanner.scan_delta(
        type_name=args.type,
        min_value=args.min,
        max_value=args.max,
        only_previous=not args.all,
        limit=args.limit,
        unchanged=args.unchanged,
    )
    print(f"差分后候选: {len(hits)}")
    _print_candidates(hits[: args.show])
    return 0


def cmd_clear_snapshot(args: argparse.Namespace) -> int:
    scanner, _backend = _scanner(args)
    scanner.clear_snapshot()
    print("快照已清除")
    return 0


def cmd_scan_values(args: argparse.Namespace) -> int:
    """扫描连续存放的多个值 (定位结构体的杀手锏)."""
    scanner, backend = _scanner(args)
    values = [float(v) if "." in v or "e" in v.lower() else int(v) for v in args.values]
    hits = scanner.scan_values(
        values, type_name=args.type, align=args.align, limit=args.limit, include_images=args.images
    )
    print(f"扫连续值 {values} ({args.type}): {len(hits)} 个候选")
    _print_candidates(hits[: args.show])
    if hits:
        print("\n提示: 记下地址, 之后移动自机, 用 `probe.py changed` 或 `probe.py read` 验证。")
    return 0


def cmd_scan_bytes(args: argparse.Namespace) -> int:
    scanner, backend = _scanner(args)
    hits = scanner.scan_pattern(
        args.pattern, align=args.align, limit=args.limit, include_images=args.images
    )
    print(f"字节模式 {args.pattern}: {len(hits)} 个候选")
    _print_candidates(hits[: args.show])
    return 0


def cmd_filter(args: argparse.Namespace) -> int:
    scanner, _backend = _scanner(args)
    cands = scanner.load()
    if not cands:
        print("没有候选, 先 scan")
        return 1
    kept = scanner.filter(cands, args.value, type_name=args.type)
    print(f"再扫 {args.value} ({args.type}): {len(cands)} -> {len(kept)}")
    _print_candidates(kept[: args.show])
    return 0


def cmd_changed(args: argparse.Namespace) -> int:
    scanner, _backend = _scanner(args)
    cands = scanner.load()
    if not cands:
        print("没有候选, 先 scan")
        return 1
    kept = scanner.filter_changed(cands, type_name=args.type, expect=args.expect)
    print(f"筛选 ({args.expect}) ({args.type}): {len(cands)} -> {len(kept)}")
    _print_candidates(kept[: args.show])
    return 0


def cmd_cands(args: argparse.Namespace) -> int:
    scanner, _backend = _scanner(args)
    cands = scanner.load()
    print(f"候选 {len(cands)} 个")
    _print_candidates(cands[: args.limit])
    return 0


def cmd_clear(args: argparse.Namespace) -> int:
    scanner, _backend = _scanner(args)
    scanner.clear()
    print("已清空候选")
    return 0


def _print_candidates(cands) -> None:
    for c in cands:
        extra = f"  {c.module}+0x{c.module_offset:X}" if c.module and c.module_offset is not None else ""
        print(f"  0x{c.address:08X} = {c.value}{extra}")


def _resolve(mem, address: str) -> int:
    if isinstance(address, str) and "+" in address:
        module, _, offset = address.partition("+")
        return mem.module_base(module.strip()) + int(offset.strip(), 0)
    return int(str(address), 0)


def cmd_dump(args: argparse.Namespace) -> int:
    """把一段内存按类型连续打印出来 (确认结构体布局用)."""
    scanner, backend = _scanner(args)
    mem = backend.mem
    base = _resolve(mem, args.address)
    print(f"从 0x{base:08X} 起 {args.count} 个 {args.type} (偏移 +4 递增):")
    for i in range(args.count):
        addr = base + i * 4
        try:
            value = mem.read_value(addr, args.type)
        except TouhouError as exc:
            print(f"  +0x{i*4:<3X} 0x{addr:08X}  读取失败")
            continue
        note = ""
        if args.type == "f32" and -4000.0 < value < 4000.0 and value != 0:
            note = "  <- 像浮点数"
        print(f"  +0x{i*4:<3X} 0x{addr:08X} = {value}{note}")
    return 0


def cmd_read(args: argparse.Namespace) -> int:
    scanner, backend = _scanner(args)
    mem = backend.mem
    address = _resolve(mem, args.address)
    print(f"地址 0x{address:08X}", end="")
    try:
        name, base, size = mem.main_module
        if base <= address < base + size:
            print(f" (= {name}+0x{address - base:X})", end="")
    except Exception:
        pass
    print()
    if args.offsets:
        offsets = [int(o, 0) for o in args.offsets.split(",")]
        # 与 offsets json 一致: [[address]+o0]+o1 —— 先解引用 address 本身
        chain = int(mem.read_value(address, "u32"))
        print(f"  -> [address] = 0x{chain:08X}")
        for off in offsets[:-1]:
            chain = int(mem.read_value(chain + off, "u32"))
            print(f"  -> deref+0x{off:X} = 0x{chain:08X}")
        address = chain + offsets[-1]
        print(f"  最终地址 0x{address:08X}")
    print(f"值 = {mem.read_value(address, args.type)} ({args.type})")
    return 0


def cmd_watch(args: argparse.Namespace) -> int:
    scanner, backend = _scanner(args)
    mem = backend.mem
    address = _resolve(mem, args.address)
    print(f"观察 0x{address:08X} ({args.type}) {args.seconds}s ...")
    deadline = time.time() + args.seconds
    last = object()
    while time.time() < deadline:
        try:
            value = mem.read_value(address, args.type)
        except TouhouError as exc:
            print(f"  读取失败: {exc}")
            break
        if value != last:
            stamp = time.strftime("%H:%M:%S")
            print(f"  [{stamp}] {value}")
            last = value
        time.sleep(args.interval)
    return 0


def cmd_write_json(args: argparse.Namespace) -> int:
    """把候选列表转成可以直接粘进 offsets json 的片段."""
    scanner, backend = _scanner(args)
    cands = scanner.load()
    if not cands:
        print("没有候选")
        return 1
    field = args.field or "unknown"
    print(f"# 候选 {len(cands)} 个; 挑一个写进 games/offsets/{backend.spec.key}.json:")
    for c in cands[: args.limit]:
        if c.module and c.module_offset is not None:
            snippet = f'"{field}": {{ "offset": "0x{c.module_offset:X}", "type": "{args.type}" }}'
        else:
            snippet = f'"{field}": {{ "address": "0x{c.address:X}", "type": "{args.type}" }}'
        print(f"  0x{c.address:08X} = {c.value}\n    {snippet}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="probe", description="东方真机探针 / 偏移标定工具")
    parser.add_argument("--config", default=None)
    parser.add_argument("--game", default="th11")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("status"); p.set_defaults(func=cmd_status)

    p = sub.add_parser("start")
    p.add_argument("--difficulty", type=int, default=1)
    p.add_argument("--character", type=int, default=0)
    p.add_argument("--shot", type=int, default=0)
    p.add_argument("--menu-index", dest="menu_index", type=int, default=0)
    p.add_argument("--menu", default="auto", choices=["auto", "none", "keys"])
    p.add_argument("--keys", nargs="*", default=None, help="menu=keys 时手动指定的按键序列")
    p.add_argument("--after", type=int, default=30)
    p.add_argument("--screenshots", action="store_true")
    p.add_argument("--restart", action="store_true", help="先杀进程重启, 保证全新一局")
    p.set_defaults(func=cmd_start)

    p = sub.add_parser("shot")
    p.add_argument("--out", default=None)
    p.add_argument("--crop", action="store_true")
    p.set_defaults(func=cmd_shot)

    p = sub.add_parser("viz"); p.add_argument("--out", default=None); p.set_defaults(func=cmd_viz)

    p = sub.add_parser("hold")
    p.add_argument("keys", nargs="+")
    p.add_argument("--frames", type=int, default=60)
    p.set_defaults(func=cmd_hold)

    p = sub.add_parser("walk"); p.add_argument("--frames", type=int, default=45); p.set_defaults(func=cmd_walk)

    for name, fn in (("scan", cmd_scan), ("filter", cmd_filter)):
        p = sub.add_parser(name)
        p.add_argument("value", type=float)
        p.add_argument("--type", default="i32")
        p.add_argument("--align", type=int, default=4)
        p.add_argument("--limit", type=int, default=2000)
        p.add_argument("--show", type=int, default=30)
        if name == "scan":
            p.add_argument("--images", action="store_true", help="连模块映像区一起扫")
            p.add_argument("--neighbour-offset", dest="neighbour_offset", type=int, default=None,
                           help="邻居约束: 偏移 (例如 4)")
            p.add_argument("--neighbour-min", dest="neighbour_min", type=float, default=None)
            p.add_argument("--neighbour-max", dest="neighbour_max", type=float, default=None)
        p.set_defaults(func=fn)

    p = sub.add_parser("scan-values", help="扫描连续存放的多个值 (例如自机 x,y)")
    p.add_argument("values", nargs="+")
    p.add_argument("--type", default="f32")
    p.add_argument("--align", type=int, default=4)
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--show", type=int, default=30)
    p.add_argument("--images", action="store_true")
    p.set_defaults(func=cmd_scan_values)

    p = sub.add_parser("scan-bytes", help="按 hex 字节模式扫描 (支持 ?? 通配)")
    p.add_argument("pattern")
    p.add_argument("--align", type=int, default=4)
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--show", type=int, default=30)
    p.add_argument("--images", action="store_true")
    p.set_defaults(func=cmd_scan_bytes)

    p = sub.add_parser("find-coord", help="全自动定位自机坐标 (撞墙法)")
    p.add_argument("--restart", action="store_true", help="先重开一局")
    p.add_argument("--difficulty", type=int, default=0)
    p.add_argument("--settle", type=int, default=150, help="撞墙需要按住的帧数")
    p.add_argument("--hold", type=int, default=50, help="撞墙后再按住多少帧测稳定性")
    p.add_argument("--type", default="f32")
    p.add_argument("--min", type=float, default=1.0)
    p.add_argument("--max", type=float, default=448.0)
    p.add_argument("--show", type=int, default=20)
    p.add_argument("--images", action="store_true")
    p.add_argument("--write", action="store_true", help="直接把 player_x/player_y 写进 offsets json")
    p.set_defaults(func=cmd_find_coord)

    p = sub.add_parser("find-frame", help="全自动定位帧计数器 (每帧 +1 的整数)")
    p.add_argument("--frames", type=int, default=60, help="每次推进的帧数")
    p.add_argument("--repeats", type=int, default=2, help="重复取交集的次数")
    p.add_argument("--tolerance", type=float, default=0.25, help="帧数误差容限")
    p.add_argument("--show", type=int, default=30)
    p.add_argument("--images", action="store_true", default=True)
    p.add_argument("--write", action="store_true", help="直接把 frame 写进 offsets json")
    p.set_defaults(func=cmd_find_frame)

    p = sub.add_parser("write-field", help="把某字段的标定结果写进 offsets json")
    p.add_argument("field", help="字段名, 例如 graze / boss_hp / player_x")
    p.add_argument("value", help='地址或偏移, 例如 0x1234ABCD 或 0xA571C')
    p.add_argument("--type", default="i32")
    p.add_argument("--as-offset", action="store_true", help="value 是相对模块基址的偏移 (默认当作绝对地址)")
    p.add_argument("--no-backup", action="store_true", help="不生成 .json.bak 备份")
    p.set_defaults(func=cmd_write_field)

    p = sub.add_parser("snapshot", help="给内存拍快照 (差分扫描用)")
    p.add_argument("--images", action="store_true", default=True)
    p.add_argument("--persist", action="store_true", default=True)
    p.set_defaults(func=cmd_snapshot)

    p = sub.add_parser("delta", help="与快照差分")
    p.add_argument("--type", default="f32")
    p.add_argument("--min", type=float, default=None)
    p.add_argument("--max", type=float, default=None)
    p.add_argument("--move", default=None, help="差分前先按住的键")
    p.add_argument("--frames", type=int, default=90)
    p.add_argument("--all", action="store_true", help="不限制在已有候选里")
    p.add_argument("--unchanged", action="store_true")
    p.add_argument("--limit", type=int, default=5000)
    p.add_argument("--show", type=int, default=40)
    p.set_defaults(func=cmd_delta)

    p = sub.add_parser("clear-snapshot"); p.set_defaults(func=cmd_clear_snapshot)

    p = sub.add_parser("changed")
    p.add_argument("--type", default="i32")
    p.add_argument("--expect", default="changed", choices=["changed", "unchanged", "any"])
    p.add_argument("--show", type=int, default=30)
    p.set_defaults(func=cmd_changed)

    p = sub.add_parser("cands"); p.add_argument("--limit", type=int, default=50); p.set_defaults(func=cmd_cands)
    p = sub.add_parser("clear"); p.set_defaults(func=cmd_clear)

    p = sub.add_parser("read")
    p.add_argument("address")
    p.add_argument("--type", default="i32")
    p.add_argument("--offsets", default=None)
    p.set_defaults(func=cmd_read)

    p = sub.add_parser("dump", help="连续打印一段内存 (看结构体布局)")
    p.add_argument("address")
    p.add_argument("--count", type=int, default=16)
    p.add_argument("--type", default="f32")
    p.set_defaults(func=cmd_dump)

    p = sub.add_parser("watch")
    p.add_argument("address")
    p.add_argument("--type", default="i32")
    p.add_argument("--seconds", type=float, default=5.0)
    p.add_argument("--interval", type=float, default=0.05)
    p.set_defaults(func=cmd_watch)

    p = sub.add_parser("write-json")
    p.add_argument("--field", default=None)
    p.add_argument("--type", default="i32")
    p.add_argument("--limit", type=int, default=10)
    p.set_defaults(func=cmd_write_json)

    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except TouhouError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
