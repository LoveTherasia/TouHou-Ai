"""自动标定自机坐标 player_x / player_y 的**通用脚本** (东方正作 th06~th14).

核心判据 —— **钳位一致性** (对"被弹瞬移"天然免疫, 因为撞墙总会到达同一常数):
  1. 反复把自机按到左墙(按住 LEFT 足够久)读一次、再按到右墙读一次;
     x 真身每次左墙都是同一个常数, 每次右墙也是同一个常数; 弹幕/相机不是;
  2. 上/下墙同理得到 y;
  3. 自机 x 与 y 通常是**相邻的两个 f32** (+4), 用这一点把真身从"子机/子弹"里挑出来;
  4. 校准期间持续射击清场, 并用 lives 偏移做命中观察 (被弹也不影响钳位读数)。

用法:
  python tools/calibrate_player.py --game th11                 # 重启一局(Easy)并标定
  python tools/calibrate_player.py --game th11 --no-restart    # 用当前正在玩的局
  python tools/calibrate_player.py --game th11 --write         # 结果写回 offsets json
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from touhou_rl_mcp.backends.memory import MemoryBackend  # noqa: E402
from touhou_rl_mcp.config import Config  # noqa: E402
from touhou_rl_mcp.games.calibration import apply_calibration  # noqa: E402
from touhou_rl_mcp.games.registry import GameRegistry  # noqa: E402
from touhou_rl_mcp.inputs.actions import parse  # noqa: E402
from touhou_rl_mcp.memory.cheats import LivesLock  # noqa: E402


def log(*a):
    print(*a, flush=True)


class PlayerFinder:
    def __init__(self, backend: MemoryBackend, *, game: str) -> None:
        self.be = backend
        self.game = game
        self.mem = backend.mem
        self.sc = backend.scanner()
        self.regions = [
            (b, s)
            for b, s, p, t in self.mem.memory_regions()
            if s <= self.sc.max_region_bytes and not (t & 0x1000000)
        ]
        self.lives_addr = backend.reader.address_of("lives")

    # ---------------- 基础 ----------------
    def cap(self) -> dict:
        out = {}
        for b, s in self.regions:
            try:
                out[b] = np.frombuffer(self.mem.read_bytes(b, s), dtype="<f4").copy()
            except Exception:
                pass
        return out

    def press(self, keys, frames: int) -> None:
        spec = keys if isinstance(keys, str) else "+".join(keys)
        if "shoot" not in spec:
            spec = f"{spec}+shoot" if spec else "shoot"
        self.be.apply(parse(spec, default_frames=int(frames)))

    def settle(self, sec: float = 0.25) -> None:
        if self.be.keyboard:
            self.be.keyboard.focus_window()
        self.be.apply(parse("shoot", default_frames=1))
        time.sleep(sec)

    def lives(self) -> int:
        try:
            return int(self.mem.read_value(self.lives_addr, "i32"))
        except Exception:
            return -1

    def readf(self, addr: int) -> float:
        try:
            v = self.mem.read_value(addr, "f32")
            return float(v) if v is not None else float("nan")
        except Exception:
            return float("nan")

    # ---------------- 钳位一致性 ----------------
    def clamp_detect(self, neg, pos, *, frames: int, reps: int, lo_max: float, hi_min: float):
        """返回 [(addr, lo_val, hi_val)]: 多次按到 neg/pos 墙都各自停在同一常数的地址."""
        nmin = nmax = pmin = pmax = None
        for k in range(reps):
            self.settle(0.3)
            self.press(neg, frames)
            self.settle(0.2)
            N = self.cap()
            lv = self.lives()
            self.press(pos, frames)
            self.settle(0.2)
            P = self.cap()
            if nmin is None:
                keep = set(N) & set(P)
                nmin = {b: N[b].copy() for b in keep}
                nmax = {b: N[b].copy() for b in keep}
                pmin = {b: P[b].copy() for b in keep}
                pmax = {b: P[b].copy() for b in keep}
            else:
                for b in list(nmin):
                    if b not in N or b not in P or N[b].shape != nmin[b].shape:
                        nmin.pop(b, None); nmax.pop(b, None); pmin.pop(b, None); pmax.pop(b, None)
                        continue
                    with np.errstate(all="ignore"):
                        nmin[b] = np.minimum(nmin[b], N[b])
                        nmax[b] = np.maximum(nmax[b], N[b])
                        pmin[b] = np.minimum(pmin[b], P[b])
                        pmax[b] = np.maximum(pmax[b], P[b])
            log(f"    [{neg}->{pos}] rep {k + 1}/{reps} lives={lv}")

        hits = []
        for b in nmin:
            with np.errstate(all="ignore"):
                nspan = nmax[b] - nmin[b]
                pspan = pmax[b] - pmin[b]
                span = pmin[b] - nmax[b]
            m = (
                (nspan < 0.5) & (pspan < 0.5)
                & (nmin[b] >= 0.0) & (nmin[b] <= lo_max)
                & (pmax[b] >= hi_min) & (pmax[b] <= 520.0)
                & (span > 300.0)
            )
            for i in np.flatnonzero(m).tolist():
                hits.append((b + 4 * i, float(nmin[b][i]), float(pmax[b][i])))
        return sorted(hits)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="th11")
    ap.add_argument("--difficulty", type=int, default=0)
    ap.add_argument("--character", type=int, default=0)
    ap.add_argument("--shot", type=int, default=0)
    ap.add_argument("--no-restart", action="store_true")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--frames", type=int, default=110)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    cfg = Config.load()
    cfg.patch({"backend": "memory", "game": args.game})
    be = MemoryBackend(GameRegistry().get(args.game), cfg)
    be.start()
    if not args.no_restart:
        log(f"[calib] 重启一局 (difficulty={args.difficulty}) ...")
        be.reset(options={"restart_process": True, "difficulty": args.difficulty,
                          "character": args.character, "shot": args.shot})
        time.sleep(3.0)
    for _ in range(60):
        s = be.read_state()
        if s.in_game and (s.power or 0) > 0.3:
            break
        be.apply(parse("noop", default_frames=20))
    log(f"[calib] pid={be.pid} in_game={s.in_game} stage={s.stage} lives={s.lives} power={s.power}")

    finder = PlayerFinder(be, game=args.game)
    lock = None
    if finder.lives_addr:
        lock = LivesLock(be.mem, finder.lives_addr, value=9, floor=1).start()

    log("[calib] === x: 左墙/右墙 钳位一致性 ===")
    xs = finder.clamp_detect("left", "right", frames=args.frames, reps=args.reps,
                             lo_max=80.0, hi_min=330.0)
    log(f"[calib] x 候选 {len(xs)} 个")
    for a, lv, hv in xs[:20]:
        log(f"    0x{a:08X} 左={lv:8.2f} 右={hv:8.2f} +4={finder.readf(a + 4):.2f}")

    if lock:
        lock.stop()

    # 自机 = x 与 y 相邻 (+4) 的 f32 对: 按"跨度大"优先, 并做 x/y 双重复核
    log("[calib] === 复核候选 (跨度优先, x/y 各自只响应自己的轴) ===")
    result = {}
    for a, lv, hv in sorted(xs, key=lambda z: z[2] - z[1], reverse=True)[:15]:
        finder.press("left", 90)
        xl = finder.readf(a)
        finder.press("right", 180)
        xr = finder.readf(a)
        finder.press("up", 90)
        yu = finder.readf(a + 4)
        finder.press("down", 140)
        yd = finder.readf(a + 4)
        x_ok = None not in (xl, xr) and xl <= 90.0 and xr >= 370.0 and (xr - xl) > 340.0
        x_is_axis = (xr - xl) > 340.0
        y_ok = None not in (yu, yd) and yu <= 90.0 and yd >= 370.0 and (yd - yu) > 340.0
        log(f"    0x{a:08X} x[{lv:.0f},{hv:.0f}] 复核左={xl} 右={xr} | "
            f"+4 上={yu} 下={yd} -> x轴={x_is_axis} y轴={y_ok}")
        if x_ok and y_ok:
            result = {"player_x": a, "player_y": a + 4}
            break

    if result:
        log("[calib] === 活体验证 ===")
        for key, addr in result.items():
            finder.press("left", 40); v1 = finder.readf(addr)
            finder.press("right", 80); v2 = finder.readf(addr)
            finder.press("up", 60); v3 = finder.readf(addr)
            finder.press("down", 130); v4 = finder.readf(addr)
            log(f"  {key} 0x{addr:08X}: 左={v1:.1f} 右={v2:.1f} 上={v3:.1f} 下={v4:.1f}")
    else:
        log("[calib] 未找到 (player_x, player_y) 相邻对")

    if args.write and result:
        # 为 player_x 找模块级指针链, 使结果跨进程稳定; y = x 的链末偏移 +4
        fields = {k: {"address": f"0x{v:X}", "type": "f32"} for k, v in result.items()}
        try:
            sc = be.scanner()
            arrays = dict(sc._capture_arrays(include_images=True))
            chains = sc.find_pointer_chain(result["player_x"], arrays=arrays, limit=40, log=None)
            best = None
            for c in chains:
                try:
                    addr = int(be.mem.read_value(be.reader.base_address + c["pointer"], "u32"))
                    for o in c["offsets"][:-1]:
                        addr = int(be.mem.read_value(addr + o, "u32"))
                    if addr + c["offsets"][-1] == result["player_x"]:
                        if best is None or len(c["offsets"]) < len(best["offsets"]):
                            best = c
                except Exception:
                    continue
            if best:
                offs = best["offsets"]
                fields = {
                    "player_x": {"pointer": f"0x{best['pointer']:X}",
                                 "offsets": [f"0x{o:X}" for o in offs], "type": "f32"},
                    "player_y": {"pointer": f"0x{best['pointer']:X}",
                                 "offsets": [f"0x{o:X}" for o in offs[:-1]] + [f"0x{offs[-1] + 4:X}"],
                                 "type": "f32"},
                }
                log(f"[calib] 指针链: [[mod+0x{best['pointer']:X}]"
                    + "".join(f"+0x{o:X}" for o in offs) + "]")
        except Exception as exc:
            log(f"[calib] 指针链搜索失败, 退回绝对地址: {exc}")
        out = apply_calibration(args.game, fields)
        log(f"[calib] 已写入 {out['path']} (备份 {out['backup']})")

    be.close()
    log("[calib] done")
    return 0 if result.get("player_x") else 1


if __name__ == "__main__":
    raise SystemExit(main())
