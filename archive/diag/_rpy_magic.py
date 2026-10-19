from pathlib import Path

files = [
    r"F:\[th07] 东方妖妖梦 (汉化版)\replay\th7_01.rpy",
    r"F:\[th08] 东方永夜抄 (汉化版)\replay\th8_01.rpy",
    r"F:\[th10] 东方风神录 (日文版+汉化版)\replay\th10_01.rpy",
    r"F:\[th11] 东方地灵殿 (汉化版+日文版)\replay\th11_01.rpy",
    r"F:\[th11] 东方地灵殿 (汉化版+日文版)\replay\th11_02.rpy",
    r"F:\[th13] 东方神灵庙 (汉化版+日文版)\replay\th13_01.rpy",
]
for f in files:
    p = Path(f)
    if not p.exists():
        print("MISSING", f)
        continue
    d = p.read_bytes()
    head = d[:16]
    asci = "".join(chr(b) if 32 <= b < 127 else "." for b in head)
    print(f"{p.name:14s} size={len(d):7d}  magic={d[:4]!r} ({d[:4].hex()})  head={asci}")
