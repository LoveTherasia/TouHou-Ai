import struct
import zlib
from pathlib import Path

p = Path(r"F:\[th11] 东方地灵殿 (汉化版+日文版)\replay\th11_01.rpy")
data = p.read_bytes()
print("size", len(data))

# 1) 扫描 zlib 头 (0x78 0x01/0x9c/0xda) 与 gzip (1f 8b)
print("\n=== 找 zlib/gzip 头 ===")
for i in range(len(data) - 2):
    if data[i] == 0x78 and data[i + 1] in (0x01, 0x5E, 0x9C, 0xDA):
        print(f"  zlib header @ 0x{i:x}")
    if data[i] == 0x1F and data[i + 1] == 0x8B:
        print(f"  gzip header @ 0x{i:x}")

# 2) 在 0..0x400 内暴力尝试 raw-deflate / zlib
print("\n=== 暴力尝试解压 (0..0x400) ===")
best = None
for start in range(0, 0x400):
    for wbits in (-15, 15):
        try:
            d = zlib.decompressobj(wbits)
            out = d.decompress(data[start:])
            if len(out) > 5000:
                print(f"  start=0x{start:x} wbits={wbits} -> {len(out)} bytes")
                if best is None or len(out) > best[2]:
                    best = (start, wbits, len(out), out)
        except Exception:
            pass
if best:
    start, wbits, n, out = best
    Path(".runtime/_rpy_body.bin").write_bytes(out)
    print(f"\n>>> 最佳: start=0x{start:x} wbits={wbits} 解压 {n} 字节 (已存 .runtime/_rpy_body.bin)")
    print("    前 96 字节:", out[:96].hex())
    print("    是否像每帧 2 字节输入:", n % 2 == 0, "帧数≈", n // 2)
else:
    print("  没有可解压的流")

# 3) 直接看尾部 256 字节的 ASCII
print("\n=== 尾部 256 字节 ASCII ===")
tail = data[-256:]
for off in range(0, len(tail), 16):
    chunk = tail[off:off + 16]
    asci = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
    print(f"{len(data)-256+off:06x}  {asci}")
