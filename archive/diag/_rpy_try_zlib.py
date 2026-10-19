import struct
import zlib
from pathlib import Path

p = Path(r"F:\[th11] 东方地灵殿 (汉化版+日文版)\replay\th11_01.rpy")
data = p.read_bytes()

u32 = lambda o: struct.unpack_from("<I", data, o)[0]
print("magic       :", data[:4])
print("ver(0x04)   :", u32(0x04))
print("f08         :", u32(0x08))
print("f0C         :", u32(0x0C))
print("f10         :", u32(0x10))
print("f14         :", u32(0x14))
print("f18         :", u32(0x18))
print("f1C         :", u32(0x1C))
print("f20         :", u32(0x20))
print("file size   :", len(data))

for start in range(0x20, 0x40, 4):
    for wbits, name in ((-15, "raw-deflate"), (15, "zlib"), (31, "gzip")):
        try:
            d = zlib.decompressobj(wbits)
            out = d.decompress(data[start:])
            if len(out) > 1000:
                print(f"\n>>> 解压成功 start=0x{start:x} ({name}) 解压后={len(out)} 字节")
                print("    前 64 字节 hex:", out[:64].hex())
                print("    期望 f20 =", u32(0x20))
                Path(".runtime/_rpy_body.bin").write_bytes(out)
                raise SystemExit
        except SystemExit:
            raise
        except Exception as e:
            pass
print("\n未找到可直接解压的流 (可能不是压缩, 或需要其它方式)")
