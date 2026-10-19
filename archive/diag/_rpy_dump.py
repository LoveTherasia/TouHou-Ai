import struct
import sys
from pathlib import Path

p = Path(r"F:\[th11] 东方地灵殿 (汉化版+日文版)\replay\th11_01.rpy")
data = p.read_bytes()
print("file:", p)
print("size:", len(data), "bytes")
print("magic:", data[:4], data[:4].hex())
print()

# 打印前 0x120 字节: 偏移 / hex / ascii
print("=== header dump (0x000 - 0x120) ===")
for off in range(0, 0x120, 16):
    chunk = data[off:off + 16]
    hexs = " ".join(f"{b:02x}" for b in chunk)
    asci = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
    print(f"{off:04x}  {hexs:<48}  {asci}")

print()
print("=== 可能的结构 (按 thtk th11 格式猜测) ===")
version = struct.unpack_from("<H", data, 4)[0]
size = struct.unpack_from("<H", data, 6)[0]
print("version = 0x%04x" % version)
print("header_size = %d" % size)
# 尾部: 输入流
print()
print("=== 尾部 64 字节 ===")
for off in range(len(data) - 64, len(data), 16):
    chunk = data[off:off + 16]
    hexs = " ".join(f"{b:02x}" for b in chunk)
    print(f"{off:04x}  {hexs}")
