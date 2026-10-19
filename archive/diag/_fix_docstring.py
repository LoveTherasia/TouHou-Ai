# -*- coding: utf-8 -*-
"""安全修复 trainer.py 头部 docstring 的乱码 (只替换第一个三引号块, 其余字节原样保留)。"""
import sys
from pathlib import Path

p = Path(__file__).resolve().parent.parent / "touhou_rl_mcp" / "training" / "trainer.py"
raw = p.read_bytes()
s = raw.decode("utf-8")

new_doc = (
    '"""训练器: 在进程内以最高速度跑 RL 循环。\n'
    "\n"
    "架构定位 (这是整个 agentic RL 的关键):\n"
    "\n"
    "    LLM Agent ──MCP──▶ Trainer            外层: 调奖励权重、看曲线、决定重启\n"
    "                         │ (进程内, 不经过 MCP)\n"
    "                         ▼\n"
    "                     TouhouEnv ──▶ 真实游戏 / 模拟器   内层: 60fps 高频闭环\n"
    "\n"
    "**训练循环绝不能走 MCP 工具调用** —— 每次 JSON-RPC 往返是毫秒级开销, 而 RL 需要\n"
    "每秒上千步。所以训练器在 Python 进程内直接驱动 TouhouEnv, MCP 只负责\n"
    '"启动/停止/观测/调参"。\n'
    "\n"
    "奖励来自 env.step() 的返回值, 而奖励权重随时可以用 MCP 的 th_configure 改 ——\n"
    '**下一个 episode 就生效**, 这就是 "Agent 调奖励" 的闭环。"""'
)

i = s.index('"""')
j = s.index('"""', i + 3) + 3
assert j > i + 3
s2 = s[:i] + new_doc + s[j:]
p.write_bytes(s2.encode("utf-8"))
print("patched", p)
