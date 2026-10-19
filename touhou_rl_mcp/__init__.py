"""Touhou RL MCP — 东方 Project 正作 RL 环境 / MCP 服务器.

提供一个可供 LLM Agent 或 RL 训练循环使用的东方 Project 环境接口:
    - 开始新一局 (reset / start_new_game)
    - 执行动作, 返回 (state, reward, done)
    - 查询当前状态
    - 屏幕截图 / 状态可视化

两种后端:
    - ``mock``   : 纯 Python 内置模拟器, 无需游戏本体, 开箱即用
    - ``memory`` : 真实游戏 (读取进程内存 + 注入按键), 需要 Windows
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
