"""MCP 服务器端到端集成测试 (真正的 stdio 会话).

用 :mod:`mcp.client.stdio` 起一个子进程服务器, 然后像真实 Agent 一样调用工具,
验证: 工具列表 / 新一局 / step 闭环 / 状态查询 / 图片返回 / 在线调参 / 统计。
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EXPECTED_TOOLS = {
    "th_list_games",
    "th_env_status",
    "th_get_config",
    "th_configure",
    "th_attach",
    "th_close_env",
    "th_start_new_game",
    "th_step",
    "th_get_state",
    "th_send_keys",
    "th_episode_stats",
    "th_screenshot",
    "th_visualize_state",
    "th_memory_scan",
    "th_memory_filter",
    "th_memory_filter_changed",
    "th_memory_read",
    "th_memory_candidates",
    "th_memory_clear_candidates",
    "th_memory_find_player",
    "th_memory_find_frame",
    "th_offsets_write",
    "th_set_game_speed",
    "th_train_start",
    "th_train_stop",
    "th_train_pause",
    "th_train_status",
    "th_train_dashboard",
    "th_train_configure",
    "th_train_policy",
}


def _params() -> StdioServerParameters:
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "touhou_rl_mcp", "--backend", "mock", "--game", "th11"],
        cwd=str(PROJECT_ROOT),
        env=env,
    )


def _text(result) -> dict:
    for block in result.content:
        if getattr(block, "type", None) == "text":
            return json.loads(block.text)
    raise AssertionError(f"工具没有返回文本块: {result.content}")


def _kinds(result) -> list[str]:
    return [getattr(b, "type", None) for b in result.content]


class TestMcpServer(unittest.TestCase):
    def setUp(self) -> None:
        import anyio

        self.anyio = anyio

    def _run(self, coro_fn) -> None:
        self.anyio.run(coro_fn)

    def test_full_loop(self) -> None:
        async def scenario() -> None:
            async with stdio_client(_params()) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()

                    tools = {t.name for t in (await session.list_tools()).tools}
                    missing = EXPECTED_TOOLS - tools
                    self.assertFalse(missing, f"缺少工具: {missing}")

                    listing = _text(await session.call_tool("th_list_games", {"detect": False}))
                    self.assertTrue(any(g["key"] == "th11" for g in listing["games"]))

                    reset = _text(
                        await session.call_tool("th_start_new_game", {"seed": 42})
                    )
                    self.assertFalse(reset["done"])
                    self.assertEqual(reset["info"]["event"], "reset")
                    self.assertEqual(reset["state"]["stage"], 1)

                    total = 0.0
                    done = False
                    for _ in range(20):
                        step = _text(
                            await session.call_tool("th_step", {"action": "dodge", "frames": 10})
                        )
                        total += step["reward"]
                        self.assertIn("reward_parts", step["info"])
                        if step["done"] or step["truncated"]:
                            done = True
                            break
                    self.assertFalse(done)
                    self.assertIsInstance(total, float)

                    state = _text(await session.call_tool("th_get_state", {"include_bullets": True}))
                    self.assertIn("bullets", state)
                    self.assertIsNotNone(state["lives"])

                    viz = await session.call_tool("th_visualize_state", {"scale": 1.0})
                    self.assertIn("image", _kinds(viz))
                    image_block = next(b for b in viz.content if b.type == "image")
                    self.assertEqual(image_block.mime_type, "image/png")
                    self.assertGreater(len(image_block.data), 1000)

                    shot = await session.call_tool("th_screenshot", {})
                    self.assertIn("image", _kinds(shot))

                    cfg = _text(
                        await session.call_tool(
                            "th_configure",
                            {"patch": {"reward": {"shaping_scale": 0.0}}, "save": False},
                        )
                    )
                    self.assertEqual(cfg["config"]["reward"]["shaping_scale"], 0.0)
                    step = _text(await session.call_tool("th_step", {"action": "shoot", "frames": 10}))
                    self.assertAlmostEqual(step["reward"], 0.0, places=6)

                    stats = _text(await session.call_tool("th_episode_stats", {"reset": True}))
                    self.assertIn("summary", stats)

        self._run(scenario)

    def test_episode_end_flow(self) -> None:
        async def scenario() -> None:
            async with stdio_client(_params()) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    await session.call_tool("th_start_new_game", {"seed": 3})
                    finished = None
                    for _ in range(600):
                        step = _text(
                            await session.call_tool("th_step", {"action": "noop", "frames": 10})
                        )
                        if step["done"] or step["truncated"]:
                            finished = step
                            break
                    self.assertIsNotNone(finished, "idle player should die within 6000 frames")
                    self.assertTrue(finished["info"]["done_reason"])
                    stats = _text(await session.call_tool("th_episode_stats", {"reset": False}))
                    self.assertEqual(stats["summary"]["episodes"], 1)

        self._run(scenario)

    def test_memory_tools_require_memory_backend(self) -> None:
        async def scenario() -> None:
            async with stdio_client(_params()) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool("th_memory_scan", {"value": 123})
                    self.assertTrue(result.is_error)
                    text = " ".join(
                        b.text for b in result.content if getattr(b, "type", None) == "text"
                    )
                    self.assertIn("memory", text.lower())

        self._run(scenario)


if __name__ == "__main__":
    unittest.main()
