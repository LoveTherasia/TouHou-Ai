"""时空规划器的基本契约测试 (不追求"会玩", 只保证不崩、方向合理)."""

from __future__ import annotations

import math
import unittest

from touhou_rl_mcp.types import GameState
from touhou_rl_mcp.training.planner import DIRECTIONS, NAMES, MacroPlanner


def _state(bullets, velocities, x=192.0, y=400.0) -> GameState:
    return GameState(player_x=x, player_y=y, bullets=list(bullets), in_game=True,
                     raw={"bullet_velocity": list(velocities)})


class PlannerContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.planner = MacroPlanner(horizon=60, beam=8, macro_frames=6)

    def test_no_bullets_is_shoot(self) -> None:
        result = self.planner.plan(_state([], []))
        self.assertEqual(result.action, "shoot")

    def test_returns_known_action(self) -> None:
        state = _state([(192.0, 360.0), (160.0, 380.0)], [(0.0, 2.0), (1.0, 1.5)])
        result = self.planner.plan(state)
        parts = result.action.split("+")
        self.assertTrue(parts[-1] == "shoot")
        self.assertTrue(all(part in NAMES or part == "shoot" for part in parts))

    def test_does_not_drive_into_bullet(self) -> None:
        # 正上方有一颗朝下飞的弹: 往上走是最差选择
        state = _state([(192.0, 360.0)], [(0.0, 2.0)])
        result = self.planner.plan(state)
        self.assertNotEqual(result.direction, 1)

    def test_velocity_length_mismatch_is_tolerated(self) -> None:
        state = _state([(192.0, 360.0), (100.0, 300.0)], [(0.0, 1.0)])
        result = self.planner.plan(state)
        self.assertTrue(result.action.endswith("shoot"))

    def test_static_ring_gap_escape(self) -> None:
        """隔离场景: 静态环留一个右上缺口, 规划器应能穿出 (允许贴边)."""
        from touhou_rl_mcp.config import Config
        from touhou_rl_mcp.env import TouhouEnv

        cfg = Config.load()
        cfg.patch({"backend": "mock", "game": "th11", "danmaku_pattern": "",
                   "step_frames": 6, "max_episode_frames": 0})
        env = TouhouEnv(cfg)
        env.start()
        try:
            state = env.reset(seed=1).state
            cx, cy = state.player_x, state.player_y
            gap = math.radians(-60.0)
            for i in range(16):
                angle = 2.0 * math.pi * i / 16
                delta = abs(((angle - gap + math.pi) % (2.0 * math.pi)) - math.pi)
                if delta < math.radians(30.0):
                    continue
                env.backend.game.spawn(cx + math.cos(angle) * 90.0,
                                       cy + math.sin(angle) * 90.0, 0.0, 0.0)
            state = env.get_state()
            planner = MacroPlanner(horizon=120, beam=16, escape_weight=0.3)
            farthest = 0.0
            for _ in range(20):
                state = env.step(planner.plan(state).action, frames=6).state
                farthest = max(farthest, math.hypot(state.player_x - cx, state.player_y - cy))
            self.assertGreater(farthest, 100.0)
        finally:
            env.close()


class PatternRegistryTest(unittest.TestCase):
    def test_closing_ring_registered(self) -> None:
        from touhou_rl_mcp.backends.patterns import PATTERN_DOCS, pattern_names

        self.assertIn("closing_ring", pattern_names())
        self.assertIn("closing_ring", PATTERN_DOCS)


if __name__ == "__main__":
    unittest.main()
