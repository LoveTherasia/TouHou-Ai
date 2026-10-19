"""模拟器后端 + 环境接口的单元测试."""

from __future__ import annotations

import unittest

from touhou_rl_mcp.config import Config
from touhou_rl_mcp.env import TouhouEnv
from touhou_rl_mcp.errors import NoActiveEpisodeError, OffsetUnavailableError
from touhou_rl_mcp.games.registry import GameRegistry


def make_env() -> TouhouEnv:
    config = Config({"backend": "mock", "game": "th11", "step_frames": 4})
    return TouhouEnv(config, registry=GameRegistry())


class TestMockEnv(unittest.TestCase):
    def setUp(self) -> None:
        self.env = make_env()
        self.env.start()
        self.addCleanup(self.env.close)

    def test_reset_returns_state(self) -> None:
        result = self.env.reset(seed=7)
        self.assertFalse(result.done)
        self.assertEqual(result.state.stage, 1)
        self.assertIsNotNone(result.state.lives)
        self.assertTrue(result.state.in_game)

    def test_seed_is_deterministic(self) -> None:
        a = self.env.reset(seed=99)
        for _ in range(40):
            self.env.step("shoot", frames=4)
        score_a = self.env.get_state().score

        self.env.reset(seed=99)
        for _ in range(40):
            self.env.step("shoot", frames=4)
        score_b = self.env.get_state().score
        self.assertEqual(score_a, score_b)
        self.assertIsNotNone(a)

    def test_step_returns_reward_and_frames(self) -> None:
        self.env.reset(seed=1)
        result = self.env.step("shoot", frames=10)
        self.assertEqual(result.info["frames"], 10)
        self.assertIn("reward_parts", result.info)
        self.assertIsInstance(result.reward, float)

    def test_step_without_episode_raises(self) -> None:
        env = make_env()
        env.start()
        self.addCleanup(env.close)
        with self.assertRaises(NoActiveEpisodeError):
            env.step("shoot")

    def test_auto_reset(self) -> None:
        self.env.reset(seed=2)
        result = self.env.step("shoot", frames=4, auto_reset=True)
        self.assertFalse(result.done)

    def test_move_to_macro_reaches_target(self) -> None:
        self.env.reset(seed=3)
        self.env.step("move_to", x=120, y=200, frames=200)
        state = self.env.get_state()
        self.assertAlmostEqual(state.player_x, 120, delta=12)
        self.assertAlmostEqual(state.player_y, 200, delta=12)

    def test_dodge_survives_longer_than_idle(self) -> None:
        self.env.reset(seed=11)
        for _ in range(150):
            result = self.env.step("dodge", frames=10, radius=40)
            if result.done:
                break
        survived = self.env.get_state().frame

        self.env.reset(seed=11)
        for _ in range(150):
            result = self.env.step("noop", frames=10)
            if result.done:
                break
        idle = self.env.get_state().frame
        self.assertGreater(survived, idle)

    def test_episode_ends_and_stats_recorded(self) -> None:
        self.env.reset(seed=5)
        done = False
        for _ in range(400):
            result = self.env.step("noop", frames=8)
            if result.done or result.truncated:
                done = True
                break
        self.assertTrue(done, "一局应该在 400*8 帧内结束 (站桩被弹)")
        self.assertEqual(len(self.env.stats.episodes), 1)
        self.assertTrue(self.env.stats.episodes[0]["game_over"])
        with self.assertRaises(NoActiveEpisodeError):
            self.env.step("shoot")

    def test_reward_weights_are_configurable(self) -> None:
        self.env.reset(seed=6)
        self.env.configure({"reward": {"shaping_scale": 0.0}})
        result = self.env.step("shoot", frames=10)
        self.assertAlmostEqual(result.reward, 0.0, places=6)

    def test_visualize_and_screenshot(self) -> None:
        self.env.reset(seed=8)
        image = self.env.visualize()
        self.assertGreater(image.width, 300)
        self.assertGreater(image.height, 300)

        shot, method = self.env.screenshot()
        self.assertEqual(method, "mock-render")
        self.assertEqual(shot.size, (640, 480))

    def test_missing_offset_error_message_is_actionable(self) -> None:
        with self.assertRaises(OffsetUnavailableError) as ctx:
            raise OffsetUnavailableError("player_x", game="th11")
        message = str(ctx.exception)
        self.assertIn("th11", message)
        self.assertIn("memory_scan", message)


if __name__ == "__main__":
    unittest.main()
