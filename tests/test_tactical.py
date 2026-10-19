import unittest
from touhou_rl_mcp.types import GameState
from touhou_rl_mcp.training.tactical import TacticalGuard


class TestTactical(unittest.TestCase):
    def test_empty_field_stays_and_shoots(self):
        guard = TacticalGuard()
        action, detail = guard.choose(GameState(player_x=192, player_y=395), 0, "shoot")
        self.assertEqual(action, "shoot+focus")
        self.assertEqual(detail["predicted_clearance"], 9999)

    def test_incoming_bullet_moves_off_axis(self):
        guard = TacticalGuard()
        state = GameState(in_game=True, player_x=192, player_y=395,
            bullets=[(192,350)], raw={"entities": {"bullets": {"status":"ok", "active_slots":[7]},
                                                 "enemies": {"status":"ok"}}})
        guard.choose(state, 10, "shoot")
        state.bullets = [(192,362)]
        action, _ = guard.choose(state, 14, "shoot")
        self.assertTrue("left" in action or "right" in action)
        self.assertIn("shoot+focus", action)

    def test_reused_slot_does_not_create_extreme_velocity(self):
        guard = TacticalGuard()
        state = GameState(player_x=192, player_y=395, bullets=[(192,20)])
        action, detail = guard.choose(state, 0, "shoot")
        self.assertGreater(detail["predicted_clearance"], 300)
