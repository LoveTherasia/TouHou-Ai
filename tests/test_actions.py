"""动作解析的单元测试."""

from __future__ import annotations

import unittest

from touhou_rl_mcp.inputs.actions import Action, parse, parse_sequence


class TestParse(unittest.TestCase):
    def test_plain_direction(self) -> None:
        action = parse("right")
        self.assertEqual(action.buttons, ("right",))
        self.assertFalse(action.is_macro)

    def test_combo(self) -> None:
        for text in ("right+shoot", "right,shoot", "right shoot", "right;shoot"):
            action = parse(text)
            self.assertEqual(set(action.buttons), {"right", "shoot"}, text)

    def test_aliases(self) -> None:
        action = parse("z")
        self.assertEqual(action.buttons, ("confirm",))

    def test_diagonal(self) -> None:
        action = parse("up_left")
        self.assertEqual(set(action.buttons), {"up", "left"})
        action = parse("down-left")
        self.assertEqual(set(action.buttons), {"down", "left"})

    def test_verbs(self) -> None:
        self.assertEqual(parse("noop").buttons, ())
        self.assertEqual(parse("focus_shoot").buttons, ("focus", "shoot"))
        self.assertEqual(parse("bomb").buttons, ("bomb",))

    def test_macros(self) -> None:
        action = parse("move_to x=100 y=200", default_frames=30)
        self.assertTrue(action.is_macro)
        self.assertEqual(action.name, "move_to")
        self.assertEqual(action.params["x"], 100)
        self.assertEqual(action.params["y"], 200)
        self.assertEqual(action.frames, 30)

    def test_macro_from_dict(self) -> None:
        action = parse({"macro": "dodge", "frames": 45, "radius": 20})
        self.assertTrue(action.is_macro)
        self.assertEqual(action.frames, 45)
        self.assertEqual(action.params["radius"], 20)

    def test_dict_buttons(self) -> None:
        action = parse({"buttons": ["up", "shoot"], "frames": 12})
        self.assertEqual(set(action.buttons), {"up", "shoot"})
        self.assertEqual(action.frames, 12)

    def test_unknown_raises(self) -> None:
        with self.assertRaises(ValueError):
            parse("teleport")

    def test_sequence(self) -> None:
        steps = parse_sequence(
            [{"buttons": ["confirm"], "frames": 8, "after": 20}, "down"]
        )
        self.assertEqual(len(steps), 2)
        self.assertEqual(steps[0].params["after"], 20)
        self.assertEqual(steps[1].buttons, ("down",))

    def test_none_and_empty(self) -> None:
        self.assertEqual(parse(None).name, "noop")
        self.assertEqual(parse("").name, "noop")


if __name__ == "__main__":
    unittest.main()
