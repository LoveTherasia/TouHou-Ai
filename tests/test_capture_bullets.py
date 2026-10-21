"""视觉弹幕检测的回归测试 (capture/bullets.py + memory 后端的接入点).

这些用例在**合成图**上跑, 不需要游戏窗口, 所以能在 CI 里保护:
  * 亮度/饱和度/尺寸三重过滤是否真的把背景和道具滤掉;
  * 像素坐标是否被正确换算成 384x448 的**场地坐标**
    (与 memory/entities.py 的实体坐标同一个系, 否则 dodge/危险度会错位);
  * ``vision.enabled`` 的开关语义: 默认关闭、内存有弹幕时不截图、截图失败不抛错。
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image, ImageDraw

from touhou_rl_mcp.backends.memory import MemoryBackend
from touhou_rl_mcp.capture.bullets import (
    bullets_from_frame,
    connected_components,
    detect_bullets,
)
from touhou_rl_mcp.config import Config
from touhou_rl_mcp.games.registry import GameSpec
from touhou_rl_mcp.types import GameState

PLAYFIELD = [32, 16, 416, 464]
WINDOW = [640, 480]
#: 窗口像素坐标下的三颗弹 (对应场地坐标要减去 PLAYFIELD 左上角)
BULLET_PIXELS = [(70, 100), (170, 250), (270, 400)]


def make_spec(playfield: list[int] | None = None, window_size: list[int] | None = None):
    return SimpleNamespace(
        playfield=playfield if playfield is not None else list(PLAYFIELD),
        window_size=window_size if window_size is not None else list(WINDOW),
    )


def build_frame() -> Image.Image:
    """640x480 合成画面: 3 颗真弹 + 1 块过大的亮色 + 1 块不饱和的亮块 + 场地外亮块."""
    image = Image.new("RGB", (WINDOW[0], WINDOW[1]), (0, 0, 0))
    draw = ImageDraw.Draw(image)
    for x, y in BULLET_PIXELS:
        draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(255, 0, 0))
    # 大块亮色 (背景花纹/敌机) -> 面积超限
    draw.rectangle([200, 200, 300, 300], fill=(0, 255, 0))
    # 亮但不饱和 (UI 文字/道具) -> 饱和度不足
    draw.rectangle([330, 380, 338, 388], fill=(210, 210, 210))
    # 场地外的亮块 (侧边 UI) -> 裁剪后不该出现
    draw.ellipse([597, 447, 603, 453], fill=(255, 255, 0))
    return image


class TestConnectedComponents(unittest.TestCase):
    def test_two_separated_blobs(self) -> None:
        import numpy as np

        mask = np.zeros((10, 10), dtype=bool)
        mask[1:3, 1:3] = True
        mask[6:9, 5:9] = True
        labels, count = connected_components(mask)
        self.assertEqual(count, 2)
        self.assertEqual(len({int(v) for v in labels[mask]}), 2)


class TestDetectOnSyntheticFrame(unittest.TestCase):
    def setUp(self) -> None:
        self.image = build_frame()

    def test_detects_only_bullets(self) -> None:
        bullets = bullets_from_frame(self.image, make_spec())
        self.assertEqual(len(bullets), 3)
        for (bx, by), (wx, wy) in zip(sorted(bullets), sorted(BULLET_PIXELS)):
            self.assertAlmostEqual(bx, wx - PLAYFIELD[0], delta=1.0)
            self.assertAlmostEqual(by, wy - PLAYFIELD[1], delta=1.0)

    def test_scaled_window(self) -> None:
        """窗口被放大到 960x720 时, 坐标必须仍然回到 384x448 的场地系."""
        scale = 1.5
        big = Image.new("RGB", (int(WINDOW[0] * scale), int(WINDOW[1] * scale)), (0, 0, 0))
        draw = ImageDraw.Draw(big)
        # 场地坐标 (100, 200) -> 裁剪图内部 (150, 300) -> 窗口像素 (48+150, 24+300)
        draw.ellipse([48 + 150 - 4, 24 + 300 - 4, 48 + 150 + 4, 24 + 300 + 4], fill=(255, 0, 0))
        bullets = bullets_from_frame(big, make_spec())
        self.assertEqual(len(bullets), 1)
        self.assertAlmostEqual(bullets[0][0], 100.0, delta=1.5)
        self.assertAlmostEqual(bullets[0][1], 200.0, delta=1.5)

    def test_black_frame_returns_empty(self) -> None:
        blank = Image.new("RGB", (WINDOW[0], WINDOW[1]), (0, 0, 0))
        self.assertEqual(bullets_from_frame(blank, make_spec()), [])

    def test_crop_excludes_ui_outside_playfield(self) -> None:
        """场地外的亮块 (侧边 UI/道具栏) 不该被当成弹幕, 坐标也不该跑出场地."""
        cropped_image = self.image.crop(
            (PLAYFIELD[0], PLAYFIELD[1], PLAYFIELD[2], PLAYFIELD[3])
        )
        raw = detect_bullets(
            cropped_image, origin=(0.0, 0.0), crop_scale=1.0,
            bright_thr=170, sat_thr=40, min_area=2, max_area=300, max_dim=40,
        )
        auto = bullets_from_frame(
            self.image, make_spec(), bright_thr=170, sat_thr=40,
            min_area=2, max_area=300, max_dim=40,
        )
        self.assertEqual(len(auto), 3)
        self.assertEqual(len(raw), 3)
        for (rx, ry), (ax, ay) in zip(sorted(raw), sorted(auto)):
            self.assertAlmostEqual(rx, ax, delta=1.0)
            self.assertAlmostEqual(ry, ay, delta=1.0)
        for x, y in auto:
            self.assertTrue(0.0 <= x <= 384.0 and 0.0 <= y <= 448.0, (x, y))


class TestVisionFallbackWiring(unittest.TestCase):
    """``MemoryBackend._maybe_vision_bullets`` 的开关语义 (不需要真游戏)."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.image = build_frame()

    def _backend(self, enabled: bool, entities: dict | None = None) -> MemoryBackend:
        spec = GameSpec(key="th06", name_zh="东方红魔乡", entities=entities or {})
        config = Config(
            {"vision": {"enabled": enabled}},
            path=Path(self._tmp.name) / "config.json",
        )
        backend = MemoryBackend(spec, config)
        backend.screenshot = lambda: (self.image, "fake")  # type: ignore[method-assign]
        return backend

    def test_disabled_by_default(self) -> None:
        backend = self._backend(False)
        state = GameState()
        backend._maybe_vision_bullets(state)
        self.assertEqual(state.bullets, [])
        self.assertNotIn("bullets_source", state.raw)

    def test_enabled_fills_playfield_coordinates(self) -> None:
        backend = self._backend(True)
        state = GameState()
        backend._maybe_vision_bullets(state)
        self.assertEqual(len(state.bullets), 3)
        self.assertEqual(state.bullet_count, 3)
        self.assertEqual(state.raw["bullets_source"], "vision")
        self.assertEqual(state.raw["vision_method"], "fake")
        self.assertAlmostEqual(state.bullets[0][0], BULLET_PIXELS[0][0] - PLAYFIELD[0], delta=1.0)

    def test_memory_source_ok_skips_screenshot(self) -> None:
        backend = self._backend(True, {"bullets": {"profile": "th11"}})
        called: list[int] = []

        def spy():
            called.append(1)
            return self.image, "fake"

        backend.screenshot = spy  # type: ignore[method-assign]
        state = GameState(raw={"entities": {"bullets": {"status": "ok"}}})
        backend._maybe_vision_bullets(state)
        self.assertEqual(called, [], "内存已有可靠弹幕时不该再截图")
        self.assertNotIn("bullets_source", state.raw)

    def test_screenshot_failure_is_recorded(self) -> None:
        backend = self._backend(True)

        def boom():
            raise RuntimeError("找不到游戏窗口")

        backend.screenshot = boom  # type: ignore[method-assign]
        state = GameState()
        backend._maybe_vision_bullets(state)  # 不该抛
        self.assertIn("vision_error", state.raw)
        self.assertIn("找不到游戏窗口", state.raw["vision_error"])
        self.assertEqual(state.bullets, [])


if __name__ == "__main__":
    unittest.main()
