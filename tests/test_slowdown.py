"""时间缩放钩子的 Python 侧测试.

* 纯 Python 逻辑 (因子校验 / 配置写入 / 可用性) 直接测;
* 若已构建 native/speedhack/bin, 再跑一次真实的"注入 + 计时"集成测试。
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

from touhou_rl_mcp.errors import TouhouError
from touhou_rl_mcp.memory.slowdown import (
    SpeedHook,
    default_binary_dir,
    speed_config_path,
)

BIN_DIR = default_binary_dir()
HAS_BINARIES = (BIN_DIR / "speedhack32.dll").exists() and (BIN_DIR / "injector32.exe").exists()
HAS_TIMER = (BIN_DIR / "timer_test32.exe").exists()


class TestSpeedHookLogic(unittest.TestCase):
    def test_missing_binaries_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            hook = SpeedHook(binary_dir=tmp)
            self.assertFalse(hook.available())
            self.assertEqual(len(hook.missing()), 2)
            with self.assertRaises(TouhouError):
                hook.apply(1234, 4.0)

    def test_write_factor_and_validation(self) -> None:
        hook = SpeedHook(binary_dir=tempfile.gettempdir())
        try:
            hook.write_factor(4.0)
            self.assertEqual(speed_config_path().read_text(encoding="ascii").strip(), "4.000")
            with self.assertRaises(TouhouError):
                hook.write_factor(0.0)
            with self.assertRaises(TouhouError):
                hook.write_factor(1000.0)
        finally:
            speed_config_path().write_text("1.000", encoding="ascii")

    def test_status_shape(self) -> None:
        hook = SpeedHook(binary_dir=tempfile.gettempdir())
        status = hook.status()
        self.assertIn("available", status)
        self.assertIn("config", status)
        self.assertIn("binary_dir", status)


@unittest.skipUnless(HAS_BINARIES and HAS_TIMER, "需要先构建 native/speedhack/bin")
class TestSpeedHookIntegration(unittest.TestCase):
    """真实注入 32 位自检程序, 验证 timeGetTime 被缩放。"""

    def _run_timer(self, factor: float, real_ms: int = 2500, inject_after_ms: int = 400) -> float:
        hook = SpeedHook(binary_dir=BIN_DIR)
        out_path = Path(tempfile.gettempdir()) / "opencode_tt.out"
        out_path.unlink(missing_ok=True)
        with out_path.open("w") as fh:
            proc = subprocess.Popen([str(BIN_DIR / "timer_test32.exe"), str(real_ms)], stdout=fh)
        try:
            time.sleep(inject_after_ms / 1000.0)
            hook.apply(proc.pid, factor)
            proc.wait(timeout=real_ms / 1000.0 + 10)
        finally:
            if proc.poll() is None:
                proc.kill()
        text = out_path.read_text(encoding="utf-8", errors="replace")
        ratio = None
        for token in text.split():
            if token.startswith("ratio="):
                ratio = float(token.split("=", 1)[1])
        self.assertIsNotNone(ratio, f"没解析到 ratio: {text!r}")
        return ratio

    def test_speedup(self) -> None:
        ratio = self._run_timer(4.0)
        # 注入前那 ~400ms 未缩放, 会把 ratio 拉低 (理论上 ~3.6); 只要明显 > 1 即证明钩子生效
        self.assertGreater(ratio, 2.0)

    def test_transparent_at_one(self) -> None:
        ratio = self._run_timer(1.0)
        self.assertAlmostEqual(ratio, 1.0, delta=0.2)


if __name__ == "__main__":
    unittest.main()
