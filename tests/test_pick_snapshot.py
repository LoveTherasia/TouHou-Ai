"""`tools/pick_snapshot.py` 的打分与扫描逻辑（不跑真实评测）。"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.pick_snapshot import find_snapshots, score_of  # noqa: E402


class TestPickSnapshot(unittest.TestCase):
    def test_greedy_snapshot_scores_lower_than_balanced(self) -> None:
        """存活高但几乎不打（756/161）必须排在 792/454 之后。

        这是《PLANNER_V2.md》§17.3 那个真实例子：`policy_best` 会把它当最优，
        联合打分要把它压下去。
        """
        greedy = {"mean_steps": 756, "mean_damage": 161}
        balanced = {"mean_steps": 792, "mean_damage": 454}
        self.assertGreater(
            score_of(balanced, 1200, 600.0),
            score_of(greedy, 1200, 600.0),
        )

    def test_score_is_normalised(self) -> None:
        """满存活 + 满伤害时分数为 1；全 0 时为 0。"""
        self.assertAlmostEqual(
            score_of({"mean_steps": 1200, "mean_damage": 600.0}, 1200, 600.0), 1.0)
        self.assertAlmostEqual(
            score_of({"mean_steps": 0, "mean_damage": 0.0}, 1200, 600.0), 0.0)

    def test_zero_damage_ref_does_not_divide_by_zero(self) -> None:
        """所有快照伤害都是 0 时不能崩（除零保护）。"""
        val = score_of({"mean_steps": 600, "mean_damage": 0.0}, 1200, 0.0)
        self.assertGreaterEqual(val, 0.0)

    def test_find_snapshots_in_subdir_and_flat(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            # snapshots/ 子目录
            (base / "snapshots").mkdir()
            (base / "snapshots" / "policy_0100.npz").write_bytes(b"x")
            found = find_snapshots(base)
            self.assertEqual([p.name for p in found], ["policy_0100.npz"])
            # 平铺（只有平铺时才会用到）
            (base / "snapshots" / "policy_0100.npz").unlink()
            (base / "policy_0200.npz").write_bytes(b"x")
            found = find_snapshots(base)
            self.assertEqual([p.name for p in found], ["policy_0200.npz"])

    def test_find_snapshots_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(find_snapshots(Path(tmp)), [])


if __name__ == "__main__":
    unittest.main()
