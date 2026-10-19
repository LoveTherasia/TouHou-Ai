"""训练看板 (dashboard) 的契约测试。

固定三件事:
  1. run 的发现规则 —— 只有含 ``train.jsonl`` 的目录才算;
  2. 指标自动发现 —— 数值字段会被扫出来, 但动作直方图不进曲线;
  3. HTTP 接口可用且**不能**读 run 目录以外的文件。
"""

from __future__ import annotations

import json
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory

from touhou_rl_mcp.dashboard.registry import (
    discover_runs, load_run, numeric_metrics, read_episodes,
)
from touhou_rl_mcp.dashboard.server import DashboardState, make_handler


def _write_run(root: Path, name: str, episodes: list[dict], *, eval_rows=None) -> Path:
    run_dir = root / name
    run_dir.mkdir(parents=True)
    with open(run_dir / "train.jsonl", "w", encoding="utf-8") as fh:
        for row in episodes:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    if eval_rows:
        with open(run_dir / "eval.jsonl", "w", encoding="utf-8") as fh:
            for row in eval_rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    (run_dir / "meta.json").write_text(json.dumps({"script": "test"}), encoding="utf-8")
    return run_dir


EPISODES = [
    {"episode": 1, "return": 100.0, "steps": 20, "max_stage": 1, "misses": 1,
     "actions": {"shoot": 10, "left+shoot": 5}},
    {"episode": 2, "return": 160.0, "steps": 32, "max_stage": 2, "misses": 0,
     "reward_breakdown": {"damage": 3.0, "graze": 1.5},
     "actions": {"shoot": 12, "right+shoot": 8}},
]


class RegistryTest(unittest.TestCase):
    def test_discovers_only_dirs_with_train_jsonl(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_run(root, "alpha", EPISODES)
            (root / "empty_dir").mkdir()
            runs = discover_runs([root])
            self.assertEqual([r.name for r in runs], ["alpha"])
            self.assertEqual(runs[0].episodes, 2)
            self.assertTrue(runs[0].meta.get("script"))

            # 嵌套一层也要能找到 (runs/<name>)
            nested = _write_run(root / "runs", "beta", EPISODES)
            names = {r.name for r in discover_runs([root])}
            self.assertEqual(names, {"alpha", "beta"})
            self.assertTrue(nested.is_dir())

    def test_incremental_read(self) -> None:
        with TemporaryDirectory() as tmp:
            run_dir = _write_run(Path(tmp), "alpha", EPISODES)
            self.assertEqual(len(read_episodes(run_dir, since=0)), 2)
            self.assertEqual(len(read_episodes(run_dir, since=1)), 1)
            self.assertEqual(read_episodes(run_dir, since=5), [])

    def test_metrics_exclude_action_histogram(self) -> None:
        metrics = numeric_metrics(EPISODES)
        self.assertIn("return", metrics)
        self.assertIn("reward_breakdown.graze", metrics)   # 嵌套字典会被拍平
        self.assertFalse([m for m in metrics if m.startswith("actions.")])
        self.assertNotIn("episode", metrics)

    def test_eval_is_merged_into_episode(self) -> None:
        with TemporaryDirectory() as tmp:
            run_dir = _write_run(Path(tmp), "alpha", EPISODES,
                                 eval_rows=[{"step": 2, "name": "heldout", "score": 123.0}])
            payload = load_run(run_dir)
            self.assertEqual(payload["episodes"][1]["eval"], 123.0)
            self.assertIn("eval", payload["metrics"])
            self.assertEqual(payload["info"]["eval"][0]["score"], 123.0)

    def test_action_totals(self) -> None:
        with TemporaryDirectory() as tmp:
            run_dir = _write_run(Path(tmp), "alpha", EPISODES)
            actions = load_run(run_dir)["actions"]
            self.assertEqual(actions["shoot"], 22)
            self.assertEqual(actions["left+shoot"], 5)


class HttpTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        _write_run(cls.root, "alpha", EPISODES,
                   eval_rows=[{"step": 1, "name": "heldout", "score": 90.0}])
        (cls.root / "alpha" / "progress.png").write_bytes(b"\x89PNG\r\n\x1a\n")
        (cls.root / "alpha" / "evil.py").write_text("print('x')", encoding="utf-8")
        (cls.root / "secret.txt").write_text("TOP SECRET", encoding="utf-8")
        cls.state = DashboardState([str(cls.root)])
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(cls.state))
        cls.port = cls.httpd.server_address[1]
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.tmp.cleanup()

    def _get(self, path: str):
        url = f"http://127.0.0.1:{self.port}{path}"
        with urllib.request.urlopen(url, timeout=5) as resp:
            return resp.status, resp.read()

    def test_index_and_static(self) -> None:
        status, body = self._get("/")
        self.assertEqual(status, 200)
        self.assertIn(b"<canvas", body)
        status, body = self._get("/static/app.js")
        self.assertEqual(status, 200)
        self.assertIn(b"fetch", body)

    def test_runs_and_run_endpoints(self) -> None:
        status, body = self._get("/api/runs")
        self.assertEqual(status, 200)
        runs = json.loads(body)["runs"]
        self.assertEqual(runs[0]["name"], "alpha")
        self.assertTrue(runs[0]["has_eval"])

        status, body = self._get("/api/run?id=alpha&since=1")
        payload = json.loads(body)
        self.assertEqual(status, 200)
        self.assertEqual(len(payload["episodes"]), 1)
        self.assertEqual(payload["total"], 2)
        self.assertIn("reward_breakdown.damage", payload["metrics"])

    def test_asset_whitelist(self) -> None:
        status, body = self._get("/api/asset?id=alpha&name=progress.png")
        self.assertEqual(status, 200)
        self.assertTrue(body.startswith(b"\x89PNG"))

    def test_asset_traversal_blocked(self) -> None:
        # run 目录之外的文件必须读不到 (即便后缀在白名单里)
        for name in ("../secret.txt", "..%2Fsecret.txt", "../../secret.txt"):
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                self._get(f"/api/asset?id=alpha&name={name}")
            self.assertEqual(ctx.exception.code, 403)

    def test_asset_suffix_whitelist(self) -> None:
        # 目录内也不放行任意后缀 (防止把脚本/可执行文件当资源发出去)
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self._get("/api/asset?id=alpha&name=evil.py")
        self.assertEqual(ctx.exception.code, 404)

    def test_unknown_run_is_404(self) -> None:
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self._get("/api/run?id=nope")
        self.assertEqual(ctx.exception.code, 404)


if __name__ == "__main__":
    unittest.main()
